use std::{collections::HashMap, hash::Hash};
use std::fmt::Write;

use crate::{Number, Value, ValueInner, Op, Module, MLP, ValueId};

#[derive(Debug)]
pub struct Program {
    heap: Vec<Number>,
    lib: libloading::Library,
    exec_mlp: extern "C" fn(*mut f64, f64) -> f64,
    get_scores: extern "C" fn(*const f64, *mut f64),
    get_parameters: extern "C" fn(*const f64, *mut [f64; 2]),
    set_inputs: extern "C" fn(*mut f64, *const f64),
    param_ptr:   usize,
    param_count: usize,
    score_ptr:   usize,
    score_count: usize,
    total_loss_ptr: usize,
}

#[derive(Debug)]
struct Codegen {
    heap: Vec<Number>,
    code: String,
    value_map: HashMap<ValueId, usize>,
}

#[repr(transparent)]
#[derive(Debug, PartialEq)]
struct HashableNumber(Number);

impl Eq for HashableNumber {}

impl Hash for HashableNumber {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_ne_bytes().hash(state)
    }
}

impl Codegen {
    fn get_heap_ptr(&mut self, node: &Value) -> usize {
        if let Some(heap_ptr) = self.value_map.get(&node.id()) {
            return *heap_ptr;
        }

        let heap_ptr = self.heap.len();
        self.value_map.insert(node.id(), heap_ptr);
        // space for value and grad
        self.heap.resize(heap_ptr + 2, 0.0);

        heap_ptr
    }

    fn forward(&mut self, node: &Value) {
        let out: &mut ValueInner = &mut node.inner.borrow_mut();

        if !out.visited {
            out.visited = true;

            let out_ptr = self.get_heap_ptr(node);

            match &mut out.op {
                Op::Value => {
                    self.heap[out_ptr] = out.value;
                    write!(&mut self.code,
                        "    heap[{:?}] = {:?};\n",
                        out_ptr, out.value);
                },
                Op::Add(lhs, rhs) => {
                    let lhs_ptr = self.get_heap_ptr(lhs);
                    let rhs_ptr = self.get_heap_ptr(rhs);

                    self.forward(lhs);
                    self.forward(rhs);

                    write!(&mut self.code,
                        "    heap[{:?}] = heap[{:?}] + heap[{:?}];\n",
                        out_ptr, lhs_ptr, rhs_ptr);
                },
                Op::Mul(lhs, rhs) => {
                    let lhs_ptr = self.get_heap_ptr(lhs);
                    let rhs_ptr = self.get_heap_ptr(rhs);

                    self.forward(lhs);
                    self.forward(rhs);

                    write!(&mut self.code,
                        "    heap[{:?}] = heap[{:?}] * heap[{:?}];\n",
                        out_ptr, lhs_ptr, rhs_ptr);
                },
                Op::Pow(lhs, rhs) => {
                    let lhs_ptr = self.get_heap_ptr(lhs);
                    let rhs = *rhs;

                    self.forward(lhs);

                    write!(&mut self.code,
                        "    heap[{:?}] = pow(heap[{:?}] * {:?});\n",
                        out_ptr, lhs_ptr, rhs);
                },
                Op::ReLu(arg) => {
                    let arg_ptr = self.get_heap_ptr(arg);

                    self.forward(arg);

                    write!(&mut self.code,
                        "    tmp = heap[{:?}]\n", arg_ptr
                    );
                    write!(&mut self.code,
                        "    heap[{:?}] = tmp < 0.0 ? 0.0 : tmp;\n",
                        out_ptr);
                },
                Op::TanH(arg) => {
                    let arg_ptr = self.get_heap_ptr(arg);

                    self.forward(arg);

                    write!(&mut self.code,
                        "    heap[{:?}] = tanh(heap[{:?}]);\n",
                        out_ptr, arg_ptr);
                },
                Op::Exp(arg) => {
                    let arg_ptr = self.get_heap_ptr(arg);

                    self.forward(arg);

                    write!(&mut self.code,
                        "    heap[{:?}] = exp(heap[{:?}]);\n",
                        out_ptr, arg_ptr);
                },
            }
        }
    }

    fn backward(&mut self, node: &Value) {
        let mut topo = Vec::new();
        node.build_topo(&mut topo);

        for node in topo.iter().rev() {
            let out: &ValueInner = &node.inner.borrow();

            let out_value_ptr = self.get_heap_ptr(node);
            let out_grad_ptr = out_value_ptr + 1;

            match &out.op {
                Op::Value => {},
                Op::Add(lhs, rhs) => {
                    let lhs_ptr = self.get_heap_ptr(lhs);
                    let rhs_ptr = self.get_heap_ptr(rhs);

                    write!(&mut self.code, "    tmp = heap[{:?}];\n", out_grad_ptr);
                    write!(&mut self.code, "    heap[{:?}] += tmp;\n", lhs_ptr + 1);
                    write!(&mut self.code, "    heap[{:?}] += tmp;\n", rhs_ptr + 1);
                },
                Op::Mul(lhs, rhs) => {
                    let lhs_ptr = self.get_heap_ptr(lhs);
                    let rhs_ptr = self.get_heap_ptr(rhs);

                    write!(&mut self.code, "    tmp = heap[{:?}];\n", out_grad_ptr);
                    write!(&mut self.code, "    heap[{:?}] += heap[{:?}] * tmp;\n", lhs_ptr + 1, rhs_ptr);
                    write!(&mut self.code, "    heap[{:?}] += heap[{:?}] * tmp;\n", rhs_ptr + 1, lhs_ptr);
                },
                Op::Pow(lhs, rhs) => {
                    let lhs_ptr = self.get_heap_ptr(lhs);
                    let rhs = *rhs;

                    write!(&mut self.code, "    heap[{:?}] += {:?} * pow(heap[{:?}], {:?}) * heap[{:?}];\n",
                        lhs_ptr + 1, rhs, lhs_ptr, rhs - 1.0, out_grad_ptr);
                },
                Op::ReLu(arg) => {
                    let arg_ptr = self.get_heap_ptr(arg);

                    write!(&mut self.code, "    heap[{:?}] += (heap[{:?}] > 0.0) * heap[{:?}];\n",
                        arg_ptr + 1, out_value_ptr, out_grad_ptr);
                },
                Op::TanH(arg) => {
                    let arg_ptr = self.get_heap_ptr(arg);

                    write!(&mut self.code, "    tmp = heap[{:?}];\n", out_value_ptr);
                    write!(&mut self.code, "    heap[{:?}] += (1.0 - tmp * tmp) * heap[{:?}];\n",
                        arg_ptr + 1, out_grad_ptr);
                },
                Op::Exp(arg) => {
                    let arg_ptr = self.get_heap_ptr(arg);

                    write!(&mut self.code, "    heap[{:?}] += heap[{:?}] * heap[{:?}];\n",
                        arg_ptr + 1, out_value_ptr, out_grad_ptr);
                },
            }
        }
    }
}

impl Program {
    pub fn compile_model(model: &MLP, scores: &[Value], total_loss: &Value, inputs: &[Value]) -> std::io::Result<Self> {
        let param_count = model.count_parameters();

        let mut value_map = HashMap::new();
        let mut heap = vec![0.0; 2 * (param_count + scores.len() + 1)];

        let mut heap_ptr = 0;

        model.for_each_parameter(|node| {
            value_map.insert(node.id(), heap_ptr);
            heap[heap_ptr] = node.value();
            heap_ptr += 2;
        });

        Program::compile_intern(heap, value_map, param_count, scores, total_loss, inputs)
    }

    pub fn compile_components(parameters: &[Value], scores: &[Value], total_loss: &Value, inputs: &[Value]) -> std::io::Result<Self> {
        let mut value_map = HashMap::new();
        let mut heap = vec![0.0; 2 * (parameters.len() + scores.len() + 1)];

        let mut heap_ptr = 0;

        let param_count = parameters.len();

        for node in parameters {
            value_map.insert(node.id(), heap_ptr);
            heap[heap_ptr] = node.value();
            heap_ptr += 2;
        }

        Program::compile_intern(heap, value_map, param_count, scores, total_loss, inputs)
    }

    fn compile_intern(
        mut heap: Vec<Number>,
        mut value_map: HashMap<ValueId, usize>,
        param_count: usize,
        scores: &[Value],
        total_loss: &Value,
        inputs: &[Value],
    ) -> std::io::Result<Self> {
        let param_ptr = 0;
        let mut heap_ptr = param_count * 2;

        let score_ptr = heap_ptr;
        let score_count = scores.len();

        for node in scores {
            value_map.insert(node.id(), heap_ptr);
            heap_ptr += 2;
        }

        let total_loss_ptr = heap_ptr;
        value_map.insert(total_loss.id(), total_loss_ptr);
        let (total_loss_value, total_loss_grad) = total_loss.get();
        heap[total_loss_ptr] = total_loss_value;
        heap[total_loss_ptr + 1] = total_loss_grad;

        let mut codegen = Codegen {
            code: String::new(),
            value_map,
            heap,
        };

        codegen.code.push_str("\
double exp(double x);
double tanh(double x);
double pow(double x, double y);

double exec_mlp(double *heap, double learning_rate) {
    double tmp;
");

        codegen.forward(total_loss);
        total_loss.clear_visited();

        write!(&mut codegen.code,
            "    heap[{:?}] = 1.0;\n",
            total_loss_ptr);

        codegen.backward(total_loss);
        total_loss.clear_visited();

        for param_ptr in (param_ptr..(param_ptr + param_count * 2)).step_by(2) {
            write!(&mut codegen.code,
                "    heap[{:?}] -= learning_rate * heap[{:?}];\n",
                param_ptr, param_ptr + 1);
        }

        codegen.code.push_str("\
}

void get_scores(const double *heap, double *scores) {
");
        // TODO

        codegen.code.push_str("\
}

void get_parameters(const double *heap, double *parameters) {
");
        // TODO

        codegen.code.push_str("\
}

void set_inputs(double *heap, const double *inputs) {
");
        // TODO

        codegen.code.push_str("\
}
");

        std::fs::write("/tmp/mlp.c", &codegen.code)?;

        // TODO: better temp names
        let mut proc = std::process::Command::new("gcc").args([
            "-O2", "-lm", "-fpic", "-shared", "/tmp/mlp.c", "/tmp/mlp.so"
        ]).spawn()?;
        proc.wait()?;

        // TODO: load shared object

        // TODO: error handling!
        let lib = unsafe { libloading::Library::new("/tmp/mlp.so").expect("failed loading library") };

        let exec_mlp: libloading::Symbol<extern "C" fn(*mut f64, f64) -> f64> = unsafe { lib.get(b"exec_mlp").expect("failed loading exec_mlp") };
todo!();
/*
        Ok(Self {
            heap: codegen.heap,
            // TODO
            lib,
            param_ptr,
            param_count,
            score_ptr,
            score_count,
            total_loss_ptr,
        })
*/
    }

    pub fn exec(&mut self, learning_rate: Number) -> Number {
        todo!();
    }

    #[inline]
    pub fn total_loss(&self) -> Number {
        self.heap[self.total_loss_ptr]
    }

    #[inline]
    pub fn scores(&self) -> Vec<Number> {
        let mut scores = Vec::with_capacity(self.score_count);
        self.get_scores(&mut scores);
        scores
    }

    #[inline]
    pub fn get_scores(&self, scores: &mut Vec<Number>) {
        for heap_ptr in (self.score_ptr..(self.score_ptr + self.score_count * 2)).step_by(2) {
            scores.push(self.heap[heap_ptr]);
        }
    }

    #[inline]
    pub fn parameters(&self) -> Vec<Number> {
        let mut parameters = Vec::with_capacity(self.param_count);
        self.get_parameters(&mut parameters);
        parameters
    }

    #[inline]
    pub fn get_parameters(&self, parameters: &mut Vec<Number>) {
        for heap_ptr in (self.param_ptr..(self.param_ptr + self.param_count * 2)).step_by(2) {
            parameters.push(self.heap[heap_ptr]);
        }
    }

    /// Copy parameters back into model.
    #[inline]
    pub fn get_model(&self, model: &mut MLP) {
        let mut offset = self.param_ptr;
        let offset_end = self.param_ptr + self.param_count * 2;
        model.for_each_parameter_mut(|value| {
            if offset < offset_end {
                value.assign_both(self.heap[offset], self.heap[offset + 1]);
            }
            offset += 2;
        });
    }

    #[inline]
    pub fn heap(&self) -> &[Number] {
        &self.heap
    }
}
