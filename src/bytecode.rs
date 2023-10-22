use std::{collections::HashMap, hash::Hash};

use crate::{Number, Value, ValueInner, Op};

// node values and grads are paired up in heap (value, grad)

#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Bytecode {
                   // arguments

    Add,           // lhs value ptr, rhs value ptr, out value ptr
    Mul,           // lhs value ptr, rhs value ptr, out value ptr
    Pow,           // lhs value ptr, rhs value ptr, out value ptr
    ReLu,          // arg value ptr, out value ptr
    TanH,          // arg value ptr, out value ptr
    Exp,           // arg value ptr, out value ptr

    GradAdd,       // lhs value ptr, rhs value ptr, out grad ptr
    GradMul,       // lhs value ptr, rhs value ptr, out grad ptr
    GradPow,       // lhs value ptr, rhs value ptr, out grad ptr
    GradReLu,      // arg grad ptr, out value ptr
    GradTanH,      // arg grad ptr, out value ptr
    GradExp,       // arg grad ptr, out value ptr

    Update,        // value ptr
}

#[derive(Debug)]
pub struct Program {
    heap:  Vec<Number>,
    ptr_args: Vec<usize>,
    code:     Vec<Bytecode>,
    param_ptr:   usize,
    param_count: usize,
    score_ptr:   usize,
    score_count: usize,
    total_loss_ptr: usize,
    heap_map: HashMap<usize, usize>,

    hits: usize,
    misses: usize,
}

struct Codegen<'a> {
    program: &'a mut Program,
    const_map: HashMap<HashableNumber, usize>,
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

impl<'a> Codegen<'a> {
    fn get_const_ptr(&mut self, value: Number) -> usize {
        let hvalue = HashableNumber(value);
        if let Some(heap_ptr) = self.const_map.get(&hvalue) {
            return *heap_ptr;
        }

        let heap_ptr = self.program.heap_map.len();
        if heap_ptr >= self.program.heap.len() {
            self.program.heap.resize(heap_ptr + 1, 0.0);
        }
        self.program.heap[heap_ptr] = value;
        self.const_map.insert(hvalue, heap_ptr);

        heap_ptr
    }

    fn forward(&mut self, node: &Value) {
        let inner: &mut ValueInner = &mut node.inner.borrow_mut();

        if !inner.visited {
            inner.visited = true;

            let out_ptr = self.program.get_heap_ptr(node);

            match &mut inner.op {
                Op::Value => {},
                Op::Add(lhs, rhs) => {
                    let lhs_ptr = self.program.get_heap_ptr(lhs);
                    let rhs_ptr = self.program.get_heap_ptr(rhs);

                    self.forward(lhs);
                    self.forward(rhs);

                    self.program.code.push(Bytecode::Add);
                    self.program.ptr_args.push(lhs_ptr);
                    self.program.ptr_args.push(rhs_ptr);
                    self.program.ptr_args.push(out_ptr);
                },
                Op::Mul(lhs, rhs) => {
                    let lhs_ptr = self.program.get_heap_ptr(lhs);
                    let rhs_ptr = self.program.get_heap_ptr(rhs);

                    self.forward(lhs);
                    self.forward(rhs);

                    self.program.code.push(Bytecode::Mul);
                    self.program.ptr_args.push(lhs_ptr);
                    self.program.ptr_args.push(rhs_ptr);
                    self.program.ptr_args.push(out_ptr);
                },
                Op::Pow(lhs, rhs) => {
                    let lhs_ptr = self.program.get_heap_ptr(lhs);
                    let rhs_ptr = self.get_const_ptr(*rhs);

                    self.forward(lhs);

                    self.program.code.push(Bytecode::Pow);
                    self.program.ptr_args.push(lhs_ptr);
                    self.program.ptr_args.push(rhs_ptr);
                    self.program.ptr_args.push(out_ptr);
                },
                Op::ReLu(arg) => {
                    let arg_ptr = self.program.get_heap_ptr(arg);

                    self.forward(arg);

                    self.program.code.push(Bytecode::ReLu);
                    self.program.ptr_args.push(arg_ptr);
                    self.program.ptr_args.push(out_ptr);
                },
                Op::TanH(arg) => {
                    let arg_ptr = self.program.get_heap_ptr(arg);

                    self.forward(arg);

                    self.program.code.push(Bytecode::TanH);
                    self.program.ptr_args.push(arg_ptr);
                    self.program.ptr_args.push(out_ptr);
                },
                Op::Exp(arg) => {
                    let arg_ptr = self.program.get_heap_ptr(arg);

                    self.forward(arg);

                    self.program.code.push(Bytecode::Exp);
                    self.program.ptr_args.push(arg_ptr);
                    self.program.ptr_args.push(out_ptr);
                },
            }
        }
    }

    fn backward(&mut self, node: &Value) {
        let inner: &mut ValueInner = &mut node.inner.borrow_mut();

        if !inner.visited {
            inner.visited = true;

            let out_value_ptr = self.program.get_heap_ptr(node);
            let out_grad_ptr = out_value_ptr + 1;

            match &mut inner.op {
                Op::Value => {},
                Op::Add(lhs, rhs) => {
                    let lhs_ptr = self.program.get_heap_ptr(lhs);
                    let rhs_ptr = self.program.get_heap_ptr(rhs);

                    self.backward(rhs);
                    self.backward(lhs);

                    self.program.code.push(Bytecode::GradAdd);
                    self.program.ptr_args.push(lhs_ptr);
                    self.program.ptr_args.push(rhs_ptr);
                    self.program.ptr_args.push(out_grad_ptr);
                },
                Op::Mul(lhs, rhs) => {
                    let lhs_ptr = self.program.get_heap_ptr(lhs);
                    let rhs_ptr = self.program.get_heap_ptr(rhs);

                    self.backward(rhs);
                    self.backward(lhs);

                    self.program.code.push(Bytecode::GradMul);
                    self.program.ptr_args.push(lhs_ptr);
                    self.program.ptr_args.push(rhs_ptr);
                    self.program.ptr_args.push(out_grad_ptr);
                },
                Op::Pow(lhs, rhs) => {
                    let lhs_ptr = self.program.get_heap_ptr(lhs);
                    let rhs_ptr = self.get_const_ptr(*rhs);

                    self.backward(lhs);

                    self.program.code.push(Bytecode::GradPow);
                    self.program.ptr_args.push(lhs_ptr);
                    self.program.ptr_args.push(rhs_ptr);
                    self.program.ptr_args.push(out_grad_ptr);
                },
                Op::ReLu(arg) => {
                    let arg_ptr = self.program.get_heap_ptr(arg);

                    self.backward(arg);

                    self.program.code.push(Bytecode::GradReLu);
                    self.program.ptr_args.push(arg_ptr + 1);
                    self.program.ptr_args.push(out_value_ptr);
                },
                Op::TanH(arg) => {
                    let arg_ptr = self.program.get_heap_ptr(arg);

                    self.backward(arg);

                    self.program.code.push(Bytecode::GradTanH);
                    self.program.ptr_args.push(arg_ptr + 1);
                    self.program.ptr_args.push(out_value_ptr);
                },
                Op::Exp(arg) => {
                    let arg_ptr = self.program.get_heap_ptr(arg);

                    self.backward(arg);

                    self.program.code.push(Bytecode::GradExp);
                    self.program.ptr_args.push(arg_ptr + 1);
                    self.program.ptr_args.push(out_value_ptr);
                },
            }
        }
    }

    fn update(&mut self, parameters: &[Value]) {
        for param in parameters {
            self.program.code.push(Bytecode::Update);
            self.program.ptr_args.push(*self.program.heap_map.get(&param.id()).unwrap());
        }
    }
}

impl Program {
    pub fn compile(parameters: &[Value], scores: &[Value], total_loss: &Value) -> Self {
        let mut heap_map = HashMap::new();
        let mut heap = vec![0.0; 2 * (parameters.len() + scores.len() + 1)];

        let mut heap_ptr = 0;

        let param_ptr = heap_ptr;
        let param_count = parameters.len();

        for node in parameters {
            heap_map.insert(node.id(), heap_ptr);
            let (value, grad) = node.get();
            heap[heap_ptr] = value;
            heap[heap_ptr + 1] = grad;
            heap_ptr += 2;
        }

        let score_ptr = heap_ptr;
        let score_count = scores.len();

        for node in scores {
            heap_map.insert(node.id(), heap_ptr);
            let (value, grad) = node.get();
            heap[heap_ptr] = value;
            heap[heap_ptr + 1] = grad;
            heap_ptr += 2;
        }

        let total_loss_ptr = heap_ptr;
        heap_map.insert(total_loss.id(), total_loss_ptr);
        let (total_loss_value, total_loss_grad) = total_loss.get();
        heap[total_loss_ptr] = total_loss_value;
        heap[total_loss_ptr + 1] = total_loss_grad;
        // heap_ptr += 2;

        let mut program = Program {
            heap,
            ptr_args: Vec::new(),
            code: Vec::new(),
            param_ptr,
            param_count,
            score_ptr,
            score_count,
            total_loss_ptr,
            heap_map,
            hits: 0,
            misses: 0,
        };

        let mut codegen = Codegen {
            program: &mut program,
            const_map: HashMap::new()
        };

        codegen.forward(total_loss);
        total_loss.clear_visited();

        codegen.backward(total_loss);
        total_loss.clear_visited();

        codegen.update(parameters);

        //println!("code:\n{:#?}", program.code);

        program
    }

    fn get_heap_ptr(&mut self, node: &Value) -> usize {
        if let Some(heap_ptr) = self.heap_map.get(&node.id()) {
            self.hits += 1;
            return *heap_ptr;
        }
        self.misses += 1;

        let heap_ptr = self.heap.len();
        self.heap_map.insert(node.id(), heap_ptr);
        // + 1 for space for value and grad
        self.heap.resize(heap_ptr + 2, 0.0);
        let (value, grad) = node.get();
        self.heap[heap_ptr] = value;
        self.heap[heap_ptr + 1] = grad;

        heap_ptr
    }

    pub fn insert(&mut self, node: &Value) -> usize {
        let heap_ptr = if let Some(heap_ptr) = self.heap_map.get(&node.id()) {
            self.hits += 1;
            *heap_ptr
        } else {
            self.misses += 1;
            let heap_ptr = self.heap.len();
            self.heap_map.insert(node.id(), heap_ptr);
            // + 1 for space for value and grad
            self.heap.resize(heap_ptr + 2, 0.0);
            heap_ptr
        };

        let (value, grad) = node.get();
        self.heap[heap_ptr] = value;
        self.heap[heap_ptr + 1] = grad;

        heap_ptr
    }

    pub fn get(&self, value: &Value) -> Option<Number> {
        self.heap_map.get(&value.id()).map(|heap_ptr| self.heap[*heap_ptr])
    }

    pub fn exec(&mut self, learning_rate: Number) -> Number {
        let heap = &mut self.heap[..];
        let ptr_args = &self.ptr_args[..];
        let mut ptr_arg_index = 0;

        // risky unsafe business for speed
        #[cfg(not(debug_assertions))]
        macro_rules! ptr_args {
            ($($expr:tt)*) => {
                *unsafe { ptr_args.get_unchecked($($expr)*) }
            };
        }

        #[cfg(not(debug_assertions))]
        macro_rules! heap {
            ($($expr:tt)*) => {
                *unsafe { heap.get_unchecked_mut($($expr)*) }
            };
        }

        #[cfg(debug_assertions)]
        macro_rules! ptr_args {
            ($($expr:tt)*) => {
                ptr_args[$($expr)*]
            };
        }

        #[cfg(debug_assertions)]
        macro_rules! heap {
            ($($expr:tt)*) => {
                heap[$($expr)*]
            };
        }

        heap![self.total_loss_ptr + 1] = 1.0;

        for op in self.code.iter().cloned() {
            match op {
                Bytecode::Add => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let res_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;
                    // println!("Add {lhs_ptr} {rhs_ptr} {res_ptr}");

                    heap![res_ptr] = heap![lhs_ptr] + heap![rhs_ptr];
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::Mul => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let res_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;
                    // println!("Mul {lhs_ptr} {rhs_ptr} {res_ptr}");

                    heap![res_ptr] = heap![lhs_ptr] * heap![rhs_ptr];
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::Pow => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let res_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;
                    // println!("Pow {lhs_ptr} {rhs_ptr} {res_ptr}");

                    heap![res_ptr] = heap![lhs_ptr].powf(heap![rhs_ptr]);
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::ReLu => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let res_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;
                    // println!("ReLu {arg_ptr} {res_ptr}");

                    let value = heap![arg_ptr];
                    heap![res_ptr] = if value < 0.0 { 0.0 } else { value };
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::TanH => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let res_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;
                    // println!("TanH {arg_ptr} {res_ptr}");

                    let value = (heap![arg_ptr] * 2.0).exp();
                    heap![res_ptr] = (value - 1.0) / (value + 1.0);
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::Exp => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let res_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;
                    // println!("Exp {arg_ptr} {res_ptr}");

                    heap![res_ptr] = heap![arg_ptr].exp();
                    heap![res_ptr + 1] = 0.0; // zero grad
                },

                Bytecode::GradAdd => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let out_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;
                    // println!("GradAdd {lhs_ptr} {rhs_ptr} {out_ptr}");

                    let out_grad = heap![out_ptr];
                    heap![lhs_ptr + 1] += out_grad;
                    heap![rhs_ptr + 1] += out_grad;
                },
                Bytecode::GradMul => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let out_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;
                    // println!("GradMul {lhs_ptr} {rhs_ptr} {out_ptr}");

                    let out_grad  = heap![out_ptr];
                    let lhs_value = heap![lhs_ptr];
                    let rhs_value = heap![rhs_ptr];
                    heap![lhs_ptr + 1] += rhs_value * out_grad;
                    heap![rhs_ptr + 1] += lhs_value * out_grad;
                },
                Bytecode::GradPow => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let out_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;
                    // println!("GradPow {lhs_ptr} {rhs_ptr} {out_ptr}");

                    let out_grad  = heap![out_ptr];
                    let lhs_value = heap![lhs_ptr];
                    let rhs_value = heap![rhs_ptr];
                    heap![lhs_ptr + 1] += rhs_value * lhs_value.powf(rhs_value - 1.0) * out_grad;
                },
                Bytecode::GradReLu => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let out_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;
                    // println!("GradReLu {arg_ptr} {out_ptr}");

                    let out_value = heap![out_ptr];
                    let out_grad  = heap![out_ptr + 1];
                    let value: Number = (out_value > 0.0).into();
                    heap![arg_ptr] += value * out_grad;
                },
                Bytecode::GradTanH => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let out_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;
                    // println!("GradTanH {arg_ptr} {out_ptr}");

                    let out_value = heap![out_ptr];
                    let out_grad  = heap![out_ptr + 1];
                    let value: Number = 1.0 - (out_value * out_value);
                    heap![arg_ptr] += value * out_grad;
                },
                Bytecode::GradExp => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let out_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;
                    // println!("GradExp {arg_ptr} {out_ptr}");

                    let out_value = heap![out_ptr];
                    let out_grad  = heap![out_ptr + 1];
                    heap![arg_ptr] += out_value * out_grad;
                },

                Bytecode::Update => {
                    let heap_ptr = ptr_args![ptr_arg_index];
                    ptr_arg_index += 1;
                    // println!("Update {heap_ptr}");
                    // let v = heap![heap_ptr];
                    // let g = heap![heap_ptr + 1];
                    // println!("Update heap_ptr: {}, value: {}, grad: {}", heap_ptr, v, g);
                    heap![heap_ptr] -= learning_rate * heap![heap_ptr + 1];
                },
            }
        }

        heap![self.total_loss_ptr]
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

    pub fn get_values(&self, values: &mut [Value]) {
        for node in values.iter_mut() {
            if let Some(heap_ptr) = self.heap_map.get(&node.id()) {
                let heap_ptr = *heap_ptr;
                let value = self.heap[heap_ptr];
                let grad  = self.heap[heap_ptr + 1];
                node.assign_both(value, grad);
            }
        }
    }

    #[inline]
    pub fn heap(&self) -> &[Number] {
        &self.heap
    }

    #[inline]
    pub fn hits(&self) -> usize {
        self.hits
    }

    #[inline]
    pub fn misses(&self) -> usize {
        self.misses
    }
}
