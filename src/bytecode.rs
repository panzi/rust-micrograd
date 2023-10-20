use std::collections::HashMap;

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
    input_ptr:   usize,
    input_count: usize,
    total_loss_ptr: usize,
}

struct Codegen<'a> {
    program: &'a mut Program,
    heap_map: HashMap<&'a Value, usize>,
}

impl<'a> Codegen<'a> {
    fn forward(&mut self, node: &'a Value) {
        let inner: &mut ValueInner = &mut node.inner.borrow_mut();

        if !inner.visited {
            inner.visited = true;

            match &mut inner.op {
                Op::Value => {
                    if !self.heap_map.contains_key(node) {
                        let heap_ptr = self.heap_map.len();
                        self.heap_map.insert(node, heap_ptr);
                        if heap_ptr >= self.program.heap.len() {
                            self.program.heap.resize(heap_ptr + 1, 0.0);
                        }
                        self.program.heap[heap_ptr] = inner.value;
                    };
                },
                Op::Add(lhs, rhs) => todo!(),
                Op::Mul(lhs, rhs) => todo!(),
                Op::Pow(lhs, rhs) => todo!(),
                Op::ReLu(arg) => todo!(),
                Op::TanH(arg) => todo!(),
                Op::Exp(arg) => todo!(),
            }
        }
    }

    fn backward(&mut self, node: &'a Value) {
        let inner: &mut ValueInner = &mut node.inner.borrow_mut();

        if !inner.visited {
            inner.visited = true;

            match &mut inner.op {
                Op::Value => todo!(),
                Op::Add(lhs, rhs) => todo!(),
                Op::Mul(lhs, rhs) => todo!(),
                Op::Pow(lhs, rhs) => todo!(),
                Op::ReLu(arg) => todo!(),
                Op::TanH(arg) => todo!(),
                Op::Exp(arg) => todo!(),
            }
        }
    }

    fn update(&mut self, parameters: &'a [Value]) {
        for param in parameters {
            self.program.code.push(Bytecode::Update);
            self.program.ptr_args.push(*self.heap_map.get(param).unwrap());
        }
    }
}

impl Program {
    pub fn compile(inputs: &[Value], parameters: &[Value], scores: &[Value], total_loss: &Value) -> Self {
        let input_ptr = 0;
        let input_count = inputs.len();

        let mut heap_map: HashMap<&Value, usize> = HashMap::new();

        for value in inputs {
            heap_map.insert(value, heap_map.get(value).cloned().unwrap_or(heap_map.len()));
        }

        let param_ptr = heap_map.len();
        let param_count = parameters.len();

        for value in parameters {
            heap_map.insert(value, heap_map.len());
        }

        let score_ptr = heap_map.len();
        let score_count = scores.len();

        for value in scores {
            heap_map.insert(value, heap_map.len());
        }

        let total_loss_ptr = heap_map.len();
        heap_map.insert(total_loss, total_loss_ptr);

        let code: Vec<Bytecode> = Vec::new();
        let ptr_args: Vec<usize> = Vec::new();

        let mut heap = Vec::with_capacity(heap_map.len());

        heap.splice(input_ptr..(input_ptr + input_count), inputs.iter().map(Value::value));

        let mut program = Program {
            heap,
            ptr_args,
            code,
            param_ptr,
            param_count,
            score_ptr,
            score_count,
            input_ptr,
            input_count,
            total_loss_ptr,
        };

        let mut codegen = Codegen {
            program: &mut program,
            heap_map,
        };

        codegen.forward(total_loss);
        total_loss.clear_visited();

        codegen.backward(total_loss);
        total_loss.clear_visited();

        codegen.update(parameters);

        program
    }

    pub fn update_inputs(&mut self, inputs: &[Number]) {
        assert!(inputs.len() == self.input_count);

        self.heap.splice(
            self.input_ptr..(self.input_ptr + self.input_count),
            inputs.iter().cloned()
        );
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

        for op in self.code.iter().cloned() {
            match op {
                Bytecode::Add => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let res_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;

                    heap![res_ptr] = heap![lhs_ptr] + heap![rhs_ptr];
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::Mul => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let res_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;

                    heap![res_ptr] = heap![lhs_ptr] * heap![rhs_ptr];
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::Pow => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let res_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;

                    heap![res_ptr] = heap![lhs_ptr].powf(heap![rhs_ptr]);
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::ReLu => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let res_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;

                    let value = heap![arg_ptr];
                    heap![res_ptr] = if value < 0.0 { 0.0 } else { value };
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::TanH => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let res_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;

                    let value = (heap![arg_ptr] * 2.0).exp();
                    heap![res_ptr] = (value - 1.0) / (value + 1.0);
                    heap![res_ptr + 1] = 0.0; // zero grad
                },
                Bytecode::Exp => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let res_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;

                    heap![res_ptr] = heap![arg_ptr].exp();
                    heap![res_ptr + 1] = 0.0; // zero grad
                },

                Bytecode::GradAdd => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let out_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;

                    let out_grad = heap![out_ptr];
                    heap![lhs_ptr] += out_grad;
                    heap![rhs_ptr] += out_grad;
                },
                Bytecode::GradMul => {
                    let lhs_ptr = ptr_args![ptr_arg_index];
                    let rhs_ptr = ptr_args![ptr_arg_index + 1];
                    let out_ptr = ptr_args![ptr_arg_index + 2];
                    ptr_arg_index += 3;

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

                    let out_grad  = heap![out_ptr];
                    let lhs_value = heap![lhs_ptr];
                    let rhs_value = heap![rhs_ptr];
                    heap![lhs_ptr + 1] += rhs_value * lhs_value.powf(rhs_value - 1.0) * out_grad;
                },
                Bytecode::GradReLu => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let out_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;

                    let out_value = heap![out_ptr];
                    let out_grad  = heap![out_ptr + 1];
                    let value: Number = (out_value > 0.0).into();
                    heap![arg_ptr] += value * out_grad;
                },
                Bytecode::GradTanH => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let out_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;

                    let out_value = heap![out_ptr];
                    let out_grad  = heap![out_ptr + 1];
                    let value: Number = 1.0 * (out_value * out_value);
                    heap![arg_ptr] += value * out_grad;
                },
                Bytecode::GradExp => {
                    let arg_ptr = ptr_args![ptr_arg_index];
                    let out_ptr = ptr_args![ptr_arg_index + 1];
                    ptr_arg_index += 2;

                    let out_value = heap![out_ptr];
                    let out_grad  = heap![out_ptr + 1];
                    heap![arg_ptr] += out_value * out_grad;
                },

                Bytecode::Update => {
                    let heap_ptr = ptr_args![ptr_arg_index];
                    ptr_arg_index += 1;
                    heap![heap_ptr] -= learning_rate * heap![heap_ptr + 1];
                },
            }
        }

        heap![self.total_loss_ptr]
    }

    pub fn total_loss(&self) -> Number {
        self.heap[self.total_loss_ptr]
    }

    pub fn scores(&self) -> &[Number] {
        &self.heap[self.score_ptr..(self.score_ptr + self.score_count)]
    }

    pub fn parameters(&self) -> &[Number] {
        &self.heap[self.param_ptr..(self.param_ptr + self.param_count)]
    }
}
