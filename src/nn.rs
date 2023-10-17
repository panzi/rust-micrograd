use std::{fmt::Display, mem::swap};

use join_string::Join;
use rand::{RngCore, Rng};

use crate::Value;

pub trait Module {
    fn zero_grad(&mut self) {
        for param in self.parameters().iter_mut() {
            param.zero_grad();
        }
    }

    fn gather_parameters(&mut self, params: &mut Vec<Value>);

    fn parameters(&mut self) -> Vec<Value> {
        let mut params = Vec::new();
        self.gather_parameters(&mut params);
        params
    }
}

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlinear: bool,
}

impl Neuron {
    #[inline]
    pub fn new(inputs: usize, nonlinear: bool, rng: &mut impl RngCore) -> Self {
        let mut weights = Vec::with_capacity(inputs);
        for _ in 0..inputs {
            weights.push(Value::new(rng.gen_range(-1.0..1.0)));
        }

        Self {
            weights,
            bias: Value::new(0.0),
            nonlinear,
        }
    }

    #[inline]
    pub fn weights(&self) -> &[Value] {
        &self.weights
    }

    #[inline]
    pub fn bias(&self) -> &Value {
        &self.bias
    }

    #[inline]
    pub fn nonlinear(&self) -> bool {
        self.nonlinear
    }

    pub fn forward(&self, xs: &[Value]) -> Value {
        let mut res = self.bias.clone();
        for (w, x) in self.weights.iter().zip(xs) {
            res += w.clone() * x.clone();
        }

        if self.nonlinear { res.relu() } else { res }
    }
}

impl Module for Neuron {
    fn zero_grad(&mut self) {
        for param in &mut self.weights {
            param.zero_grad();
        }
        self.bias.zero_grad();
    }

    fn gather_parameters(&mut self, params: &mut Vec<Value>) {
        params.reserve(self.weights.len() + 1);
        params.extend_from_slice(&self.weights);
        params.push(self.bias.clone());
    }
}

impl Display for Neuron {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}Neuron([{}])", if self.nonlinear { "ReLu" } else { "Linear" }, self.weights.iter().join(", "))
    }
}

pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    #[inline]
    pub fn new(inputs: usize, outputs: usize, nonlinear: bool, rng: &mut impl RngCore) -> Self {
        Self {
            neurons: (0..outputs).map(|_| Neuron::new(inputs, nonlinear, rng)).collect()
        }
    }

    #[inline]
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    #[inline]
    pub fn forward(&self, input: &[Value], output: &mut Vec<Value>) {
        for neuron in &self.neurons {
            output.push(neuron.forward(input));
        }
    }
}

impl Module for Layer {
    fn zero_grad(&mut self) {
        for neuron in &mut self.neurons {
            neuron.zero_grad();
        }
    }

    fn gather_parameters(&mut self, params: &mut Vec<Value>) {
        for neuron in &mut self.neurons {
            neuron.gather_parameters(params);
        }
    }

}

impl Display for Layer {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer of [{}]", self.neurons.iter().join(", "))
    }
}

pub struct MLP {
    layers: Vec<Layer>
}

impl MLP {
    #[inline]
    pub fn new(inputs: usize, outputs: &[usize], rng: &mut impl RngCore) -> Self {
        let mut sz = Vec::with_capacity(outputs.len() + 1);
        sz.push(inputs);
        sz.extend_from_slice(outputs);

        let linear_index = outputs.len() - 1;

        Self {
            layers: (0..outputs.len()).map(|index| Layer::new(sz[index], sz[index + 1], index != linear_index, rng)).collect()
        }
    }

    #[inline]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn forward(&self, input: &[Value], output: &mut Vec<Value>) {
        let mut inbuf = Vec::from(input);

        for layer in &self.layers {
            output.truncate(0);
            layer.forward(&inbuf, output);
            swap(&mut inbuf, output);
        }

        swap(&mut inbuf, output);
    }
}

impl Module for MLP {
    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    fn gather_parameters(&mut self, params: &mut Vec<Value>) {
        for layer in &mut self.layers {
            layer.gather_parameters(params);
        }
    }
}
