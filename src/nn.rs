use std::{fmt::Display, mem::swap};

use join_string::Join;

use crate::{Value, Number};

pub trait Module {
    fn zero_grad(&mut self);

    fn update(&mut self, learning_rate: Number);

    fn count_parameters(&self) -> usize;
}

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlinear: bool,
}

impl Neuron {
    #[inline]
    pub fn new(inputs: usize, nonlinear: bool, mut rng: impl FnMut(Number, Number) -> Number) -> Self {
        Self {
            weights: (0..inputs).map(|_| Value::new(rng(-1.0, 1.0))).collect(),
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

    pub fn forward(&mut self, xs: &[Value]) -> Value {
        let mut res = self.bias.clone();
        for (w, x) in self.weights.iter().zip(xs) {
            res += w * x;
        }

        if self.nonlinear { res.relu() } else { res }
    }

    #[inline]
    pub fn parameters(&self) -> impl std::iter::Iterator<Item = &Value> {
        self.weights.iter().chain(Some(&self.bias))
    }

    #[inline]
    pub fn parameters_mut(&mut self) -> impl std::iter::Iterator<Item = &mut Value> {
        self.weights.iter_mut().chain(Some(&mut self.bias))
    }
}

impl Module for Neuron {
    #[inline]
    fn zero_grad(&mut self) {
        for param in &mut self.weights {
            param.zero_grad();
        }
        self.bias.zero_grad();
    }

    #[inline]
    fn update(&mut self, learning_rate: Number) {
        for param in &mut self.weights {
            param.update(learning_rate);
        }
        self.bias.update(learning_rate);
    }

    #[inline]
    fn count_parameters(&self) -> usize {
        self.weights.len() + 1
    }
}

impl Display for Neuron {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}Neuron([{:?}])", if self.nonlinear { "ReLu" } else { "Linear" }, self.weights.iter().join(", "))
    }
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
    inputs: usize,
    outputs: usize,
}

impl Layer {
    #[inline]
    pub fn new(inputs: usize, outputs: usize, nonlinear: bool, mut rng: impl FnMut(Number, Number) -> Number) -> Self {
        Self {
            neurons: (0..outputs).map(|_| Neuron::new(inputs, nonlinear, &mut rng)).collect(),
            inputs,
            outputs,
        }
    }

    #[inline]
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    #[inline]
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    #[inline]
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    #[inline]
    pub fn forward_into(&mut self, input: &[Value], output: &mut Vec<Value>) {
        output.clear();
        output.extend(self.neurons.iter_mut().map(
            |neuron| neuron.forward(input)
        ));
    }

    #[inline]
    pub fn forward(&mut self, input: &[Value]) -> Vec<Value> {
        let mut output = Vec::with_capacity(self.neurons.len());
        self.forward_into(input, &mut output);
        output
    }

    #[inline]
    pub fn parameters(&self) -> impl std::iter::Iterator<Item = &Value> {
        self.neurons.iter().flat_map(|neuron| neuron.parameters())
    }

    #[inline]
    pub fn parameters_mut(&mut self) -> impl std::iter::Iterator<Item = &mut Value> {
        self.neurons.iter_mut().flat_map(|neuron| neuron.parameters_mut())
    }
}

impl Module for Layer {
    #[inline]
    fn zero_grad(&mut self) {
        for neuron in &mut self.neurons {
            neuron.zero_grad();
        }
    }

    #[inline]
    fn update(&mut self, learning_rate: Number) {
        for neuron in &mut self.neurons {
            neuron.update(learning_rate);
        }
    }

    #[inline]
    fn count_parameters(&self) -> usize {
        self.outputs * (self.inputs + 1)
    }
}

impl Display for Layer {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer of [{}]", self.neurons.iter().join(", "))
    }
}

/// Multilayer perceptron
#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
    max_layer_size: usize,
}

impl MLP {
    #[inline]
    pub fn new(inputs: usize, outputs: &[usize], mut rng: impl FnMut(Number, Number) -> Number) -> Self {
        let mut sz = Vec::with_capacity(outputs.len() + 1);
        sz.push(inputs);
        sz.extend_from_slice(outputs);

        let linear_index = outputs.len() - 1;

        Self {
            layers: (0..outputs.len()).map(|index|
                Layer::new(sz[index], sz[index + 1], index != linear_index, &mut rng)
            ).collect(),
            max_layer_size: sz.iter().cloned().fold(0, std::cmp::max),
        }
    }

    #[inline]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    #[inline]
    pub fn max_layer_size(&self) -> usize {
        self.max_layer_size
    }

    pub fn forward_into(&mut self, input: &[Value], output: &mut Vec<Value>) {
        let mut inbuf = Vec::from(input);

        for layer in self.layers.iter_mut() {
            layer.forward_into(&inbuf, output);
            swap(&mut inbuf, output);
        }

        swap(&mut inbuf, output);
    }

    #[inline]
    pub fn forward(&mut self, input: &[Value]) -> Vec<Value> {
        let mut output = Vec::with_capacity(self.max_layer_size);
        self.forward_into(input, &mut output);
        output
    }

    #[inline]
    pub fn forward_unwrap(&mut self, input: &[Value]) -> Value {
        self.forward(input).first().unwrap().clone()
    }

    pub fn optimize<'a, L>(&'a mut self, steps: usize, mut loss: L) -> impl std::iter::Iterator<Item = (usize, Loss)> + 'a
    where
        L: FnMut(&MLP, usize) -> Loss + 'a,
    {
        let mut topo = Vec::new();
        (0..steps).map(move |k| {
            // forward
            let mut res = loss(self, k);

            // backward
            // the loss function needs to apply zero_grad() or do it as a side effect of e.g. refresh()
            // self.zero_grad();
            res.total.backward_buffered(&mut topo);

            // update (sgd)
            let learning_rate: Number = 1.0 - 0.9 * k as Number / steps as Number;
            self.update(learning_rate);

            (k, res)
        })
    }

    pub fn optimize_batched<'a, L>(&'a mut self, steps: usize, batch_size: usize, mut loss: L) -> impl std::iter::Iterator<Item = (usize, Loss)> + 'a
    where
        L: FnMut(&MLP, usize, usize) -> Loss + 'a,
    {
        let mut topo = Vec::new();
        (0..steps).map(move |k| {
            // forward
            let mut res = loss(self, k, batch_size);

            // backward
            // the loss function needs to apply zero_grad() or do it as a side effect of e.g. refresh()
            // self.zero_grad();
            res.total.backward_buffered(&mut topo);

            // update (sgd)
            let learning_rate: Number = 1.0 - 0.9 * k as Number / steps as Number;
            self.update(learning_rate);

            (k, res)
        })
    }

    #[inline]
    pub fn parameters(&self) -> impl std::iter::Iterator<Item = &Value> {
        self.layers.iter().flat_map(|layer| layer.parameters())
    }

    #[inline]
    pub fn parameters_mut(&mut self) -> impl std::iter::Iterator<Item = &mut Value> {
        self.layers.iter_mut().flat_map(|layer| layer.parameters_mut())
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Loss {
    pub total: Value,
    pub accuracy: Number,
}

impl Display for Loss {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Loss {{ total: {}, accuracy: {} }}", self.total.value(), self.accuracy)
    }
}

impl Module for MLP {
    #[inline]
    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    #[inline]
    fn update(&mut self, learning_rate: Number) {
        for layer in &mut self.layers {
            layer.update(learning_rate);
        }
    }

    #[inline]
    fn count_parameters(&self) -> usize {
        self.layers.iter().map(Layer::count_parameters).sum()
    }
}

impl Display for MLP {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MLP of [{}]", self.layers.iter().join(", "))
    }
}
