use micrograd::Program;
use micrograd::{MLP, Module, Value, Number};

mod common;

use crate::common::{make_moons, plot_moons};

#[allow(non_snake_case)]
fn main() {
    let (X, y) = make_moons(100, 0.1);

    let mut rng = rand::thread_rng();
    // 2-layer neural network
    let mut model = MLP::new(2, &[16, 16, 1], &mut rng);

    println!("{:#?}", model);
    println!();
    println!("number of parameters: {}", model.count_parameters());
    println!();

    let mut buf = Vec::with_capacity(model.max_size());
    let mut scores = Vec::with_capacity(X.len());

    // forward the model to get scores
    let mut inputs = Vec::with_capacity(X.len());
    for xrow in &X {
        let input: Vec<_> = xrow.iter().cloned().map(Value::new).collect();
        model.forward_into(&input, &mut buf);
        scores.push(buf.first().unwrap().clone());
        inputs.push(input);
    }

    // sum "max-margin" loss
    let loss_sum: Value = y.iter().cloned().zip(scores.iter()).map(
        |(yi, scorei)| (scorei * -yi + 1.0).relu()
    ).sum();
    let data_loss = loss_sum / y.len() as Number;

    // L2 regularization
    let alpha: Number = 1e-4;
    let reg_loss = model.fold_paramters(
        Value::new(0.0),
        |acc, value| acc + (value * value)
    ) * alpha;
    let total = data_loss + reg_loss;

    let mut program = Program::compile(&model.parameters(), &scores, &total);

    println!("optimizing:");
    let steps = 100;
    for k in 0..steps {
        let learning_rate: Number = 1.0 - 0.9 * k as Number / steps as Number;
        program.exec(learning_rate);

        let total = program.total_loss();
        let scores = program.scores();

        // also get accuracy
        let sum_accuracy: usize = y.iter().cloned().zip(scores.iter()).map(
            |(yi, scorei)| ((yi > 0.0) == (*scorei > 0.0)) as usize
        ).sum();

        let accuracy = sum_accuracy as Number / y.len() as Number;

        println!("step {} loss {}, accuracy {}%", k, total, accuracy * 100.0);
        if accuracy > 0.99 {
            println!("stopping because accuracy > 99%");
            break;
        }
    }

    for (node, value) in model.parameters().iter_mut().zip(program.parameters().iter().cloned()) {
        node.assign(value);
    }

    // visualize decision boundary
    plot_moons(&X, &y, &mut model);
}
