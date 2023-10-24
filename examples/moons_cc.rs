use micrograd::CCProgram;
use micrograd::{MLP, Module, Value, Number};
// use rand::{SeedableRng, Rng};

mod common;

use crate::common::{plot_moons, DEFAULT_X, DEFAULT_Y, FixedRandom};

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let X = DEFAULT_X;
    let y = DEFAULT_Y;

    // let mut rng = rand::rngs::StdRng::from_seed([0u8; 32]);
    // let mut rng = rand::thread_rng();

    // 2-layer neural network
    // let mut model = MLP::new(2, &[16, 16, 1], |lower, upper| rng.gen_range(lower..upper));
    let mut rng = FixedRandom::new();
    let mut model = MLP::new(2, &[16, 16, 1], |_, _| rng.next());

    println!("{:#?}", model);
    println!();
    println!("number of parameters: {}", model.count_parameters());
    println!();

    let mut buf = Vec::with_capacity(model.max_layer_size());
    let mut scores = Vec::with_capacity(X.len());

    // forward the model to get scores
    let mut inputs = Vec::with_capacity(X.len());
    for xrow in X {
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
    let reg_loss = model.map_parameters(&|value| value * value).sum::<Value>() * alpha;
    let total = data_loss + reg_loss;

    let mut program = CCProgram::compile_model(&model, &scores, &total, &[])?;

    println!("optimizing:");
    let steps = 100;
    let mut scores_buf = Vec::with_capacity(scores.len());
    for k in 0..steps {
        let learning_rate: Number = 1.0 - 0.9 * k as Number / steps as Number;
        let total = program.exec(learning_rate);

        scores_buf.clear();
        program.get_scores(&mut scores_buf);

        // also get accuracy
        let sum_accuracy: usize = y.iter().cloned().zip(scores_buf.iter()).map(
            |(yi, scorei)| ((yi > 0.0) == (*scorei > 0.0)) as usize
        ).sum();

        let accuracy = sum_accuracy as Number / y.len() as Number;

        println!("step {} loss {}, accuracy {}%", k, total, accuracy * 100.0);
    }

    // copy paramters back into model
    program.get_model(&mut model);

    // visualize decision boundary
    plot_moons(X, y, &mut model);

    Ok(())
}
