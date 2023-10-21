use micrograd::{MLP, Module, Value, Number, Loss};
use rand::SeedableRng;

mod common;

use crate::common::DEFAULT_X;
use crate::common::DEFAULT_Y;
use crate::common::plot_moons;

#[allow(non_snake_case)]
fn main() {
    let X = DEFAULT_X;
    let y = DEFAULT_Y;

    let mut rng = rand::rngs::StdRng::from_seed([0u8; 32]);

    // 2-layer neural network
    let mut model = MLP::new(2, &[16, 16, 1], &mut rng);

    println!("{:#?}", model);
    println!();
    println!("number of parameters: {}", model.count_parameters());
    println!();

    let mut buf = Vec::with_capacity(model.max_size());
    let mut scores = Vec::with_capacity(X.len());

    // forward the model to get scores
    for xrow in X {
        let input: Vec<_> = xrow.iter().cloned().map(Value::new).collect();
        model.forward_into(&input, &mut buf);
        scores.push(buf.first().unwrap().clone());
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
    let mut total = data_loss + reg_loss;

    let loss = |_model: &MLP, k: usize| {
        // Re-use the existing expression tree and just recalculate the values in-place.
        // This also zeroes the grad field.
        total.refresh(k);

        // also get accuracy
        let sum_accuracy: usize = y.iter().cloned().zip(scores.iter()).map(
            |(yi, scorei)| ((yi > 0.0) == (scorei.value() > 0.0)) as usize
        ).sum();

        let accuracy = sum_accuracy as Number / y.len() as Number;

        Loss {
            total: total.clone(),
            accuracy,
        }
    };

    println!("optimizing:");
    for (k, loss) in model.optimize(100, loss) {
        println!("step {} loss {}, accuracy {}%", k, loss.total.value(), loss.accuracy * 100.0);
        if loss.accuracy > 0.99 {
            println!("stopping because accuracy > 99%");
            break;
        }
    }

    // visualize decision boundary
    plot_moons(X, &y, &mut model);
}
