use micrograd::{MLP, Module, Value, Number, Loss};
use rand::SeedableRng;

mod common;

use crate::common::DEFAULT_X;
use crate::common::DEFAULT_Y;
use crate::common::FixedRandom;
use crate::common::plot_moons;

use rand::seq::SliceRandom;

#[allow(non_snake_case)]
fn main() {
    let X = DEFAULT_X;
    let y = DEFAULT_Y;

    let mut rng = rand::rngs::StdRng::from_seed([0u8; 32]);

    // 2-layer neural network
    // let mut model = MLP::new(2, &[16, 16, 1], |lower, upper| rng.gen_range(lower..upper));
    let mut fixrng = FixedRandom::new();
    let mut model = MLP::new(2, &[16, 16, 1], |_, _| fixrng.next());
    let batch_size = 50;

    println!("{:#?}", model);
    println!();
    println!("number of parameters: {}", model.count_parameters());
    println!();

    // for later random selection
    let indices: Vec<_> = (0..X.len()).collect();

    let mut buf = Vec::with_capacity(model.max_layer_size());
    let mut scores = Vec::with_capacity(batch_size);

    // prepare batches which will later be updated
    let mut Xb: Vec<Vec<Value>> = (0..batch_size).map(|_| vec![Value::new(0.0), Value::new(0.0)]).collect();
    let mut yb: Vec<Value> = (0..batch_size).map(|_| Value::new(0.0)).collect();

    // forward the model to get scores
    for xrow in &Xb {
        model.forward_into(&xrow, &mut buf);
        scores.push(buf.first().unwrap().clone());
    }

    // sum "max-margin" loss
    let loss_sum: Value = yb.iter().zip(scores.iter()).map(
        |(yi, scorei)| (scorei * -yi + 1.0).relu()
    ).sum();
    let data_loss = loss_sum / batch_size as Number;

    // L2 regularization
    let alpha: Number = 1e-4;
    let reg_loss = model.fold_paramters(
        Value::new(0.0),
        |acc, value| acc + (value * value)
    ) * alpha;
    let mut total = data_loss + reg_loss;

    let loss = |_model: &MLP, k: usize, batch_size: usize| {
        for (batch_index, src_index) in indices.choose_multiple(&mut rng, batch_size).cloned().enumerate() {
            for (xb, x) in Xb[batch_index].iter_mut().zip(X[src_index].iter().cloned()) {
                xb.assign(x);
            }

            yb[batch_index].assign(y[src_index]);
        }

        // Re-use the existing expression tree and just recalculate the values in-place.
        // This also zeroes the grad field.
        total.refresh(k);

        // also get accuracy
        let sum_accuracy: usize = yb.iter().zip(scores.iter()).map(
            |(yi, scorei)| ((yi.value() > 0.0) == (scorei.value() > 0.0)) as usize
        ).sum();

        let accuracy = sum_accuracy as Number / batch_size as Number;

        Loss {
            total: total.clone(),
            accuracy,
        }
    };

    println!("optimizing:");
    for (k, loss) in model.optimize_batched(200, batch_size, loss) {
        println!("step {} loss {}, accuracy {}%", k, loss.total.value(), loss.accuracy * 100.0);
        if loss.accuracy > 0.99 {
            println!("stopping because accuracy > 99%");
            break;
        }
    }

    // visualize decision boundary
    plot_moons(X, y, &mut model);
}
