use std::process::{Command, Stdio};

use micrograd::{MLP, Module, Value, Number, Loss, cmp_number_ref};

#[allow(non_snake_case)]
fn main() {
    let pycode = format!("\
#!/usr/bin/env python3
from sklearn.datasets import make_moons
import json
import sys

X, y = make_moons(n_samples=100, noise=0.1)
y = y * 2 - 1

json.dump([X.tolist(), y.tolist()], sys.stdout)
    ");

    let output = Command::new("python")
        .args(["-c", &pycode])
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed creating moons");

    let (X, y): (Vec<Vec<Number>>, Vec<Number>) = serde_json::from_reader(output.stdout.unwrap())
        .expect("failed creating moons");

    let mut rng = rand::thread_rng();
    // 2-layer neural network
    let mut model = MLP::new(2, &[16, 16, 1], &mut rng);

    println!("{:#?}", model);
    println!();
    println!("number of parameters: {}", model.count_parameters());
    println!();

    let mut buf = Vec::with_capacity(model.max_size());
    let mut scores: Vec<Value> = Vec::with_capacity(X.len());

    let loss = |model: &MLP| {
        // forward the model to get scores
        scores.clear();
        for xrow in &X {
            let input: Vec<_> = xrow.iter().cloned().map(Value::new).collect();
            model.forward_into(&input, &mut buf);
            scores.push(buf.first().unwrap().clone());
        }

        // sum "max-margin" loss
        let losses: Vec<_> = y.iter().cloned().zip(scores.iter()).map(
            |(yi, scorei)| (scorei * -yi + 1.0).relu()
        ).collect();
        let data_loss = losses.iter().cloned().sum::<Value>() / losses.len() as Number;

        // L2 regularization
        let alpha: Number = 1e-4;
        let reg_loss = model.fold_paramters(
            Value::new(0.0),
            |acc, value| acc + (value * value)
        ) * alpha;
        let total = data_loss + reg_loss;

        // also get accuracy
        let sum_accuracy: usize = y.iter().cloned().zip(scores.iter()).map(
            |(yi, scorei)| ((yi > 0.0) == (scorei.value() > 0.0)) as usize
        ).sum();

        let accuracy = sum_accuracy as Number / y.len() as Number;

        Loss {
            total,
            accuracy,
        }
    };

    println!("optimizing:");
    for (k, loss) in model.optimize(100, loss) {
        println!("step {} loss {}, accuracy {}%", k, loss.total.value(), loss.accuracy * 100.0);
    }

    // visualize decision boundary
    let h = 0.25;
    let x_min = X.iter().map(|row| row[0]).min_by(cmp_number_ref).unwrap_or(0.0);
    let x_max = X.iter().map(|row| row[0]).max_by(cmp_number_ref).unwrap_or(0.0);
    let y_min = X.iter().map(|row| row[1]).min_by(cmp_number_ref).unwrap_or(0.0);
    let y_max = X.iter().map(|row| row[1]).max_by(cmp_number_ref).unwrap_or(0.0);

    let mut xmesh = Vec::new();

    {
        let mut y = y_min;
        while y < y_max {
            let mut x = x_min;
            while x < x_max {
                xmesh.push([x, y]);
                x += h;
            }
            y += h;
        }
    }

    let x_len = ((x_max - x_min) / h).ceil() as usize;
    // let y_len = ((y_max - y_min) / h).ceil() as usize;

    let (xx, yy) = meshgrid(
        &arange(x_min, x_max, h),
        &arange(y_min, y_max, h),
    );

    let scores: Vec<_> = xmesh.iter().map(
        |xrow| model.forward(&xrow.map(Value::new)).first().unwrap().value()
    ).collect();

    let Z: Vec<_> = scores.iter().cloned().map(|value| (value > 0.0) as u32).collect();
    let Z: Vec<_> = Z.chunks(x_len).collect();

    let pycode = format!("\
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

X = np.array({:?})
y = np.array({:?})
xx = np.array({:?})
yy = np.array({:?})
Z = np.array({:?}, dtype=bool)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
",
        X, y, xx, yy, Z);

    std::fs::write("/tmp/plot.py", &pycode).expect("writing /tmp/plot.py");

    Command::new("python").args(["-c", &pycode]).spawn().expect("spawning python failed");
}

#[inline]
fn arange(start: Number, end: Number, step: Number) -> Vec<Number> {
    let mut xs = Vec::new();
    let mut x = start;

    while x < end {
        xs.push(x);
        x += step;
    }

    xs
}

#[inline]
fn meshgrid(xs: &[Number], ys: &[Number]) -> (Vec<Vec<Number>>, Vec<Vec<Number>>) {
    (
        (0..ys.len()).map(|_| Vec::from(xs)).collect(),
        ys.iter().cloned().map(|y| vec![y; xs.len()]).collect(),
    )
}
