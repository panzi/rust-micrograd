use std::process::{Command, Stdio};

use micrograd::Number;

#[allow(non_snake_case)]
pub fn make_moons(n_samples: usize, noise: f64) -> (Vec<Vec<Number>>, Vec<Number>) {
    let pycode = format!("\
#!/usr/bin/env python3
from sklearn.datasets import make_moons
import json
import sys

X, y = make_moons(n_samples={n_samples:?}, noise={noise:?})
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

    (X, y)
}
