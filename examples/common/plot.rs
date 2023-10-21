use std::process::Command;

use micrograd::{cmp_number_ref, Value, Number, MLP};

use super::{meshgrid, arange};

#[allow(non_snake_case)]
#[allow(dead_code)]
pub fn plot_moons(X: &[impl AsRef<[Number]> + std::fmt::Debug], y: &[Number], model: &mut MLP) {
    // visualize decision boundary
    let h = 0.25;
    let x_min = X.iter().map(|row| row.as_ref()[0]).min_by(cmp_number_ref).unwrap_or(0.0);
    let x_max = X.iter().map(|row| row.as_ref()[0]).max_by(cmp_number_ref).unwrap_or(0.0);
    let y_min = X.iter().map(|row| row.as_ref()[1]).min_by(cmp_number_ref).unwrap_or(0.0);
    let y_max = X.iter().map(|row| row.as_ref()[1]).max_by(cmp_number_ref).unwrap_or(0.0);

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
