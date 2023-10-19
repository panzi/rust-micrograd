pub mod plot;
pub mod dataset;

pub use plot::*;
pub use dataset::*;

use micrograd::Number;

#[inline]
pub fn arange(start: Number, end: Number, step: Number) -> Vec<Number> {
    let mut xs = Vec::new();
    let mut x = start;

    while x < end {
        xs.push(x);
        x += step;
    }

    xs
}

#[inline]
pub fn meshgrid(xs: &[Number], ys: &[Number]) -> (Vec<Vec<Number>>, Vec<Vec<Number>>) {
    (
        (0..ys.len()).map(|_| Vec::from(xs)).collect(),
        ys.iter().cloned().map(|y| vec![y; xs.len()]).collect(),
    )
}
