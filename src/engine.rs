use std::cmp::Ordering;
use std::mem::replace;
use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::fmt::{Display, Debug};
use std::{rc::Rc, cell::RefCell};

pub type Number = f64;
pub(crate) type ValueId = usize;

#[inline]
pub fn cmp_number(lhs: Number, rhs: Number) -> Ordering {
    if lhs < rhs {
        Ordering::Less
    } else if lhs > rhs {
        Ordering::Greater
    } else {
        // all NaNs are equal
        Ordering::Equal
    }
}

#[inline]
pub fn cmp_number_ref(lhs: &Number, rhs: &Number) -> Ordering {
    cmp_number(*lhs, *rhs)
}

#[derive(Debug, Clone)]
pub enum Op {
    Value,
    Add(Value, Value),
    Mul(Value, Value),
    Pow(Value, Number),
    ReLU(Value),
    TanH(Value),
    Exp(Value),
}

impl Op {
    #[inline]
    pub fn is_value(&self) -> bool {
        matches!(self, Op::Value)
    }

    #[inline]
    pub fn is_add(&self) -> bool {
        matches!(self, Op::Add(_, _))
    }

    #[inline]
    pub fn is_mul(&self) -> bool {
        matches!(self, Op::Mul(_, _))
    }

    #[inline]
    pub fn is_pow(&self) -> bool {
        matches!(self, Op::Pow(_, _))
    }

    #[inline]
    pub fn is_relu(&self) -> bool {
        matches!(self, Op::ReLU(_))
    }

    #[inline]
    pub fn is_tanh(&self) -> bool {
        matches!(self, Op::TanH(_))
    }

    #[inline]
    pub fn is_exp(&self) -> bool {
        matches!(self, Op::Exp(_))
    }
}

#[derive(Debug)]
pub(crate) struct ValueInner {
    pub(crate) value: Number,
    pub(crate) grad: Number,
    pub(crate) op: Op,
    pub(crate) visited: bool,
    pub(crate) k: usize,
}

#[repr(transparent)]
#[derive(Clone)]
pub struct Value {
    pub(crate) inner: Rc<RefCell<ValueInner>>
}

impl Value {
    #[inline]
    pub fn new(value: Number) -> Self {
        Value::from_op(value, Op::Value)
    }

    #[inline]
    fn from_op(value: Number, op: Op) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                value,
                grad: 0.0,
                op,
                visited: false,
                k: 0,
            }))
        }
    }

    #[inline]
    pub fn value(&self) -> Number {
        self.inner.borrow().value
    }

    #[inline]
    pub fn grad(&self) -> Number {
        self.inner.borrow().grad
    }

    #[inline]
    pub fn get(&self) -> (Number, Number) {
        let inner: &ValueInner = &self.inner.borrow();
        (inner.value, inner.grad)
    }

    #[inline]
    pub fn k(&self) -> usize {
        self.inner.borrow().k
    }

    #[inline]
    pub fn zero_grad(&mut self) {
        self.inner.borrow_mut().grad = 0.0;
    }

    #[inline]
    pub fn update(&mut self, learning_rate: Number) {
        let inner: &mut ValueInner = &mut self.inner.borrow_mut();
        inner.value -= learning_rate * inner.grad;
    }

    /// For re-using an expression tree and just updating the values.
    /// It also sets grad to zero.
    pub fn refresh(&mut self, k: usize) -> Number {
        let inner: &mut ValueInner = &mut self.inner.borrow_mut();

        if inner.k >= k {
            return inner.value;
        }

        match &mut inner.op {
            Op::Value => {},
            Op::Add(lhs, rhs) => {
                let lhs_value = lhs.refresh(k);
                let rhs_value = rhs.refresh(k);
                inner.value = lhs_value + rhs_value;
            },
            Op::Mul(lhs, rhs) => {
                let lhs_value = lhs.refresh(k);
                let rhs_value = rhs.refresh(k);
                inner.value = lhs_value * rhs_value;
            },
            Op::Pow(lhs, rhs) => {
                inner.value = lhs.refresh(k).powf(*rhs);
            },
            Op::ReLU(arg) => {
                inner.value = Number::max(arg.refresh(k), 0.0);
            },
            Op::TanH(arg) => {
                inner.value = arg.refresh(k).tanh();
            },
            Op::Exp(arg) => {
                inner.value = arg.refresh(k).exp();
            },
        }
        inner.k = k;
        inner.grad = 0.0;

        inner.value
    }

    #[inline]
    pub fn children(&self) -> impl std::iter::Iterator<Item = Value> {
        // Always the same number of `Optional`s in a chain so that
        // the return value of all the expressions is the same.
        match &self.inner.borrow().op {
            Op::Value =>
                None.into_iter().chain(None),
            Op::Add(lhs, rhs) | Op::Mul(lhs, rhs) =>
                Some(lhs.clone()).into_iter().chain(Some(rhs.clone())),
            Op::Pow(lhs, _) =>
                Some(lhs.clone()).into_iter().chain(None),
            Op::ReLU(arg) | Op::TanH(arg) | Op::Exp(arg) =>
                Some(arg.clone()).into_iter().chain(None),
        }
    }

    #[inline]
    pub fn relu(&self) -> Self {
        let value = Number::max(self.value(), 0.0);

        Value::from_op(value, Op::ReLU(self.clone()))
    }

    #[inline]
    pub fn pow(&self, rhs: Number) -> Self {
        if rhs == 1.0 {
            return self.clone();
        }
        Value::from_op(self.value().powf(rhs), Op::Pow(self.clone(), rhs))
    }

    #[inline]
    pub fn tanh(&self) -> Self {
        Value::from_op(self.value().tanh(), Op::TanH(self.clone()))
    }

    #[inline]
    pub fn exp(&self) -> Self {
        Value::from_op(self.value().exp(), Op::Exp(self.clone()))
    }

    fn backward_intern(&mut self) {
        let out: &mut ValueInner = &mut self.inner.borrow_mut();
        let out_grad = out.grad;
        let out_value = out.value;
        match &out.op {
            Op::Value => {},
            Op::Add(lhs, rhs) => {
                lhs.inner.borrow_mut().grad += out_grad;
                rhs.inner.borrow_mut().grad += out_grad;
            },
            Op::Mul(lhs, rhs) => {
                let lhs_value = lhs.inner.borrow().value;
                let rhs_value = rhs.inner.borrow().value;

                lhs.inner.borrow_mut().grad += rhs_value * out_grad;
                rhs.inner.borrow_mut().grad += lhs_value * out_grad;
            },
            Op::Pow(lhs, rhs) => {
                let lhs_inner: &mut ValueInner = &mut lhs.inner.borrow_mut();
                lhs_inner.grad += rhs * lhs_inner.value.powf(rhs - 1.0) * out_grad;
            },
            Op::ReLU(arg) => {
                if out_value > 0.0 {
                    arg.inner.borrow_mut().grad += out_grad;
                }
            },
            Op::TanH(arg) => {
                let value = 1.0 - (out_value * out_value);
                arg.inner.borrow_mut().grad += value * out_grad;
            },
            Op::Exp(arg) => {
                arg.inner.borrow_mut().grad += out_value * out_grad;
            },
        }
    }

    #[inline]
    pub fn backward(&mut self) {
        let mut topo = Vec::new();
        self.backward_buffered(&mut topo);
    }

    pub fn build_topo(&self, topo: &mut Vec<Value>) {
        let out: &mut ValueInner = &mut self.inner.borrow_mut();
        if !out.visited {
            out.visited = true;
            match &out.op {
                Op::Value => {},
                Op::Add(lhs, rhs) | Op::Mul(lhs, rhs) => {
                    lhs.build_topo(topo);
                    rhs.build_topo(topo);
                },
                Op::Pow(lhs, _) => {
                    lhs.build_topo(topo);
                },
                Op::ReLU(arg) | Op::TanH(arg) | Op::Exp(arg) => {
                    arg.build_topo(topo);
                },
            }
            topo.push(self.clone());
        }
    }

    /// To minimize allocations pass a topology buffer.
    pub fn backward_buffered(&mut self, topo: &mut Vec<Value>) {
        {
            self.inner.borrow_mut().grad = 1.0;

            topo.clear();
            self.build_topo(topo);

            for node in topo.iter_mut().rev() {
                node.backward_intern();
            }
        }

        self.clear_visited();
    }

    pub fn count_nodes(&self) -> usize {
        fn count_nodes(node: &Value) -> usize {
            let inner: &mut ValueInner = &mut node.inner.borrow_mut();

            if inner.visited {
                return 0;
            }

            let mut count = 1;

            inner.visited = true;

            match &inner.op {
                Op::Value => {},
                Op::Add(lhs, rhs) | Op::Mul(lhs, rhs) => {
                    count += count_nodes(lhs) + count_nodes(rhs);
                },
                Op::Pow(lhs, _) => {
                    count += count_nodes(lhs);
                },
                Op::ReLU(arg) | Op::TanH(arg) | Op::Exp(arg) => {
                    count += count_nodes(arg);
                }
            }

            count
        }

        let count = count_nodes(self);

        self.clear_visited();

        count
    }

    pub(crate) fn clear_visited(&self) {
        let inner: &mut ValueInner = &mut self.inner.borrow_mut();
        if inner.visited {
            inner.visited = false;
            match &inner.op {
                Op::Value => {},
                Op::Add(lhs, rhs) | Op::Mul(lhs, rhs) => {
                    lhs.clear_visited();
                    rhs.clear_visited();
                },
                Op::Pow(lhs, _) => {
                    lhs.clear_visited();
                },
                Op::ReLU(arg) | Op::TanH(arg) | Op::Exp(arg) => {
                    arg.clear_visited();
                },
            }
        }
    }

    #[inline]
    pub(crate) fn id(&self) -> ValueId {
        self.inner.as_ptr() as ValueId
    }

    #[inline]
    pub fn precedence(&self) -> u32 {
        let inner = self.inner.borrow();
        match &inner.op {
            Op::Value     => 4,
            Op::Add(_, _) => 1,
            Op::Mul(_, _) => 2,
            Op::Pow(_, _) => 3,
            Op::ReLU(_)   => 4,
            Op::TanH(_)   => 4,
            Op::Exp(_)    => 4,
        }
    }

    #[inline]
    fn fmt_binary(&self, lhs: &Value, op: &str, rhs: &Value, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let precedence = self.precedence();
        if lhs.precedence() < precedence {
            write!(f, "({})", lhs)?;
        } else {
            write!(f, "{}", lhs)?;
        }
        std::fmt::Display::fmt(op, f)?;
        if rhs.precedence() < precedence {
            write!(f, "({})", rhs)?;
        } else {
            write!(f, "{}", rhs)?;
        }
        Ok(())
    }

    #[inline]
    pub fn is_value(&self) -> bool {
        self.inner.borrow().op.is_value()
    }

    #[inline]
    pub fn is_add(&self) -> bool {
        self.inner.borrow().op.is_add()
    }

    #[inline]
    pub fn is_mul(&self) -> bool {
        self.inner.borrow().op.is_mul()
    }

    #[inline]
    pub fn is_pow(&self) -> bool {
        self.inner.borrow().op.is_pow()
    }

    #[inline]
    pub fn is_relu(&self) -> bool {
        self.inner.borrow().op.is_relu()
    }

    #[inline]
    pub fn is_tanh(&self) -> bool {
        self.inner.borrow().op.is_tanh()
    }

    #[inline]
    pub fn is_exp(&self) -> bool {
        self.inner.borrow().op.is_exp()
    }

    /// This is meant for changing inputs.
    /// After changing inputs with this calling `Value::refresh()` on `Loss::total` is required.
    /// Don't mess with this otherwise.
    #[inline]
    pub fn assign(&mut self, value: Number) {
        let inner: &mut ValueInner = &mut self.inner.borrow_mut();
        inner.value = value;
        inner.grad  = 0.0;
        inner.op    = Op::Value;
        inner.k     = 0; // not sure if needed, probably not
    }

    #[inline]
    pub(crate) fn assign_both(&mut self, value: Number, grad: Number) {
        let inner: &mut ValueInner = &mut self.inner.borrow_mut();
        inner.value = value;
        inner.grad  = grad;
    }
}

impl Default for Value {
    #[inline]
    fn default() -> Self {
        Value::new(0.0)
    }
}

impl AsRef<Value> for Value {
    #[inline]
    fn as_ref(&self) -> &Value {
        self
    }
}

impl std::iter::Sum<Value> for Value {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(first) = iter.next() {
            let mut sum = first.as_ref().clone();
            for value in iter {
                sum += value.as_ref();
            }
            return sum;
        }
        Value::new(0.0)
    }
}

impl std::iter::Sum<&Value> for Value {
    #[inline]
    fn sum<I: Iterator>(mut iter: I) -> Value
    where I::Item: AsRef<Value> {
        if let Some(first) = iter.next() {
            let mut sum = first.as_ref().clone();
            for value in iter {
                sum += value.as_ref();
            }
            return sum;
        }
        Value::new(0.0)
    }
}

impl std::iter::Product<Value> for Value {
    #[inline]
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(first) = iter.next() {
            let mut product = first.as_ref().clone();
            for value in iter {
                product *= value.as_ref();
            }
            return product;
        }
        Value::new(1.0)
    }
}

impl std::iter::Product<&Value> for Value {
    #[inline]
    fn product<I: Iterator>(mut iter: I) -> Self
    where I::Item: AsRef<Value> {
        if let Some(first) = iter.next() {
            let mut product = first.as_ref().clone();
            for value in iter {
                product *= value.as_ref();
            }
            return product;
        }
        Value::new(1.0)
    }
}

impl From<f32> for Value {
    #[inline]
    fn from(value: f32) -> Self {
        Value::new(value as Number)
    }
}

impl From<f64> for Value {
    #[inline]
    fn from(value: f64) -> Self {
        Value::new(value as Number)
    }
}

impl From<u32> for Value {
    #[inline]
    fn from(value: u32) -> Self {
        Value::new(value as Number)
    }
}

impl From<u64> for Value {
    #[inline]
    fn from(value: u64) -> Self {
        Value::new(value as Number)
    }
}

impl From<i32> for Value {
    #[inline]
    fn from(value: i32) -> Self {
        Value::new(value as Number)
    }
}

impl From<i64> for Value {
    #[inline]
    fn from(value: i64) -> Self {
        Value::new(value as Number)
    }
}

impl From<Value> for Number {
    #[inline]
    fn from(value: Value) -> Self {
        value.value()
    }
}

impl PartialEq for Value {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.as_ptr() == other.inner.as_ptr()
    }
}

impl Eq for Value {}

impl Debug for Value {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner: &ValueInner = &self.inner.borrow();
        write!(f, "Value(value={}, grad={}, k={})", inner.value, inner.grad, inner.k)
    }
}

impl Display for Value {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.borrow();
        match &inner.op {
            Op::Value => std::fmt::Display::fmt(&self.value(), f),
            Op::Add(lhs, rhs) => self.fmt_binary(lhs, " + ", rhs, f),
            Op::Mul(lhs, rhs) => self.fmt_binary(lhs, " * ", rhs, f),
            Op::Pow(lhs, rhs) => {
                let precedence = self.precedence();
                if lhs.precedence() < precedence {
                    write!(f, "({}) ^ {}", lhs, rhs)
                } else {
                    write!(f, "{} ^ {}", lhs, rhs)
                }
            },
            Op::ReLU(arg) => write!(f, "relu({})", arg),
            Op::TanH(arg) => write!(f, "tanh({})", arg),
            Op::Exp(arg)  => write!(f, "exp({})", arg),
        }
    }
}

impl Add for Value {
    type Output = Value;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Value::from_op(self.value() + rhs.value(), Op::Add(self, rhs))
    }
}

impl Add<Number> for Value {
    type Output = Value;

    #[inline]
    fn add(self, rhs: Number) -> Self::Output {
        if rhs == 0.0 {
            self
        } else {
            self + Value::new(rhs)
        }
    }
}

impl<B> Add<B> for &Value
where
    B: std::borrow::Borrow<Value>,
{
    type Output = Value;

    #[inline]
    fn add(self, rhs: B) -> Self::Output {
        self.clone() + rhs.borrow().clone()
    }
}

impl Add<Number> for &Value {
    type Output = Value;

    #[inline]
    fn add(self, rhs: Number) -> Self::Output {
        self.clone() + rhs
    }
}

impl AddAssign for Value {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        let _ = replace(self, self.clone() + rhs);
    }
}

impl AddAssign<&Value> for Value {
    #[inline]
    fn add_assign(&mut self, rhs: &Value) {
        let _ = replace(self, self.clone() + rhs.clone());
    }
}

impl AddAssign<Number> for Value {
    #[inline]
    fn add_assign(&mut self, rhs: Number) {
        let _ = replace(self, self.clone() + rhs);
    }
}

impl Mul for Value {
    type Output = Value;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Value::from_op(self.value() * rhs.value(), Op::Mul(self, rhs))
    }
}

impl Mul<Number> for Value {
    type Output = Value;

    #[inline]
    fn mul(self, rhs: Number) -> Self::Output {
        if rhs == 1.0 {
            self
        } else {
            self * Value::new(rhs)
        }
    }
}

impl<B> Mul<B> for &Value
where
    B: std::borrow::Borrow<Value>,
{
    type Output = Value;

    #[inline]
    fn mul(self, rhs: B) -> Self::Output {
        self.clone() * rhs.borrow().clone()
    }
}

impl Mul<Number> for &Value {
    type Output = Value;

    #[inline]
    fn mul(self, rhs: Number) -> Self::Output {
        self.clone() * rhs
    }
}

impl MulAssign for Value {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        let _ = replace(self, self.clone() * rhs);
    }
}

impl MulAssign<Number> for Value {
    #[inline]
    fn mul_assign(&mut self, rhs: Number) {
        let _ = replace(self, self.clone() * rhs);
    }
}

impl MulAssign<&Value> for Value {
    #[inline]
    fn mul_assign(&mut self, rhs: &Value) {
        let _ = replace(self, self.clone() * rhs.clone());
    }
}

impl Neg for Value {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Neg for &Value {
    type Output = Value;

    #[inline]
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl Sub for Value {
    type Output = Value;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.is_value() {
            return self + -rhs.value();
        }
        self + -rhs
    }
}

impl Sub<Number> for Value {
    type Output = Value;

    #[inline]
    fn sub(self, rhs: Number) -> Self::Output {
        self + -rhs
    }
}

impl<B> Sub<B> for &Value
where
    B: std::borrow::Borrow<Value>,
{
    type Output = Value;

    #[inline]
    fn sub(self, rhs: B) -> Self::Output {
        self.clone() - rhs.borrow().clone()
    }
}

impl Sub<Number> for &Value {
    type Output = Value;

    #[inline]
    fn sub(self, rhs: Number) -> Self::Output {
        self.clone() + -rhs
    }
}

impl SubAssign for Value {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        let _ = replace(self, self.clone() - rhs);
    }
}

impl SubAssign<Number> for Value {
    #[inline]
    fn sub_assign(&mut self, rhs: Number) {
        let _ = replace(self, self.clone() - rhs);
    }
}

impl SubAssign<&Value> for Value {
    #[inline]
    fn sub_assign(&mut self, rhs: &Value) {
        let _ = replace(self, self.clone() - rhs.clone());
    }
}

impl Div for Value {
    type Output = Value;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_value() {
            return self * (1.0 / rhs.value());
        }
        self * rhs.pow(-1.0)
    }
}

impl Div<Number> for Value {
    type Output = Value;

    #[inline]
    fn div(self, rhs: Number) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<B> Div<B> for &Value
where
    B: std::borrow::Borrow<Value>,
{
    type Output = Value;

    #[inline]
    fn div(self, rhs: B) -> Self::Output {
        self.clone() / rhs.borrow().clone()
    }
}

impl Div<Number> for &Value {
    type Output = Value;

    #[inline]
    fn div(self, rhs: Number) -> Self::Output {
        self.clone() / rhs
    }
}

impl DivAssign for Value {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        let _ = replace(self, self.clone() / rhs);
    }
}

impl DivAssign<Number> for Value {
    #[inline]
    fn div_assign(&mut self, rhs: Number) {
        let _ = replace(self, self.clone() / rhs);
    }
}

impl DivAssign<&Value> for Value {
    #[inline]
    fn div_assign(&mut self, rhs: &Value) {
        let _ = replace(self, self.clone() / rhs.clone());
    }
}
