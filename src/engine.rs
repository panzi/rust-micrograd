use std::cmp::Ordering;
use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::fmt::{Display, Debug};
use std::{rc::Rc, cell::RefCell, mem::swap};

pub type Number = f64;

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
    ReLu(Value),
    TanH(Value),
    Exp(Value),
}

impl Op {
    #[inline]
    pub fn is_value(&self) -> bool {
        match self {
            Op::Value => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_add(&self) -> bool {
        match self {
            Op::Add(_, _) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_mul(&self) -> bool {
        match self {
            Op::Mul(_, _) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_pow(&self) -> bool {
        match self {
            Op::Pow(_, _) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_relu(&self) -> bool {
        match self {
            Op::ReLu(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_tanh(&self) -> bool {
        match self {
            Op::TanH(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_exp(&self) -> bool {
        match self {
            Op::Exp(_) => true,
            _ => false,
        }
    }
}

#[derive(Debug)]
struct ValueInner {
    value: Number,
    grad: Number,
    op: Op,
    visited: bool,
}

#[repr(transparent)]
#[derive(Clone)]
pub struct Value {
    inner: Rc<RefCell<ValueInner>>
}

impl Value {
    #[inline]
    pub fn new(value: Number) -> Self {
        Value::new_inner(value, Op::Value)
    }

    #[inline]
    fn new_inner(value: Number, op: Op) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                value,
                grad: 0.0,
                op,
                visited: false,
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
    pub fn zero_grad(&mut self) {
        self.inner.borrow_mut().grad = 0.0;
    }

    #[inline]
    pub fn update(&mut self, learning_rate: Number) {
        let mut inner = self.inner.borrow_mut();
        inner.value -= learning_rate * inner.grad;
    }

    #[inline]
    pub fn relu(&self) -> Self {
        let value = self.value();
        let value = if value < 0.0 { 0.0 } else { value };

        Value::new_inner(value, Op::ReLu(self.clone()))
    }

    #[inline]
    pub fn pow(&self, rhs: Number) -> Self {
        Value::new_inner(self.value().powf(rhs), Op::Pow(self.clone(), rhs))
    }

    #[inline]
    pub fn tanh(&self) -> Self {
        let value = (self.value() * 2.0).exp();
        let value = (value - 1.0) / (value + 1.0);
        Value::new_inner(value, Op::TanH(self.clone()))
    }

    #[inline]
    pub fn exp(&self) -> Self {
        Value::new_inner(self.value().exp(), Op::Exp(self.clone()))
    }

    pub fn backward(&mut self) {
        fn backward(node: &Value) {
            let mut out = node.inner.borrow_mut();
            if !out.visited {
                out.visited = true;
                let out_grad = out.grad;
                let out_value = out.value;
                match out.op.clone() {
                    Op::Value => {},
                    Op::Add(lhs, rhs) => {
                        lhs.inner.borrow_mut().grad += out_grad;
                        rhs.inner.borrow_mut().grad += out_grad;

                        drop(out);
                        backward(&rhs);
                        backward(&lhs);
                    },
                    Op::Mul(lhs, rhs) => {
                        {
                            let lhs_value = lhs.inner.borrow().value;
                            let rhs_value = rhs.inner.borrow().value;

                            lhs.inner.borrow_mut().grad += rhs_value * out_grad;
                            rhs.inner.borrow_mut().grad += lhs_value * out_grad;
                        }

                        drop(out);
                        backward(&rhs);
                        backward(&lhs);
                    },
                    Op::Pow(lhs, rhs) => {
                        {
                            let lhs_inner: &mut ValueInner = &mut lhs.inner.borrow_mut();
                            lhs_inner.grad += rhs * lhs_inner.value.powf(rhs - 1.0) * out_grad;
                        }

                        drop(out);
                        backward(&lhs);
                    },
                    Op::ReLu(arg) => {
                        let value: Number = (out_value > 0.0).into();
                        arg.inner.borrow_mut().grad += value * out_grad;

                        drop(out);
                        backward(&arg);
                    },
                    Op::TanH(arg) => {
                        let value = 1.0 - (out_value * out_value);
                        arg.inner.borrow_mut().grad += value * out_grad;

                        drop(out);
                        backward(&arg);
                    },
                    Op::Exp(arg) => {
                        arg.inner.borrow_mut().grad += out_value * out_grad;

                        drop(out);
                        backward(&arg);
                    },
                }
            }
        }

        {
            self.inner.borrow_mut().grad = 1.0;

            backward(self);
        }

        fn clear_visited(node: &mut Value) {
            let mut inner = node.inner.borrow_mut();
            if inner.visited {
                inner.visited = false;
                match &mut inner.op {
                    Op::Value => {},
                    Op::Add(lhs, rhs) | Op::Mul(lhs, rhs) => {
                        clear_visited(rhs);
                        clear_visited(lhs);
                    },
                    Op::Pow(lhs, _) => {
                        clear_visited(lhs);
                    },
                    Op::ReLu(arg) | Op::TanH(arg) | Op::Exp(arg) => {
                        clear_visited(arg);
                    },
                }
            }
        }

        clear_visited(self);
    }

    #[inline]
    pub fn precedence(&self) -> u32 {
        let inner = self.inner.borrow();
        match &inner.op {
            Op::Value     => 4,
            Op::Add(_, _) => 1,
            Op::Mul(_, _) => 2,
            Op::Pow(_, _) => 3,
            Op::ReLu(_)   => 4,
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
}

impl Default for Value {
    #[inline]
    fn default() -> Self {
        Value::new(0.0)
    }
}

impl std::iter::Sum for Value {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut sum = Value::new(0.0);
        for value in iter {
            sum = sum + value;
        }
        sum
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

impl Into<Number> for Value {
    #[inline]
    fn into(self) -> Number {
        self.value()
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
        write!(f, "Value(value={}, grad={})", self.value(), self.grad())
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
            Op::ReLu(arg) => write!(f, "relu({})", arg),
            Op::TanH(arg) => write!(f, "tanh({})", arg),
            Op::Exp(arg)  => write!(f, "exp({})", arg),
        }
    }
}

impl Add for Value {
    type Output = Value;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Value::new_inner(self.value() + rhs.value(), Op::Add(self, rhs))
    }
}

impl Add<Number> for Value {
    type Output = Value;

    #[inline]
    fn add(self, rhs: Number) -> Self::Output {
        self + Value::new(rhs)
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
        let mut value = self.clone() + rhs;
        swap(self, &mut value);
    }
}

impl AddAssign<Number> for Value {
    #[inline]
    fn add_assign(&mut self, rhs: Number) {
        let mut value = self.clone() + rhs;
        swap(self, &mut value);
    }
}

impl Mul for Value {
    type Output = Value;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Value::new_inner(self.value() * rhs.value(), Op::Mul(self, rhs))
    }
}

impl Mul<Number> for Value {
    type Output = Value;

    #[inline]
    fn mul(self, rhs: Number) -> Self::Output {
        self * Value::new(rhs)
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
        let mut value = self.clone() * rhs;
        swap(self, &mut value);
    }
}

impl MulAssign<Number> for Value {
    #[inline]
    fn mul_assign(&mut self, rhs: Number) {
        let mut value = self.clone() * rhs;
        swap(self, &mut value);
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
        self.clone() - rhs
    }
}

impl SubAssign for Value {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        let mut value = self.clone() - rhs;
        swap(self, &mut value);
    }
}

impl SubAssign<Number> for Value {
    #[inline]
    fn sub_assign(&mut self, rhs: Number) {
        let mut value = self.clone() - rhs;
        swap(self, &mut value);
    }
}

impl Div for Value {
    type Output = Value;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_value() {
            return self * (1.0 / rhs.value())
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
        let mut value = self.clone() / rhs;
        swap(self, &mut value);
    }
}

impl DivAssign<Number> for Value {
    #[inline]
    fn div_assign(&mut self, rhs: Number) {
        let mut value = self.clone() / rhs;
        swap(self, &mut value);
    }
}
