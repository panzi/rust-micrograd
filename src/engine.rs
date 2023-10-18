use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::fmt::{Display, Debug};
use std::{rc::Rc, cell::RefCell, collections::HashSet, hash::Hash, mem::swap};

pub type Number = f64;

#[derive(Debug)]
pub enum Op {
    Value,
    Add(Value, Value),
    Mul(Value, Value),
    Pow(Value, Number),
    ReLu(Value),
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
}

#[derive(Debug)]
struct ValueInner {
    value: Number,
    grad: Number,
    op: Op,
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
    fn inner_backward(&mut self) {
        let out = self.inner.borrow_mut();
        match &out.op {
            Op::Value => {},
            Op::Add(ref lhs, ref rhs) => {
                lhs.inner.borrow_mut().grad += out.grad;
                rhs.inner.borrow_mut().grad += out.grad;
            },
            Op::Mul(ref lhs, ref rhs) => {
                lhs.inner.borrow_mut().grad += rhs.inner.borrow().value * out.grad;
                rhs.inner.borrow_mut().grad += lhs.inner.borrow().value * out.grad;
            },
            Op::Pow(ref lhs, rhs) => {
                lhs.inner.borrow_mut().grad += rhs * lhs.value().powf(rhs - 1.0) * out.grad;
            },
            Op::ReLu(ref child) => {
                let value: Number = (out.value > 0.0).into();
                child.inner.borrow_mut().grad += value * out.grad;
            }
        }
    }

    #[inline]
    fn id(&self) -> usize {
        self.inner.as_ptr() as usize
    }

    pub fn topo_order(&self) -> Vec<Self> {
        let mut topo = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();

        fn build_topo(topo: &mut Vec<Value>, visited: &mut HashSet<usize>, node: &Value) {
            let id = node.id();
            if !visited.contains(&id) {
                visited.insert(id);
                let inner = node.inner.borrow();
                match &inner.op {
                    Op::Value => {},
                    Op::Add(lhs, rhs) => {
                        build_topo(topo, visited, lhs);
                        build_topo(topo, visited, rhs);
                    },
                    Op::Mul(lhs, rhs) => {
                        build_topo(topo, visited, lhs);
                        build_topo(topo, visited, rhs);
                    },
                    Op::Pow(lhs, _) => {
                        build_topo(topo, visited, lhs);
                    },
                    Op::ReLu(child) => {
                        build_topo(topo, visited, child);
                    }
                }
                topo.push(node.clone());
            }
        }

        build_topo(&mut topo, &mut visited, self);

        topo
    }

    /// As long as the structure of the network doesn't change you
    /// can cache the result of `Value::topo_order()` and call
    /// `backward()` on that instead.
    #[inline]
    pub fn backward_slow(&mut self) {
        let mut topo = self.topo_order();
        backward(&mut topo);
    }

    #[inline]
    pub fn precedence(&self) -> u32 {
        let inner = self.inner.borrow();
        match &inner.op {
            Op::Value => 4,
            Op::Add(_, _) => 1,
            Op::Mul(_, _) => 2,
            Op::Pow(_, _) => 3,
            Op::ReLu(_) => 4,
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
}

impl Default for Value {
    #[inline]
    fn default() -> Self {
        Value::new(0.0)
    }
}

pub fn backward(topo_order: &mut [Value]) {
    if let Some(node) = topo_order.last() {
        node.inner.borrow_mut().grad = 1.0;
    }

    for node in topo_order.iter_mut().rev() {
        node.inner_backward();
    }
}

impl PartialEq for Value {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.inner.as_ptr() == other.inner.as_ptr()
    }
}

impl Eq for Value {}

impl Hash for Value {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.as_ptr().hash(state)
    }
}

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
            Op::ReLu(child) => write!(f, "relu({})", child),
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

    fn neg(self) -> Self::Output {
        self * -1.0
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
