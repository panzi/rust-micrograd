use std::{ops::{Add, Mul, Neg, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign}, fmt::Display, rc::Rc, cell::RefCell, collections::HashSet, hash::Hash, mem::swap};

#[derive(Debug)]
pub enum Op {
    Value,
    Add(Value, Value),
    Mul(Value, Value),
    Pow(Value, f64),
    ReLu(Value),
}

#[derive(Debug)]
struct ValueInner {
    value: f64,
    grad: f64,
    op: Op,
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Value {
    inner: Rc<RefCell<ValueInner>>
}

impl Value {
    #[inline]
    pub fn new(value: f64) -> Self {
        Value::new_inner(value, Op::Value)
    }

    #[inline]
    fn new_inner(value: f64, op: Op) -> Self {
        Self {
            inner: Rc::new(RefCell::new(ValueInner {
                value,
                grad: 0.0,
                op,
            }))
        }
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.inner.borrow().value
    }

    #[inline]
    pub fn grad(&self) -> f64 {
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
    pub fn pow(&self, rhs: f64) -> Self {
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
                let value: f64 = (out.value > 0.0).into();
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
    pub fn backward_slow(&mut self) {
        let mut topo = self.topo_order();
        backward(&mut topo);
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

impl Display for Value {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(value={}, grad={})", self.value(), self.grad())
    }
}

impl Add for Value {
    type Output = Value;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Value::new_inner(self.value() + rhs.value(), Op::Add(self, rhs))
    }
}

impl Add<f64> for Value {
    type Output = Value;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
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

impl AddAssign<f64> for Value {
    #[inline]
    fn add_assign(&mut self, rhs: f64) {
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

impl Mul<f64> for Value {
    type Output = Value;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
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

impl MulAssign<f64> for Value {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
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
        self + -rhs
    }
}

impl Sub<f64> for Value {
    type Output = Value;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
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

impl SubAssign<f64> for Value {
    #[inline]
    fn sub_assign(&mut self, rhs: f64) {
        let mut value = self.clone() - rhs;
        swap(self, &mut value);
    }
}

impl Div for Value {
    type Output = Value;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.pow(-1.0)
    }
}

impl Div<f64> for Value {
    type Output = Value;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}

impl DivAssign for Value {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        let mut value = self.clone() / rhs;
        swap(self, &mut value);
    }
}

impl DivAssign<f64> for Value {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        let mut value = self.clone() / rhs;
        swap(self, &mut value);
    }
}
