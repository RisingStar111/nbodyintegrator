use std::io::Read;

/// 3-vector with some helpful functions.
/// 
/// Notably, +,-,*,/ are symbolic, and [f64; 3] can directly .into() a V3.
#[derive(Debug, Clone, Copy)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl V3 {
    pub fn zero() -> Self {
        [0., 0., 0.].into()
    }
    /// Returns the magnitude of this vector, squared.
    pub fn sq_mag(&self) -> f64 {
        self.x*self.x + self.y*self.y + self.z*self.z
    }
    pub fn to_le_bytes(&self) -> Vec<u8> {
        [self.x.to_le_bytes(), self.y.to_le_bytes(), self.z.to_le_bytes()].concat()
    }
    /// Returns the dot product of this with another V3.
    /// 
    /// `x1*x2+y1*y2+z1*z2`
    pub fn dot(&self, other: &V3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    /// Returns a new V3 that has been rotated ('accurately') around the z-axis by angle.
    pub fn rotated_accurate(&self, angle: f64) -> V3 {
        let (x, y) = rotate_xy_accurate(self.x, self.y, angle);
        [x, y, self.z].into()
    }

    // Returns a size 3 array containing copies of the elements of this vector, in order [x,y,z].
    pub fn array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

impl From<[f64; 3]> for V3 {
    fn from(value: [f64; 3]) -> Self {
        V3 {x: value[0], y: value[1], z: value[2]}
    }
}

impl core::ops::Sub for V3 {
    type Output = V3;
    fn sub(self, rhs: Self) -> Self::Output {
        V3 {x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z}
    }
}
impl core::ops::Add for V3 {
    type Output = V3;
    fn add(self, rhs: Self) -> Self::Output {
        V3 {x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z}
    }
}
impl core::ops::Mul<f64> for V3 {
    type Output = V3;
    fn mul(self, rhs: f64) -> Self::Output {
        V3 {x: self.x * rhs, y: self.y * rhs, z: self.z * rhs}
    }
}
impl core::ops::Div<f64> for V3 {
    type Output = V3;
    fn div(self, rhs: f64) -> Self::Output {
        V3 {x: self.x / rhs, y: self.y / rhs, z: self.z / rhs}
    }
}

pub fn fast_2sum(lhs: f64, rhs: f64) -> (f64, f64) { // lhs larger, returns (sum, error)
    let error_add = lhs + rhs;
    let error_rhs = error_add - lhs;
    let error = rhs - error_rhs; // need to be wary of compiler optimising out '0'
    (error_add, error)
}

pub fn compensated_sum<'a, T: Iterator<Item = &'a f64>>(iter: T) -> f64 {
    iter.fold((0., 0.), |(acc, e), element| {
        let x = element - e;
        let t = acc + x;
        (t, (t - acc) - x)
    }).0
}
pub fn compensated_add(primary: &mut f64, value: f64, error: &mut f64) {
    let x = value - *error;
    let t = *primary + x;
    *error = (t - *primary) - x;
    *primary = t;
}

/// Rotate radians counter-clockwise in the x-y plane.
pub fn rotate_xy(x: f64, y: f64, a: f64) -> (f64, f64) {
    let cos = a.cos(); // unspecified precision - use libm crate or custom library if determinism is needed
    let sin = a.sin();
    return (
        cos * x - sin * y,
        sin * x + cos * y
    )
}

// 2.5x slower (tan is really slow) - there's likely a way to pipe this better than the compiler but even just the addmul is costly (likely already puts the tan before the sin)
// tan seems to be just sin/cos but should look into that
// realistically static mut the sin/tan constants (with a separate function?) or maybe the integrator will just have its own versions
/// Rotate radians counter-clockwise in the x-y plane using a double shear to preserve length.
/// 
/// Is only consistently more accurate if performing many thousands of rotations/small angles.
pub fn rotate_xy_accurate(x: f64, y: f64, a: f64) -> (f64, f64) {
    let sin = -a.sin(); // unspecified precision - use libm crate or custom library if determinism is needed
    let tanhalf = (a/2.).tan();
    let y = tanhalf * x + y;
    let x = x + sin * y;
    let y = tanhalf * x + y;
    return (x, y)
}