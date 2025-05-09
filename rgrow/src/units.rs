
use std::{iter::Sum, ops::{Add, AddAssign, Mul, Sub, SubAssign}};
use serde::{Deserialize, Serialize};
use num_traits::identities::Zero;

trait Temperature {
    fn to_kelvin(self) -> f64;
}

trait Energy {
    fn times_beta(self, temperature: impl Temperature) -> f64;
}

/// Energy in kcal/mol.
pub struct EnergyKCM(f64);

impl Add for EnergyKCM {
    type Output = EnergyKCM;
    fn add(self, other: EnergyKCM) -> EnergyKCM {
        EnergyKCM(self.0 + other.0)
    }
}

pub trait Rate {
    fn to_per_second(self) -> RatePS;

    fn from_per_second(r: RatePS) -> Self;
}

impl Rate for f64 {
    fn to_per_second(self) -> RatePS {
        RatePS(self)
    }

    fn from_per_second(r: RatePS) -> Self {
        r.0
    }
}

impl Rate for RatePS {
    fn to_per_second(self) -> RatePS {
        self
    }

    fn from_per_second(r: RatePS) -> Self {
        r
    }
}


impl Into<RatePS> for f64 {
    fn into(self) -> RatePS {
        RatePS(self)
    }
}


/// Entropy in kcal/mol/K.
pub struct EntropyKCMK(f64);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]

/// Concentration in M
pub struct ConcM(f64);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ConcM2(f64);

impl ConcM {
    pub fn new(value: f64) -> ConcM {
        ConcM(value)
    }
}

impl Add for ConcM {
    type Output = ConcM;
    fn add(self, other: ConcM) -> ConcM {
        ConcM(self.0 + other.0)
    }
}

impl Mul<f64> for ConcM {
    type Output = ConcM;
    fn mul(self, other: f64) -> ConcM {
        ConcM(self.0 * other)
    }
}

impl Mul<ConcM> for f64 {
    type Output = ConcM;
    fn mul(self, other: ConcM) -> ConcM {
        ConcM(self * other.0)
    }
}

impl Sub for ConcM {
    type Output = ConcM;
    fn sub(self, other: ConcM) -> ConcM {
        ConcM(self.0 - other.0)
    }
}

impl Mul<ConcM> for ConcM {
    type Output = ConcM2;
    fn mul(self, other: ConcM) -> ConcM2 {
        ConcM2(self.0 * other.0)
    }
}

impl Sum for ConcM {
    fn sum<I: Iterator<Item = ConcM>>(iter: I) -> ConcM {
        iter.fold(ConcM::zero(), |acc, x| acc + x)
    }
}

impl ConcM {
    pub fn squared(self) -> ConcM2 {
        ConcM2(self.0 * self.0)
    }

    pub fn u0_times(unitless: f64) -> ConcM {
        ConcM(unitless)
    }

    pub fn over_u0(self) -> f64 {
        self.0
    }
}

impl Zero for ConcM {
    fn zero() -> Self {
        ConcM(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl ConcM2 {
    pub fn sqrt(self) -> ConcM {
        ConcM(self.0.sqrt())
    }
}

impl Add for ConcM2 {
    type Output = ConcM2;
    fn add(self, other: ConcM2) -> ConcM2 {
        ConcM2(self.0 + other.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct RatePMS(f64);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct RatePS(f64);

impl Add for RatePS {
    type Output = RatePS;
    fn add(self, other: RatePS) -> RatePS {
        RatePS(self.0 + other.0)
    }
}

impl AddAssign for RatePS {
    fn add_assign(&mut self, other: RatePS) {
        self.0 += other.0;
    }
}

impl SubAssign for RatePS {
    fn sub_assign(&mut self, other: RatePS) {
        self.0 -= other.0;
    }
}

impl Zero for RatePS {
    fn zero() -> Self {
        RatePS(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Mul<ConcM> for RatePMS {
    type Output = RatePS;
    fn mul(self, other: ConcM) -> RatePS {
        RatePS(self.0 * other.0)
    }
}

impl Mul<ConcM> for RatePS {
    type Output = RatePMS;
    fn mul(self, other: ConcM) -> RatePMS {
        RatePMS(self.0 * other.0)
    }
}

impl RatePS {
    pub fn new(value: f64) -> RatePS {
        RatePS(value)
    }
}

impl RatePMS {
    pub fn new(value: f64) -> RatePMS {
        RatePMS(value)
    }
}

impl Into<ConcM> for f64 {
    fn into(self) -> ConcM {
        ConcM(self)
    }
}

impl Into<ConcM2> for f64 {
    fn into(self) -> ConcM2 {
        ConcM2(self)
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct RateMPS(f64);

impl Mul<ConcM2> for RatePMS {
    type Output = RateMPS;
    fn mul(self, other: ConcM2) -> RateMPS {
        RateMPS(self.0 * other.0)
    }
}

impl RateMPS {
    pub fn new(value: f64) -> RateMPS {
        RateMPS(value)
    }
}
