use std::{fmt::Display, iter::Sum, ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign}};
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

pub trait Rate: Clone + Copy + num_traits::Zero + std::fmt::Debug {
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

impl From<f64> for RatePS {
    fn from(value: f64) -> Self {
        RatePS(value)
    }
}

/// Entropy in kcal/mol/K.
pub struct EntropyKCMK(f64);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]

/// Concentration in M
pub struct ConcM(pub(crate) f64);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct ConcM2(f64);

impl ConcM {
    pub fn new(value: f64) -> ConcM {
        ConcM(value)
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

impl Div<ConcM> for ConcM {
    type Output = f64;
    fn div(self, other: ConcM) -> f64 {
        self.0 / other.0
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

impl From<ConcM> for f64 {
    fn from(value: ConcM) -> Self {
        value.0
    }
}

impl From<f64> for ConcM {
    fn from(value: f64) -> Self {
        ConcM(value)
    }
}

impl ConcM2 {
    pub fn sqrt(self) -> ConcM {
        ConcM(self.0.sqrt())
    }

    pub fn over_u0(self) -> ConcM {
        ConcM(self.0)
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

impl Zero for RatePMS {
    fn zero() -> Self {
        RatePMS(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Add for RatePMS {
    type Output = RatePMS;
    fn add(self, other: RatePMS) -> RatePMS {
        RatePMS(self.0 + other.0)
    }
}

impl Zero for ConcM2 {
    fn zero() -> Self {
        ConcM2(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

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

impl Neg for RatePS {
    type Output = RatePS;
    fn neg(self) -> RatePS {
        RatePS(-self.0)
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

impl Mul<f64> for RatePS {
    type Output = RatePS;
    fn mul(self, other: f64) -> RatePS {
        RatePS(self.0 * other)
    }
}


impl Sub for RatePS {
    type Output = RatePS;
    fn sub(self, other: RatePS) -> RatePS {
        RatePS(self.0 - other.0)
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

impl From<f64> for ConcM2 {
    fn from(value: f64) -> Self {
        ConcM2(value)
    }
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct RateMPS(f64);

impl Zero for RateMPS {
    fn zero() -> Self {
        RateMPS(0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Add for RateMPS {
    type Output = RateMPS;
    fn add(self, other: RateMPS) -> RateMPS {
        RateMPS(self.0 + other.0)
    }
}

impl AddAssign for RateMPS {
    fn add_assign(&mut self, other: RateMPS) {
        self.0 += other.0;
    }
}

impl SubAssign for RateMPS {
    fn sub_assign(&mut self, other: RateMPS) {
        self.0 -= other.0;
    }
}

impl rand::distr::weighted::Weight for RateMPS {
    const ZERO: Self = RateMPS(0.0);
    
    fn checked_add_assign(&mut self, v: &Self) -> Result<(), ()> {
        self.0.checked_add_assign(&v.0)
    }
}

impl From<RateMPS> for f64 {
    fn from(value: RateMPS) -> Self {
        value.0
    }
}

impl Mul<ConcM2> for RatePMS {
    type Output = RateMPS;
    fn mul(self, other: ConcM2) -> RateMPS {
        RateMPS(self.0 * other.0)
    }
}

impl Mul<f64> for RateMPS {
    type Output = RateMPS;
    fn mul(self, other: f64) -> RateMPS {
        RateMPS(self.0 * other)
    }
}

impl Div<RateMPS> for RateMPS {
    type Output = f64;
    fn div(self, other: RateMPS) -> f64 {
        self.0 / other.0
    }
}

impl From<f64> for RatePMS {
    fn from(value: f64) -> Self {
        RatePMS(value)
    }
}

impl From<RatePMS> for f64 {
    fn from(value: RatePMS) -> Self {
        value.0
    }
}

impl From<RatePS> for f64 {
    fn from(value: RatePS) -> Self {
        value.0
    }
}

impl RateMPS {
    pub fn new(value: f64) -> RateMPS {
        RateMPS(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct TimeS(f64);

impl TimeS {
    pub fn new(value: f64) -> TimeS {
        TimeS(value)
    }
}

impl Add for TimeS {
    type Output = TimeS;
    fn add(self, other: TimeS) -> TimeS {
        TimeS(self.0 + other.0)
    }
}

impl AddAssign for TimeS {
    fn add_assign(&mut self, other: TimeS) {
        self.0 += other.0;
    }
}

impl Div<RatePS> for f64 {
    type Output = TimeS;
    fn div(self, other: RatePS) -> TimeS {
        TimeS(self / other.0)
    }
}

impl Display for TimeS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<TimeS> for f64 {
    fn from(value: TimeS) -> Self {
        value.0
    }
}

impl Display for RatePS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl TimeS {
    pub fn min(self, other: TimeS) -> TimeS {
        TimeS(self.0.min(other.0))
    }

    pub fn max(self, other: TimeS) -> TimeS {
        TimeS(self.0.max(other.0))
    }
    
}

impl Sub for TimeS {
    type Output = TimeS;
    fn sub(self, other: TimeS) -> TimeS {
        TimeS(self.0 - other.0)
    }
}

impl SubAssign for TimeS {
    fn sub_assign(&mut self, other: TimeS) {
        self.0 -= other.0;
    }
}

impl std::fmt::LowerExp for TimeS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerExp::fmt(&self.0, f)
    }
}

impl std::fmt::UpperExp for TimeS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperExp::fmt(&self.0, f)
    }
}

impl std::fmt::LowerExp for RatePS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerExp::fmt(&self.0, f)
    }
}

impl std::fmt::UpperExp for RatePS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperExp::fmt(&self.0, f)
    }
}  