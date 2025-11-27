use num_traits::identities::Zero;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

const R_VAL: KcalPerMolKelvin = KcalPerMolKelvin(1.98720425864083e-3); // in kcal/mol/K

// ===================
// Temperature
// ===================
pub trait Temperature: Sized {
    fn to_kelvin_m(self) -> f64;

    fn to_kelvin(self) -> Kelvin {
        Kelvin(self.to_kelvin_m())
    }

    fn to_celsius(self) -> Celsius {
        Celsius(self.to_kelvin_m() - 273.15)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct Kelvin(pub f64);

impl Kelvin {
    pub fn new(value: f64) -> Self {
        Kelvin(value)
    }
}

impl Temperature for Kelvin {
    fn to_kelvin_m(self) -> f64 {
        self.0
    }
}

impl Add for Kelvin {
    type Output = Kelvin;
    fn add(self, other: Kelvin) -> Kelvin {
        Kelvin(self.0 + other.0)
    }
}

impl Sub for Kelvin {
    type Output = Kelvin;
    fn sub(self, other: Kelvin) -> Kelvin {
        Kelvin(self.0 - other.0)
    }
}

impl AddAssign for Kelvin {
    fn add_assign(&mut self, other: Kelvin) {
        self.0 += other.0;
    }
}

impl SubAssign for Kelvin {
    fn sub_assign(&mut self, other: Kelvin) {
        self.0 -= other.0;
    }
}

impl From<Celsius> for Kelvin {
    fn from(value: Celsius) -> Self {
        Kelvin(value.to_kelvin_m())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct Celsius(pub f64);

impl Celsius {
    pub fn new(value: f64) -> Self {
        Celsius(value)
    }
}

impl Temperature for Celsius {
    fn to_kelvin_m(self) -> f64 {
        self.0 + 273.15
    }

    fn to_celsius(self) -> Celsius {
        self
    }
}

impl From<Celsius> for f64 {
    fn from(value: Celsius) -> Self {
        value.0
    }
}

impl From<f64> for Celsius {
    fn from(value: f64) -> Self {
        Celsius(value)
    }
}

impl From<f64> for Kelvin {
    fn from(value: f64) -> Self {
        Kelvin(value)
    }
}

impl From<Kelvin> for f64 {
    fn from(value: Kelvin) -> Self {
        value.0
    }
}

impl Sub<Kelvin> for Celsius {
    type Output = Celsius;
    fn sub(self, other: Kelvin) -> Celsius {
        Celsius(self.0 - other.0)
    }
}

impl Sub<Celsius> for Kelvin {
    type Output = Kelvin;
    fn sub(self, other: Celsius) -> Kelvin {
        Kelvin(self.0 - other.to_kelvin_m())
    }
}

// ===================
// Energy
// ===================
pub trait Energy {
    fn times_beta(self, temperature: impl Temperature) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct KcalPerMol(pub(crate) f64);

impl KcalPerMol {
    pub fn new(value: f64) -> Self {
        KcalPerMol(value)
    }
}

impl Energy for KcalPerMol {
    fn times_beta(self, temperature: impl Temperature) -> f64 {
        self / (R_VAL * temperature)
    }
}

impl Div<KcalPerMol> for KcalPerMol {
    type Output = f64;
    fn div(self, other: KcalPerMol) -> f64 {
        self.0 / other.0
    }
}

impl Default for KcalPerMol {
    fn default() -> Self {
        KcalPerMol(0.0)
    }
}

impl Add for KcalPerMol {
    type Output = KcalPerMol;
    fn add(self, other: KcalPerMol) -> KcalPerMol {
        KcalPerMol(self.0 + other.0)
    }
}

impl AddAssign for KcalPerMol {
    fn add_assign(&mut self, other: KcalPerMol) {
        self.0 += other.0;
    }
}

impl Sub for KcalPerMol {
    type Output = KcalPerMol;
    fn sub(self, other: KcalPerMol) -> KcalPerMol {
        KcalPerMol(self.0 - other.0)
    }
}

impl From<f64> for KcalPerMol {
    fn from(value: f64) -> Self {
        KcalPerMol(value)
    }
}

impl Zero for KcalPerMol {
    fn zero() -> Self {
        KcalPerMol(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Mul<f64> for KcalPerMol {
    type Output = KcalPerMol;
    fn mul(self, other: f64) -> KcalPerMol {
        KcalPerMol(self.0 * other)
    }
}

impl Mul<i32> for KcalPerMol {
    type Output = KcalPerMol;
    fn mul(self, other: i32) -> KcalPerMol {
        KcalPerMol(self.0 * other as f64)
    }
}

impl From<KcalPerMol> for f64 {
    fn from(value: KcalPerMol) -> Self {
        value.0
    }
}

impl approx::AbsDiffEq for KcalPerMol {
    type Epsilon = f64;
    fn default_epsilon() -> f64 {
        f64::default_epsilon()
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        f64::abs_diff_eq(&self.0, &other.0, epsilon)
    }
}

impl approx::RelativeEq for KcalPerMol {
    fn default_max_relative() -> f64 {
        f64::default_max_relative()
    }
    fn relative_eq(&self, other: &Self, epsilon: f64, max_relative: f64) -> bool {
        f64::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

impl approx::UlpsEq for KcalPerMol {
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }
    fn ulps_eq(&self, other: &Self, epsilon: f64, max_ulps: u32) -> bool {
        f64::ulps_eq(&self.0, &other.0, epsilon, max_ulps)
    }
}

// ===================
// Entropy
// ===================
pub trait Entropy {
    fn to_kcal_mol_k(self) -> KcalPerMolKelvin;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct KcalPerMolKelvin(pub(crate) f64);

impl KcalPerMolKelvin {
    pub fn new(value: f64) -> Self {
        KcalPerMolKelvin(value)
    }
}

impl Entropy for KcalPerMolKelvin {
    fn to_kcal_mol_k(self) -> KcalPerMolKelvin {
        self
    }
}

impl Default for KcalPerMolKelvin {
    fn default() -> Self {
        KcalPerMolKelvin(0.0)
    }
}

impl From<f64> for KcalPerMolKelvin {
    fn from(value: f64) -> Self {
        KcalPerMolKelvin(value)
    }
}

impl<T: Temperature> Mul<T> for KcalPerMolKelvin {
    type Output = KcalPerMol;
    fn mul(self, other: T) -> KcalPerMol {
        KcalPerMol(self.0 * other.to_kelvin_m())
    }
}

impl Mul<KcalPerMolKelvin> for Kelvin {
    type Output = KcalPerMol;
    fn mul(self, other: KcalPerMolKelvin) -> KcalPerMol {
        KcalPerMol(self.0 * other.0)
    }
}

impl Mul<KcalPerMolKelvin> for Celsius {
    type Output = KcalPerMol;
    fn mul(self, other: KcalPerMolKelvin) -> KcalPerMol {
        KcalPerMol(self.to_kelvin_m() * other.0)
    }
}

impl From<KcalPerMolKelvin> for f64 {
    fn from(value: KcalPerMolKelvin) -> Self {
        value.0
    }
}

impl Add<KcalPerMolKelvin> for KcalPerMolKelvin {
    type Output = KcalPerMolKelvin;
    fn add(self, other: KcalPerMolKelvin) -> KcalPerMolKelvin {
        KcalPerMolKelvin(self.0 + other.0)
    }
}

impl AddAssign<KcalPerMolKelvin> for KcalPerMolKelvin {
    fn add_assign(&mut self, other: KcalPerMolKelvin) {
        self.0 += other.0;
    }
}

impl Sub<KcalPerMolKelvin> for KcalPerMolKelvin {
    type Output = KcalPerMolKelvin;
    fn sub(self, other: KcalPerMolKelvin) -> KcalPerMolKelvin {
        KcalPerMolKelvin(self.0 - other.0)
    }
}

impl approx::AbsDiffEq for KcalPerMolKelvin {
    type Epsilon = f64;
    fn default_epsilon() -> f64 {
        f64::default_epsilon()
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: f64) -> bool {
        f64::abs_diff_eq(&self.0, &other.0, epsilon)
    }
}

impl approx::RelativeEq for KcalPerMolKelvin {
    fn default_max_relative() -> f64 {
        f64::default_max_relative()
    }
    fn relative_eq(&self, other: &Self, epsilon: f64, max_relative: f64) -> bool {
        f64::relative_eq(&self.0, &other.0, epsilon, max_relative)
    }
}

impl approx::UlpsEq for KcalPerMolKelvin {
    fn default_max_ulps() -> u32 {
        f64::default_max_ulps()
    }
    fn ulps_eq(&self, other: &Self, epsilon: f64, max_ulps: u32) -> bool {
        f64::ulps_eq(&self.0, &other.0, epsilon, max_ulps)
    }
}

impl num_traits::Zero for KcalPerMolKelvin {
    fn zero() -> Self {
        KcalPerMolKelvin(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

// ===================
// Concentration
// ===================
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct Molar(pub(crate) f64);

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct MolarSq(f64);

impl Molar {
    pub fn new(value: f64) -> Molar {
        Molar(value)
    }
    pub fn squared(self) -> MolarSq {
        MolarSq(self.0 * self.0)
    }
    pub fn u0_times(unitless: f64) -> Molar {
        Molar(unitless)
    }
    pub fn over_u0(self) -> f64 {
        self.0
    }
}

impl Zero for Molar {
    fn zero() -> Self {
        Molar(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Add for Molar {
    type Output = Molar;
    fn add(self, other: Molar) -> Molar {
        Molar(self.0 + other.0)
    }
}

impl Mul<f64> for Molar {
    type Output = Molar;
    fn mul(self, other: f64) -> Molar {
        Molar(self.0 * other)
    }
}

impl Mul<Molar> for f64 {
    type Output = Molar;
    fn mul(self, other: Molar) -> Molar {
        Molar(self * other.0)
    }
}

impl Div<Molar> for Molar {
    type Output = f64;
    fn div(self, other: Molar) -> f64 {
        self.0 / other.0
    }
}

impl Sub for Molar {
    type Output = Molar;
    fn sub(self, other: Molar) -> Molar {
        Molar(self.0 - other.0)
    }
}

impl Mul<Molar> for Molar {
    type Output = MolarSq;
    fn mul(self, other: Molar) -> MolarSq {
        MolarSq(self.0 * other.0)
    }
}

impl Sum for Molar {
    fn sum<I: Iterator<Item = Molar>>(iter: I) -> Molar {
        iter.fold(Molar::zero(), |acc, x| acc + x)
    }
}

impl From<Molar> for f64 {
    fn from(value: Molar) -> Self {
        value.0
    }
}

impl From<f64> for Molar {
    fn from(value: f64) -> Self {
        Molar(value)
    }
}

impl MolarSq {
    pub fn sqrt(self) -> Molar {
        Molar(self.0.sqrt())
    }
    pub fn over_u0(self) -> Molar {
        Molar(self.0)
    }
}

impl Add for MolarSq {
    type Output = MolarSq;
    fn add(self, other: MolarSq) -> MolarSq {
        MolarSq(self.0 + other.0)
    }
}

impl Zero for MolarSq {
    fn zero() -> Self {
        MolarSq(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl From<f64> for MolarSq {
    fn from(value: f64) -> Self {
        MolarSq(value)
    }
}

// ===================
// Rate
// ===================
pub trait Rate: Clone + Copy + num_traits::Zero + std::fmt::Debug {
    fn to_per_second(self) -> PerSecond;
    fn from_per_second(r: PerSecond) -> Self;
}

impl Rate for f64 {
    fn to_per_second(self) -> PerSecond {
        PerSecond(self)
    }
    fn from_per_second(r: PerSecond) -> Self {
        r.0
    }
}

impl Rate for PerSecond {
    fn to_per_second(self) -> PerSecond {
        self
    }
    fn from_per_second(r: PerSecond) -> Self {
        r
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct PerSecond(pub f64);

impl PerSecond {
    pub fn new(value: f64) -> PerSecond {
        PerSecond(value)
    }
}

impl Add for PerSecond {
    type Output = PerSecond;
    fn add(self, other: PerSecond) -> PerSecond {
        PerSecond(self.0 + other.0)
    }
}

impl AddAssign for PerSecond {
    fn add_assign(&mut self, other: PerSecond) {
        self.0 += other.0;
    }
}

impl SubAssign for PerSecond {
    fn sub_assign(&mut self, other: PerSecond) {
        self.0 -= other.0;
    }
}

impl Neg for PerSecond {
    type Output = PerSecond;
    fn neg(self) -> PerSecond {
        PerSecond(-self.0)
    }
}

impl Zero for PerSecond {
    fn zero() -> Self {
        PerSecond(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Mul<Molar> for PerSecond {
    type Output = PerMolarSecond;
    fn mul(self, other: Molar) -> PerMolarSecond {
        PerMolarSecond(self.0 * other.0)
    }
}

impl Mul<f64> for PerSecond {
    type Output = PerSecond;
    fn mul(self, other: f64) -> PerSecond {
        PerSecond(self.0 * other)
    }
}

impl Sub for PerSecond {
    type Output = PerSecond;
    fn sub(self, other: PerSecond) -> PerSecond {
        PerSecond(self.0 - other.0)
    }
}

impl From<f64> for PerSecond {
    fn from(value: f64) -> Self {
        PerSecond(value)
    }
}

impl From<PerSecond> for f64 {
    fn from(value: PerSecond) -> Self {
        value.0
    }
}

// RatePMS
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct PerMolarSecond(f64);

impl PerMolarSecond {
    pub fn new(value: f64) -> PerMolarSecond {
        PerMolarSecond(value)
    }
}

impl Zero for PerMolarSecond {
    fn zero() -> Self {
        PerMolarSecond(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Add for PerMolarSecond {
    type Output = PerMolarSecond;
    fn add(self, other: PerMolarSecond) -> PerMolarSecond {
        PerMolarSecond(self.0 + other.0)
    }
}

impl Mul<Molar> for PerMolarSecond {
    type Output = PerSecond;
    fn mul(self, other: Molar) -> PerSecond {
        PerSecond(self.0 * other.0)
    }
}

impl Mul<MolarSq> for PerMolarSecond {
    type Output = MolarPerSecond;
    fn mul(self, other: MolarSq) -> MolarPerSecond {
        MolarPerSecond(self.0 * other.0)
    }
}

impl From<f64> for PerMolarSecond {
    fn from(value: f64) -> Self {
        PerMolarSecond(value)
    }
}

impl From<PerMolarSecond> for f64 {
    fn from(value: PerMolarSecond) -> Self {
        value.0
    }
}

// RateMPS
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct MolarPerSecond(f64);

impl MolarPerSecond {
    pub fn new(value: f64) -> MolarPerSecond {
        MolarPerSecond(value)
    }
}

impl Zero for MolarPerSecond {
    fn zero() -> Self {
        MolarPerSecond(0.0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl Add for MolarPerSecond {
    type Output = MolarPerSecond;
    fn add(self, other: MolarPerSecond) -> MolarPerSecond {
        MolarPerSecond(self.0 + other.0)
    }
}

impl AddAssign for MolarPerSecond {
    fn add_assign(&mut self, other: MolarPerSecond) {
        self.0 += other.0;
    }
}

impl SubAssign for MolarPerSecond {
    fn sub_assign(&mut self, other: MolarPerSecond) {
        self.0 -= other.0;
    }
}

impl rand::distr::weighted::Weight for MolarPerSecond {
    const ZERO: Self = MolarPerSecond(0.0);
    fn checked_add_assign(&mut self, v: &Self) -> Result<(), ()> {
        self.0.checked_add_assign(&v.0)
    }
}

impl Mul<f64> for MolarPerSecond {
    type Output = MolarPerSecond;
    fn mul(self, other: f64) -> MolarPerSecond {
        MolarPerSecond(self.0 * other)
    }
}

impl Div<MolarPerSecond> for MolarPerSecond {
    type Output = f64;
    fn div(self, other: MolarPerSecond) -> f64 {
        self.0 / other.0
    }
}

impl From<MolarPerSecond> for f64 {
    fn from(value: MolarPerSecond) -> Self {
        value.0
    }
}

// ===================
// Time
// ===================
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject, IntoPyObject))]
pub struct Second(f64);

impl Second {
    pub fn new(value: f64) -> Second {
        Second(value)
    }
    pub fn min(self, other: Second) -> Second {
        Second(self.0.min(other.0))
    }
    pub fn max(self, other: Second) -> Second {
        Second(self.0.max(other.0))
    }
}

impl Add for Second {
    type Output = Second;
    fn add(self, other: Second) -> Second {
        Second(self.0 + other.0)
    }
}

impl AddAssign for Second {
    fn add_assign(&mut self, other: Second) {
        self.0 += other.0;
    }
}

impl Sub for Second {
    type Output = Second;
    fn sub(self, other: Second) -> Second {
        Second(self.0 - other.0)
    }
}

impl SubAssign for Second {
    fn sub_assign(&mut self, other: Second) {
        self.0 -= other.0;
    }
}

impl Div<PerSecond> for f64 {
    type Output = Second;
    fn div(self, other: PerSecond) -> Second {
        Second(self / other.0)
    }
}

impl Display for Second {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Second> for f64 {
    fn from(value: Second) -> Self {
        value.0
    }
}

impl Display for PerSecond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::fmt::LowerExp for Second {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerExp::fmt(&self.0, f)
    }
}

impl std::fmt::UpperExp for Second {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperExp::fmt(&self.0, f)
    }
}

impl std::fmt::LowerExp for PerSecond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::LowerExp::fmt(&self.0, f)
    }
}

impl std::fmt::UpperExp for PerSecond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::UpperExp::fmt(&self.0, f)
    }
}
