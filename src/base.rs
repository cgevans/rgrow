use thiserror;

pub type Point = (usize, usize);
pub type NumTiles = u32;
pub type NumEvents = u64;
pub type Tile = usize;
pub type Energy = f64;
pub type Glue = u32;
pub type CanvasLength = usize;

#[derive(thiserror::Error, Debug)]
pub enum GrowError {
    #[error("can't create canvas from array of size ({0}, {1})")]
    WrongCanvasSize(usize, usize),
}

pub type GrowResult<T> = Result<T, GrowError>;

pub type Rate = f64;

/// The kTAM unitless rate, *not* including $\alpha$.  Thus, in this regime, the attachment rate of a tile
/// with $G_{mc}$ is just $e^{-G_{mc}}$, and the detachment rate of a tile bound by $bG_{se}$ is $e^{-bG_{se}}$.
struct UnitlessAdjRate(Rate);

impl UnitlessAdjRate {
    fn into_ratems(self, k_f_h: Rate) -> RateMS {
        RateMS(k_f_h * self.0)
    }
}

/// A real (M/s) rate.
struct RateMS(Rate);
