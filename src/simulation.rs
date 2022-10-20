//$ Simulations hold both a model and a state, so that they can be handled without knowing the specific model, state, or canvas being used.

use rand::prelude::SmallRng;

use crate::base::{GrowError, NumEvents, NumTiles};
use crate::canvas::PointSafe2;
use crate::state::{State, StateCreate};
use crate::system::{StepOutcome, System};
use crate::system::{SystemWithStateCreate, TileBondInfo};

#[derive(Debug, Copy, Clone)]
pub struct EvolveBounds {
    pub events: Option<NumEvents>,
    pub time: Option<f64>,
    pub size_min: Option<NumTiles>,
    pub size_max: Option<NumTiles>,
}

pub enum EvolveOutcome {
    Events,
    Time,
    SizeMin,
    SizeMax,
    NoStep(StepOutcome),
}

pub(crate) struct Simulation<Sy: System<St>, St: State> {
    pub system: Sy,
    pub states: Vec<St>,
    pub rng: SmallRng,
}

pub trait Sim {
    fn evolve(&mut self, state_index: usize, bounds: EvolveBounds) -> EvolveOutcome;
    fn state_ref(&self, state_index: usize) -> &dyn State;
    fn add_state(&mut self, shape: (usize, usize)) -> Result<usize, GrowError>;
    fn draw(&self, state_index: usize, frame: &mut [u8], scaled: usize);
}

impl<Sy: System<St> + SystemWithStateCreate<St> + TileBondInfo, St: State + StateCreate> Sim
    for Simulation<Sy, St>
{
    fn evolve(&mut self, state_index: usize, bounds: EvolveBounds) -> EvolveOutcome {
        self.system.evolve(
            &mut self.states[state_index],
            &mut self.rng,
            bounds.events,
            bounds.time,
            bounds.size_min,
            bounds.size_max,
        );
        EvolveOutcome::Events
    }
    fn state_ref(&self, state_index: usize) -> &dyn State {
        &self.states[state_index]
    }
    fn draw(&self, state_index: usize, frame: &mut [u8], scaled: usize) {
        let state = &self.states[state_index];
        let mut scy = scaled;
        let mut scx = scaled;
        let mut x = 0;
        let mut y = 0;
        for (pixel) in frame.chunks_exact_mut(4) {
            //println!("{} {}", x/scaled, y/scaled);

            let tv = unsafe { state.tile_at_point(PointSafe2((x, y))) };

            scy -= 1;
            if scy == 0 {
                scy = scaled;
                y += 1;
                if y == state.ncols() {
                    y = 0;
                    scx -= 1;
                    if scx == 0 {
                        scx = scaled;
                        x += 1;
                        if x == state.nrows() {
                            break;
                        }
                    }
                }
            }

            pixel.copy_from_slice(
                &(if tv > 0 {
                    self.system.tile_color(tv)
                } else {
                    [0, 0, 0, 0x00]
                }),
            );
        }
    }
    fn add_state(&mut self, shape: (usize, usize)) -> Result<usize, GrowError> {
        self.states.push(self.system.new_state(shape)?);
        Ok(self.states.len() - 1)
    }
}
