//$ Simulations hold both a model and a state, so that they can be handled without knowing the specific model, state, or canvas being used.

use std::time::Duration;

use rand::prelude::SmallRng;

use crate::base::{GrowError, NumEvents, NumTiles};
use crate::state::{State, StateCreate};
use crate::system::{EvolveOutcome, System};
use crate::system::{SystemWithStateCreate, TileBondInfo};

#[derive(Debug, Copy, Clone)]
pub struct EvolveBounds {
    pub events: Option<NumEvents>,
    pub time: Option<f64>,
    pub size_min: Option<NumTiles>,
    pub size_max: Option<NumTiles>,
    pub wall_time: Option<Duration>,
}
pub(crate) struct ConcreteSimulation<Sy: System<St>, St: State> {
    pub system: Sy,
    pub states: Vec<St>,
    pub rng: SmallRng,
}
pub trait Simulation: Send {
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError>;
    fn state_ref(&self, state_index: usize) -> &dyn State;
    fn add_state(&mut self, shape: (usize, usize)) -> Result<usize, GrowError>;
    fn add_n_states(&mut self, n: usize, shape: (usize, usize)) -> Result<Vec<usize>, GrowError> {
        let mut indices = Vec::with_capacity(n);
        for _ in 0..n {
            indices.push(self.add_state(shape)?);
        }
        Ok(indices)
    }
    fn draw_size(&self, state_index: usize) -> (u32, u32);
    fn draw(&self, state_index: usize, frame: &mut [u8]);

    #[cfg(feature = "use_rayon")]
    fn evolve_all(&mut self, bounds: EvolveBounds) -> Vec<Result<EvolveOutcome, GrowError>>;
}

impl<
        Sy: System<St> + SystemWithStateCreate<St> + TileBondInfo + Send + Sync,
        St: State + StateCreate,
    > Simulation for ConcreteSimulation<Sy, St>
{
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError> {
        self.system.evolve(
            &mut self.states[state_index],
            &mut self.rng,
            bounds.events,
            bounds.time,
            bounds.size_min,
            bounds.size_max,
            bounds.wall_time,
        )
    }
    fn state_ref(&self, state_index: usize) -> &dyn State {
        &self.states[state_index]
    }
    fn draw_size(&self, state_index: usize) -> (u32, u32) {
        self.states[state_index].draw_size()
    }
    fn draw(&self, state_index: usize, frame: &mut [u8]) {
        let state = &self.states[state_index];
        state.draw(frame, self.system.tile_colors());
    }

    fn add_state(&mut self, shape: (usize, usize)) -> Result<usize, GrowError> {
        self.states.push(self.system.new_state(shape)?);
        Ok(self.states.len() - 1)
    }

    #[cfg(feature = "use_rayon")]
    fn evolve_all(&mut self, bounds: EvolveBounds) -> Vec<Result<EvolveOutcome, GrowError>> {
        use rand::SeedableRng;
        use rayon::prelude::*;
        let sys = &self.system;
        self.states
            .par_iter_mut()
            .map(|state| {
                sys.evolve(
                    state,
                    &mut SmallRng::from_entropy(),
                    bounds.events,
                    bounds.time,
                    bounds.size_min,
                    bounds.size_max,
                    bounds.wall_time,
                )
            })
            .collect()
    }
}
