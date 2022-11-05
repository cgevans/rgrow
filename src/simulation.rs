//$ Simulations hold both a model and a state, so that they can be handled without knowing the specific model, state, or canvas being used.

use rand::prelude::SmallRng;

use crate::base::GrowError;
use crate::state::{State, StateCreate};
use crate::system::{EvolveBounds, EvolveOutcome, System, SystemInfo};
use crate::system::{SystemWithStateCreate, TileBondInfo};

pub(crate) struct ConcreteSimulation<Sy: System<St>, St: State> {
    pub system: Sy,
    pub states: Vec<St>,
    pub default_state_size: (usize, usize),
    pub rng: SmallRng,
}
pub trait Simulation: Send + Sync + SystemInfo {
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError>;
    fn state_ref(&self, state_index: usize) -> &dyn State;
    fn n_states(&self) -> usize;
    fn add_state(&mut self) -> Result<usize, GrowError>;
    fn add_n_states(&mut self, n: usize) -> Result<Vec<usize>, GrowError> {
        let mut indices = Vec::with_capacity(n);
        for _ in 0..n {
            indices.push(self.add_state()?);
        }
        Ok(indices)
    }
    fn draw_size(&self, state_index: usize) -> (u32, u32);
    fn draw(&self, state_index: usize, frame: &mut [u8]);

    #[cfg(feature = "use_rayon")]
    fn evolve_all(&mut self, bounds: EvolveBounds) -> Vec<Result<EvolveOutcome, GrowError>>;
}

impl<
        Sy: System<St> + SystemWithStateCreate<St> + TileBondInfo + Send + Sync + SystemInfo,
        St: State + StateCreate,
    > Simulation for ConcreteSimulation<Sy, St>
{
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError> {
        self.system
            .evolve(&mut self.states[state_index], &mut self.rng, bounds)
    }
    fn n_states(&self) -> usize {
        self.states.len()
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

    fn add_state(&mut self) -> Result<usize, GrowError> {
        self.states
            .push(self.system.new_state(self.default_state_size)?);
        Ok(self.states.len() - 1)
    }

    #[cfg(feature = "use_rayon")]
    fn evolve_all(&mut self, bounds: EvolveBounds) -> Vec<Result<EvolveOutcome, GrowError>> {
        use rand::SeedableRng;
        use rayon::prelude::*;
        let sys = &self.system;
        self.states
            .par_iter_mut()
            .map(|state| sys.evolve(state, &mut SmallRng::from_entropy(), bounds))
            .collect()
    }
}

impl<
        Sy: System<St> + SystemWithStateCreate<St> + TileBondInfo + Send + Sync + SystemInfo,
        St: State + StateCreate,
    > SystemInfo for ConcreteSimulation<Sy, St>
{
    fn tile_concs(&self) -> Vec<f64> {
        self.system.tile_concs()
    }

    fn tile_stoics(&self) -> Vec<f64> {
        self.system.tile_stoics()
    }
}
