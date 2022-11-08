//$ Simulations hold both a model and a state, so that they can be handled without knowing the specific model, state, or canvas being used.

use std::sync::{Arc, RwLock};

use rand::prelude::SmallRng;

use crate::base::GrowError;
use crate::canvas::Canvas;
use crate::state::{State, StateCreate};
use crate::system::{EvolveBounds, EvolveOutcome, System, SystemInfo};
use crate::system::{SystemWithStateCreate, TileBondInfo};

pub(crate) struct ConcreteSimulation<Sy: System> {
    pub system: Sy,
    pub states: Vec<Arc<RwLock<Sy::S>>>,
    pub default_state_size: (usize, usize),
    pub rng: SmallRng,
}

pub trait Simulation: Send + Sync + SystemInfo {
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError>;
    fn state_ref(&self, state_index: usize) -> std::sync::Arc<RwLock<dyn State>>;
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

impl<Sy: SystemWithStateCreate + TileBondInfo + SystemInfo> Simulation for ConcreteSimulation<Sy>
where
    Sy::S: StateCreate + 'static,
{
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError> {
        let mut state = self.states[state_index].write().unwrap();
        self.system.evolve(&mut state, &mut self.rng, bounds)
    }
    fn n_states(&self) -> usize {
        self.states.len()
    }
    fn state_ref(&self, state_index: usize) -> std::sync::Arc<RwLock<dyn State>> {
        self.states[state_index].clone()
    }
    fn draw_size(&self, state_index: usize) -> (u32, u32) {
        self.states[state_index].read().unwrap().draw_size()
    }
    fn draw(&self, state_index: usize, frame: &mut [u8]) {
        let state = &self.states[state_index].read().unwrap();
        state.draw(frame, self.system.tile_colors());
    }

    fn add_state(&mut self) -> Result<usize, GrowError> {
        self.states.push(Arc::new(RwLock::new(
            self.system.new_state(self.default_state_size)?,
        )));
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
                    &mut state.write().unwrap(),
                    &mut SmallRng::from_entropy(),
                    bounds,
                )
            })
            .collect()
    }
}

impl<Sy: System + SystemWithStateCreate + TileBondInfo + SystemInfo> SystemInfo
    for ConcreteSimulation<Sy>
{
    fn tile_concs(&self) -> Vec<f64> {
        self.system.tile_concs()
    }

    fn tile_stoics(&self) -> Vec<f64> {
        self.system.tile_stoics()
    }
}
