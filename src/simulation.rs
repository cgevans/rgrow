//$ Simulations hold both a model and a state, so that they can be handled without knowing the specific model, state, or canvas being used.

use crate::base::GrowError;

use crate::state::{State, StateCreate};
use crate::system::TileBondInfo;
use crate::system::{EvolveBounds, EvolveOutcome, System, SystemInfo};

pub trait Simulation: Send + Sync + SystemInfo {
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError>;
    fn state_ref(&self, state_index: usize) -> &dyn State; //std::sync::Arc<RwLock<dyn State>>;
    fn state_mut_ref(&mut self, state_index: usize) -> &mut dyn State; //std::sync::Arc<RwLock<dyn State>>;
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

pub(crate) struct ConcreteSimulation<Sy: System, St: State> {
    pub system: Sy,
    pub states: Vec<St>,
    pub default_state_size: (usize, usize),
}

impl<Sy: System + TileBondInfo + SystemInfo, St: State + StateCreate + 'static> Simulation
    for ConcreteSimulation<Sy, St>
{
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError> {
        let state = self.states.get_mut(state_index).unwrap(); //.lock().unwrap();
        self.system.evolve(state, bounds)
    }
    fn n_states(&self) -> usize {
        self.states.len()
    }
    fn state_ref(&self, state_index: usize) -> &dyn State {
        //std::sync::Arc<RwLock<dyn State>> {
        &self.states[state_index] //.clone()
    }
    fn state_mut_ref(&mut self, state_index: usize) -> &mut dyn State {
        //std::sync::Arc<RwLock<dyn State>> {
        &mut self.states[state_index] //.clone()
    }
    fn draw_size(&self, state_index: usize) -> (u32, u32) {
        self.states[state_index].draw_size() //.read().unwrap().draw_size()
    }
    fn draw(&self, state_index: usize, frame: &mut [u8]) {
        let state = &self.states[state_index]; //.lock().unwrap();
        state.draw(frame, self.system.tile_colors());
    }

    fn add_state(&mut self) -> Result<usize, GrowError> {
        self.states.push(
            //Arc::new(RwLock::new(
            self.system.new_state(self.default_state_size)?,
        );
        //)));
        Ok(self.states.len() - 1)
    }

    #[cfg(feature = "use_rayon")]
    fn evolve_all(&mut self, bounds: EvolveBounds) -> Vec<Result<EvolveOutcome, GrowError>> {
        use rayon::prelude::*;
        let sys = &self.system;
        self.states
            .par_iter_mut()
            .map(|state| sys.evolve(state, bounds))
            .collect()
    }
}

impl<Sy: System + TileBondInfo + SystemInfo, St: State> SystemInfo for ConcreteSimulation<Sy, St> {
    fn tile_concs(&self) -> Vec<f64> {
        self.system.tile_concs()
    }

    fn tile_stoics(&self) -> Vec<f64> {
        self.system.tile_stoics()
    }
}
