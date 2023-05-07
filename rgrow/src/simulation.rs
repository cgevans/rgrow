//$ Simulations hold both a model and a state, so that they can be handled without knowing the specific model, state, or canvas being used.

use std::any::Any;

use ndarray::Array2;

use crate::base::{GrowError, Tile};

use crate::state::{State, StateCreate};
use crate::system::TileBondInfo;
use crate::system::{EvolveBounds, EvolveOutcome, System, SystemInfo};

pub trait Simulation: Send + Sync + SystemInfo + TileBondInfo {
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError>;
    fn state_ref(&self, state_index: usize) -> &dyn State; //std::sync::Arc<RwLock<dyn State>>;
    fn state_mut_ref(&mut self, state_index: usize) -> &mut dyn State; //std::sync::Arc<RwLock<dyn State>>;
    fn n_states(&self) -> usize;

    fn mismatch_array(&self, state_index: usize) -> Array2<usize>;
    fn n_mismatches(&self, state_index: usize) -> usize;

    fn state_keys(&self) -> Vec<usize>;

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
    fn draw_scaled(&self, state_index: usize, frame: &mut [u8], tile_size: usize, edge_size: usize);

    #[cfg(feature = "use_rayon")]
    fn evolve_all(&mut self, bounds: EvolveBounds) -> Vec<Result<EvolveOutcome, GrowError>>;

    #[cfg(feature = "use_rayon")]
    fn evolve_some(
        &mut self,
        state_indices: &[usize],
        bounds: EvolveBounds,
    ) -> Vec<Result<EvolveOutcome, GrowError>>;

    fn set_system_param(
        &mut self,
        param_name: &str,
        _value: Box<dyn Any>,
    ) -> Result<(), GrowError> {
        Err(GrowError::NoParameter(param_name.to_string()))
    }

    fn get_system_param(&self, param_name: &str) -> Result<Box<dyn Any>, GrowError> {
        Err(GrowError::NoParameter(param_name.to_string()))
    }
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

    fn mismatch_array(&self, state_index: usize) -> Array2<usize> {
        let state = &self.states[state_index]; //.lock().unwrap();
        self.system.calc_mismatch_locations(state)
    }

    fn n_mismatches(&self, state_index: usize) -> usize {
        let state = &self.states[state_index]; //.lock().unwrap();
        self.system.calc_mismatches(state) as usize
    }

    fn draw_size(&self, state_index: usize) -> (u32, u32) {
        self.states[state_index].draw_size() //.read().unwrap().draw_size()
    }
    fn draw(&self, state_index: usize, frame: &mut [u8]) {
        let state = &self.states[state_index]; //.lock().unwrap();
        state.draw(frame, self.system.tile_colors());
    }
    fn draw_scaled(
        &self,
        state_index: usize,
        frame: &mut [u8],
        tile_size: usize,
        edge_size: usize,
    ) {
        let state = &self.states[state_index]; //.lock().unwrap();
        if edge_size == 0 {
            state.draw_scaled(frame, self.system.tile_colors(), tile_size, edge_size);
        } else {
            state.draw_scaled_with_mm(
                frame,
                self.system.tile_colors(),
                self.system.calc_mismatch_locations(state),
                tile_size,
                edge_size,
            );
        }
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

    // FIXME: this implementation could be better.
    #[cfg(feature = "use_rayon")]
    fn evolve_some(
        &mut self,
        state_indices: &[usize],
        bounds: EvolveBounds,
    ) -> Vec<Result<EvolveOutcome, GrowError>> {
        use rayon::prelude::*;
        let sys = &self.system;
        self.states
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| state_indices.contains(i))
            .map(|(_, state)| sys.evolve(state, bounds))
            .collect()
    }

    fn state_keys(&self) -> Vec<usize> {
        (0..self.states.len()).collect()
    }

    fn set_system_param(&mut self, param_name: &str, value: Box<dyn Any>) -> Result<(), GrowError> {
        let needed_update = self.system.set_param(param_name, value)?;

        #[cfg(feature = "use_rayon")]
        {
            use rayon::prelude::*;
            self.states.par_iter_mut().for_each(|state| {
                self.system.update_all(state, &needed_update);
            });
        }
        #[cfg(not(feature = "use_rayon"))]
        {
            for state in self.states.iter_mut() {
                self.system.update_all(state);
            }
        }
        Ok(())
    }

    fn get_system_param(&self, param_name: &str) -> Result<Box<dyn Any>, GrowError> {
        self.system.get_param(param_name)
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

impl<Sy: System + TileBondInfo + System, St: State> TileBondInfo for ConcreteSimulation<Sy, St> {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.system.tile_color(tile_number)
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.system.tile_name(tile_number)
    }

    fn bond_name(&self, bond_number: usize) -> &str {
        self.system.bond_name(bond_number)
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        self.system.tile_colors()
    }

    fn tile_names(&self) -> Vec<String> {
        self.system.tile_names()
    }

    fn bond_names(&self) -> Vec<String> {
        self.system.bond_names()
    }
}
