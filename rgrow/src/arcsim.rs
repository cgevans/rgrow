use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use ndarray::ArrayView2;

use crate::{
    base::Tile,
    canvas::Canvas,
    simulation::Simulation,
    state::{State, StateCreate},
    system::{System, SystemInfo, TileBondInfo},
};

trait ArcSimulation: Simulation + AMDStateRef {}

struct ArcSim<Sy: System, St: State> {
    system: Sy,
    state: Vec<Arc<Mutex<St>>>,
    default_state_size: (usize, usize),
}

trait AMDStateRef {
    fn state_ref(&self, state_index: usize) -> Arc<Mutex<dyn State>>;
    // fn canvas_ref(&self, state_index: usize) -> ArrayView2<Tile>;
}

impl<Sy: System + SystemInfo, St: State + Canvas + 'static> AMDStateRef for ArcSim<Sy, St> {
    fn state_ref(&self, state_index: usize) -> Arc<Mutex<dyn State>> {
        self.state[state_index].clone()
    }
}

impl<Sy: System + SystemInfo + TileBondInfo, St: State + StateCreate> Simulation
    for ArcSim<Sy, St>
{
    fn evolve(
        &mut self,
        state_index: usize,
        bounds: crate::system::EvolveBounds,
    ) -> Result<crate::system::EvolveOutcome, crate::base::GrowError> {
        self.system
            .evolve(self.state[state_index].lock().unwrap().deref_mut(), bounds)
    }
    fn n_states(&self) -> usize {
        self.state.len()
    }

    fn add_state(&mut self) -> Result<usize, crate::base::GrowError> {
        self.state.push(Arc::new(Mutex::new(
            self.system.new_state(self.default_state_size)?,
        )));
        Ok(self.state.len() - 1)
    }

    fn draw_size(&self, state_index: usize) -> (u32, u32) {
        self.state[state_index].lock().unwrap().deref().draw_size()
    }

    fn draw(&self, state_index: usize, frame: &mut [u8]) {
        let state = &self.state[state_index].lock().unwrap();
        state.draw(frame, self.system.tile_colors());
    }

    fn state_ref(&self, state_index: usize) -> &dyn State {
        todo!()
    }
}

impl<Sy: System + SystemInfo, St: State> SystemInfo for ArcSim<Sy, St> {
    fn tile_concs(&self) -> Vec<f64> {
        self.system.tile_concs()
    }

    fn tile_stoics(&self) -> Vec<f64> {
        self.system.tile_stoics()
    }
}
