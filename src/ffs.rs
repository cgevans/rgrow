use super::*;
//use ndarray::prelude::*;
//use ndarray::Zip;
use rand::thread_rng;
use rand::seq::SliceRandom;
//use std::convert::{TryFrom, TryInto};

pub struct FFSLevel<S: System<CanvasSquare>, T: StateTracker> {
    pub state_list: Vec<State2DQT<S,T>>,
    pub p_f: f64,
    pub target_size: NumTiles
}

impl<S: System<CanvasSquare> + Clone, T: StateTracker + Clone> FFSLevel<S, T> {
    fn next_level(&mut self, system: &S) -> Self {
        let mut rng = thread_rng();
        
        let mut state_list = Vec::new();
        let mut i = 0usize;
        let target_size = self.target_size + 1;

        while state_list.len() < self.state_list.len() {
            let mut state = self.state_list.choose(&mut rng).unwrap().clone();

            state.evolve_in_size_range_events_max(system, 0, target_size, 50_000);

            if state.ntiles() == target_size {
                state_list.push(state)
            }

            i += 1;
        }
        let p_f = (target_size as f64) / (i as f64);

        Self { state_list, p_f, target_size }
    }

    fn from_dimers(system: &S, num_states: usize) -> Self {
        
        todo!()
    }

}
