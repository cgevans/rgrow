use super::*;
//use ndarray::prelude::*;
//use ndarray::Zip;
use rand::thread_rng;
use rand::{distributions::Uniform, distributions::WeightedIndex, prelude::Distribution};
//use std::convert::{TryFrom, TryInto};

pub struct FFSRun<S: System<CanvasSquare> + Clone> {
    pub level_list: Vec<FFSLevel<S>>,
    pub dimerization_rate: f64
}

impl<S: System<CanvasSquare> + Clone> FFSRun<S> {
    pub fn create(
        system: &S,
        num_states: usize,
        target_size: NumTiles,
        canvas_size: CanvasLength,
        max_init_events: NumEvents,
        max_subseq_events: NumEvents,
        start_size: NumTiles,
        size_step: NumTiles
    ) -> Self {
        let level = FFSLevel::nmers_from_dimers(system, num_states, canvas_size, max_init_events, start_size);

        let mut level_list = Vec::new();

        level_list.push(level);

        while level_list.last().unwrap().target_size < target_size {
            level_list.push(
                level_list
                    .last()
                    .unwrap()
                    .next_level(system, size_step, max_subseq_events),
            );
        }

        let dimerization_rate = system.calc_dimers().iter().fold(0., |acc, d| acc + d.formation_rate);

        Self { level_list, dimerization_rate }
    }

    pub fn nucleation_rate(&self) -> Rate {
        self.dimerization_rate *
        self.level_list
            .iter()
            .fold(1., |acc, level| acc * level.p_r)
    }

    pub fn forward_vec(&self) -> Vec<f64> {
        self.level_list.iter().map(|level| level.p_r).collect()
    }

    pub fn dimer_conc(&self) -> f64 {
        self.level_list[0].p_r
    }
}

pub struct FFSLevel<S: System<CanvasSquare> + Clone> {
    pub state_list: Vec<State2DQT<S, NullStateTracker>>,
    pub previous_list: Vec<usize>,
    pub p_r: f64,
    pub target_size: NumTiles,
}

impl<S: System<CanvasSquare> + Clone> FFSLevel<S> {
    pub fn next_level(&self, system: &S, size_step: u32, max_events: u64) -> Self {
        let mut rng = thread_rng();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;
        let target_size = self.target_size + size_step;

        let chooser = Uniform::new(0, self.state_list.len());

        while state_list.len() < self.state_list.len() {
            let i_old_state = chooser.sample(&mut rng);
            let mut state = self.state_list[i_old_state].clone();
            state.evolve_in_size_range_events_max(system, 0, target_size, max_events);

            if state.ntiles() == target_size {
                state_list.push(state);
                previous_list.push(i_old_state);
            }

            i += 1;
        }
        let p_r = (state_list.len() as f64) / (i as f64);

        Self {
            state_list,
            previous_list,
            p_r,
            target_size,
        }
    }

    pub fn nmers_from_dimers(
        system: &S,
        num_states: usize,
        canvas_size: CanvasLength,
        max_events: u64,
        next_size: NumTiles
    ) -> Self {
        let mut rng = thread_rng();

        let dimers = system.calc_dimers();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;

        let weights: Vec<_> = dimers.iter().map(|d| d.formation_rate).collect();
        let chooser = WeightedIndex::new(&weights).unwrap();

        while state_list.len() < num_states {
            let i_old_state = chooser.sample(&mut rng);

            let dimer = &dimers[i_old_state];

            let mut state = match dimer.orientation {
                Orientation::NS => State2DQT::create_ns_pair_with_tracker(
                    system,
                    dimer.t1,
                    dimer.t2,
                    canvas_size,
                    NullStateTracker(),
                ),
                Orientation::WE => State2DQT::create_we_pair_with_tracker(
                    system,
                    dimer.t1,
                    dimer.t2,
                    canvas_size,
                    NullStateTracker(),
                ),
            };

            state.evolve_in_size_range_events_max(system, 0, next_size, max_events);

            if state.ntiles() == next_size {
                state_list.push(state);
                previous_list.push(i_old_state);
            }

            i += 1;
        }

        let p_r = (num_states as f64) / (i as f64);

        Self {
            state_list,
            previous_list,
            p_r,
            target_size: next_size,
        }
    }
}
