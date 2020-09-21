use super::*;
//use ndarray::prelude::*;
//use ndarray::Zip;
use rand::thread_rng;
use rand::{distributions::Uniform, distributions::WeightedIndex, prelude::Distribution};
#[cfg(feature = "rayon")] use rayon::prelude::*;
#[cfg(feature = "rayon")] use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "rayon")] use std::sync::Arc;
//use std::convert::{TryFrom, TryInto};

pub struct FFSRun {
    pub level_list: Vec<FFSLevel>,
    pub dimerization_rate: f64
}

impl FFSRun {
    pub fn create(
        system: &StaticKTAM,
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

        let dimerization_rate = System::<CanvasSquare>::calc_dimers(system).iter().fold(0., |acc, d| acc + d.formation_rate);

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

pub struct FFSLevel {
    pub state_list: Vec<State2DQT<StaticKTAM, NullStateTracker>>,
    pub previous_list: Vec<usize>,
    pub p_r: f64,
    pub target_size: NumTiles,
}

impl FFSLevel {
    #[cfg(not(feature = "use_rayon"))]
    pub fn next_level(&self, system: &StaticKTAM, size_step: u32, max_events: u64) -> Self {
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

    #[cfg(feature = "use_rayon")]
    pub fn next_level(&self, system: &StaticKTAM, size_step: u32, max_events: u64) -> Self {
        let mut rng = thread_rng();

        let mut n_attempts = AtomicUsize::new(0);
        let mut n_successes = AtomicUsize::new(0);

        let sysref = Arc::new(system);

        let num_states = self.state_list.len() as usize;

        let target_size = (self.target_size + size_step);

        let chooser = Uniform::new(0, self.state_list.len());

        let state_list = rayon::iter::repeat(()).map( |_| {
            if n_successes.load(Ordering::SeqCst) >= num_states {return None};
            let mut rng = rand::thread_rng();
            let i_old_state = chooser.sample(&mut rng);
            let mut state = self.state_list[i_old_state].clone();
            state.evolve_in_size_range_events_max(*sysref, 0, target_size, max_events);
            if state.ntiles() == target_size {
                n_attempts.fetch_add(1, Ordering::SeqCst);
                n_successes.fetch_add(1, Ordering::SeqCst);
                Some(Some(state))
            } else {
                n_attempts.fetch_add(1, Ordering::SeqCst);
                Some(None)
            }
        }).while_some().filter_map(|x| x).collect::<Vec<_>>();


        let mut previous_list = Vec::new();


        let p_r = (n_successes.load(Ordering::SeqCst) as f64) / (n_attempts.load(Ordering::SeqCst) as f64);

        Self {
            state_list: state_list,
            previous_list: previous_list,
            p_r,
            target_size,
        }
    }

    pub fn nmers_from_dimers(
        system: &StaticKTAM,
        num_states: usize,
        canvas_size: CanvasLength,
        max_events: u64,
        next_size: NumTiles
    ) -> Self {
        let mut rng = thread_rng();

        let dimers = System::<CanvasSquare>::calc_dimers(system);

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
