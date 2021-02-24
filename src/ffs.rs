use super::*;
//use ndarray::prelude::*;
//use ndarray::Zip;
use base::{CanvasLength, NumEvents, NumTiles, Rate};

use ndarray::Array2;
use rand::{distributions::Uniform, distributions::WeightedIndex, prelude::Distribution};
use rand::{prelude::SmallRng, SeedableRng};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use state::{DangerousStateClone, State, StateCreate};
#[cfg(feature = "rayon")]
use std::sync::Arc;
#[cfg(feature = "rayon")]
use std::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};
use system::{Orientation, System};
//use std::convert::{TryFrom, TryInto};

const MAX_SAMPLES: usize = 100000;

pub struct FFSRun<St: State, Sy: System<St>> {
    pub system: Sy,
    pub level_list: Vec<FFSLevel<St, Sy>>,
    pub dimerization_rate: f64,
    pub forward_prob: Vec<f64>,
}

impl<St: State + StateCreate + DangerousStateClone, Sy: System<St>> FFSRun<St, Sy> {
    pub fn create(
        system: Sy,
        num_states: usize,
        target_size: NumTiles,
        canvas_size: CanvasLength,
        max_init_events: NumEvents,
        max_subseq_events: NumEvents,
        start_size: NumTiles,
        size_step: NumTiles,
    ) -> Self {
        let level_list = Vec::new();

        let dimerization_rate = system
            .calc_dimers()
            .iter()
            .fold(0., |acc, d| acc + d.formation_rate);

        let forward_prob = Vec::with_capacity((target_size / size_step) as usize);

        let mut ret = Self {
            level_list,
            dimerization_rate,
            system: system,
            forward_prob: forward_prob,
        };

        ret.level_list.push(FFSLevel::nmers_from_dimers(
            &mut ret.system,
            num_states,
            canvas_size,
            max_init_events,
            start_size,
        ));

        while ret.level_list.last().unwrap().target_size < target_size {
            ret.level_list
                .push(ret.level_list.last().unwrap().next_level(
                    &mut ret.system,
                    size_step,
                    max_subseq_events,
                ));
            println!(
                "Done with target size {}.",
                ret.level_list.last().unwrap().target_size
            );
        }

        ret.forward_prob
            .extend(ret.level_list.iter().map(|x| x.p_r));

        ret
    }

    pub fn create_without_history(
        system: Sy,
        num_states: usize,
        target_size: NumTiles,
        canvas_size: CanvasLength,
        max_init_events: NumEvents,
        max_subseq_events: NumEvents,
        start_size: NumTiles,
        size_step: NumTiles,
    ) -> Self {
        let level_list = Vec::new();

        let dimerization_rate = system
            .calc_dimers()
            .iter()
            .fold(0., |acc, d| acc + d.formation_rate);

        let mut ret = Self {
            level_list,
            dimerization_rate,
            system: system,
            forward_prob: Vec::new(),
        };

        let first_level = FFSLevel::nmers_from_dimers(
            &mut ret.system,
            num_states,
            canvas_size,
            max_init_events,
            start_size,
        );
        ret.forward_prob.push(first_level.p_r);
        ret.level_list.push(first_level);

        while ret.level_list.last().unwrap().target_size < target_size {
            let next = ret.level_list.pop().unwrap().next_level(
                &mut ret.system,
                size_step,
                max_subseq_events,
            );
            ret.forward_prob.push(next.p_r);
            ret.level_list.push(next);
            println!(
                "Done with target size {}.",
                ret.level_list.last().unwrap().target_size
            );
        }

        ret
    }

    pub fn create_with_constant_variance_and_size_cutoff(
        system: Sy,
        varpermean2: f64,
        min_states: usize,
        target_size: NumTiles,
        cutoff_prob: f64,
        cutoff_number: usize,
        canvas_size: CanvasLength,
        max_init_events: NumEvents,
        max_subseq_events: NumEvents,
        start_size: NumTiles,
        size_step: NumTiles,
    ) -> Self {
        let level_list = Vec::new();

        let dimerization_rate = system
            .calc_dimers()
            .iter()
            .fold(0., |acc, d| acc + d.formation_rate);

        let mut ret = Self {
            level_list,
            dimerization_rate,
            system: system,
            forward_prob: Vec::new(),
        };

        let first_level = FFSLevel::nmers_from_dimers_cvar(
            &mut ret.system,
            varpermean2,
            min_states,
            canvas_size,
            max_init_events,
            start_size,
        );
        ret.forward_prob.push(first_level.p_r);
        ret.level_list.push(first_level);

        let mut above_cutoff: usize = 0;

        while ret.level_list.last().unwrap().target_size < target_size {
            let next = ret.level_list.last().unwrap().next_level_cvar(
                &mut ret.system,
                varpermean2,
                min_states,
                size_step,
                max_subseq_events,
            );
            ret.level_list.last_mut().unwrap().drop_states();
            let pf = next.p_r;
            ret.forward_prob.push(pf);
            println!(
                "Done with target size {}: p_f {}, used {} trials for {} states.",
                ret.level_list.last().unwrap().target_size,
                pf,
                next.num_trails,
                next.num_states
            );
            ret.level_list.push(next);
            if pf > cutoff_prob {
                above_cutoff += 1;
                if above_cutoff > cutoff_number {
                    break
                }
            } else {
                above_cutoff = 0;
            }
        }

        ret
    }

    pub fn nucleation_rate(&self) -> Rate {
        self.dimerization_rate * self.forward_prob.iter().fold(1., |acc, level| acc * *level)
    }

    pub fn forward_vec(&self) -> &Vec<f64> {
        &self.forward_prob
    }

    pub fn dimer_conc(&self) -> f64 {
        self.level_list[0].p_r
    }
}

pub struct FFSLevel<St: State, Sy: System<St>> {
    pub system: std::marker::PhantomData<Sy>,
    pub state_list: Vec<St>,
    pub previous_list: Vec<usize>,
    pub p_r: f64,
    pub num_states: usize,
    pub num_trails: usize,
    pub target_size: NumTiles,
}

impl<'a, St: State + StateCreate + DangerousStateClone, Sy: System<St>> FFSLevel<St, Sy> {
    pub fn drop_states(&mut self) -> &Self {
        self.state_list.drain(..);
        self
    }

    pub fn next_level(&self, system: &mut Sy, size_step: u32, max_events: u64) -> Self {
        let mut rng = SmallRng::from_entropy();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;
        let target_size = self.target_size + size_step;

        let chooser = Uniform::new(0, self.state_list.len());

        let canvas_size = (self.state_list[0].nrows(), self.state_list[0].ncols());

        while state_list.len() < self.state_list.len() {
            let mut state = St::create_raw(Array2::zeros(canvas_size)).unwrap();

            let mut i_old_state: usize = 0;

            while state.ntiles() == 0 {
                assert!(state.total_rate() == 0.);
                i_old_state = chooser.sample(&mut rng);

                state.zeroed_copy_from_state_nonzero_rate(&self.state_list[i_old_state]);
                system.evolve_in_size_range_events_max(
                    &mut state,
                    0,
                    target_size,
                    max_events,
                    &mut rng,
                );
                i += 1;
            }

            if state.ntiles() == target_size {
                state_list.push(state);
                previous_list.push(i_old_state);
            } else {
                println!(
                    "Ran out of events: {} tiles, {} events, {} time, {} total rate.",
                    state.ntiles(),
                    state.total_events(),
                    state.time(),
                    state.total_rate(),
                );
            }
        }
        let p_r = (state_list.len() as f64) / (i as f64);
        let num_states = state_list.len();

        Self {
            state_list,
            previous_list,
            p_r,
            target_size,
            system: std::marker::PhantomData::<Sy>,
            num_states: num_states,
            num_trails: i,
        }
    }

    pub fn next_level_cvar(
        &self,
        system: &mut Sy,
        cvar: f64,
        min_samples: usize,
        size_step: u32,
        max_events: u64,
    ) -> Self {
        let mut rng = SmallRng::from_entropy();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;
        let target_size = self.target_size + size_step;

        let chooser = Uniform::new(0, self.state_list.len());

        let canvas_size = (self.state_list[0].nrows(), self.state_list[0].ncols());

        while state_list.len() < MAX_SAMPLES {
            let mut state = St::create_raw(Array2::zeros(canvas_size)).unwrap();

            let mut i_old_state: usize = 0;

            while state.ntiles() == 0 {
                assert!(state.total_rate() == 0.);
                i_old_state = chooser.sample(&mut rng);

                state.zeroed_copy_from_state_nonzero_rate(&self.state_list[i_old_state]);
                system.evolve_in_size_range_events_max(
                    &mut state,
                    0,
                    target_size,
                    max_events,
                    &mut rng,
                );
                i += 1;
            }

            if state.ntiles() == target_size {
                state_list.push(state);
                previous_list.push(i_old_state);
            } else {
                println!(
                    "Ran out of events: {} tiles, {} events, {} time, {} total rate.",
                    state.ntiles(),
                    state.total_events(),
                    state.time(),
                    state.total_rate(),
                );
            }

            if (variance_over_mean2(state_list.len(), i) < cvar) & (state_list.len() >= min_samples)
            {
                break;
            }
        }
        let p_r = (state_list.len() as f64) / (i as f64);
        let num_states = state_list.len();

        Self {
            state_list,
            previous_list,
            p_r,
            target_size,
            system: std::marker::PhantomData::<Sy>,
            num_states: num_states,
            num_trails: i,
        }
    }

    pub fn nmers_from_dimers(
        system: &mut Sy,
        num_states: usize,
        canvas_size: CanvasLength,
        max_events: u64,
        next_size: NumTiles,
    ) -> Self {
        let mut rng = SmallRng::from_entropy();

        let dimers = system.calc_dimers();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;

        let weights: Vec<_> = dimers.iter().map(|d| d.formation_rate).collect();
        let chooser = WeightedIndex::new(&weights).unwrap();

        let mid = canvas_size / 2;

        while state_list.len() < num_states {
            let mut state = St::create_raw(Array2::zeros((canvas_size, canvas_size))).unwrap();

            while state.ntiles() == 0 {
                let i_old_state = chooser.sample(&mut rng);
                let dimer = &dimers[i_old_state];

                match dimer.orientation {
                    Orientation::NS => {
                        system.set_point(&mut state, (mid, mid), dimer.t1);
                        system.set_point(&mut state, (mid + 1, mid), dimer.t2);
                    }
                    Orientation::WE => {
                        system.set_point(&mut state, (mid, mid), dimer.t1);
                        system.set_point(&mut state, (mid, mid + 1), dimer.t2);
                    }
                };

                system.evolve_in_size_range_events_max(
                    &mut state, 0, next_size, max_events, &mut rng,
                );
                i += 1;

                if state.ntiles() == next_size {
                    state_list.push(state);
                    previous_list.push(i_old_state);
                    break;
                } else {
                    if state.ntiles() != 0 {
                        panic!("{}", state.panicinfo())
                    }
                    if state.total_rate() != 0. {
                        panic!("{}", state.panicinfo())
                    };
                }
            }
        }

        let p_r = (num_states as f64) / (i as f64);
        let num_states = state_list.len();
        Self {
            system: std::marker::PhantomData::<Sy>,
            state_list,
            previous_list,
            p_r,
            target_size: next_size,
            num_states: num_states,
            num_trails: i,
        }
    }

    pub fn nmers_from_dimers_cvar(
        system: &mut Sy,
        cvar: f64,
        min_samples: usize,
        canvas_size: CanvasLength,
        max_events: u64,
        next_size: NumTiles,
    ) -> Self {
        let mut rng = SmallRng::from_entropy();

        let dimers = system.calc_dimers();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;

        let weights: Vec<_> = dimers.iter().map(|d| d.formation_rate).collect();
        let chooser = WeightedIndex::new(&weights).unwrap();

        let mid = canvas_size / 2;

        let mut num_states = 0usize;

        while state_list.len() < MAX_SAMPLES {
            let mut state = St::create_raw(Array2::zeros((canvas_size, canvas_size))).unwrap();

            while state.ntiles() == 0 {
                let i_old_state = chooser.sample(&mut rng);
                let dimer = &dimers[i_old_state];

                match dimer.orientation {
                    Orientation::NS => {
                        system.set_point(&mut state, (mid, mid), dimer.t1);
                        system.set_point(&mut state, (mid + 1, mid), dimer.t2);
                    }
                    Orientation::WE => {
                        system.set_point(&mut state, (mid, mid), dimer.t1);
                        system.set_point(&mut state, (mid, mid + 1), dimer.t2);
                    }
                };

                system.evolve_in_size_range_events_max(
                    &mut state, 0, next_size, max_events, &mut rng,
                );
                i += 1;

                if state.ntiles() == next_size {
                    state_list.push(state);
                    previous_list.push(i_old_state);
                    num_states += 1;
                    break;
                } else {
                    if state.ntiles() != 0 {
                        panic!("{}", state.panicinfo())
                    }
                    if state.total_rate() != 0. {
                        panic!("{}", state.panicinfo())
                    };
                }
            }

            if (variance_over_mean2(num_states, i) < cvar) & (num_states >= min_samples) {
                break;
            }
        }

        let p_r = (num_states as f64) / (i as f64);

        Self {
            system: std::marker::PhantomData::<Sy>,
            state_list,
            previous_list,
            p_r,
            target_size: next_size,
            num_states: num_states,
            num_trails: i,
        }
    }
}

fn variance_over_mean2(num_success: usize, num_trials: usize) -> f64 {
    let ns = num_success as f64;
    let nt = num_trials as f64;
    let p = ns / nt;
    (1. - p) / (ns)
}
