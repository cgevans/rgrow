use crate::base::GrowError;
use crate::canvas::{CanvasPeriodic, CanvasSquare, CanvasTube, PointSafe2, PointSafeHere};
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::state::{NullStateTracker, QuadTreeState, StateTracked};
use crate::system::SystemWithDimers;
use crate::tileset::{CanvasType, FromTileSet, Model, Size, TileSet};

use super::*;
//use ndarray::prelude::*;
//use ndarray::Zip;
use base::{CanvasLength, NumEvents, NumTiles, Rate};

use ndarray::Array2;
use rand::Rng;
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

pub trait FFSResult: Send {
    fn nucleation_rate(&self) -> f64;
    fn forward_vec(&self) -> &Vec<f64>;
}

impl<St: State + StateTracked<NullStateTracker> + Send, Sy: System<St> + Send> FFSResult
    for FFSRun<St, Sy>
{
    fn nucleation_rate(&self) -> Rate {
        self.dimerization_rate * self.forward_prob.iter().fold(1., |acc, level| acc * *level)
    }

    fn forward_vec(&self) -> &Vec<f64> {
        &self.forward_prob
    }
}

impl TileSet {
    pub fn run_ffs(
        &self,
        varpermean2: f64,
        min_states: usize,
        target_size: NumTiles,
        cutoff_prob: f64,
        cutoff_number: usize,
        min_cutoff_size: NumTiles,
        max_init_events: NumEvents,
        max_subseq_events: NumEvents,
        start_size: NumTiles,
        size_step: NumTiles,
        keep_states: bool,
    ) -> Result<Box<dyn FFSResult>, GrowError> {
        match self.options.model {
            Model::KTAM => match self.options.canvas_type {
                CanvasType::Square => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasSquare, NullStateTracker>,
                    KTAM<QuadTreeState<CanvasSquare, NullStateTracker>>,
                >::create_from_tileset(
                    self,
                    varpermean2,
                    min_states,
                    target_size,
                    cutoff_prob,
                    cutoff_number,
                    min_cutoff_size,
                    max_init_events,
                    max_subseq_events,
                    start_size,
                    size_step,
                    keep_states,
                )?)),
                CanvasType::Periodic => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasPeriodic, NullStateTracker>,
                    KTAM<QuadTreeState<CanvasPeriodic, NullStateTracker>>,
                >::create_from_tileset(
                    self,
                    varpermean2,
                    min_states,
                    target_size,
                    cutoff_prob,
                    cutoff_number,
                    min_cutoff_size,
                    max_init_events,
                    max_subseq_events,
                    start_size,
                    size_step,
                    keep_states,
                )?)),
                CanvasType::Tube => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasTube, NullStateTracker>,
                    KTAM<QuadTreeState<CanvasTube, NullStateTracker>>,
                >::create_from_tileset(
                    self,
                    varpermean2,
                    min_states,
                    target_size,
                    cutoff_prob,
                    cutoff_number,
                    min_cutoff_size,
                    max_init_events,
                    max_subseq_events,
                    start_size,
                    size_step,
                    keep_states,
                )?)),
            },
            Model::ATAM => Err(GrowError::FFSCannotRunATAM),
            Model::OldKTAM => match self.options.canvas_type {
                CanvasType::Square => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasSquare, NullStateTracker>,
                    OldKTAM<QuadTreeState<CanvasSquare, NullStateTracker>>,
                >::create_from_tileset(
                    self,
                    varpermean2,
                    min_states,
                    target_size,
                    cutoff_prob,
                    cutoff_number,
                    min_cutoff_size,
                    max_init_events,
                    max_subseq_events,
                    start_size,
                    size_step,
                    keep_states,
                )?)),
                CanvasType::Periodic => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasPeriodic, NullStateTracker>,
                    OldKTAM<QuadTreeState<CanvasPeriodic, NullStateTracker>>,
                >::create_from_tileset(
                    self,
                    varpermean2,
                    min_states,
                    target_size,
                    cutoff_prob,
                    cutoff_number,
                    min_cutoff_size,
                    max_init_events,
                    max_subseq_events,
                    start_size,
                    size_step,
                    keep_states,
                )?)),
                CanvasType::Tube => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasTube, NullStateTracker>,
                    OldKTAM<QuadTreeState<CanvasTube, NullStateTracker>>,
                >::create_from_tileset(
                    self,
                    varpermean2,
                    min_states,
                    target_size,
                    cutoff_prob,
                    cutoff_number,
                    min_cutoff_size,
                    max_init_events,
                    max_subseq_events,
                    start_size,
                    size_step,
                    keep_states,
                )?)),
            },
        }
    }
}

pub struct FFSRun<St: State + StateTracked<NullStateTracker>, Sy: System<St>> {
    pub system: Sy,
    pub level_list: Vec<FFSLevel<St, Sy>>,
    pub dimerization_rate: f64,
    pub forward_prob: Vec<f64>,
}

impl<
        St: State + StateCreate + DangerousStateClone + StateTracked<NullStateTracker>,
        Sy: SystemWithDimers<St> + FromTileSet,
    > FFSRun<St, Sy>
{
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
            // println!(
            //     "Done with target size {}.",
            //     ret.level_list.last().unwrap().target_size
            // );
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
            // println!(
            //     "Done with target size {}.",
            //     ret.level_list.last().unwrap().target_size
            // );
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
        min_cutoff_size: NumTiles,
        canvas_size: (CanvasLength, CanvasLength),
        max_init_events: NumEvents,
        max_subseq_events: NumEvents,
        start_size: NumTiles,
        size_step: NumTiles,
        keep_states: bool,
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

        let (first_level, dimer_level) = FFSLevel::nmers_from_dimers_cvar(
            &mut ret.system,
            varpermean2,
            min_states,
            canvas_size,
            max_init_events,
            start_size,
        );

        ret.forward_prob.push(first_level.p_r);

        let mut current_size = first_level.target_size;

        ret.level_list.push(dimer_level);
        ret.level_list.push(first_level);

        let mut above_cutoff: usize = 0;

        while current_size < target_size {
            let last = ret.level_list.last_mut().unwrap();

            let next = last.next_level_cvar(
                &mut ret.system,
                varpermean2,
                min_states,
                size_step,
                max_subseq_events,
            );
            if !keep_states {
                last.drop_states();
            }
            let pf = next.p_r;
            ret.forward_prob.push(pf);
            // println!(
            //     "Done with target size {}: p_f {}, used {} trials for {} states.",
            //     last.target_size, pf, next.num_trials, next.num_states
            // );
            current_size = next.target_size;
            ret.level_list.push(next);
            if pf > cutoff_prob {
                above_cutoff += 1;
                if (above_cutoff > cutoff_number) & (current_size >= min_cutoff_size) {
                    break;
                }
            } else {
                above_cutoff = 0;
            }
        }

        ret
    }

    pub fn create_from_tileset(
        tileset: &TileSet,
        varpermean2: f64,
        min_states: usize,
        target_size: NumTiles,
        cutoff_prob: f64,
        cutoff_number: usize,
        min_cutoff_size: NumTiles,
        max_init_events: NumEvents,
        max_subseq_events: NumEvents,
        start_size: NumTiles,
        size_step: NumTiles,
        keep_states: bool,
    ) -> Result<Self, GrowError> {
        let sys = Sy::from_tileset(tileset);
        Ok(Self::create_with_constant_variance_and_size_cutoff(
            sys,
            varpermean2,
            min_states,
            target_size,
            cutoff_prob,
            cutoff_number,
            min_cutoff_size,
            match tileset.options.size {
                Size::Single(x) => (x, x),
                Size::Pair(p) => p,
            },
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
            keep_states,
        ))
    }

    pub fn dimer_conc(&self) -> f64 {
        self.level_list[0].p_r
    }
}

pub struct FFSLevel<St: State + StateTracked<NullStateTracker>, Sy: System<St>> {
    pub system: std::marker::PhantomData<Sy>,
    pub state_list: Vec<St>,
    pub previous_list: Vec<usize>,
    pub p_r: f64,
    pub num_states: usize,
    pub num_trials: usize,
    pub target_size: NumTiles,
}

impl<
        'a,
        St: State + StateCreate + DangerousStateClone + StateTracked<NullStateTracker>,
        Sy: SystemWithDimers<St>,
    > FFSLevel<St, Sy>
{
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

                if system.calc_ntiles(&state) != state.ntiles() {
                    panic!("sink {:?}", state);
                }

                system.evolve_in_size_range_events_max(
                    &mut state,
                    0,
                    target_size,
                    max_events,
                    &mut rng,
                );
                i += 1;
            }

            if state.ntiles() >= target_size {
                // duple hack >=
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
            num_trials: i,
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
                if state.total_rate() != 0. {
                    panic!("Total rate is not zero! {:?}", state);
                };
                i_old_state = chooser.sample(&mut rng);

                state.zeroed_copy_from_state_nonzero_rate(&self.state_list[i_old_state]);
                if system.calc_ntiles(&state) != state.ntiles() {
                    panic!(
                        "{:?} {:?} {:?}",
                        system.calc_ntiles(&state),
                        state,
                        &self.state_list[i_old_state]
                    );
                }
                system.evolve_in_size_range_events_max(
                    &mut state,
                    0,
                    target_size,
                    max_events,
                    &mut rng,
                );
                i += 1;
            }

            if state.ntiles() >= target_size {
                // >= hack for duples
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
            num_trials: i,
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
            num_trials: i,
        }
    }

    pub fn nmers_from_dimers_cvar(
        system: &mut Sy,
        cvar: f64,
        min_samples: usize,
        canvas_size: (CanvasLength, CanvasLength),
        max_events: u64,
        next_size: NumTiles,
    ) -> (Self, Self) {
        let mut rng = SmallRng::from_entropy();

        let dimers = system.calc_dimers();

        let mut state_list = Vec::with_capacity(min_samples);
        let mut previous_list = Vec::with_capacity(min_samples);
        let mut i = 0usize;

        let mut dimer_state_list = Vec::with_capacity(min_samples);

        let weights: Vec<_> = dimers.iter().map(|d| d.formation_rate).collect();
        let chooser = WeightedIndex::new(&weights).unwrap();

        if canvas_size.0 < 2 || canvas_size.1 < 2 {
            panic!("Canvas size too small for dimers");
        }
        let mid = PointSafe2((canvas_size.0 / 2, canvas_size.1 / 2));

        let mut num_states = 0usize;

        let mut tile_list = Vec::with_capacity(min_samples);

        let mut other: (usize, usize);

        while state_list.len() < MAX_SAMPLES {
            let mut state = St::create_raw(Array2::zeros(canvas_size)).unwrap();

            while state.ntiles() == 0 {
                let i_old_state = chooser.sample(&mut rng);
                let dimer = &dimers[i_old_state];

                match dimer.orientation {
                    Orientation::NS => {
                        other = state.move_sa_s(mid).0;
                        system.set_point(&mut state, mid.0, dimer.t1);
                        system.set_point(&mut state, other, dimer.t2);
                    }
                    Orientation::WE => {
                        other = state.move_sa_e(mid).0;
                        system.set_point(&mut state, mid.0, dimer.t1);
                        system.set_point(&mut state, other, dimer.t2);
                    }
                };

                if state.ntiles() != system.calc_ntiles(&state) {
                    println!(
                        "{:?} {} {} {:?} {:?} {:?}",
                        dimer,
                        state.ntiles(),
                        system.calc_ntiles(&state),
                        state,
                        system.event_rate_at_point(&state, PointSafeHere((6, 4))),
                        state.tile_at_point(PointSafe2((6, 4))),
                    );
                    // wait for 0.5 seconds
                    std::thread::sleep(std::time::Duration::from_millis(500));
                }

                system.evolve_in_size_range_events_max(
                    &mut state, 0, next_size, max_events, &mut rng,
                );
                i += 1;

                if state.ntiles() >= next_size {
                    // FIXME: >= for duples is a hack.  Should count properly
                    // Create (retrospectively) a dimer state
                    let mut dimer_state = St::create_raw(Array2::zeros(canvas_size)).unwrap();
                    match dimer.orientation {
                        Orientation::NS => {
                            other = dimer_state.move_sa_s(mid).0;
                            system.set_point(&mut dimer_state, mid.0, dimer.t1);
                            system.set_point(&mut dimer_state, other, dimer.t2);
                        }
                        Orientation::WE => {
                            other = dimer_state.move_sa_e(mid).0;
                            system.set_point(&mut dimer_state, mid.0, dimer.t1);
                            system.set_point(&mut dimer_state, other, dimer.t2);
                        }
                    };

                    state_list.push(state);

                    dimer_state_list.push(dimer_state);

                    if rng.gen::<bool>() {
                        tile_list.push(dimer.t1 as usize);
                    } else {
                        tile_list.push(dimer.t2 as usize);
                    }

                    previous_list.push(num_states);

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

        (
            Self {
                system: std::marker::PhantomData::<Sy>,
                state_list,
                previous_list,
                p_r,
                target_size: next_size,
                num_states: num_states,
                num_trials: i,
            },
            Self {
                system: std::marker::PhantomData::<Sy>,
                state_list: dimer_state_list,
                previous_list: tile_list,
                p_r: 1.0,
                target_size: 2,
                num_states: num_states,
                num_trials: num_states,
            },
        )
    }
}

fn variance_over_mean2(num_success: usize, num_trials: usize) -> f64 {
    let ns = num_success as f64;
    let nt = num_trials as f64;
    let p = ns / nt;
    (1. - p) / (ns)
}
