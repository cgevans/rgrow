#![allow(clippy::too_many_arguments)]

use crate::base::{GrowError, RgrowError};
use crate::canvas::{CanvasPeriodic, CanvasSquare, CanvasTube, PointSafe2};
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::state::{NullStateTracker, QuadTreeState, StateTracked};
use crate::system::{EvolveBounds, SystemWithDimers};
use crate::tileset::{CanvasType, FromTileSet, Model, TileSet};

use super::*;
//use ndarray::prelude::*;
//use ndarray::Zip;
use base::{NumTiles, Rate};

use ndarray::{Array2, ArrayView2};
use rand::Rng;
use rand::{distributions::Uniform, distributions::WeightedIndex, prelude::Distribution};
use rand::{prelude::SmallRng, SeedableRng};

use state::{DangerousStateClone, State, StateCreate};

use system::{Orientation, System};
//use std::convert::{TryFrom, TryInto};

#[derive(Debug, Clone)]
pub struct FFSRunConfig {
    pub constance_variance: bool,
    pub varpermean2: Option<f64>,
    pub min_configs: usize,
    pub max_configs: usize,
    pub early_cutoff: bool,
    pub cutoff_prob: Option<f64>,
    pub cutoff_number: Option<usize>,
    pub min_cutoff_size: Option<NumTiles>,
    pub init_bound: Option<EvolveBounds>,
    pub subseq_bound: Option<EvolveBounds>,
    pub start_size: NumTiles,
    pub size_step: NumTiles,
    pub keep_configs: bool,
    pub min_nuc_rate: Option<Rate>,
    pub canvas_size: (usize, usize),
    pub target_size: NumTiles,
}

impl Default for FFSRunConfig {
    fn default() -> Self {
        Self {
            constance_variance: true,
            varpermean2: Some(0.01),
            min_configs: 1000,
            max_configs: 100000,
            early_cutoff: true,
            cutoff_prob: Some(0.99),
            cutoff_number: Some(4),
            min_cutoff_size: Some(30),
            init_bound: None,
            subseq_bound: None,
            start_size: 0,
            size_step: 1,
            keep_configs: false,
            min_nuc_rate: None,
            canvas_size: (64, 64),
            target_size: 100,
        }
    }
}

pub trait FFSResult: Send + Sync {
    fn nucleation_rate(&self) -> f64;
    fn forward_vec(&self) -> &Vec<f64>;
    fn dimerization_rate(&self) -> f64;
    fn surfaces(&self) -> Vec<&dyn FFSSurface>;
}

pub trait FFSSurface: Send + Sync {
    fn get_config(&self, i: usize) -> ArrayView2<usize>;
    fn configs(&self) -> Vec<ArrayView2<usize>> {
        (0..self.num_configs())
            .map(|i| self.get_config(i))
            .collect()
    }
    fn num_configs(&self) -> usize;
}

impl TileSet {
    pub fn run_ffs(&self, config: &FFSRunConfig) -> Result<Box<dyn FFSResult>, RgrowError> {
        match self.options.model {
            Model::KTAM => match self.options.canvas_type {
                CanvasType::Square => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasSquare, NullStateTracker>,
                    KTAM<QuadTreeState<CanvasSquare, NullStateTracker>>,
                >::create_from_tileset(
                    self, config
                )?)),
                CanvasType::Periodic => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasPeriodic, NullStateTracker>,
                    KTAM<QuadTreeState<CanvasPeriodic, NullStateTracker>>,
                >::create_from_tileset(
                    self, config
                )?)),
                CanvasType::Tube => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasTube, NullStateTracker>,
                    KTAM<QuadTreeState<CanvasTube, NullStateTracker>>,
                >::create_from_tileset(
                    self, config
                )?)),
            },
            Model::ATAM => Err(GrowError::FFSCannotRunATAM.into()),
            Model::OldKTAM => match self.options.canvas_type {
                CanvasType::Square => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasSquare, NullStateTracker>,
                    OldKTAM<QuadTreeState<CanvasSquare, NullStateTracker>>,
                >::create_from_tileset(
                    self, config
                )?)),
                CanvasType::Periodic => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasPeriodic, NullStateTracker>,
                    OldKTAM<QuadTreeState<CanvasPeriodic, NullStateTracker>>,
                >::create_from_tileset(
                    self, config
                )?)),
                CanvasType::Tube => Ok(Box::new(FFSRun::<
                    QuadTreeState<CanvasTube, NullStateTracker>,
                    OldKTAM<QuadTreeState<CanvasTube, NullStateTracker>>,
                >::create_from_tileset(
                    self, config
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

impl<St: State + StateTracked<NullStateTracker>, Sy: SystemWithDimers<St> + Send + Sync> FFSResult
    for FFSRun<St, Sy>
{
    fn nucleation_rate(&self) -> Rate {
        self.dimerization_rate * self.forward_prob.iter().fold(1., |acc, level| acc * *level)
    }

    fn forward_vec(&self) -> &Vec<f64> {
        &self.forward_prob
    }

    fn surfaces(&self) -> Vec<&dyn FFSSurface> {
        self.level_list
            .iter()
            .map(|level| level as &dyn FFSSurface)
            .collect()
    }

    fn dimerization_rate(&self) -> f64 {
        self.dimerization_rate
    }
}

impl<
        St: State + StateCreate + DangerousStateClone + StateTracked<NullStateTracker>,
        Sy: SystemWithDimers<St> + FromTileSet + Send + Sync,
    > FFSRun<St, Sy>
{
    pub fn create(system: Sy, config: &FFSRunConfig) -> Self {
        let level_list = Vec::new();

        let dimerization_rate = system
            .calc_dimers()
            .iter()
            .fold(0., |acc, d| acc + d.formation_rate);

        let mut ret = Self {
            level_list,
            dimerization_rate,
            system,
            forward_prob: Vec::new(),
        };

        let (first_level, dimer_level) = FFSLevel::nmers_from_dimers(&mut ret.system, config);

        ret.forward_prob.push(first_level.p_r);

        let mut current_size = first_level.target_size;

        ret.level_list.push(dimer_level);
        ret.level_list.push(first_level);

        let mut above_cutoff: usize = 0;

        while current_size < config.target_size {
            let last = ret.level_list.last_mut().unwrap();

            let next = last.next_level(&mut ret.system, config);
            if !config.keep_configs {
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

            if config.early_cutoff {
                if pf > config.cutoff_prob.unwrap() {
                    above_cutoff += 1;
                    if (above_cutoff > config.cutoff_number.unwrap())
                        & (current_size >= config.min_cutoff_size.unwrap())
                    {
                        break;
                    }
                } else {
                    above_cutoff = 0;
                }
            }

            if let Some(min_nuc_rate) = config.min_nuc_rate {
                if ret.nucleation_rate() < min_nuc_rate {
                    break;
                }
            }
        }

        ret
    }

    pub fn create_from_tileset(
        tileset: &TileSet,
        config: &FFSRunConfig,
    ) -> Result<Self, RgrowError> {
        let sys = Sy::from_tileset(tileset)?;
        let c = {
            let mut c = config.clone();
            c.canvas_size = match tileset.options.size {
                tileset::Size::Single(x) => (x, x),
                tileset::Size::Pair(p) => p,
            };
            c
        };

        Ok(Self::create(sys, &c))
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

impl<St: State + StateTracked<NullStateTracker>, Sy: SystemWithDimers<St> + Sync + Send> FFSSurface
    for FFSLevel<St, Sy>
{
    fn get_config(&self, i: usize) -> ArrayView2<usize> {
        self.state_list[i].raw_array()
    }

    fn num_configs(&self) -> usize {
        self.state_list.len()
    }
}

impl<
        St: State + StateCreate + DangerousStateClone + StateTracked<NullStateTracker>,
        Sy: SystemWithDimers<St> + Sync + Send,
    > FFSLevel<St, Sy>
{
    pub fn drop_states(&mut self) -> &Self {
        self.state_list.drain(..);
        self
    }

    pub fn next_level(&self, system: &mut Sy, config: &FFSRunConfig) -> Self {
        let mut rng = SmallRng::from_entropy();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;
        let target_size = self.target_size + config.size_step;

        let bounds = {
            let mut b = match config.subseq_bound {
                Some(b) => b,
                None => EvolveBounds::default(),
            };
            b.size_max = Some(target_size);
            b.size_min = Some(0);
            b
        };

        let chooser = Uniform::new(0, self.state_list.len());

        let canvas_size = (self.state_list[0].nrows(), self.state_list[0].ncols());

        let cvar = if config.constance_variance {
            config.varpermean2.unwrap()
        } else {
            0.
        };

        while state_list.len() < config.max_configs {
            let mut state = St::create_raw(Array2::zeros(canvas_size)).unwrap();

            let mut i_old_state: usize = 0;

            while state.ntiles() == 0 {
                if state.total_rate() != 0. {
                    panic!("Total rate is not zero! {state:?}");
                };
                i_old_state = chooser.sample(&mut rng);

                state.zeroed_copy_from_state_nonzero_rate(&self.state_list[i_old_state]);
                debug_assert_eq!(system.calc_ntiles(&state), state.ntiles());

                system.evolve(&mut state, &mut rng, bounds).unwrap();
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

            if (variance_over_mean2(state_list.len(), i) < cvar)
                & (state_list.len() >= config.min_configs)
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
            num_states,
            num_trials: i,
        }
    }

    pub fn nmers_from_dimers(system: &mut Sy, config: &FFSRunConfig) -> (Self, Self) {
        let mut rng = SmallRng::from_entropy();

        let dimers = system.calc_dimers();

        let mut state_list = Vec::with_capacity(config.min_configs);
        let mut previous_list = Vec::with_capacity(config.min_configs);
        let mut i = 0usize;

        let mut dimer_state_list = Vec::with_capacity(config.min_configs);

        let weights: Vec<_> = dimers.iter().map(|d| d.formation_rate).collect();
        let chooser = WeightedIndex::new(&weights).unwrap();

        if config.canvas_size.0 < 4 || config.canvas_size.1 < 4 {
            panic!("Canvas size too small for dimers");
        }
        let mid = PointSafe2((config.canvas_size.0 / 2, config.canvas_size.1 / 2));

        let mut num_states = 0usize;

        let mut tile_list = Vec::with_capacity(config.min_configs);

        let mut other: (usize, usize);

        let cvar = if config.constance_variance {
            config.varpermean2.unwrap()
        } else {
            0.
        };

        let bounds = {
            let mut b = match config.subseq_bound {
                Some(b) => b,
                None => EvolveBounds::default(),
            };
            b.size_max = Some(config.start_size);
            b.size_min = Some(0);
            b
        };

        while state_list.len() < config.max_configs {
            let mut state = St::create_raw(Array2::zeros(config.canvas_size)).unwrap();

            while state.ntiles() == 0 {
                let i_old_state = chooser.sample(&mut rng);
                let dimer = &dimers[i_old_state];

                other = match dimer.orientation {
                    Orientation::NS => state.move_sa_s(mid).0,
                    Orientation::WE => state.move_sa_e(mid).0,
                };
                system.set_points(&mut state, &[(mid.0, dimer.t1), (other, dimer.t2)]);

                debug_assert_eq!(system.calc_ntiles(&state), state.ntiles());

                system.evolve(&mut state, &mut rng, bounds).unwrap();
                i += 1;

                if state.ntiles() >= config.start_size {
                    // FIXME: >= is a hack
                    // Create (retrospectively) a dimer state
                    let mut dimer_state =
                        St::create_raw(Array2::zeros(config.canvas_size)).unwrap();
                    other = match dimer.orientation {
                        Orientation::NS => dimer_state.move_sa_s(mid).0,
                        Orientation::WE => dimer_state.move_sa_e(mid).0,
                    };
                    system.set_points(&mut dimer_state, &[(mid.0, dimer.t1), (other, dimer.t2)]);

                    state_list.push(state);

                    dimer_state_list.push(dimer_state);

                    if rng.gen::<bool>() {
                        tile_list.push(dimer.t1);
                    } else {
                        tile_list.push(dimer.t2);
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

            if (variance_over_mean2(num_states, i) < cvar) & (num_states >= config.min_configs) {
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
                target_size: config.start_size,
                num_states,
                num_trials: i,
            },
            Self {
                system: std::marker::PhantomData::<Sy>,
                state_list: dimer_state_list,
                previous_list: tile_list,
                p_r: 1.0,
                target_size: 2,
                num_states,
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
