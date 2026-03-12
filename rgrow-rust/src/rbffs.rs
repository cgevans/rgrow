// Choose random dimer

// k0 trials to next surface

use num_traits::Zero;
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    Rng,
};

use crate::{
    base::GrowError,
    canvas::PointSafe2,
    ffs::FFSRunConfig,
    state::{ClonableState, StateEnum, StateWithCreate},
    system::{self, DimerInfo, EvolveOutcome, Orientation, System},
    units::PerSecond,
};

struct RBFFSResult {
    successes_per_surf: Vec<Vec<u64>>,
    config_trajectories: Vec<Vec<StateEnum>>,
    n_trials: usize,
    n_surfs: usize,
}

impl RBFFSResult {
    pub fn trajectory_weights_to_i(&self, i: usize) -> Vec<f64> {
        self.successes_per_surf[0..i]
            .iter()
            .map(|successes| {
                successes
                    .iter()
                    .map(|x| (*x as f64) / (self.n_trials as f64))
                    .product::<f64>()
            })
            .collect()
    }

    pub fn trajectory_weights(&self) -> Vec<f64> {
        self.trajectory_weights_to_i(self.n_surfs)
    }

    fn forward_probability_i(&self, i: usize) -> f64 {
        let twi = self.trajectory_weights_to_i(i);
        let tot: f64 = twi.iter().sum();
        let wt: f64 = twi
            .iter()
            .zip(self.successes_per_surf.iter())
            .map(|(w, ss)| *w * (ss[i] as f64) / (self.n_trials as f64))
            .sum();
        wt / tot
    }

    pub fn forward_probabilities(&self) -> Vec<f64> {
        (1..self.n_surfs)
            .map(|i| self.forward_probability_i(i))
            .collect()
    }

    pub fn trajectories(&self) -> &Vec<Vec<StateEnum>> {
        &self.config_trajectories
    }
}

pub fn run_rbffs<Sy: System, St: ClonableState + StateWithCreate<Params = (usize, usize)>>(
    system: &mut Sy,
    config: &FFSRunConfig,
) -> Result<RBFFSResult, GrowError>
where
    StateEnum: From<St>,
{
    let _ = config;
    let dimers = system.calc_dimers()?;

    let mut n_surfs = 1;

    let weights: Vec<_> = dimers.iter().map(|d| f64::from(d.formation_rate)).collect();
    let chooser = WeightedIndex::new(weights).unwrap();

    let mut successes_per_surf = Vec::new();
    let mut config_trajectories = Vec::new();

    let desired_traj = 1000;
    let mut n_traj = 0;
    let n_trials = config.min_configs; // FIXME: use other param name

    'outer: while n_traj < desired_traj {
        let mut traj = Vec::<St>::new(); // FIXME: preallocate
        let mut successes = Vec::new(); // FIXME: preallocate

        let mut base_state = state_from_dimer::<Sy, St>(
            system,
            &dimers[chooser.sample(&mut rand::rng())],
            config.canvas_size,
        )?;
        let mut next_size = 3;

        while next_size <= config.target_size {
            // FIXME: may miss by a bit
            n_surfs += 1;
            let mut successful_states = Vec::new();
            let mut n_success: u64 = 0;
            let bounds = {
                let mut b = config.subseq_bound;
                b.size_max = Some(next_size);
                b.size_min = Some(0);
                b
            };
            for _ in 0..n_trials {
                let mut state = base_state.clone();
                let outcome = system.evolve(&mut state, bounds)?;
                match outcome {
                    EvolveOutcome::ReachedSizeMin => continue,
                    EvolveOutcome::ReachedSizeMax => {
                        n_success += 1;
                        successful_states.push(state);
                    }
                    EvolveOutcome::ReachedTimeMax => continue,
                    _ => {
                        panic!("Unexpected outcome: {:?}", outcome)
                    }
                }
            }
            if n_success == 0 {
                continue 'outer;
            }
            successes.push(n_success);
            traj.push(base_state);
            base_state = successful_states.pop().unwrap(); // FIXME: choose randomly
            next_size += 1;
        }

        successes_per_surf.push(successes);
        config_trajectories.push(traj);
        n_traj += 1;
    }

    Ok(RBFFSResult {
        successes_per_surf,
        config_trajectories: config_trajectories
            .into_iter()
            .map(|traj| traj.into_iter().map(|s| s.into()).collect())
            .collect(),
        n_trials,
        n_surfs,
    })
}

pub fn state_from_dimer<
    Sy: System,
    St: ClonableState + StateWithCreate<Params = (usize, usize)>,
>(
    system: &Sy,
    dimer: &DimerInfo,
    canvas_shape: (usize, usize),
) -> Result<St, GrowError> {
    let mut state = St::empty(canvas_shape)?;
    if canvas_shape.0 < 4 || canvas_shape.1 < 4 {
        panic!("Canvas size too small for dimers");
    }
    let mid = PointSafe2((canvas_shape.0 / 2, canvas_shape.1 / 2));
    let other = match dimer.orientation {
        Orientation::NS => PointSafe2(state.move_sa_s(mid).0),
        Orientation::WE => PointSafe2(state.move_sa_e(mid).0),
    };
    // Use place_tile to properly handle double tiles
    let energy_change = system.place_tile(&mut state, mid, dimer.t1, true)?
        + system.place_tile(&mut state, other, dimer.t2, true)?;
    let cl = [(mid, dimer.t1), (other, dimer.t2)];
    state.record_event(
        &system::Event::PolymerAttachment(cl.to_vec()),
        PerSecond::zero(),
        f64::NAN,
        energy_change,
        energy_change,
        2,
    );
    Ok(state)
}
