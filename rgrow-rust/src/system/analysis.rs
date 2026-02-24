use bpci::Interval;
use bpci::WilsonScore;
use rayon::prelude::*;

use crate::base::{GrowError, NumEvents, NumTiles};
use crate::state::{StateEnum, StateStatus};

use super::core::System;
use super::dispatch::SystemEnum;
use super::types::*;

pub(super) fn calc_committor<S: System>(
    sys: &mut S,
    initial_state: &StateEnum,
    cutoff_size: NumTiles,
    max_time: Option<f64>,
    max_events: Option<NumEvents>,
    num_trials: usize,
) -> Result<f64, GrowError>
where
    SystemEnum: From<S>,
{
    if num_trials == 0 {
        return Err(GrowError::NotSupported(
            "Number of trials must be greater than 0".to_string(),
        ));
    }

    let mut successes = 0;

    let mut trial_states = (0..num_trials)
        .map(|_| initial_state.clone())
        .collect::<Vec<_>>();

    let bounds = EvolveBounds {
        size_min: Some(0),
        size_max: Some(cutoff_size),
        for_time: max_time,
        for_events: max_events,
        ..Default::default()
    };

    let outcomes = sys.evolve_states(&mut trial_states, bounds);

    for outcome in outcomes.iter() {
        let outcome = outcome
            .as_ref()
            .map_err(|e| GrowError::NotSupported(e.to_string()))?;
        match outcome {
            EvolveOutcome::ReachedSizeMax => successes += 1,
            EvolveOutcome::ReachedSizeMin => {}
            _ => {
                return Err(GrowError::NotSupported(
                    "Evolve outcome not supported".to_string(),
                )); // FIXME: this should make more sense
            }
        }
    }

    Ok(successes as f64 / num_trials as f64)
}

pub(super) fn calc_committor_adaptive<S: System>(
    sys: &S,
    initial_state: &StateEnum,
    cutoff_size: NumTiles,
    max_time: Option<f64>,
    max_events: Option<NumEvents>,
    conf_interval_margin: f64,
) -> Result<(f64, usize), GrowError>
where
    SystemEnum: From<S>,
{
    use bpci::{NSuccessesSample, WilsonScore};

    let mut successes = 0u32;
    let mut num_trials = 0u32;

    let mut trial_state = initial_state.clone();

    let bounds = EvolveBounds {
        size_min: Some(0),
        size_max: Some(cutoff_size),
        for_time: max_time,
        for_events: max_events,
        ..Default::default()
    };

    while (NSuccessesSample::new(num_trials, successes)
        .unwrap()
        .wilson_score(1.960)
        .margin
        > conf_interval_margin)
        || num_trials < 1
    {
        let outcome = sys.evolve(&mut trial_state, bounds)?;
        match outcome {
            EvolveOutcome::ReachedSizeMax => {
                successes += 1;
                num_trials += 1;
                initial_state.clone_into(&mut trial_state);
            }
            EvolveOutcome::ReachedSizeMin => {
                num_trials += 1;
                initial_state.clone_into(&mut trial_state);
            }
            _ => {
                return Err(GrowError::NotSupported(
                    "Evolve outcome not supported".to_string(),
                )); // FIXME: this should make more sense
            }
        }
    }

    Ok((successes as f64 / num_trials as f64, num_trials as usize))
}

pub(super) fn calc_committors_adaptive<S: System>(
    sys: &S,
    initial_states: &[&StateEnum],
    cutoff_size: NumTiles,
    max_time: Option<f64>,
    max_events: Option<NumEvents>,
    conf_interval_margin: f64,
) -> Result<(Vec<f64>, Vec<usize>), GrowError>
where
    SystemEnum: From<S>,
{
    let results = initial_states
        .par_iter()
        .map(|initial_state| {
            calc_committor_adaptive(
                sys,
                initial_state,
                cutoff_size,
                max_time,
                max_events,
                conf_interval_margin,
            )
        })
        .collect::<Vec<_>>();

    let results: Vec<(f64, usize)> = results.into_iter().map(|r| r.unwrap()).collect();

    let committors: Vec<f64> = results.iter().map(|(c, _)| *c).collect();
    let trials: Vec<usize> = results.iter().map(|(_, t)| *t).collect();

    Ok((committors, trials))
}

pub(super) fn calc_forward_probability<S: System>(
    sys: &mut S,
    initial_state: &StateEnum,
    forward_step: NumTiles,
    max_time: Option<f64>,
    max_events: Option<NumEvents>,
    num_trials: usize,
) -> Result<f64, GrowError>
where
    SystemEnum: From<S>,
{
    if num_trials == 0 {
        return Err(GrowError::NotSupported(
            "Number of trials must be greater than 0".to_string(),
        ));
    }

    let initial_size = initial_state.n_tiles();
    let cutoff_size = initial_size + forward_step;

    let mut successes = 0;

    let mut trial_states = (0..num_trials)
        .map(|_| initial_state.clone())
        .collect::<Vec<_>>();

    let bounds = EvolveBounds {
        size_min: Some(0),
        size_max: Some(cutoff_size),
        for_time: max_time,
        for_events: max_events,
        ..Default::default()
    };

    let outcomes = sys.evolve_states(&mut trial_states, bounds);

    for outcome in outcomes.iter() {
        let outcome = outcome
            .as_ref()
            .map_err(|e| GrowError::NotSupported(e.to_string()))?;
        match outcome {
            EvolveOutcome::ReachedSizeMax => successes += 1,
            EvolveOutcome::ReachedSizeMin => {}
            _ => {
                return Err(GrowError::NotSupported(
                    "Evolve outcome not supported".to_string(),
                ));
            }
        }
    }

    Ok(successes as f64 / num_trials as f64)
}

pub(super) fn calc_forward_probability_adaptive<S: System>(
    sys: &S,
    initial_state: &StateEnum,
    forward_step: NumTiles,
    max_time: Option<f64>,
    max_events: Option<NumEvents>,
    conf_interval_margin: f64,
) -> Result<(f64, usize), GrowError>
where
    SystemEnum: From<S>,
{
    use bpci::{NSuccessesSample, WilsonScore};

    let initial_size = initial_state.n_tiles();
    let cutoff_size = initial_size + forward_step;

    let mut successes = 0u32;
    let mut num_trials = 0u32;

    let mut trial_state = initial_state.clone();

    let bounds = EvolveBounds {
        size_min: Some(0),
        size_max: Some(cutoff_size),
        for_time: max_time,
        for_events: max_events,
        ..Default::default()
    };

    while (NSuccessesSample::new(num_trials, successes)
        .unwrap()
        .wilson_score(1.960)
        .margin
        > conf_interval_margin)
        || num_trials < 1
    {
        let outcome = sys.evolve(&mut trial_state, bounds)?;
        match outcome {
            EvolveOutcome::ReachedSizeMax => {
                successes += 1;
                num_trials += 1;
                initial_state.clone_into(&mut trial_state);
            }
            EvolveOutcome::ReachedSizeMin => {
                num_trials += 1;
                initial_state.clone_into(&mut trial_state);
            }
            _ => {
                return Err(GrowError::NotSupported(
                    "Evolve outcome not supported".to_string(),
                ));
            }
        }
    }

    Ok((successes as f64 / num_trials as f64, num_trials as usize))
}

pub(super) fn calc_forward_probabilities_adaptive<S: System>(
    sys: &S,
    initial_states: &[&StateEnum],
    forward_step: NumTiles,
    max_time: Option<f64>,
    max_events: Option<NumEvents>,
    conf_interval_margin: f64,
) -> Result<(Vec<f64>, Vec<usize>), GrowError>
where
    SystemEnum: From<S>,
{
    let results = initial_states
        .par_iter()
        .map(|initial_state| {
            calc_forward_probability_adaptive(
                sys,
                initial_state,
                forward_step,
                max_time,
                max_events,
                conf_interval_margin,
            )
        })
        .collect::<Vec<_>>();

    let results: Vec<(f64, usize)> = results.into_iter().map(|r| r.unwrap()).collect();

    let probabilities: Vec<f64> = results.iter().map(|(p, _)| *p).collect();
    let trials: Vec<usize> = results.iter().map(|(_, t)| *t).collect();

    Ok((probabilities, trials))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn calc_committor_threshold_test<S: System>(
    sys: &mut S,
    initial_state: &StateEnum,
    cutoff_size: NumTiles,
    threshold: f64,
    z_level: f64,
    max_time: Option<f64>,
    max_events: Option<NumEvents>,
    max_trials: Option<usize>,
    return_on_max_trials: bool,
) -> Result<(bool, f64, usize, bool), GrowError>
where
    SystemEnum: From<S>,
{
    use bpci::NSuccessesSample;

    let n_par = rayon::current_num_threads();

    if !(0.0..=1.0).contains(&threshold) {
        return Err(GrowError::NotSupported(
            "Threshold must be between 0.0 and 1.0".to_string(),
        ));
    }

    let max_trials = max_trials.unwrap_or(100000);
    let mut successes = 0u32;
    let mut num_trials = 0u32;

    let mut trial_states = Vec::new();
    for _ in 0..n_par {
        trial_states.push(initial_state.clone());
    }

    let bounds = EvolveBounds {
        size_min: Some(0),
        size_max: Some(cutoff_size),
        for_time: max_time,
        for_events: max_events,
        ..Default::default()
    };

    // Continue sampling until we can determine with confidence whether
    // the probability is above or below the threshold
    loop {
        let outcomes = sys.evolve_states(&mut trial_states, bounds);
        for outcome in outcomes {
            match outcome? {
                EvolveOutcome::ReachedSizeMax => {
                    successes += 1;
                    num_trials += 1;
                }
                EvolveOutcome::ReachedSizeMin => {
                    num_trials += 1;
                }
                x => {
                    return Err(GrowError::NotSupported(format!(
                        "Evolve outcome not supported: {:?}",
                        x
                    )));
                }
            }
        }

        // Need at least a few trials before we can make any statistical determination
        if num_trials < 10 {
            continue;
        }

        // Calculate Wilson score confidence interval for the threshold test
        // This uses the test confidence level to determine if we can make a decision
        let sample = NSuccessesSample::new(num_trials, successes).unwrap();
        let test_wilson = sample.wilson_score_with_cc(z_level);

        let test_lower_bound = test_wilson.lower();
        let test_upper_bound = test_wilson.upper();

        // Check if the test confidence interval excludes the threshold (definitive determination)
        if test_upper_bound < threshold {
            // We're confident the probability is below the threshold
            let probability_estimate = successes as f64 / num_trials as f64;
            return Ok((false, probability_estimate, num_trials as usize, false));
        } else if test_lower_bound > threshold {
            // We're confident the probability is above the threshold
            let probability_estimate = successes as f64 / num_trials as f64;
            return Ok((true, probability_estimate, num_trials as usize, false));
        }

        // Check if we've exceeded the maximum number of trials without reaching a decision
        if num_trials >= max_trials as u32 {
            let probability_estimate = successes as f64 / num_trials as f64;

            if return_on_max_trials {
                // Return current best estimate with warning flag set
                // Use simple point estimate comparison since we couldn't reach statistical confidence
                let is_above_threshold = probability_estimate > threshold;
                return Ok((
                    is_above_threshold,
                    probability_estimate,
                    num_trials as usize,
                    true,
                ));
            } else {
                // Raise error when max trials exceeded and user doesn't want fallback result
                return Err(GrowError::NotSupported(format!(
                    "Maximum number of trials ({}) exceeded without reaching confidence",
                    max_trials
                )));
            }
        }
        for state in trial_states.iter_mut() {
            initial_state.clone_into(state);
        }
    }
}

pub(super) fn find_first_critical_state<S: System>(
    sys: &mut S,
    end_state: &StateEnum,
    config: &CriticalStateConfig,
) -> Result<Option<CriticalStateResult>, GrowError>
where
    SystemEnum: From<S>,
{
    let _tracker = if let Some(tracker) = end_state.get_movie_tracker() {
        tracker
    } else {
        return Err(GrowError::NotSupported(
            "State does not have a movie tracker".to_string(),
        ));
    };

    let filtered_indices = end_state.filtered_movie_indices()?;

    if filtered_indices.is_empty() {
        return Ok(None);
    }

    for &item in &filtered_indices {
        let mut state = end_state.replay(Some(item as u64))?;
        sys.update_state(&mut state, &NeededUpdate::All);

        let (is_above, prob, trials, exceeded) = calc_committor_threshold_test(
            sys,
            &state,
            config.cutoff_size,
            config.threshold,
            config.confidence_level,
            None, // max_time
            None, // max_events
            Some(config.max_trials),
            true, // return_on_max_trials
        )?;

        if is_above {
            let orig_idx = item;
            let energy = state.energy();

            return Ok(Some(CriticalStateResult {
                state,
                energy,
                trajectory_index: orig_idx,
                is_above_threshold: is_above,
                probability: prob,
                num_trials: trials,
                max_trials_exceeded: exceeded,
            }));
        }
    }

    Ok(None)
}

pub(super) fn find_last_critical_state<S: System>(
    sys: &mut S,
    end_state: &StateEnum,
    config: &CriticalStateConfig,
) -> Result<Option<CriticalStateResult>, GrowError>
where
    SystemEnum: From<S>,
{
    let _tracker = if let Some(tracker) = end_state.get_movie_tracker() {
        tracker
    } else {
        return Err(GrowError::NotSupported(
            "State does not have a movie tracker".to_string(),
        ));
    };

    let filtered_indices = end_state.filtered_movie_indices()?;

    if filtered_indices.is_empty() {
        return Ok(None);
    }

    for i in (0..filtered_indices.len()).rev() {
        let mut state = end_state.replay(Some(filtered_indices[i] as u64))?;
        sys.update_state(&mut state, &NeededUpdate::All);

        let (is_above, prob, trials, exceeded) = calc_committor_threshold_test(
            sys,
            &state,
            config.cutoff_size,
            config.threshold,
            config.confidence_level,
            None, // max_time
            None, // max_events
            Some(config.max_trials),
            true, // return_on_max_trials
        )?;

        if !is_above {
            let orig_idx = filtered_indices[i];
            let energy = state.energy();

            return Ok(Some(CriticalStateResult {
                state,
                energy,
                trajectory_index: orig_idx,
                is_above_threshold: is_above,
                probability: prob,
                num_trials: trials,
                max_trials_exceeded: exceeded,
            }));
        }
    }

    Ok(None)
}
