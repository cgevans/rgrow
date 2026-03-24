use rgrow::base::GrowError;
use rgrow::canvas::{Canvas, PointSafe2, PointSafeHere};
use rgrow::models::sdc1d::{GsOrSeq, RefOrPair, SDCParams, SDCStrand, SingleOrMultiScaffold};
use rgrow::models::sdc1d_bindreplace::SDC1DBindReplace;
use rgrow::state::StateEnum;
use rgrow::system::{Event, EvolveBounds, NeededUpdate, System, TileBondInfo};
use rgrow::tileset::CanvasType::SquareCompact;
use rgrow::tileset::TrackingType;
use rgrow::units::PerSecond;
use std::collections::HashMap;

/// Build a bitcopy system of length `n` with a given `input_bit` (0 or 1).
///
/// - Input position (scaffold 0): one strand with right_glue matching c{input_bit}.
/// - Cascade positions 1..n-1: two strands each (bit 0 and bit 1) that propagate
///   the bit from west to east via glue matching.
fn make_bitcopy(n: usize, input_bit: u32) -> SDC1DBindReplace {
    assert!(n >= 2);
    assert!(input_bit <= 1);

    let mut strands = Vec::new();

    // Input strand at position 0
    strands.push(SDCStrand {
        name: Some(format!("input_{input_bit}")),
        color: None,
        concentration: 1e6,
        btm_glue: Some("sc0".to_string()),
        left_glue: None,
        right_glue: Some(format!("c{input_bit}*")),
    });

    // Cascade positions 1..n-1: two strands each (bit 0 first, then bit 1)
    for i in 1..n {
        for bit in 0..=1u32 {
            strands.push(SDCStrand {
                name: Some(format!("pos{i}_bit{bit}")),
                color: None,
                concentration: 1e6,
                btm_glue: Some(format!("sc{i}")),
                left_glue: Some(format!("c{bit}")),
                right_glue: Some(format!("c{bit}*")),
            });
        }
    }

    let scaffold = (0..n).map(|i| Some(format!("sc{i}*"))).collect::<Vec<_>>();

    let params = SDCParams {
        strands,
        quencher_name: None,
        quencher_concentration: 0.0,
        reporter_name: None,
        fluorophore_concentration: 0.0,
        scaffold: SingleOrMultiScaffold::Single(scaffold),
        scaffold_concentration: 1e-100,
        glue_dg_s: HashMap::new(),
        k_f: 1e6,
        k_n: 0.0,
        k_c: 0.0,
        temperature: 37.0,
        junction_penalty_dg: None,
        junction_penalty_ds: None,
    };

    SDC1DBindReplace::from_params(params)
}

/// Returns the tile index for the correct strand at position `pos` for the given `input_bit`.
fn correct_tile(pos: usize, input_bit: u32) -> u32 {
    if pos == 0 {
        1 // input strand is always tile 1
    } else {
        // Cascade strands: pos 1 has tiles 2 (bit0), 3 (bit1); pos 2 has 4 (bit0), 5 (bit1); ...
        let base = 2 + 2 * (pos - 1) as u32;
        base + input_bit
    }
}

/// Returns the tile index for the wrong strand at position `pos` for the given `input_bit`.
fn wrong_tile(pos: usize, input_bit: u32) -> u32 {
    correct_tile(pos, 1 - input_bit)
}

#[test]
fn test_bitcopy_empty_state_rates() {
    let n = 5;
    let sys = make_bitcopy(n, 0);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // All positions with matching strands should have rate 1
    for col in 0..n {
        let rate = sys.event_rate_at_point(&state, PointSafeHere((0, col)));
        assert_eq!(
            f64::from(rate),
            1.0,
            "empty position {col} should have rate 1"
        );
    }
}

#[test]
fn test_bitcopy_perfect_filled_state_rates() {
    let n = 5;
    let input_bit = 0;
    let sys = make_bitcopy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // Fill every position with the correct bit
    for col in 0..n {
        let tile = correct_tile(col, input_bit);
        sys.place_tile(&mut state, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    // Every position should have rate 0
    for col in 0..n {
        let rate = sys.event_rate_at_point(&state, PointSafeHere((0, col)));
        assert_eq!(
            f64::from(rate),
            0.0,
            "position {col} should have rate 0 when perfectly filled"
        );
    }
}

#[test]
fn test_bitcopy_mismatched_filled_state_rates() {
    let n = 5;
    let input_bit = 0;
    let mismatch_pos = 2;
    let sys = make_bitcopy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // Fill all positions correctly except mismatch_pos which gets the wrong bit
    for col in 0..n {
        let tile = if col == mismatch_pos {
            wrong_tile(col, input_bit)
        } else {
            correct_tile(col, input_bit)
        };
        sys.place_tile(&mut state, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    for col in 0..n {
        let rate = sys.event_rate_at_point(&state, PointSafeHere((0, col)));
        let expected = if col == mismatch_pos {
            // Mismatched: correct strand has more bonds → rate 1
            1.0
        } else if col == mismatch_pos - 1 || col == mismatch_pos + 1 {
            // Adjacent to mismatch: alternative has equal bonds → rate 1
            1.0
        } else {
            0.0
        };
        assert_eq!(
            f64::from(rate),
            expected,
            "position {col} rate mismatch (mismatch at {mismatch_pos})"
        );
    }
}

#[test]
fn test_bitcopy_reaches_perfect_copy() {
    for input_bit in [0u32, 1] {
        let n = 5;
        let sys = make_bitcopy(n, input_bit);
        let mut state = StateEnum::empty(
            (1, n),
            SquareCompact,
            TrackingType::None,
            sys.tile_names().len(),
        )
        .unwrap();
        sys.update_state(&mut state, &NeededUpdate::All);

        let bounds = EvolveBounds::default().for_events(100_000);
        System::evolve(&sys, &mut state, bounds).unwrap();

        // Every position should end up with the correct strand
        for col in 0..n {
            let tile = state.tile_at_point(PointSafe2((0, col)));
            let expected = correct_tile(col, input_bit);
            assert_eq!(
                tile, expected,
                "input_bit={input_bit}, position {col}: expected tile {expected}, got {tile}"
            );
        }
    }
}

// --- Energy-aware tests ---

/// Build a bitcopy system with energy data and account_for_energy enabled.
/// Uses strong binding (ΔG = -10 kcal/mol, ΔS = 0) for all glue pairs.
fn make_bitcopy_with_energy(n: usize, input_bit: u32) -> SDC1DBindReplace {
    assert!(n >= 2);
    assert!(input_bit <= 1);

    let mut strands = Vec::new();

    strands.push(SDCStrand {
        name: Some(format!("input_{input_bit}")),
        color: None,
        concentration: 1e6,
        btm_glue: Some("sc0".to_string()),
        left_glue: None,
        right_glue: Some(format!("c{input_bit}*")),
    });

    for i in 1..n {
        for bit in 0..=1u32 {
            strands.push(SDCStrand {
                name: Some(format!("pos{i}_bit{bit}")),
                color: None,
                concentration: 1e6,
                btm_glue: Some(format!("sc{i}")),
                left_glue: Some(format!("c{bit}")),
                right_glue: Some(format!("c{bit}*")),
            });
        }
    }

    let scaffold = (0..n).map(|i| Some(format!("sc{i}*"))).collect::<Vec<_>>();

    // Provide energy data for all glue pairs
    let mut glue_dg_s: HashMap<RefOrPair, GsOrSeq> = HashMap::new();

    // Scaffold glues: sc0, sc1, ..., sc(n-1)
    for i in 0..n {
        glue_dg_s.insert(RefOrPair::Ref(format!("sc{i}")), GsOrSeq::GS((-10.0, 0.0)));
    }
    // Cascade glues: c0, c1
    for bit in 0..=1u32 {
        glue_dg_s.insert(RefOrPair::Ref(format!("c{bit}")), GsOrSeq::GS((-10.0, 0.0)));
    }

    let params = SDCParams {
        strands,
        quencher_name: None,
        quencher_concentration: 0.0,
        reporter_name: None,
        fluorophore_concentration: 0.0,
        scaffold: SingleOrMultiScaffold::Single(scaffold),
        scaffold_concentration: 1e-100,
        glue_dg_s,
        k_f: 1e6,
        k_n: 0.0,
        k_c: 0.0,
        temperature: 37.0,
        junction_penalty_dg: None,
        junction_penalty_ds: None,
    };

    let mut sys = SDC1DBindReplace::from_params(params);
    sys.account_for_energy = true;
    // Re-run update to fill energy arrays
    sys.update();
    sys
}

#[test]
fn test_energy_empty_state_rates() {
    let n = 5;
    let sys = make_bitcopy_with_energy(n, 0);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // Empty state rates should be 1 (energy doesn't affect empty sites)
    for col in 0..n {
        let rate = sys.event_rate_at_point(&state, PointSafeHere((0, col)));
        assert_eq!(
            f64::from(rate),
            1.0,
            "empty position {col} should have rate 1"
        );
    }
}

#[test]
fn test_energy_perfect_filled_rates() {
    let n = 5;
    let input_bit = 0;
    let sys = make_bitcopy_with_energy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    for col in 0..n {
        let tile = correct_tile(col, input_bit);
        sys.place_tile(&mut state, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    // Perfect fill: n_others=0 everywhere, so rate=0 regardless of energy
    for col in 0..n {
        let rate = sys.event_rate_at_point(&state, PointSafeHere((0, col)));
        assert_eq!(
            f64::from(rate),
            0.0,
            "position {col} should have rate 0 when perfectly filled"
        );
    }
}

#[test]
fn test_energy_mismatched_rates() {
    let n = 5;
    let input_bit = 0;
    let mismatch_pos = 2;
    let sys = make_bitcopy_with_energy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    for col in 0..n {
        let tile = if col == mismatch_pos {
            wrong_tile(col, input_bit)
        } else {
            correct_tile(col, input_bit)
        };
        sys.place_tile(&mut state, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    for col in 0..n {
        let rate = f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, col))));
        if col == mismatch_pos || col == mismatch_pos - 1 || col == mismatch_pos + 1 {
            // These positions have n_others > 0, so rate = n_others * kf * exp(β·ΔG)
            // With strong negative ΔG, the rate should be very small (but > 0)
            assert!(
                rate > 0.0,
                "position {col} should have positive rate with energy"
            );
            // ΔG is strongly negative → exp(β·ΔG) ≈ very small → rate << 1
            // Without energy, these would be rate 1.0; with energy they should be much less
            assert!(
                rate < 1.0,
                "position {col} rate {rate} should be much less than 1.0 due to negative ΔG"
            );
        } else {
            assert_eq!(rate, 0.0, "position {col} should have rate 0");
        }
    }
}

#[test]
fn test_energy_reaches_perfect_copy() {
    for input_bit in [0u32, 1] {
        let n = 5;
        let sys = make_bitcopy_with_energy(n, input_bit);
        let mut state = StateEnum::empty(
            (1, n),
            SquareCompact,
            TrackingType::None,
            sys.tile_names().len(),
        )
        .unwrap();
        sys.update_state(&mut state, &NeededUpdate::All);

        // With energy, convergence may take more events due to slower replacement
        let bounds = EvolveBounds::default().for_events(1_000_000);
        System::evolve(&sys, &mut state, bounds).unwrap();

        for col in 0..n {
            let tile = state.tile_at_point(PointSafe2((0, col)));
            let expected = correct_tile(col, input_bit);
            assert_eq!(
                tile, expected,
                "energy: input_bit={input_bit}, position {col}: expected tile {expected}, got {tile}"
            );
        }
    }
}

#[test]
fn test_no_energy_backward_compat() {
    // A system constructed with glue_dg_s data but account_for_energy=false
    // should behave identically to one without energy data
    let n = 5;
    let input_bit = 0;
    let mismatch_pos = 2;

    let sys_no_energy = make_bitcopy(n, input_bit);
    let mut sys_with_data = make_bitcopy_with_energy(n, input_bit);
    sys_with_data.account_for_energy = false;
    // Note: energy arrays not filled since account_for_energy is false, but that's fine
    // because the rate code checks account_for_energy before using them

    let mut state1 = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys_no_energy.tile_names().len(),
    )
    .unwrap();
    let mut state2 = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys_with_data.tile_names().len(),
    )
    .unwrap();
    sys_no_energy.update_state(&mut state1, &NeededUpdate::All);
    sys_with_data.update_state(&mut state2, &NeededUpdate::All);

    // Fill with a mismatch
    for col in 0..n {
        let tile = if col == mismatch_pos {
            wrong_tile(col, input_bit)
        } else {
            correct_tile(col, input_bit)
        };
        sys_no_energy
            .place_tile(&mut state1, PointSafe2((0, col)), tile, false)
            .unwrap();
        sys_with_data
            .place_tile(&mut state2, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    for col in 0..n {
        let rate1 = f64::from(sys_no_energy.event_rate_at_point(&state1, PointSafeHere((0, col))));
        let rate2 = f64::from(sys_with_data.event_rate_at_point(&state2, PointSafeHere((0, col))));
        assert_eq!(
            rate1, rate2,
            "position {col}: account_for_energy=false should give same rates as no-energy system"
        );
    }
}

// --- Weak replacement tests ---

#[test]
fn test_weak_replacement_allows_less_matching() {
    let n = 5;
    let input_bit = 0;
    let mut sys = make_bitcopy_with_energy(n, input_bit);
    sys.allow_weak_replacement = true;

    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // Fill every position with the correct bit (2 glue matches each)
    for col in 0..n {
        let tile = correct_tile(col, input_bit);
        sys.place_tile(&mut state, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    // With allow_weak_replacement, positions with alternative strands should have nonzero rates
    // even when perfectly filled (because any different scaffold-matching strand can replace).
    // Position 0 has only one matching strand (input), so rate is still 0.
    // Positions 1..n-1 each have two matching strands, so rate should be nonzero.
    for col in 0..n {
        let rate = f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, col))));
        if col == 0 {
            assert_eq!(
                rate, 0.0,
                "position 0 has only one matching strand, rate should be 0"
            );
        } else {
            assert!(
                rate > 0.0,
                "position {col} should have nonzero rate with allow_weak_replacement"
            );
            // Rate should be kf * exp(βΔG) where ΔG is strongly negative → small but positive
            assert!(
                rate < 1.0,
                "position {col} rate {rate} should be small due to strong binding energy"
            );
        }
    }
}

/// Build a bitcopy system with a specified per-glue ΔG (kcal/mol) and energy + weak replacement enabled.
fn make_bitcopy_weak_replacement(n: usize, input_bit: u32, dg: f64) -> SDC1DBindReplace {
    assert!(n >= 2);
    assert!(input_bit <= 1);

    let mut strands = Vec::new();

    strands.push(SDCStrand {
        name: Some(format!("input_{input_bit}")),
        color: None,
        concentration: 1e6,
        btm_glue: Some("sc0".to_string()),
        left_glue: None,
        right_glue: Some(format!("c{input_bit}*")),
    });

    for i in 1..n {
        for bit in 0..=1u32 {
            strands.push(SDCStrand {
                name: Some(format!("pos{i}_bit{bit}")),
                color: None,
                concentration: 1e6,
                btm_glue: Some(format!("sc{i}")),
                left_glue: Some(format!("c{bit}")),
                right_glue: Some(format!("c{bit}*")),
            });
        }
    }

    let scaffold = (0..n).map(|i| Some(format!("sc{i}*"))).collect::<Vec<_>>();

    let mut glue_dg_s: HashMap<RefOrPair, GsOrSeq> = HashMap::new();
    for i in 0..n {
        glue_dg_s.insert(RefOrPair::Ref(format!("sc{i}")), GsOrSeq::GS((dg, 0.0)));
    }
    for bit in 0..=1u32 {
        glue_dg_s.insert(RefOrPair::Ref(format!("c{bit}")), GsOrSeq::GS((dg, 0.0)));
    }

    let params = SDCParams {
        strands,
        quencher_name: None,
        quencher_concentration: 0.0,
        reporter_name: None,
        fluorophore_concentration: 0.0,
        scaffold: SingleOrMultiScaffold::Single(scaffold),
        scaffold_concentration: 1e-100,
        glue_dg_s,
        k_f: 1e6,
        k_n: 0.0,
        k_c: 0.0,
        temperature: 37.0,
        junction_penalty_dg: None,
        junction_penalty_ds: None,
    };

    let mut sys = SDC1DBindReplace::from_params(params);
    sys.account_for_energy = true;
    sys.allow_weak_replacement = true;
    sys.update();
    sys
}

#[test]
fn test_weak_replacement_fills_and_evolves() {
    // With weak replacement, any strand can displace the current occupant, so the
    // system is ergodic and will NOT reliably converge to the perfect copy.
    // Instead, verify that all sites fill and that replacement dynamics occur
    // (i.e. events happen beyond the initial filling).
    for input_bit in [0u32, 1] {
        let n = 5;
        let sys = make_bitcopy_weak_replacement(n, input_bit, -5.0);

        let mut state = StateEnum::empty(
            (1, n),
            SquareCompact,
            TrackingType::None,
            sys.tile_names().len(),
        )
        .unwrap();
        sys.update_state(&mut state, &NeededUpdate::All);

        let bounds = EvolveBounds::default().for_events(1_000);
        System::evolve(&sys, &mut state, bounds).unwrap();

        // All sites should be filled.
        for col in 0..n {
            let tile = state.tile_at_point(PointSafe2((0, col)));
            assert_ne!(
                tile, 0,
                "weak_replacement: input_bit={input_bit}, position {col} should be filled"
            );
        }

        // Rates should still be nonzero at sites with multiple candidates
        // (position 0 has only one input strand, so its rate is correctly zero).
        for col in 1..n {
            let rate = sys.event_rate_at_point(&state, PointSafeHere((0, col)));
            assert!(
                rate.0 > 0.0,
                "weak_replacement: input_bit={input_bit}, position {col} should have nonzero rate"
            );
        }
    }
}

/// Build a bitcopy system with entropy (nonzero ΔS) for temperature-dependence tests.
fn make_bitcopy_with_entropy(n: usize, input_bit: u32, dg: f64, ds: f64) -> SDC1DBindReplace {
    assert!(n >= 2);
    assert!(input_bit <= 1);

    let mut strands = Vec::new();

    strands.push(SDCStrand {
        name: Some(format!("input_{input_bit}")),
        color: None,
        concentration: 1e6,
        btm_glue: Some("sc0".to_string()),
        left_glue: None,
        right_glue: Some(format!("c{input_bit}*")),
    });

    for i in 1..n {
        for bit in 0..=1u32 {
            strands.push(SDCStrand {
                name: Some(format!("pos{i}_bit{bit}")),
                color: None,
                concentration: 1e6,
                btm_glue: Some(format!("sc{i}")),
                left_glue: Some(format!("c{bit}")),
                right_glue: Some(format!("c{bit}*")),
            });
        }
    }

    let scaffold = (0..n).map(|i| Some(format!("sc{i}*"))).collect::<Vec<_>>();

    let mut glue_dg_s: HashMap<RefOrPair, GsOrSeq> = HashMap::new();
    for i in 0..n {
        glue_dg_s.insert(RefOrPair::Ref(format!("sc{i}")), GsOrSeq::GS((dg, ds)));
    }
    for bit in 0..=1u32 {
        glue_dg_s.insert(RefOrPair::Ref(format!("c{bit}")), GsOrSeq::GS((dg, ds)));
    }

    let params = SDCParams {
        strands,
        quencher_name: None,
        quencher_concentration: 0.0,
        reporter_name: None,
        fluorophore_concentration: 0.0,
        scaffold: SingleOrMultiScaffold::Single(scaffold),
        scaffold_concentration: 1e-100,
        glue_dg_s,
        k_f: 1e6,
        k_n: 0.0,
        k_c: 0.0,
        temperature: 37.0,
        junction_penalty_dg: None,
        junction_penalty_ds: None,
    };

    let mut sys = SDC1DBindReplace::from_params(params);
    sys.account_for_energy = true;
    sys.allow_weak_replacement = true;
    sys.update();
    sys
}

/// With `bindunbind_replacement_rate`, the filled-site rate should be
/// 1/(1/r_detach + 1/r_attach) = r_detach·r_attach / (r_detach + r_attach).
#[test]
fn test_bindunbind_replacement_rate() {
    let n = 5;
    let mut sys = make_bitcopy_with_energy(n, 0);
    sys.account_for_energy = true;
    sys.physical_attachment_rate = true;
    sys.allow_weak_replacement = true;

    // Rates without bindunbind_replacement_rate (pure detachment rate)
    sys.bindunbind_replacement_rate = false;
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);
    for col in 0..n {
        sys.place_tile(
            &mut state,
            PointSafe2((0, col)),
            correct_tile(col, 0),
            false,
        )
        .unwrap();
    }
    sys.update_state(&mut state, &NeededUpdate::All);
    let rates_detach: Vec<f64> = (0..n)
        .map(|c| f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, c)))))
        .collect();

    // Rates with bindunbind_replacement_rate (combined detach + attach)
    sys.bindunbind_replacement_rate = true;
    sys.update_state(&mut state, &NeededUpdate::All);
    let rates_combined: Vec<f64> = (0..n)
        .map(|c| f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, c)))))
        .collect();

    // Attachment rate at each position = sum(kf * conc) over valid replacers.
    // Each cascade position has 1 valid replacer at conc 1e6, so r_attach = 1e6 * 1e6 = 1e12.
    // Position 0 has no replacer (single input strand), so rate should be 0 for both.
    let kf = 1e6_f64;
    let conc = 1e6_f64;
    let r_attach_cascade = kf * conc; // one replacer per cascade position

    for col in 0..n {
        let r_d = rates_detach[col];
        let r_c = rates_combined[col];
        if col == 0 {
            // Position 0: only one input strand, no replacer
            assert_eq!(r_d, 0.0, "pos 0: no replacer, detach rate should be 0");
            assert_eq!(r_c, 0.0, "pos 0: no replacer, combined rate should be 0");
        } else {
            let expected = (r_d * r_attach_cascade) / (r_d + r_attach_cascade);
            assert!(
                (r_c - expected).abs() < 1e-10 * expected.abs(),
                "pos {col}: combined={r_c}, expected={expected}, detach={r_d}"
            );
            // Combined rate should always be <= detach rate
            assert!(
                r_c <= r_d,
                "pos {col}: combined rate should be <= detach rate"
            );
        }
    }
}

// --- physical_attachment_rate on empty sites ---

#[test]
fn test_physical_attachment_rate_empty_sites() {
    let n = 5;
    let mut sys = make_bitcopy(n, 0);
    sys.physical_attachment_rate = true;

    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    let kf = 1e6_f64;
    let conc = 1e6_f64;

    // Position 0: 1 matching tile (input strand) → kf * conc
    let rate0 = f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, 0))));
    assert!(
        (rate0 - kf * conc).abs() < 1e-6 * (kf * conc),
        "position 0: expected {}, got {rate0}",
        kf * conc
    );

    // Positions 1..4: 2 matching tiles each → 2 * kf * conc
    for col in 1..n {
        let rate = f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, col))));
        assert!(
            (rate - 2.0 * kf * conc).abs() < 1e-6 * (2.0 * kf * conc),
            "position {col}: expected {}, got {rate}",
            2.0 * kf * conc
        );
    }
}

// --- allow_same_replacement ---

#[test]
fn test_allow_same_replacement_increases_rate() {
    let n = 5;
    let input_bit = 0;

    let mut sys = make_bitcopy_with_energy(n, input_bit);
    sys.allow_weak_replacement = true;
    sys.bindunbind_replacement_rate = true;
    sys.physical_attachment_rate = true;
    sys.allow_same_replacement = false;
    sys.update();

    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    for col in 0..n {
        sys.place_tile(
            &mut state,
            PointSafe2((0, col)),
            correct_tile(col, input_bit),
            false,
        )
        .unwrap();
    }
    sys.update_state(&mut state, &NeededUpdate::All);

    let rates_no_same: Vec<f64> = (0..n)
        .map(|c| f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, c)))))
        .collect();

    // Position 0: only one matching strand (input), and it's the same tile → rate 0
    assert_eq!(
        rates_no_same[0], 0.0,
        "pos 0: only one candidate (itself), should be filtered"
    );

    // Enable allow_same_replacement
    sys.allow_same_replacement = true;
    sys.update_state(&mut state, &NeededUpdate::All);

    let rates_with_same: Vec<f64> = (0..n)
        .map(|c| f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, c)))))
        .collect();

    // Position 0 now allows self-replacement → rate > 0
    assert!(
        rates_with_same[0] > 0.0,
        "pos 0: self-replacement allowed, rate should be > 0"
    );

    // Cascade positions: rate with self-replacement >= rate without
    // (r_attach doubles but since r_detach << r_attach the combined rate barely changes)
    for col in 1..n {
        assert!(
            rates_with_same[col] >= rates_no_same[col],
            "pos {col}: rate with same ({}) should be >= without ({})",
            rates_with_same[col],
            rates_no_same[col]
        );
    }
}

// --- calc_mismatch_locations ---

#[test]
fn test_mismatch_locations_all_correct() {
    let n = 5;
    let input_bit = 0;
    let sys = make_bitcopy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    for col in 0..n {
        sys.place_tile(
            &mut state,
            PointSafe2((0, col)),
            correct_tile(col, input_bit),
            false,
        )
        .unwrap();
    }

    let mm = sys.calc_mismatch_locations(&state);
    for col in 0..n {
        assert_eq!(
            mm[(0, col)],
            0,
            "position {col} should have no mismatches when all correct"
        );
    }
}

#[test]
fn test_mismatch_locations_single_mismatch() {
    let n = 5;
    let input_bit = 0;
    let mismatch_pos = 2;
    let sys = make_bitcopy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    for col in 0..n {
        let tile = if col == mismatch_pos {
            wrong_tile(col, input_bit)
        } else {
            correct_tile(col, input_bit)
        };
        sys.place_tile(&mut state, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    let mm = sys.calc_mismatch_locations(&state);

    // Position 0: correct tile, neighbor at pos 1 is correct → 0
    assert_eq!(mm[(0, 0)], 0, "position 0 should have no mismatches");

    // Position 1: correct tile, east neighbor (pos 2) is wrong → mm_e=1, mm_w=0 → 4*1+0=4
    assert_eq!(
        mm[(0, 1)],
        4,
        "position 1: east mismatch with wrong tile at pos 2"
    );

    // Position 2: wrong tile, west neighbor (pos 1) is correct → mm_w=1,
    //             east neighbor (pos 3) is correct → mm_e=1 → 4*1+1=5
    assert_eq!(mm[(0, 2)], 5, "position 2: both east and west mismatches");

    // Position 3: correct tile, west neighbor (pos 2) is wrong → mm_w=1, mm_e=0 → 0*4+1=1
    assert_eq!(
        mm[(0, 3)],
        1,
        "position 3: west mismatch with wrong tile at pos 2"
    );

    // Position 4: correct tile, west neighbor (pos 3) is correct → 0
    assert_eq!(mm[(0, 4)], 0, "position 4 should have no mismatches");
}

#[test]
fn test_mismatch_locations_empty_neighbors() {
    let n = 5;
    let input_bit = 0;
    let sys = make_bitcopy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // Only fill positions 0 and 2 (leave 1, 3, 4 empty)
    sys.place_tile(
        &mut state,
        PointSafe2((0, 0)),
        correct_tile(0, input_bit),
        false,
    )
    .unwrap();
    sys.place_tile(
        &mut state,
        PointSafe2((0, 2)),
        correct_tile(2, input_bit),
        false,
    )
    .unwrap();

    let mm = sys.calc_mismatch_locations(&state);

    // Empty cells should be 0
    assert_eq!(mm[(0, 1)], 0, "empty position 1 should be 0");
    assert_eq!(mm[(0, 3)], 0, "empty position 3 should be 0");
    assert_eq!(mm[(0, 4)], 0, "empty position 4 should be 0");

    // Filled cells with empty neighbors: empty tile has Glue(0), so te/tw != 0 is false → 0
    assert_eq!(
        mm[(0, 0)],
        0,
        "position 0: east neighbor empty → no mismatch counted"
    );
    assert_eq!(
        mm[(0, 2)],
        0,
        "position 2: both neighbors empty → no mismatch counted"
    );
}

// --- set_param / get_param ---

#[test]
fn test_set_get_param_roundtrip() {
    let mut sys = make_bitcopy(5, 0);

    // kf: initial 1e6
    let kf_val: f64 = *sys.get_param("kf").unwrap().downcast::<f64>().unwrap();
    assert!((kf_val - 1e6).abs() < 1.0, "initial kf should be 1e6");

    sys.set_param("kf", Box::new(2e6_f64)).unwrap();
    let kf_val: f64 = *sys.get_param("kf").unwrap().downcast::<f64>().unwrap();
    assert!((kf_val - 2e6).abs() < 1.0, "kf should be 2e6 after set");

    // temperature: initial 37.0
    let temp: f64 = *sys
        .get_param("temperature")
        .unwrap()
        .downcast::<f64>()
        .unwrap();
    assert!(
        (temp - 37.0).abs() < 0.01,
        "initial temperature should be 37.0"
    );

    sys.set_param("temperature", Box::new(50.0_f64)).unwrap();
    let temp: f64 = *sys
        .get_param("temperature")
        .unwrap()
        .downcast::<f64>()
        .unwrap();
    assert!(
        (temp - 50.0).abs() < 0.01,
        "temperature should be 50.0 after set"
    );
}

#[test]
fn test_set_param_unknown_returns_error() {
    let mut sys = make_bitcopy(5, 0);
    let result = sys.set_param("nonexistent", Box::new(1.0_f64));
    assert!(result.is_err(), "unknown param should return error");
    match result.unwrap_err() {
        GrowError::NoParameter(name) => assert_eq!(name, "nonexistent"),
        other => panic!("expected NoParameter, got: {other:?}"),
    }
}

#[test]
fn test_set_param_kf_changes_rates() {
    let n = 5;
    let mut sys = make_bitcopy(n, 0);
    sys.physical_attachment_rate = true;

    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    let rate_before = f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, 0))));

    // Double kf
    let needed = sys.set_param("kf", Box::new(2e6_f64)).unwrap();
    sys.update_state(&mut state, &needed);

    let rate_after = f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, 0))));

    assert!(
        (rate_after - 2.0 * rate_before).abs() < 1e-6 * rate_before,
        "doubling kf should double rate: before={rate_before}, after={rate_after}"
    );
}

// --- choose_event_at_point correctness ---

#[test]
fn test_choose_event_empty_site_returns_attachment() {
    let n = 5;
    let sys = make_bitcopy(n, 0);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // Position 0 has one candidate: input strand (tile 1)
    let (event, _rate) = sys.choose_event_at_point(&state, PointSafe2((0, 0)), PerSecond(0.0));
    match event {
        Event::MonomerAttachment(p, tile) => {
            assert_eq!(p, PointSafe2((0, 0)));
            assert_eq!(tile, 1, "should attach the input strand (tile 1)");
        }
        other => panic!("expected MonomerAttachment, got: {other:?}"),
    }
}

#[test]
fn test_choose_event_filled_site_returns_change() {
    let n = 5;
    let input_bit = 0;
    let mismatch_pos = 2;
    let sys = make_bitcopy(n, input_bit);
    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    for col in 0..n {
        let tile = if col == mismatch_pos {
            wrong_tile(col, input_bit)
        } else {
            correct_tile(col, input_bit)
        };
        sys.place_tile(&mut state, PointSafe2((0, col)), tile, false)
            .unwrap();
    }

    // Position 2 has a wrong tile; the correct tile should replace it
    let (event, _rate) =
        sys.choose_event_at_point(&state, PointSafe2((0, mismatch_pos)), PerSecond(0.0));
    match event {
        Event::MonomerChange(p, tile) => {
            assert_eq!(p, PointSafe2((0, mismatch_pos)));
            assert_eq!(
                tile,
                correct_tile(mismatch_pos, input_bit),
                "should change to the correct tile"
            );
        }
        other => panic!("expected MonomerChange, got: {other:?}"),
    }
}

// --- Minimum system size (n=2) ---

#[test]
fn test_n2_empty_rates_and_evolve() {
    for input_bit in [0u32, 1] {
        let n = 2;
        let sys = make_bitcopy(n, input_bit);
        let mut state = StateEnum::empty(
            (1, n),
            SquareCompact,
            TrackingType::None,
            sys.tile_names().len(),
        )
        .unwrap();
        sys.update_state(&mut state, &NeededUpdate::All);

        // Empty rates should both be 1.0
        for col in 0..n {
            let rate = f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, col))));
            assert_eq!(
                rate, 1.0,
                "n=2 input_bit={input_bit}: empty position {col} rate should be 1.0"
            );
        }

        // Evolve and check correctness
        let bounds = EvolveBounds::default().for_events(1_000);
        System::evolve(&sys, &mut state, bounds).unwrap();

        for col in 0..n {
            let tile = state.tile_at_point(PointSafe2((0, col)));
            let expected = correct_tile(col, input_bit);
            assert_eq!(
                tile, expected,
                "n=2 input_bit={input_bit}, position {col}: expected {expected}, got {tile}"
            );
        }
    }
}

// --- Temperature-dependent energy with nonzero entropy ---

#[test]
fn test_entropy_temperature_dependence() {
    let n = 5;
    let input_bit = 0;
    let dg = -10.0;
    let ds = -0.03;

    let mut sys = make_bitcopy_with_entropy(n, input_bit, dg, ds);

    let mut state = StateEnum::empty(
        (1, n),
        SquareCompact,
        TrackingType::None,
        sys.tile_names().len(),
    )
    .unwrap();
    sys.update_state(&mut state, &NeededUpdate::All);

    // Fill all correctly
    for col in 0..n {
        sys.place_tile(
            &mut state,
            PointSafe2((0, col)),
            correct_tile(col, input_bit),
            false,
        )
        .unwrap();
    }
    sys.update_state(&mut state, &NeededUpdate::All);

    // Record rates at T=37
    let rates_37: Vec<f64> = (1..n)
        .map(|c| f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, c)))))
        .collect();

    // Change temperature to 50
    let needed = sys.set_param("temperature", Box::new(50.0_f64)).unwrap();
    sys.update_state(&mut state, &needed);

    let rates_50: Vec<f64> = (1..n)
        .map(|c| f64::from(sys.event_rate_at_point(&state, PointSafeHere((0, c)))))
        .collect();

    // With ds < 0, at higher T the ΔG term becomes less negative (weaker binding),
    // so the detach rate increases → rates should be higher at T=50 vs T=37
    for (i, col) in (1..n).enumerate() {
        assert!(
            rates_50[i] > rates_37[i],
            "position {col}: rate at T=50 ({}) should exceed rate at T=37 ({})",
            rates_50[i],
            rates_37[i]
        );
    }
}
