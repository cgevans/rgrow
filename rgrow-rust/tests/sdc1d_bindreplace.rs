use rgrow::canvas::{Canvas, PointSafe2, PointSafeHere};
use rgrow::models::sdc1d::{GsOrSeq, RefOrPair, SDCParams, SDCStrand, SingleOrMultiScaffold};
use rgrow::models::sdc1d_bindreplace::SDC1DBindReplace;
use rgrow::state::StateEnum;
use rgrow::system::{EvolveBounds, NeededUpdate, System, TileBondInfo};
use rgrow::tileset::CanvasType::SquareCompact;
use rgrow::tileset::TrackingType;
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
fn test_weak_replacement_reaches_perfect_copy() {
    // Use -5 kcal/mol: strong enough for reliable convergence but still
    // allows replacement dynamics (wrong strands have ~300/s off-rate).
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

        let bounds = EvolveBounds::default().for_events(1_000_000);
        System::evolve(&sys, &mut state, bounds).unwrap();

        for col in 0..n {
            let tile = state.tile_at_point(PointSafe2((0, col)));
            let expected = correct_tile(col, input_bit);
            assert_eq!(
                tile, expected,
                "weak_replacement: input_bit={input_bit}, position {col}: expected tile {expected}, got {tile}"
            );
        }
    }
}

/// With `physical_replacement_rate`, the filled-site rate should be
/// 1/(1/r_detach + 1/r_attach) = r_detach·r_attach / (r_detach + r_attach).
#[test]
fn test_physical_replacement_rate() {
    let n = 5;
    let mut sys = make_bitcopy_with_energy(n, 0);
    sys.account_for_energy = true;
    sys.physical_attachment_rate = true;
    sys.allow_weak_replacement = true;

    // Rates without physical_replacement_rate (pure detachment rate)
    sys.physical_replacement_rate = false;
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

    // Rates with physical_replacement_rate (combined detach + attach)
    sys.physical_replacement_rate = true;
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
