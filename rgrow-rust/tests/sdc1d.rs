/*
Test that SDC 1D simulations actually run.
*/

use rgrow::models::sdc1d::{GsOrSeq, RefOrPair, SDCParams, SDCStrand, SingleOrMultiScaffold, SDC};
use rgrow::state::{StateEnum, StateStatus};
use rgrow::system::{EvolveBounds, NeededUpdate, System, TileBondInfo};
use rgrow::tileset::CanvasType::Square;
use rgrow::tileset::TrackingType;
use std::collections::HashMap;

fn strands() -> Vec<SDCStrand> {
    let strand = |base: String, input: usize| SDCStrand {
        name: None,
        color: None,
        concentration: 1e6,
        btm_glue: Some(base),
        left_glue: Some(format!("{input}")),
        right_glue: Some(format!("{input}*")),
    };

    let mut strands = vec![];
    for base in ["A", "B", "C", "D", "E"] {
        for inp in [0, 1] {
            strands.push(strand(base.to_string(), inp));
        }
    }

    strands.push(SDCStrand {
        name: None,
        color: None,
        concentration: 1e6,
        btm_glue: Some("E*".to_string()),
        left_glue: None,
        right_glue: None,
    });
    strands
}

/// Creates a HashMap<RefOrPair, GsOrSeq> from the given sequences.
///
/// This function converts the Python dictionary from the issue description into a Rust HashMap.
/// The sequences A, B, C, D, E are represented as RefOrPair::Ref.
/// The numbered sequences (0, 0', 1, 1', etc.) are also represented as RefOrPair::Ref.
pub fn create_sequences_map() -> HashMap<RefOrPair, GsOrSeq> {
    HashMap::from([
        // Single sequences (A, B, C, D, E)
        (
            RefOrPair::Ref("A".to_string()),
            GsOrSeq::Seq("TCTTTCCAGAGCCTAATTTGCCAG".to_string()),
        ),
        (
            RefOrPair::Ref("B".to_string()),
            GsOrSeq::Seq("AGCGTCCAATACTGCGGAATCGTC".to_string()),
        ),
        (
            RefOrPair::Ref("C".to_string()),
            GsOrSeq::Seq("ATAAATATTCATTGAATCCCCCTC".to_string()),
        ),
        (
            RefOrPair::Ref("D".to_string()),
            GsOrSeq::Seq("AAATGCTTTAAACAGTTCAGAAAA".to_string()),
        ),
        (
            RefOrPair::Ref("E".to_string()),
            GsOrSeq::Seq("AAAGAGGACAGATGAACGGTGTAC".to_string()),
        ),
        // Numbered sequences (0, 0', 1, 1', etc.)
        (
            RefOrPair::Ref("0".to_string()),
            GsOrSeq::Seq("CTCATCCTGACC".to_string()),
        ),
        (
            RefOrPair::Ref("0'".to_string()),
            GsOrSeq::Seq("CCTCTTCTCAGC".to_string()),
        ),
        (
            RefOrPair::Ref("1".to_string()),
            GsOrSeq::Seq("TCAACTCCGTTC".to_string()),
        ),
        (
            RefOrPair::Ref("1'".to_string()),
            GsOrSeq::Seq("CATCTCCGATCC".to_string()),
        ),
        (
            RefOrPair::Ref("2".to_string()),
            GsOrSeq::Seq("AATGCCACCATT".to_string()),
        ),
        (
            RefOrPair::Ref("2'".to_string()),
            GsOrSeq::Seq("TCTTTCCAAGCC".to_string()),
        ),
        (
            RefOrPair::Ref("3".to_string()),
            GsOrSeq::Seq("ACAACCCTTGTC".to_string()),
        ),
        (
            RefOrPair::Ref("3'".to_string()),
            GsOrSeq::Seq("TCAATCCTTGCC".to_string()),
        ),
        (
            RefOrPair::Ref("4".to_string()),
            GsOrSeq::Seq("CTGTTCCCAACA".to_string()),
        ),
        (
            RefOrPair::Ref("4'".to_string()),
            GsOrSeq::Seq("CACATCCCTGTT".to_string()),
        ),
        (
            RefOrPair::Ref("5".to_string()),
            GsOrSeq::Seq("CACTACCAGTCC".to_string()),
        ),
        (
            RefOrPair::Ref("5'".to_string()),
            GsOrSeq::Seq("CCATGTCCCATT".to_string()),
        ),
        (
            RefOrPair::Ref("6".to_string()),
            GsOrSeq::Seq("ACACACACTGTC".to_string()),
        ),
        (
            RefOrPair::Ref("6'".to_string()),
            GsOrSeq::Seq("CAACCAACGTTC".to_string()),
        ),
        (
            RefOrPair::Ref("7".to_string()),
            GsOrSeq::Seq("TCACTTTCGTCC".to_string()),
        ),
        (
            RefOrPair::Ref("7'".to_string()),
            GsOrSeq::Seq("TCACACTTCGTC".to_string()),
        ),
    ])
}

fn params() -> SDCParams {
    let scaffold = [
        None,
        None,
        Some("A*"),
        Some("B*"),
        Some("C*"),
        Some("D*"),
        Some("E*"),
        None,
        None,
    ]
    .map(|x| x.map(String::from))
    .to_vec();

    let strands = strands();

    SDCParams {
        strands,
        quencher_name: Some("D0".to_string()),
        quencher_concentration: 1e6,
        reporter_name: Some("Rep".to_string()),
        fluorophore_concentration: 1e6,
        scaffold: SingleOrMultiScaffold::Single(scaffold),
        scaffold_concentration: 1e-100,
        glue_dg_s: create_sequences_map(),
        k_f: 1e6,
        k_n: 0.0,
        k_c: 0.0,
        temperature: 49.0,
        junction_penalty_dg: None,
        junction_penalty_ds: None,
    }
}

fn make_system() -> SDC {
    SDC::from_params(params())
}

#[test]
fn run_sdc_system() {
    let sdc_sys = make_system();
    let mut state = StateEnum::empty(
        (100, 9),
        Square,
        TrackingType::None,
        sdc_sys.tile_names().len(),
    )
    .unwrap();
    sdc_sys.update_state(&mut state, &NeededUpdate::All);
    let bounds = EvolveBounds::default().for_time(1.0 * 60.0 * 60.0);
    let _eo = System::evolve(&sdc_sys, &mut state, bounds).unwrap();
    assert_ne!(state.total_events(), 0)
}
