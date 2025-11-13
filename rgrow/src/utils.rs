use std::f64;

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::units::*;

// For testing
#[macro_export]
macro_rules! assert_all {
    ($($e:expr),*) => {
        $(assert!($e);)*
    };
}

const INITIATION_DG: KcalPerMol = KcalPerMol(1.96);
const INITIATION_DS: KcalPerMolKelvin = KcalPerMolKelvin(0.0057);

const T37C: Kelvin = Kelvin(310.15);

// Gas constant in kcal / mol / K
//
// (same unit as delta G needed)
const R: KcalPerMolKelvin = KcalPerMolKelvin(1.98720425864083 / 1000.0);

/// Index this as follows:
///
/// Given the following MISMATCH
/// PX/(P*)Y then the penalty is given
/// by index [P][X][Y]
const PENALTY_TABLE: [[[f64; 4]; 4]; 4] = [
    // AX/TY
    [
        // X = A
        [0.61, 0.0, 0.14, 0.88],
        // X = T
        [0.0, 0.69, 0.07, 0.73],
        // X = G
        [0.02, 0.71, -0.13, 0.0],
        // X = C
        [0.77, 0.64, 0.0, 1.33],
    ],
    // TX/AY
    [
        [0.69, 0.0, 0.42, 0.92],
        [0.0, 0.68, 0.34, 0.75],
        [0.74, 0.43, 0.44, 0.0],
        [1.33, 0.97, 0.0, 1.05],
    ],
    // GX/CY
    [
        [0.17, 0.0, -0.25, 0.81],
        [0.0, 0.45, -0.59, 0.98],
        [-0.52, 0.08, -1.11, 0.0],
        [0.47, 0.62, 0.0, 0.79],
    ],
    // CX/GY
    [
        [0.43, 0.0, 0.03, 0.75],
        [0.0, -0.12, -0.32, 0.40],
        [0.11, -0.47, -0.11, 0.0],
        [0.79, 0.62, 0.0, 0.70],
    ],
];

pub enum LoopKind {
    Internal = 0,
    Bulge = 1,
    HairPin = 2,
}

const LOOP_TABLE: [[f64; 15]; 3] = [
    // Internal Loops
    [
        3.2, 3.6, 4.0, 4.4, 4.6, 4.8, 4.9, 4.9, 5.2, 5.4, 5.6, 5.8, 5.9, 6.3, 6.6,
    ],
    // Bulge Loops
    [
        3.1, 3.2, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.8, 5.0, 5.2, 5.3, 5.6, 5.9,
    ],
    // Hairpin Loops
    [
        3.5, 3.5, 3.3, 4.0, 4.2, 4.3, 4.5, 4.6, 5.0, 5.1, 5.3, 5.5, 5.7, 6.1, 6.3,
    ],
];

const LENGTHS: [usize; 15] = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30];

/*
* A G A A A
* --------->
* <---------
* T C T T T
*
* dG =
*   g(T, C) + (temp - 37) s(T, C)
*   + g(C, A) + (temp - 37) s(C, A)
*   + g(A, A) + (temp - 37) s(A, A)
*   + g(A, T) + (temp - 37) s(A, T)
* */

/// 2-sliding window generic implementation for any iterator with a fold function
///
/// None will be returned if the iterator is too short
fn two_window_fold<T, K, F>(mut iter: impl Iterator<Item = T>, fold: F) -> Option<K>
where
    K: Default,
    F: Fn(K, (&T, &T)) -> K,
{
    let mut ans = K::default();
    let mut last = iter.next()?;
    let mut current = iter.next()?;

    loop {
        ans = fold(ans, (&last, &current));
        if let Some(next) = iter.next() {
            last = current;
            current = next;
        } else {
            break;
        }
    }

    Some(ans)
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DnaNucleotideBase {
    A = 0,
    T = 1,
    G = 2,
    C = 3,
}

impl DnaNucleotideBase {
    pub fn connects_to(&self) -> Self {
        match self {
            Self::A => Self::T,
            Self::T => Self::A,
            Self::G => Self::C,
            Self::C => Self::G,
        }
    }
}

impl From<char> for DnaNucleotideBase {
    fn from(value: char) -> Self {
        match value {
            'a' | 'A' => DnaNucleotideBase::A,
            'c' | 'C' => DnaNucleotideBase::C,
            'g' | 'G' => DnaNucleotideBase::G,
            't' | 'T' => DnaNucleotideBase::T,
            _ => panic!("DNA sequence must contain only a,c,g,t characters in upper/lower"),
        }
    }
}

/// For some given pair 5' - a, b - 3', find (Delta G at 37 degrees C, Delta S)
///
/// By default the values found in santalucia_thermodynamics_2004 are used
#[inline(always)]
#[allow(non_snake_case)]
fn dG_dS(a: &DnaNucleotideBase, b: &DnaNucleotideBase) -> (KcalPerMol, KcalPerMolKelvin) {
    // Full name made the match statment horrible
    use DnaNucleotideBase::*;
    let (dg, ds) = match (a, b) {
        (T, T) | (A, A) => (-1.0, -0.0213),
        (C, C) | (G, G) => (-1.84, -0.0199),
        (G, T) | (A, C) => (-1.44, -0.0224),
        (C, A) | (T, G) => (-1.45, -0.0227),
        (G, A) | (T, C) => (-1.30, -0.0222),
        (C, T) | (A, G) => (-1.28, -0.0210),
        (T, A) => (-0.58, -0.0213),
        (A, T) => (-0.88, -0.0204),
        (C, G) => (-2.17, -0.0272),
        (G, C) => (-2.24, -0.0244),
    };
    (KcalPerMol(dg), KcalPerMolKelvin(ds))
}

/// Get the binding strength of two sequences
///
/// Right now this function can handle:
/// - Single Mismatches
/// - Many mismatches back to back
///
/// It can not yet handle:
/// - Mismatches at end
///
///
/// If only one dna is provided, then this function will
/// use the given dna sequence eg TAGGCGTA to find dG
/// of said sequence with its "perfect fit"
/// (in this case ATCCGCAT)
///
/// the sum of all neighbours a, b -- dG_(37 degrees C) (a, b) - (temperature - 37) dS(a, b)
fn sequences_strength(
    dna: Vec<DnaNucleotideBase>,
    other_dna: Option<Vec<DnaNucleotideBase>>,
    temperature: impl Temperature,
) -> KcalPerMol {
    let (total_dg, total_ds) = match other_dna {
        None => single_sequence_dg_ds(dna.into_iter()),
        Some(other) => sequence_pair_dg_ds(dna, other),
    };
    (total_dg + INITIATION_DG) - ((temperature.to_kelvin() - T37C) * (total_ds + INITIATION_DS))
}

/// Calculate ΔG37 and ΔS for a single sequence and its reverse complement, provided as an iterator.
fn single_sequence_dg_ds(
    dna: impl Iterator<Item = DnaNucleotideBase>,
) -> (KcalPerMol, KcalPerMolKelvin) {
    let (dg, ds) = two_window_fold(dna, |(acc_dg, acc_ds), (a, b)| {
        let (dg, ds) = dG_dS(a, b);
        (dg + acc_dg, ds + acc_ds)
    })
    .expect("DNA must have length of at least 2");
    (dg, ds)
}

fn good_match(a: &DnaNucleotideBase, b: &DnaNucleotideBase) -> bool {
    a.connects_to() == *b
}

/// Calculate the penalty introduced by a single mismatch
#[inline(always)]
fn calc_penalty(
    prior: &DnaNucleotideBase,
    x: &DnaNucleotideBase,
    y: &DnaNucleotideBase,
) -> KcalPerMol {
    KcalPerMol(PENALTY_TABLE[*prior as usize][*x as usize][*y as usize])
}

fn calculate_single_mismatch_penalty(
    (a1, a2): (&DnaNucleotideBase, &DnaNucleotideBase),
    (b1, b2): (&DnaNucleotideBase, &DnaNucleotideBase),
) -> KcalPerMol {
    if good_match(a1, b1) {
        // Case 1: PX/(P*)Y
        calc_penalty(a1, a2, b2)
    } else if good_match(a2, b2) {
        // Case 2: XP/Y(P*)
        calc_penalty(b2, b1, a1)
    } else {
        KcalPerMol(0.0)
    }
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn sequence_pair_str_dg_ds(dna_a: &'_ str, dna_b: &'_ str) -> (KcalPerMol, KcalPerMolKelvin) {
    let dna_seq_a = dna_a.chars().map(DnaNucleotideBase::from);
    let dna_seq_b = dna_b.chars().map(DnaNucleotideBase::from);
    sequence_pair_dg_ds(dna_seq_a.collect(), dna_seq_b.collect())
}

/// Calculate ΔG37 and ΔS for a pair of sequences.
fn sequence_pair_dg_ds(
    dna_a: Vec<DnaNucleotideBase>,
    dna_b: Vec<DnaNucleotideBase>,
) -> (KcalPerMol, KcalPerMolKelvin) {
    if dna_a.len() != dna_b.len() {
        panic!("Dnas must be same length to compare");
    }

    let mut current_loop_length = 0;
    let (mut dg, mut ds) = (KcalPerMol(0.0), KcalPerMolKelvin(0.0));
    let a_windows = dna_a.windows(2);
    let b_windows = dna_b.windows(2);

    for (a, b) in std::iter::zip(a_windows, b_windows) {
        let (a1, a2) = (a[0], a[1]);
        let (b1, b2) = (b[0], b[1]);
        if good_match(&a2, &b2) && good_match(&a1, &b1) {
            let (ndg, nds) = dG_dS(&a1, &a2);
            dg += ndg;
            ds += nds;

            // When we find a good match we know that the loop is over
            if current_loop_length > 2 {
                dg += loop_penalty(current_loop_length - 1, "internal");
            }
            current_loop_length = 0;
        } else {
            dg += calculate_single_mismatch_penalty((&a1, &a2), (&b1, &b2));
            current_loop_length += 1;
        }
    }
    (dg, ds)
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn string_dna_dg_ds(dna_sequence: &str) -> (KcalPerMol, KcalPerMolKelvin) {
    let (g37, s) = single_sequence_dg_ds(dna_sequence.chars().map(DnaNucleotideBase::from));
    (g37 + INITIATION_DG, s + INITIATION_DS)
}

/// Get delta g for some string dna sequence and its "perfect match".  For example:
///
/// ```rust
/// use rgrow::utils::string_dna_delta_g;
/// use rgrow::units::*;
/// let seq = "cgatg";
/// assert_eq!(string_dna_delta_g(seq, Celsius::new(37.0)), KcalPerMol::new(-5.8+1.96));
/// ```
///
pub fn string_dna_delta_g(dna_sequence: &str, temperature: impl Temperature) -> KcalPerMol {
    sequences_strength(
        // Convert dna_sequence string into an iterator of nucleotide bases
        dna_sequence.chars().map(DnaNucleotideBase::from).collect(),
        None,
        temperature,
    )
}

fn _loop_penalty(length: usize, kind: LoopKind) -> KcalPerMol {
    let (g_diff, len) = LOOP_TABLE[kind as usize]
        .iter()
        .zip(LENGTHS)
        .rev()
        .find(|(_, len)| len <= &length)
        .expect("Please enter a valid length");

    KcalPerMol(*g_diff) + R * T37C * (length as f64 / (len as f64)).ln() * 2.44
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn loop_penalty(length: usize, kind: &str) -> KcalPerMol {
    match kind {
        "bulge" => _loop_penalty(length, LoopKind::Bulge),
        "internal" => _loop_penalty(length, LoopKind::Internal),
        "hairpin" => _loop_penalty(length, LoopKind::HairPin),
        _ => panic!(),
    }
}
#[cfg(test)]
mod test_utils {

    use crate::units::Celsius;
    use crate::units::KcalPerMol;
    use crate::utils::sequence_pair_dg_ds;
    use crate::utils::INITIATION_DG;
    use crate::utils::INITIATION_DS;
    use crate::utils::LOOP_TABLE;
    use crate::utils::T37C;

    use super::string_dna_dg_ds;
    use super::DnaNucleotideBase;
    use super::_loop_penalty;
    use super::string_dna_delta_g;
    use super::two_window_fold;
    use approx::assert_ulps_eq;

    #[test]
    #[allow(clippy::neg_multiply)]
    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    fn test_sliding_window() {
        let v = [1., 2., 0., -1., 5.];
        let expected = ((1 + 2) + (2 + 0) + (0 + (-1)) + ((-1) + 5)) as f64;
        let acc = two_window_fold(v.iter(), |acc: f64, (a, b)| acc + (*a + *b));
        assert_eq!(Some(expected), acc);
        let expected = ((1 * 2) + (2 * 0) + (0 * (-1)) + ((-1) * 5)) as f64;
        let acc = two_window_fold(v.iter(), |acc: f64, (a, b)| acc + (*a * *b));
        assert_eq!(Some(expected), acc);
        let v = Vec::<f64>::new();
        let expected = None;
        let acc = two_window_fold(v.iter(), |acc: f64, (_, _)| acc);
        assert_eq!(expected, acc);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dna_strength() {
        // random sequences
        let seqs = [
            "cg",
            "cttcgccac",
            "gacggcattatgtc",
            "ct",
            "tc",
            "aatacgacggccag",
            "caga",
            "ttaaccctta",
            "actatg",
            "cttaatccgagaataaaaa",
            "gccggggttaaaac",
            "tacaaagggtg",
            "tgg",
            "tggtcgccatctcccgt",
            "ccgttcctagat",
            "agttagagcttttggacta",
            "cacctttccgcagg",
            "tttaacttctc",
            "gcgccct",
            "tatttcgtaacttgcacat",
        ];

        /*
        Values are taken from stickydesign 0.9.0.a3, using

        ```python
        # T is temperature, x is sequence
        -sd.EnergeticsBasic(temperature=T).matching_uniform(sd.endarray([x],'S'))[0]-1.96+(T-37)*0.0057
        ```

        The correction here is because stickydesign includes the initiation penalty
        from SantaLucia.  It's actually unclear whether that should be included here,
        or in other places where it has been included in the past.  It's worth a discussion.
         */

        let dG_at_37 = [
            -2.17,
            -12.719999999999999,
            -17.970000000000002,
            -1.28,
            -1.3,
            -19.630000000000003,
            -4.03,
            -10.560000000000002,
            -5.63,
            -20.39,
            -19.23,
            -13.32,
            -3.29,
            -25.78,
            -14.91,
            -22.57,
            -20.130000000000003,
            -11.18,
            -11.61,
            -22.58,
        ];

        let dG_at_50 = [
            -1.8164,
            -10.365699999999999,
            -14.2065,
            -1.0070000000000001,
            -1.0114,
            -15.8301,
            -3.1733000000000002,
            -8.0939,
            -4.2286,
            -15.3434,
            -15.557500000000003,
            -10.526299999999997,
            -2.7362,
            -21.144200000000005,
            -11.8056,
            -17.513,
            -16.4133,
            -8.381099999999998,
            -9.8316,
            -17.396900000000002,
        ];

        for (&seq, &dG) in seqs.iter().zip(dG_at_37.iter()) {
            let result = string_dna_delta_g(seq, Celsius(37.0));
            println!("{seq}");
            // TODO: Undo dG properly
            assert_ulps_eq!(KcalPerMol(dG) + INITIATION_DG, result, max_ulps = 10);
        }

        for (&seq, &dG) in seqs.iter().zip(dG_at_50.iter()) {
            let result = string_dna_delta_g(seq, Celsius(50.0));
            println!("{seq}");
            assert_ulps_eq!(
                KcalPerMol(dG + 1.96) - (Celsius(50.0) - T37C) * INITIATION_DS,
                result,
                max_ulps = 10
            );
        }
    }

    #[test]
    fn test_loops() {
        let val29 = _loop_penalty(29, super::LoopKind::Internal);
        assert!(val29 > KcalPerMol(LOOP_TABLE[0][13]));
        assert!(val29 < KcalPerMol(LOOP_TABLE[0][14]));
    }

    #[test]
    fn test_mismatch_penalty() {
        let dna_a = "GGACTGACG".chars().map(DnaNucleotideBase::from).collect();
        let dna_b = "CCTGGCTGC".chars().map(DnaNucleotideBase::from).collect();
        let (total, _) = sequence_pair_dg_ds(dna_a, dna_b);
        assert_eq!(total + INITIATION_DG, KcalPerMol(-8.32));
    }

    #[test]
    fn test_no_mismatches() {
        let dna_a = "GGACTGAC".chars().map(DnaNucleotideBase::from).collect();
        let dna_b = {
            let v: &Vec<DnaNucleotideBase> = &dna_a;
            v.iter().map(|s| s.connects_to()).collect()
        };
        let (g, s) = sequence_pair_dg_ds(dna_a, dna_b);
        let (pg, ps) = string_dna_dg_ds("GGACTGAC");
        assert_eq!(g, pg - INITIATION_DG);
        assert_eq!(s, ps - INITIATION_DS);
    }

    #[test]
    fn internal_loop_mismatch() {
        /*
         * Make sure that the way I calculated -0.17 by hand is right
         *
         *       A G C T G
         * A T T | | | | | G T C
         * | | |           | | |
         * T A A | | | | | C A G
         *       G A T G A
         *
         * Delta G =
         *   G(A, T) + G(T, T)
         *    + SingleMismatch(T A / A G)
         *    + InternalLoop(5)
         *    + SingleMismatch(G G / A C)
         *    + G(G, T)  + G(T, C)
         *    = -0.17
         * */
        let (g, _) = sequence_pair_dg_ds(
            "attagctggtc".chars().map(DnaNucleotideBase::from).collect(),
            "taagatgacag".chars().map(DnaNucleotideBase::from).collect(),
        );
        approx::assert_relative_eq!(g, KcalPerMol(-0.17));
    }
}
