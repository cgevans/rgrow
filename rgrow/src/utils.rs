#[cfg(feature = "python")]
use pyo3::prelude::*;

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

/// 2-sliding window generic implementatin for any iterator with a fold function
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

#[derive(Debug, Clone)]
enum DnaNucleotideBase {
    A,
    T,
    G,
    C,
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
fn dG_dS(a: &DnaNucleotideBase, b: &DnaNucleotideBase) -> (f64, f64) {
    // Full name made the match statment horrible
    use DnaNucleotideBase::*;
    match (a, b) {
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
    }
}

/// Given some dna sequence eg TAGGCGTA, find dG
/// of said sequence with its "perfect fit"
/// (in this case ATCCGCAT)
///
/// the sum of all neighbours a, b -- dG_(37 degrees C) (a, b) - (temperature - 37) dS(a, b)
fn dna_strength(dna: impl Iterator<Item = DnaNucleotideBase>, temperature: f64) -> f64 {
    let (total_dg, total_ds) = dna_dg_ds(dna);
    total_dg - (temperature - 37.0) * total_ds
}

fn dna_dg_ds(dna: impl Iterator<Item = DnaNucleotideBase>) -> (f64, f64) {
    two_window_fold(dna, |(acc_dg, acc_ds), (a, b)| {
        let (dg, ds) = dG_dS(a, b);
        (dg + acc_dg, ds + acc_ds)
    })
    .expect("DNA must have length of at least 2")
}

#[cfg_attr(feature = "python", pyfunction)]
pub fn string_dna_dg_ds(dna_sequence: &str) -> (f64, f64) {
    dna_dg_ds(dna_sequence.chars().map(DnaNucleotideBase::from))
}

/// Get delta g for some string dna sequence and its "perfect match".  For example:
///
/// ```rust
/// use rgrow::utils::string_dna_delta_g;
/// let seq = "cgatg";
/// assert_eq!(string_dna_delta_g(seq, 37.0), -5.8);
/// ```
///
pub fn string_dna_delta_g(dna_sequence: &str, temperature: f64) -> f64 {
    dna_strength(
        // Convert dna_sequence string into an iterator of nucleotide bases
        dna_sequence.chars().map(DnaNucleotideBase::from),
        temperature,
    )
}

#[cfg(test)]
mod test_utils {

    use super::string_dna_delta_g;
    use super::two_window_fold;
    use approx::assert_ulps_eq;

    #[test]
    fn test_sliding_window() {
        let v = vec![1., 2., 0., -1., 5.];
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
            let result = string_dna_delta_g(seq, 37.0);
            println!("{}", seq);
            assert_ulps_eq!(dG, result, max_ulps = 10);
        }

        for (&seq, &dG) in seqs.iter().zip(dG_at_50.iter()) {
            let result = string_dna_delta_g(seq, 50.0);
            println!("{}", seq);
            assert_ulps_eq!(dG, result, max_ulps = 10);
        }
    }
}
