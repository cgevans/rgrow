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

/// For some given pair a, b, find (Delta G at 37 degrees C, Delta S)
///
/// By default the values found in santalucia_thermodynamics_2004 are used
#[inline(always)]
fn dG_dS(a: &DnaNucleotideBase, b: &DnaNucleotideBase) -> (f64, f64) {
    // Full name made the match statment horrible
    use DnaNucleotideBase::*;
    match (a, b) {
        (T, T) => (-1.0, -0.0213),
        (T, A) => (-0.88, -0.0204),
        (A, T) => (-0.58, -0.0213),
        (G, T) => (-1.45, -0.0227),
        (C, A) => (-1.44, -0.0224),
        (G, A) => (-1.28, -0.0210),
        (C, T) => (-1.30, -0.0222),
        (G, C) => (-2.17, -0.0272),
        (C, G) => (-2.24, -0.0244),
        (C, C) => (-1.84, -0.0199),
        // TODO:Is there missing data that needs to be filled ?
        _ => panic!("Could not get dG/dS of pair!!!"),
    }
}

/// Given some dna sequence eg TAGGCGTA, find dG
/// of said sequence with its "perfect fit"
/// (in this case ATCCGCAT)
///
/// the sum of all neighbours a, b -- dG_(37 degrees C) (a, b) - (temperature - 37) dS(a, b)
fn dna_strength(dna: impl Iterator<Item = DnaNucleotideBase>, temperature: f64) -> f64 {
    two_window_fold(dna, |acc, (a, b)| {
        let (dg, ds) = dG_dS(a, b);
        // Calculate the sum of dG(a, b) - (T - 37) * dS(a, b)
        acc + (dg - (temperature - 37.) * ds)
    })
    .expect("DNA must have length of at least 2")
}

/// Get delta g for some string dna sequence and its "perfect match"
pub fn string_dna_delta_g(dna_sequence: String, temperature: f64) -> f64 {
    dna_strength(
        // Convert dna_sequence string into an iterator of nucleotide bases
        dna_sequence
            .chars()
            .into_iter()
            .map(DnaNucleotideBase::from),
        temperature,
    )
}

#[cfg(test)]
mod test_utils {
    use crate::utils::string_dna_delta_g;

    use super::two_window_fold;

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
    fn test_dna_strength() {}
}
