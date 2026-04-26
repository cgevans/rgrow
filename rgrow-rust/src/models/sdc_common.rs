//! Helpers shared between SDC1D and SDC2D models.
//!
//! These were originally defined in `sdc1d.rs` and imported across SDC
//! variants. They are extracted here so that SDC2D can use them without
//! depending on `sdc1d.rs` internals.

use std::collections::HashMap;

use crate::units::{KcalPerMol, KcalPerMolKelvin};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum RefOrPair {
    Ref(String),
    Pair(String, String),
}

impl From<String> for RefOrPair {
    fn from(r: String) -> Self {
        RefOrPair::Ref(r)
    }
}

impl From<(String, String)> for RefOrPair {
    fn from(p: (String, String)) -> Self {
        RefOrPair::Pair(p.0, p.1)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum GsOrSeq {
    GS((f64, f64)),
    Seq(String),
}

pub fn gsorseq_to_gs(gsorseq: &GsOrSeq) -> (KcalPerMol, KcalPerMolKelvin) {
    match gsorseq {
        GsOrSeq::GS(x) => (KcalPerMol(x.0), KcalPerMolKelvin(x.1)),
        GsOrSeq::Seq(s) => crate::utils::string_dna_dg_ds(s.as_str()),
    }
}

/// Triple (is_from, from, inverse).
///
/// - is_from: true if `value` is the "from" form (no trailing star, or even number of stars)
/// - from:    canonical form (e.g. "h")
/// - inverse: starred form (e.g. "h*")
pub fn self_and_inverse(value: &str) -> (bool, String, String) {
    let filtered = value.trim_end_matches('*');
    let star_count = value.len() - filtered.len();
    let is_from = star_count.is_multiple_of(2);
    (is_from, filtered.to_string(), format!("{filtered}*"))
}

/// Look up or allocate a glue ID for `val`.
///
/// `None` resolves to glue ID 0 (the null glue, which never binds).
/// Both the canonical and starred form are inserted on first sight.
pub fn get_or_generate(
    map: &mut HashMap<String, usize>,
    count: &mut usize,
    val: Option<String>,
) -> usize {
    let str = match val {
        Some(x) => x,
        None => return 0,
    };

    let (is_from, fromval, toval) = self_and_inverse(&str);
    let simpl = if is_from { &fromval } else { &toval };
    if let Some(u) = map.get(simpl) {
        return *u;
    }

    map.insert(fromval, *count);
    map.insert(toval, *count + 1);
    *count += 2;

    if is_from {
        *count - 2
    } else {
        *count - 1
    }
}
