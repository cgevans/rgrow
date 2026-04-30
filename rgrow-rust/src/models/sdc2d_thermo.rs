//! Exact finite-grid thermodynamics for SDC2D.
//!
//! These methods enumerate a frontier of length `min(nrows, ncols)`, so they
//! are exact but exponential in the smaller scaffold dimension.

use std::collections::{hash_map::Entry, HashMap};

use crate::base::Tile;
use crate::canvas::PointSafe2;
use crate::units::Temperature;

use super::sdc2d::SDC2D;

const R: f64 = 1.98720425864083 / 1000.0; // kcal/mol/K

#[derive(Debug, Clone, Copy)]
struct ScanOrder {
    major_len: usize,
    minor_len: usize,
    transposed: bool,
}

impl ScanOrder {
    fn new(nrows: usize, ncols: usize) -> Self {
        if ncols <= nrows {
            Self {
                major_len: nrows,
                minor_len: ncols,
                transposed: false,
            }
        } else {
            Self {
                major_len: ncols,
                minor_len: nrows,
                transposed: true,
            }
        }
    }

    fn site(&self, major: usize, minor: usize) -> (usize, usize) {
        if self.transposed {
            (minor, major)
        } else {
            (major, minor)
        }
    }
}

fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    if m.is_infinite() {
        return m;
    }
    m + ((a - m).exp() + (b - m).exp()).ln()
}

fn logsumexp(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(f64::NEG_INFINITY, logaddexp)
}

impl SDC2D {
    #[inline(always)]
    fn rtval(&self) -> f64 {
        R * self.temperature().to_kelvin_m()
    }

    fn has_state_shape(&self, state: &[Vec<Tile>]) -> bool {
        state.len() == self.nrows() && state.iter().all(|row| row.len() == self.ncols())
    }

    fn validate_state_shape(&self, state: &[Vec<Tile>]) {
        assert_eq!(
            state.len(),
            self.nrows(),
            "SDC2D state has {} rows but system has {} rows",
            state.len(),
            self.nrows()
        );
        for (row_idx, row) in state.iter().enumerate() {
            assert_eq!(
                row.len(),
                self.ncols(),
                "SDC2D state row {row_idx} has {} columns but system has {} columns",
                row.len(),
                self.ncols()
            );
        }
    }

    fn validate_constraints_shape(&self, constraints: &[Vec<Vec<Tile>>]) {
        assert_eq!(
            constraints.len(),
            self.nrows(),
            "SDC2D constraints have {} rows but system has {} rows",
            constraints.len(),
            self.nrows()
        );
        for (row_idx, row) in constraints.iter().enumerate() {
            assert_eq!(
                row.len(),
                self.ncols(),
                "SDC2D constraints row {row_idx} has {} columns but system has {} columns",
                row.len(),
                self.ncols()
            );
        }
    }

    fn validate_state_tile_ids(&self, state: &[Vec<Tile>]) {
        for (row_idx, row) in state.iter().enumerate() {
            for (col_idx, &tile) in row.iter().enumerate() {
                assert!(
                    (tile as usize) < self.n_strands(),
                    "SDC2D state tile {tile} at ({row_idx}, {col_idx}) is out of range for {} strands",
                    self.n_strands()
                );
            }
        }
    }

    fn base_allowed_tiles_at(&self, row: usize, col: usize) -> Vec<Tile> {
        if let Some(&seed_tile) = self.seed.get(&PointSafe2((row, col))) {
            return vec![seed_tile];
        }

        let friends = &self.friends_btm[(row, col)];
        let mut allowed = Vec::with_capacity(friends.len() + 1);
        allowed.push(0);
        for &tile in friends {
            if !allowed.contains(&tile) {
                allowed.push(tile);
            }
        }
        allowed
    }

    fn constrained_allowed_tiles_at(
        &self,
        row: usize,
        col: usize,
        constraints: Option<&[Vec<Vec<Tile>>]>,
    ) -> Vec<Tile> {
        let base = self.base_allowed_tiles_at(row, col);
        let Some(constraints) = constraints else {
            return base;
        };
        let constrained = &constraints[row][col];
        if constrained.is_empty() {
            return base;
        }

        let mut allowed = Vec::new();
        for &tile in constrained {
            if base.contains(&tile) && !allowed.contains(&tile) {
                allowed.push(tile);
            }
        }
        allowed
    }

    fn is_tile_allowed_at(&self, row: usize, col: usize, tile: Tile) -> bool {
        if (tile as usize) >= self.n_strands() {
            return false;
        }
        self.base_allowed_tiles_at(row, col).contains(&tile)
    }

    fn site_beta_energy(&self, row: usize, col: usize, tile: Tile) -> f64 {
        if tile == 0 {
            0.0
        } else {
            assert!(
                (tile as usize) < self.n_strands(),
                "SDC2D tile {tile} at ({row}, {col}) is out of range for {} strands",
                self.n_strands()
            );
            self.bond_with_scaffold(row, col, tile)
                - self.strand_concentration[tile as usize].over_u0().ln()
        }
    }

    fn edge_beta_between(
        &self,
        a_pos: (usize, usize),
        a_tile: Tile,
        b_pos: (usize, usize),
        b_tile: Tile,
    ) -> f64 {
        match (a_pos, b_pos) {
            ((ar, ac), (br, bc)) if ar == br && bc == ac + 1 => {
                if a_tile == 0 || b_tile == 0 {
                    0.0
                } else {
                    self.bond_we(a_tile, b_tile)
                }
            }
            ((ar, ac), (br, bc)) if ar == br && ac == bc + 1 => {
                if a_tile == 0 || b_tile == 0 {
                    0.0
                } else {
                    self.bond_we(b_tile, a_tile)
                }
            }
            ((ar, ac), (br, bc)) if ac == bc && br == ar + 1 => {
                if a_tile == 0 || b_tile == 0 {
                    0.0
                } else {
                    self.bond_ns(a_tile, b_tile)
                }
            }
            ((ar, ac), (br, bc)) if ac == bc && ar == br + 1 => {
                if a_tile == 0 || b_tile == 0 {
                    0.0
                } else {
                    self.bond_ns(b_tile, a_tile)
                }
            }
            _ => panic!("SDC2D thermo edge positions {a_pos:?} and {b_pos:?} are not adjacent"),
        }
    }

    fn state_beta_energy(&self, state: &[Vec<Tile>]) -> f64 {
        self.validate_state_shape(state);
        self.validate_state_tile_ids(state);

        let mut energy = 0.0;
        for row in 0..self.nrows() {
            for col in 0..self.ncols() {
                let tile = state[row][col];
                energy += self.site_beta_energy(row, col, tile);
                if col > 0 {
                    energy += self.edge_beta_between(
                        (row, col - 1),
                        state[row][col - 1],
                        (row, col),
                        tile,
                    );
                }
                if row > 0 {
                    energy += self.edge_beta_between(
                        (row - 1, col),
                        state[row - 1][col],
                        (row, col),
                        tile,
                    );
                }
            }
        }
        energy
    }

    /// Physical free energy of a 2D state in kcal/mol.
    pub fn state_g(&self, state: &[Vec<Tile>]) -> f64 {
        self.state_beta_energy(state) * self.rtval()
    }

    fn log_partition_function_inner(&self, constraints: Option<&[Vec<Vec<Tile>>]>) -> f64 {
        if let Some(constraints) = constraints {
            self.validate_constraints_shape(constraints);
        }

        let order = ScanOrder::new(self.nrows(), self.ncols());
        let mut table: HashMap<Vec<Tile>, f64> = HashMap::new();
        table.insert(vec![0; order.minor_len], 0.0);

        for major in 0..order.major_len {
            for minor in 0..order.minor_len {
                let (row, col) = order.site(major, minor);
                let allowed = self.constrained_allowed_tiles_at(row, col, constraints);
                if allowed.is_empty() {
                    return f64::NEG_INFINITY;
                }

                let pos = (row, col);
                let mut next: HashMap<Vec<Tile>, f64> =
                    HashMap::with_capacity(table.len() * allowed.len());
                for (frontier, &log_weight) in table.iter() {
                    if log_weight == f64::NEG_INFINITY {
                        continue;
                    }

                    let previous_major_tile = if major > 0 { frontier[minor] } else { 0 };
                    let previous_minor_tile = if minor > 0 { frontier[minor - 1] } else { 0 };
                    for &tile in allowed.iter() {
                        let mut added_energy = self.site_beta_energy(row, col, tile);
                        if major > 0 {
                            let previous_pos = order.site(major - 1, minor);
                            added_energy += self.edge_beta_between(
                                previous_pos,
                                previous_major_tile,
                                pos,
                                tile,
                            );
                        }
                        if minor > 0 {
                            let previous_pos = order.site(major, minor - 1);
                            added_energy += self.edge_beta_between(
                                previous_pos,
                                previous_minor_tile,
                                pos,
                                tile,
                            );
                        }

                        let mut new_frontier = frontier.clone();
                        new_frontier[minor] = tile;
                        let new_log_weight = log_weight - added_energy;
                        match next.entry(new_frontier) {
                            Entry::Vacant(entry) => {
                                entry.insert(new_log_weight);
                            }
                            Entry::Occupied(mut entry) => {
                                *entry.get_mut() = logaddexp(*entry.get(), new_log_weight);
                            }
                        }
                    }
                }
                table = next;
            }
        }

        logsumexp(table.values().copied())
    }

    /// Exact log partition function.
    ///
    /// This dynamic program is exact on the finite grid, with memory exponential
    /// in `min(nrows, ncols)`.
    pub fn log_partition_function(&self) -> f64 {
        self.log_partition_function_inner(None)
    }

    /// Exact partition function as an `f64`.
    ///
    /// Prefer [`SDC2D::log_partition_function`] for large systems, since this
    /// value can overflow.
    pub fn partition_function(&self) -> f64 {
        self.log_partition_function().exp()
    }

    /// Exact log partition function subject to per-site allowed-tile constraints.
    ///
    /// The innermost constraint list is empty for an unconstrained site; otherwise
    /// it lists the allowed tile IDs at that site. Tile `0` is the empty state.
    pub fn log_partial_partition_function(&self, constraints: Vec<Vec<Vec<Tile>>>) -> f64 {
        self.log_partition_function_inner(Some(&constraints))
    }

    /// Exact partition function subject to per-site allowed-tile constraints.
    pub fn partial_partition_function(&self, constraints: Vec<Vec<Vec<Tile>>>) -> f64 {
        self.log_partial_partition_function(constraints).exp()
    }

    pub fn probability_of_state(&self, state: &[Vec<Tile>]) -> f64 {
        if !self.has_state_shape(state) {
            return 0.0;
        }
        for row in 0..self.nrows() {
            for col in 0..self.ncols() {
                if !self.is_tile_allowed_at(row, col, state[row][col]) {
                    return 0.0;
                }
            }
        }

        (-self.state_beta_energy(state) - self.log_partition_function()).exp()
    }

    pub fn probability_of_constrained_configurations(
        &self,
        constraints: Vec<Vec<Vec<Tile>>>,
    ) -> f64 {
        (self.log_partial_partition_function(constraints) - self.log_partition_function()).exp()
    }

    fn mfe_configuration_inner(
        &self,
        constraints: Option<&[Vec<Vec<Tile>>]>,
    ) -> (Vec<Vec<Tile>>, f64) {
        if let Some(constraints) = constraints {
            self.validate_constraints_shape(constraints);
        }

        let order = ScanOrder::new(self.nrows(), self.ncols());
        let mut table: HashMap<Vec<Tile>, f64> = HashMap::new();
        let mut backpointers: Vec<HashMap<Vec<Tile>, (Vec<Tile>, Tile)>> =
            Vec::with_capacity(self.nrows() * self.ncols());
        table.insert(vec![0; order.minor_len], 0.0);

        for major in 0..order.major_len {
            for minor in 0..order.minor_len {
                let (row, col) = order.site(major, minor);
                let allowed = self.constrained_allowed_tiles_at(row, col, constraints);
                assert!(
                    !allowed.is_empty(),
                    "SDC2D MFE has no legal tile at ({row}, {col}) under constraints"
                );

                let pos = (row, col);
                let mut next: HashMap<Vec<Tile>, f64> =
                    HashMap::with_capacity(table.len() * allowed.len());
                let mut step_backpointers: HashMap<Vec<Tile>, (Vec<Tile>, Tile)> =
                    HashMap::with_capacity(table.len() * allowed.len());

                for (frontier, &energy) in table.iter() {
                    let previous_major_tile = if major > 0 { frontier[minor] } else { 0 };
                    let previous_minor_tile = if minor > 0 { frontier[minor - 1] } else { 0 };
                    for &tile in allowed.iter() {
                        let mut added_energy = self.site_beta_energy(row, col, tile);
                        if major > 0 {
                            let previous_pos = order.site(major - 1, minor);
                            added_energy += self.edge_beta_between(
                                previous_pos,
                                previous_major_tile,
                                pos,
                                tile,
                            );
                        }
                        if minor > 0 {
                            let previous_pos = order.site(major, minor - 1);
                            added_energy += self.edge_beta_between(
                                previous_pos,
                                previous_minor_tile,
                                pos,
                                tile,
                            );
                        }

                        let mut new_frontier = frontier.clone();
                        new_frontier[minor] = tile;
                        let new_energy = energy + added_energy;
                        match next.entry(new_frontier.clone()) {
                            Entry::Vacant(entry) => {
                                entry.insert(new_energy);
                                step_backpointers.insert(new_frontier, (frontier.clone(), tile));
                            }
                            Entry::Occupied(mut entry) if new_energy < *entry.get() => {
                                *entry.get_mut() = new_energy;
                                step_backpointers.insert(new_frontier, (frontier.clone(), tile));
                            }
                            Entry::Occupied(_) => {}
                        }
                    }
                }

                table = next;
                backpointers.push(step_backpointers);
            }
        }

        let (mut frontier, min_energy_beta) = table
            .into_iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("SDC2D MFE has no legal configuration");

        let mut tiles_reversed = Vec::with_capacity(backpointers.len());
        for step_backpointers in backpointers.iter().rev() {
            let (previous_frontier, tile) = step_backpointers
                .get(&frontier)
                .unwrap_or_else(|| panic!("SDC2D MFE backpointer missing for {frontier:?}"))
                .clone();
            tiles_reversed.push(tile);
            frontier = previous_frontier;
        }
        tiles_reversed.reverse();

        let mut state = vec![vec![0; self.ncols()]; self.nrows()];
        for (step, tile) in tiles_reversed.into_iter().enumerate() {
            let major = step / order.minor_len;
            let minor = step % order.minor_len;
            let (row, col) = order.site(major, minor);
            state[row][col] = tile;
        }

        (state, min_energy_beta * self.rtval())
    }

    /// Minimum free energy configuration and its physical free energy in kcal/mol.
    pub fn mfe_configuration(&self) -> (Vec<Vec<Tile>>, f64) {
        self.mfe_configuration_inner(None)
    }

    #[cfg(test)]
    fn visit_legal_states_for_tests(
        &self,
        constraints: Option<&[Vec<Vec<Tile>>]>,
        state: &mut [Vec<Tile>],
        idx: usize,
        visit: &mut dyn FnMut(&[Vec<Tile>]),
    ) {
        if idx == self.nrows() * self.ncols() {
            visit(state);
            return;
        }

        let row = idx / self.ncols();
        let col = idx % self.ncols();
        for tile in self.constrained_allowed_tiles_at(row, col, constraints) {
            state[row][col] = tile;
            self.visit_legal_states_for_tests(constraints, state, idx + 1, visit);
        }
    }

    #[cfg(test)]
    fn partition_function_full_slow_for_tests(&self) -> f64 {
        let mut state = vec![vec![0; self.ncols()]; self.nrows()];
        let mut z = 0.0;
        self.visit_legal_states_for_tests(None, &mut state, 0, &mut |state| {
            z += (-self.state_beta_energy(state)).exp();
        });
        z
    }

    #[cfg(test)]
    fn partial_partition_function_full_slow_for_tests(
        &self,
        constraints: &[Vec<Vec<Tile>>],
    ) -> f64 {
        self.validate_constraints_shape(constraints);
        let mut state = vec![vec![0; self.ncols()]; self.nrows()];
        let mut z = 0.0;
        self.visit_legal_states_for_tests(Some(constraints), &mut state, 0, &mut |state| {
            z += (-self.state_beta_energy(state)).exp();
        });
        z
    }

    #[cfg(test)]
    fn brute_force_mfe_for_tests(&self) -> (Vec<Vec<Tile>>, f64) {
        let mut state = vec![vec![0; self.ncols()]; self.nrows()];
        let mut best_state = Vec::new();
        let mut best_energy = f64::INFINITY;
        self.visit_legal_states_for_tests(None, &mut state, 0, &mut |state| {
            let energy = self.state_beta_energy(state);
            if energy < best_energy {
                best_energy = energy;
                best_state = state.to_vec();
            }
        });
        (best_state, best_energy * self.rtval())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::base::Tile;

    use super::super::sdc2d::{SDC2DParams, SDC2DStrand};
    use super::super::sdc_common::{GsOrSeq, RefOrPair};
    use super::*;

    fn assert_close(left: f64, right: f64, rel_tol: f64) {
        let scale = left.abs().max(right.abs()).max(1.0);
        assert!(
            (left - right).abs() <= rel_tol * scale,
            "left={left}, right={right}, diff={}, rel_tol={rel_tol}",
            (left - right).abs()
        );
    }

    fn empty_constraints(sys: &SDC2D) -> Vec<Vec<Vec<Tile>>> {
        vec![vec![Vec::new(); sys.ncols()]; sys.nrows()]
    }

    fn independent_sys(nrows: usize, ncols: usize, dg: f64, concentration: f64) -> SDC2D {
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("g".into()), GsOrSeq::GS((dg, 0.0)));
        SDC2D::from_params(SDC2DParams {
            strands: vec![SDC2DStrand {
                name: Some("A".into()),
                color: None,
                concentration,
                west_glue: None,
                north_glue: None,
                east_glue: None,
                south_glue: None,
                bottom_glue: Some("g".into()),
            }],
            scaffold: vec![vec![Some("g*".into()); ncols]; nrows],
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        })
    }

    fn lateral_sys(nrows: usize, ncols: usize) -> SDC2D {
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("g".into()), GsOrSeq::GS((-1.0, 0.0)));
        glue_dg_s.insert(RefOrPair::Ref("h".into()), GsOrSeq::GS((-0.7, 0.0)));
        glue_dg_s.insert(RefOrPair::Ref("v".into()), GsOrSeq::GS((-1.3, 0.0)));
        SDC2D::from_params(SDC2DParams {
            strands: vec![SDC2DStrand {
                name: Some("A".into()),
                color: None,
                concentration: 1.0,
                west_glue: Some("h*".into()),
                north_glue: Some("v*".into()),
                east_glue: Some("h".into()),
                south_glue: Some("v".into()),
                bottom_glue: Some("g".into()),
            }],
            scaffold: vec![vec![Some("g*".into()); ncols]; nrows],
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        })
    }

    #[test]
    fn test_thermo_one_by_one_manual() {
        let sys = independent_sys(1, 1, -1.0, 1.0);
        let site_beta = sys.site_beta_energy(0, 0, 1);
        let expected_z = 1.0 + (-site_beta).exp();

        assert_close(sys.partition_function(), expected_z, 1e-12);
        assert_close(sys.log_partition_function(), expected_z.ln(), 1e-12);
        assert_close(sys.state_g(&[vec![1]]), site_beta * sys.rtval(), 1e-12);

        let (mfe_state, mfe_g) = sys.mfe_configuration();
        let expected_tile = if site_beta < 0.0 { 1 } else { 0 };
        assert_eq!(mfe_state, vec![vec![expected_tile]]);
        assert_close(mfe_g, site_beta.min(0.0) * sys.rtval(), 1e-12);

        let p_empty = sys.probability_of_state(&[vec![0]]);
        let p_filled = sys.probability_of_state(&[vec![1]]);
        assert_close(p_empty + p_filled, 1.0, 1e-12);
    }

    #[test]
    fn test_thermo_two_by_two_independent_sites() {
        let sys = independent_sys(2, 2, -2.0, 1.0);
        let mut expected_z = 1.0;
        for row in 0..2 {
            for col in 0..2 {
                expected_z *= 1.0 + (-sys.site_beta_energy(row, col, 1)).exp();
            }
        }
        assert_close(sys.partition_function(), expected_z, 1e-12);

        let mut constraints = empty_constraints(&sys);
        constraints[0][0] = vec![0];
        let expected_partial = expected_z / (1.0 + (-sys.site_beta_energy(0, 0, 1)).exp());
        assert_close(
            sys.partial_partition_function(constraints),
            expected_partial,
            1e-12,
        );
    }

    #[test]
    fn test_thermo_two_by_two_lateral_bonds_match_bruteforce() {
        let sys = lateral_sys(2, 2);
        let brute_z = sys.partition_function_full_slow_for_tests();
        assert_close(sys.partition_function(), brute_z, 1e-12);
        assert_close(sys.log_partition_function(), brute_z.ln(), 1e-12);

        let mut constraints = empty_constraints(&sys);
        constraints[0][0] = vec![0];
        let brute_partial = sys.partial_partition_function_full_slow_for_tests(&constraints);
        assert_close(
            sys.partial_partition_function(constraints),
            brute_partial,
            1e-12,
        );

        let (mfe_state, mfe_g) = sys.mfe_configuration();
        let (_brute_state, brute_g) = sys.brute_force_mfe_for_tests();
        assert_close(mfe_g, brute_g, 1e-12);
        assert_close(sys.state_g(&mfe_state), mfe_g, 1e-12);
    }

    #[test]
    fn test_thermo_per_position_friends_and_impossible_constraints() {
        let mut glue_dg_s = HashMap::new();
        glue_dg_s.insert(RefOrPair::Ref("p".into()), GsOrSeq::GS((-1.0, 0.0)));
        glue_dg_s.insert(RefOrPair::Ref("q".into()), GsOrSeq::GS((-2.0, 0.0)));
        let sys = SDC2D::from_params(SDC2DParams {
            strands: vec![
                SDC2DStrand {
                    name: Some("P".into()),
                    color: None,
                    concentration: 1.0,
                    west_glue: None,
                    north_glue: None,
                    east_glue: None,
                    south_glue: None,
                    bottom_glue: Some("p".into()),
                },
                SDC2DStrand {
                    name: Some("Q".into()),
                    color: None,
                    concentration: 1.0,
                    west_glue: None,
                    north_glue: None,
                    east_glue: None,
                    south_glue: None,
                    bottom_glue: Some("q".into()),
                },
            ],
            scaffold: vec![vec![Some("p*".into()), Some("q*".into())]],
            scaffold_concentration: 1e-9,
            glue_dg_s,
            k_f: 1e6,
            temperature: 37.0,
            seed: vec![],
        });

        assert_eq!(sys.base_allowed_tiles_at(0, 0), vec![0, 1]);
        assert_eq!(sys.base_allowed_tiles_at(0, 1), vec![0, 2]);

        let mut constraints = empty_constraints(&sys);
        constraints[0][0] = vec![2];
        assert_eq!(
            sys.log_partial_partition_function(constraints),
            f64::NEG_INFINITY
        );

        let mut constraints = empty_constraints(&sys);
        constraints[0][0] = vec![1, 2, 1];
        let expected =
            (-sys.site_beta_energy(0, 0, 1)).exp() * (1.0 + (-sys.site_beta_energy(0, 1, 2)).exp());
        assert_close(sys.partial_partition_function(constraints), expected, 1e-12);
    }

    #[test]
    fn test_thermo_seed_constraints() {
        let mut params = {
            let mut glue_dg_s = HashMap::new();
            glue_dg_s.insert(RefOrPair::Ref("g".into()), GsOrSeq::GS((-1.0, 0.0)));
            SDC2DParams {
                strands: vec![SDC2DStrand {
                    name: Some("A".into()),
                    color: None,
                    concentration: 1.0,
                    west_glue: None,
                    north_glue: None,
                    east_glue: None,
                    south_glue: None,
                    bottom_glue: Some("g".into()),
                }],
                scaffold: vec![
                    vec![Some("g*".into()), Some("g*".into())],
                    vec![Some("g*".into()), Some("g*".into())],
                ],
                scaffold_concentration: 1e-9,
                glue_dg_s,
                k_f: 1e6,
                temperature: 37.0,
                seed: vec![],
            }
        };
        params.seed = vec![(0, 0, "A".into())];
        let sys = SDC2D::from_params(params);

        assert_eq!(sys.base_allowed_tiles_at(0, 0), vec![1]);
        assert_eq!(sys.probability_of_state(&[vec![0, 0], vec![0, 0]]), 0.0);

        let (mfe_state, _) = sys.mfe_configuration();
        assert_eq!(mfe_state[0][0], 1);
        assert_close(
            sys.partition_function(),
            sys.partition_function_full_slow_for_tests(),
            1e-12,
        );
    }

    #[test]
    fn test_thermo_transposed_orientation_matches_bruteforce() {
        let sys = lateral_sys(2, 4);
        let brute_z = sys.partition_function_full_slow_for_tests();
        assert_close(sys.partition_function(), brute_z, 1e-12);
        assert_close(sys.log_partition_function(), brute_z.ln(), 1e-12);

        let (mfe_state, mfe_g) = sys.mfe_configuration();
        let (_brute_state, brute_g) = sys.brute_force_mfe_for_tests();
        assert_close(mfe_g, brute_g, 1e-12);
        assert_close(sys.state_g(&mfe_state), mfe_g, 1e-12);
    }

    #[test]
    fn test_thermo_probability_identities() {
        let sys = lateral_sys(2, 2);
        let state = vec![vec![1, 0], vec![0, 1]];
        let constraints = state
            .iter()
            .map(|row| row.iter().map(|&tile| vec![tile]).collect())
            .collect::<Vec<Vec<Vec<Tile>>>>();

        assert_close(
            sys.probability_of_constrained_configurations(constraints),
            sys.probability_of_state(&state),
            1e-12,
        );
        assert_close(
            sys.partial_partition_function(empty_constraints(&sys)),
            sys.partition_function(),
            1e-12,
        );
        let (mfe_state, mfe_g) = sys.mfe_configuration();
        assert_close(sys.state_g(&mfe_state), mfe_g, 1e-12);
    }
}
