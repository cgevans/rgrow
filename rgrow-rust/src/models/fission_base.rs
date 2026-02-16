use crate::base::HashMapType;
use crate::base::Tile;
use crate::canvas::PointSafe2;
use rand::{distr::weighted::WeightedIndex, distr::Distribution};

/// An array specifying whether or not the 1 bits of the index are in a single
/// group, with no 0s between them, when the number is seed as a u8 that is a ring,
/// eg, such that 0b11100111 is true.  This is useful because it tells us whether
/// tiles arranged in a ring (eg, around a single point) are a single connected
/// group.
///
/// The array is calculated with the code below. A test verifies that the
/// generating code produces the same array as the pre-computed constant.
///
/// ```ignore
/// let mut v = Vec::<bool>::with_capacity(2usize.pow(8));
/// v.push(false); // All zeros
/// for i in 0b1u8..0b11111111 {
///     let i = i.rotate_right(i.trailing_ones());
///     let i = i.rotate_right(i.trailing_zeros());
///     v.push((i+1).is_power_of_two())
/// }
/// v.push(true); // All ones
/// ```
pub(super) static CONNECTED_RING: &[bool] = &[
    false, true, true, true, true, false, true, true, true, false, false, false, true, false, true,
    true, true, false, false, false, false, false, false, false, true, false, false, false, true,
    false, true, true, true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, true, false, false, false, false, false, false, false, true,
    false, false, false, true, false, true, true, true, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, true,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, true, false, false, false, false, false, false, false, true, false, false, false,
    true, false, true, true, true, true, false, true, false, false, false, true, false, false,
    false, false, false, false, false, true, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, true, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, true, true, true, false, true, false, false, false, true, false, false, false, false,
    false, false, false, true, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, true, true, true, false, true, false, false, false,
    true, false, false, false, false, false, false, false, true, true, true, false, true, false,
    false, false, true, true, true, false, true, true, true, true, true,
];

type GroupNum = usize;

#[derive(Debug)]
pub struct GroupInfo {
    /// Contains mappings of point -> (unmerged) group number.
    pub map: HashMapType<PointSafe2, GroupNum>,
    /// Contains mappings of unmerged group number to merged group number.
    groupmerges: Vec<GroupNum>,
    /// Contains lists of points in each (unmerged) group number.
    pointlist: Vec<Vec<PointSafe2>>,
}

impl GroupInfo {
    pub(super) fn new(start_points: &[&PointSafe2], now_empty: &[PointSafe2]) -> Self {
        let groupmerges = (0usize..=start_points.len()).collect();

        let mut map = HashMapType::default();

        let mut pointlist = Vec::new();

        for point in now_empty {
            map.insert(*point, 0usize);
            pointlist.push(vec![*point]);
        }

        for (i, point) in start_points.iter().enumerate() {
            map.insert(**point, i + 1);
            pointlist.push(vec![**point]);
        }

        GroupInfo {
            map,
            groupmerges,
            pointlist,
        }
    }

    /// If point1 and point2 are in different (nonzero) groups,
    /// merge the groups, and return true (no further movement needed).  
    /// If they are in the same group, return true.
    /// If point2 is not in a group, add point2 to point1's group, then
    /// return false (further movement into point2 needed).
    /// Point1 must be in a group, and debug checks to make sure it isn't in zero.
    pub(super) fn merge_or_add(&mut self, point1: &PointSafe2, point2: &PointSafe2) -> bool {
        let g1 = self.groupmerges[*self.map.get(point1).unwrap()];

        assert!(g1 != 0);

        let mp2 = self.map.get(point2);

        if mp2 == Some(&0) {
            return true;
        }

        if let Some(g2) = mp2 {
            let g2 = self.groupmerges[*g2];
            if g1 != g2 {
                let new_group = g1.min(g2);
                for gv in self.groupmerges.iter_mut() {
                    if (*gv == g1) | (*gv == g2) {
                        *gv = new_group;
                    }
                }
            }
            true
        } else {
            self.map.insert(*point2, g1);
            self.pointlist[g1].push(*point2);
            false
        }
    }

    pub fn choose_deletions_size_weighted(&self) -> Vec<PointSafe2> {
        let mpl = self.merged_pointlist();
        let mut rng = rand::rng();

        let sizes: Vec<usize> = mpl.iter().map(|x| x.len()).collect();
        let dist = WeightedIndex::new(sizes).unwrap();

        let keep = dist.sample(&mut rng);

        let mut deletions = Vec::new();

        for (i, pv) in mpl.iter().enumerate() {
            if i == keep {
                continue;
            } else {
                deletions.extend(pv);
            }
        }
        deletions.extend(&self.pointlist[0]);

        deletions
    }

    pub fn choose_deletions_keep_largest_group(&self) -> Vec<PointSafe2> {
        let mut mpl = self.merged_pointlist();

        let mut deletions = Vec::new();

        mpl.sort_by(|a, b| a.len().cmp(&b.len()).reverse());

        let mi = mpl.iter().skip(1);

        for pv in mi {
            deletions.extend(pv)
        }
        deletions.extend(&self.pointlist[0]);

        //println!("{:?} {:?}", deletions, self.groupmerges);

        deletions
    }

    pub fn choose_deletions_seed_unattached(
        &self,
        seeds: Vec<(PointSafe2, Tile)>,
    ) -> Vec<PointSafe2> {
        let mut deletions = Vec::new();

        let seed_points = seeds.iter().map(|x| x.0).collect::<Vec<_>>();

        let mergedpoints = self.merged_pointlist();

        for group in mergedpoints {
            let mut contains_seed = false;
            for seed_point in &seed_points {
                if group.contains(seed_point) {
                    contains_seed = true;
                }
            }
            if contains_seed {
                continue;
            } else {
                deletions.extend(group);
            }
        }

        deletions.extend(&self.pointlist[0]);

        deletions
    }

    pub fn merged_pointlist(&self) -> Vec<Vec<PointSafe2>> {
        let mut mergedpointhash = HashMapType::<usize, Vec<PointSafe2>>::default();
        for (pointvec, i) in self.pointlist.iter().zip(&self.groupmerges) {
            // Exclude deletion group
            if *i == 0 {
                continue;
            }
            match mergedpointhash.get_mut(i) {
                Some(v) => {
                    v.extend(pointvec);
                }
                None => {
                    mergedpointhash.insert(*i, pointvec.clone());
                }
            };
        }
        mergedpointhash.into_values().collect()
    }

    pub(super) fn n_groups(&self) -> usize {
        let mut sg = self.groupmerges.clone();
        sg.sort_unstable();
        sg.dedup();
        sg.len()
    }
}

#[derive(Debug)]
pub enum FissionResult {
    NoFission,
    FissionGroups(GroupInfo),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connected_ring_generation() {
        let mut v = Vec::<bool>::with_capacity(2usize.pow(8));
        v.push(false); // All zeros
        for i in 0b1u8..0b11111111 {
            let i = i.rotate_right(i.trailing_ones());
            let i = i.rotate_right(i.trailing_zeros());
            v.push((i + 1).is_power_of_two())
        }
        v.push(true); // All ones

        assert_eq!(v, CONNECTED_RING);
    }
}
