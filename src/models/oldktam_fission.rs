use crate::state::State;
use crate::{canvas::PointSafe2, models::oldktam::OldKTAM};

use crate::base::HashMapType;
use crate::base::Tile;
use rand::{distributions::weighted::WeightedIndex, distributions::Distribution};
use std::collections::VecDeque;

// lazy_static! {
//     /// A vector specifying whether or not the 1 bits of the index are in a single
//     /// group, with no 0s between them, when the number is seed as a u8 that is a ring,
//     /// eg, such that 0b11100111 is true.  This is useful because it tells us whether
//     /// tiles arranged in a ring (eg, around a single point) are a single connected
//     /// group.
//     pub static ref CONNECTED_RING: Vec<bool> = {
//         let mut v = Vec::<bool>::with_capacity(2usize.pow(8));
//         v.push(false); // All zeros
//         for i in 0b1u8..0b11111111 {
//             let i = i.rotate_right(i.trailing_ones());
//             let i = i.rotate_right(i.trailing_zeros());
//             v.push((i+1).is_power_of_two())
//         }
//         v.push(true); // All ones
//         v
//     };
// }

static CONNECTED_RING: &[bool] = &[
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
    fn new(start_points: &Vec<&PointSafe2>, now_empty: &[PointSafe2]) -> Self {
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
    fn merge_or_add(&mut self, point1: &PointSafe2, point2: &PointSafe2) -> bool {
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
        let mut rng = rand::thread_rng();

        let sizes: Vec<usize> = mpl.iter().map(|x| x.len()).collect();
        let dist = WeightedIndex::new(&sizes).unwrap();

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

    fn n_groups(&self) -> usize {
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

impl OldKTAM {
    pub fn determine_fission<C: State>(
        &self,
        canvas: &C,
        possible_start_points: &[PointSafe2],
        now_empty: &[PointSafe2],
    ) -> FissionResult {
        // Optimizations for a single empty site.
        if now_empty.len() == 1 {
            let p = now_empty[0];

            let tn = canvas.tile_to_n(p);
            let tw = canvas.tile_to_w(p);
            let te = canvas.tile_to_e(p);
            let ts = canvas.tile_to_s(p);
            let tnw = canvas.tile_to_nw(p);
            let tne = canvas.tile_to_ne(p);
            let tsw = canvas.tile_to_sw(p);
            let tse = canvas.tile_to_se(p);

            let ri: u8 = (((tn != 0) as u8) << 7)
                + ((((self.get_energy_we(tn, tne) != 0.) & (self.get_energy_ns(tne, te) != 0.))
                    as u8)
                    << 6)
                + (((te != 0) as u8) << 5)
                + ((((self.get_energy_ns(te, tse) != 0.) & (self.get_energy_we(ts, tse) != 0.))
                    as u8)
                    << 4)
                + (((ts != 0) as u8) << 3)
                + ((((self.get_energy_we(tsw, ts) != 0.) & (self.get_energy_ns(tw, tsw) != 0.))
                    as u8)
                    << 2)
                + (((tw != 0) as u8) << 1)
                + (((self.get_energy_ns(tnw, tw) != 0.) & (self.get_energy_we(tnw, tn) != 0.))
                    as u8);

            if CONNECTED_RING[ri as usize] {
                return FissionResult::NoFission;
            }

            if ri == 0 {
                //println!("Unattached tile detaching!");
                return FissionResult::NoFission;
            }

            //println!("Ring check failed");
        }

        let start_points = (*possible_start_points)
            .iter()
            .filter(|x| canvas.tile_at_point(**x) != 0)
            .collect();

        let mut groupinfo = GroupInfo::new(&start_points, now_empty);

        let mut queue = VecDeque::new();

        // Put all the starting points in the queue.
        for (_i, point) in start_points.iter().enumerate() {
            queue.push_back(**point);
        }

        //println!("Start queue {:?}", queue);
        while let Some(p) = queue.pop_front() {
            let t = canvas.tile_at_point(p);
            let pn = canvas.move_sa_n(p);
            let tn = canvas.v_sh(pn);
            let pw = canvas.move_sa_w(p);
            let tw = canvas.v_sh(pw);
            let pe = canvas.move_sa_e(p);
            let te = canvas.v_sh(pe);
            let ps = canvas.move_sa_s(p);
            let ts = canvas.v_sh(ps);

            if (unsafe { *self.energy_ns.uget((tn as usize, t as usize)) } != 0.) {
                let pn = PointSafe2(pn.0); // FIXME
                match groupinfo.merge_or_add(&p, &pn) {
                    true => {}
                    false => {
                        queue.push_back(pn);
                    }
                }
            }

            if (unsafe { *self.energy_we.uget((t as usize, te as usize)) } != 0.) {
                let pe = PointSafe2(pe.0); // FIXME
                match groupinfo.merge_or_add(&p, &pe) {
                    true => {}
                    false => {
                        queue.push_back(pe);
                    }
                }
            }

            if (unsafe { *self.energy_ns.uget((t as usize, ts as usize)) } != 0.) {
                let ps = PointSafe2(ps.0); // FIXME
                match groupinfo.merge_or_add(&p, &ps) {
                    true => {}
                    false => {
                        queue.push_back(ps);
                    }
                }
            }

            if (unsafe { *self.energy_we.uget((tw as usize, t as usize)) } != 0.) {
                let pw = PointSafe2(pw.0); // FIXME
                match groupinfo.merge_or_add(&p, &pw) {
                    true => {}
                    false => {
                        queue.push_back(pw);
                    }
                }
            }

            // We break on *two* groups, because group 0 is the removed area,
            // and so there will be two groups (0 and something) if there
            // is one contiguous area.
            if groupinfo.n_groups() <= 2 {
                //println!("Found 2 groups");
                return FissionResult::NoFission;
            }
        }

        //println!("Finished queue");
        //println!("{:?}", groupinfo);
        FissionResult::FissionGroups(groupinfo)
    }
}
