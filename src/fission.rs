use super::base::Point;
use super::Canvas;
use crate::StaticKTAM;
use fnv::FnvHashMap;
use std::collections::VecDeque;
use std::iter::repeat;

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

struct GroupInfo {
    /// Contains mappings of point -> (unmerged) group number.
    map: FnvHashMap<Point, GroupNum>,
    /// Contains mappings of unmerged group number to merged group number.
    groupmerges: Vec<GroupNum>,
    /// Contains lists of points in each (unmerged) group number.
    pointlist: Vec<Vec<Point>>
}

impl GroupInfo {
    /// If point1 and point2 are in different (nonzero) groups,
    /// merge the groups, and return true (no further movement needed).  
    /// If they are in the same group, return true.
    /// If point2 is not in a group, add point2 to point1's group, then
    /// return false (further movement into point2 needed).
    /// Point1 must be in a group, and debug checks to make sure it isn't in zero.
    fn merge_or_add(&mut self, point1: &Point, point2: &Point) -> bool {
        debug_assert!(self.map.get(point1) != Some(&0));

        if let Some(g2) = self.map.get(point2) {
            let g1 = self.map.get(point1).unwrap(); // point1 must be in group.
            if *g1 != *g2 {
                let new_group = *g1.min(g2);
                for gv in self.groupmerges.iter_mut() {
                    if (*gv == *g1) | (*gv == *g2) {
                        *gv = new_group;
                    }
                }
            }
            true
        } else {
            let g1 = *self.map.get(point1).unwrap(); // point1 must be in group.
            self.map.insert(*point2, g1);
            self.pointlist[g1].push(*point2);
            false
        }
    }

    fn choose_deletions_size_weighted(&self) {
        todo!();
    }

    fn choose_deletions_largest_group(&self) {
        todo!();
    }

    fn choose_deletions_seed_unattached(&self) {
        todo!();
    }

    fn n_groups(&self) -> usize {
        let mut sg = self.groupmerges.clone();
        sg.sort();
        sg.dedup();
        sg.len()
    }
}

struct GroupVec{
    ident: Vec<usize>,
}

impl GroupVec {
    fn merge_groups(&mut self, g1: usize, g2: usize) -> usize {
        let ng = g1.min(g2);

        for gv in self.ident.iter_mut() {
            if (*gv == g1) | (*gv == g2) {
                *gv = ng;
            }
        }

        ng
    }

    fn n_groups(&self) -> usize {
        let mut sg = self.ident.clone();
        sg.sort();
        sg.dedup();
        sg.len()
    }
}

pub enum FissionResult {
    NoFission,
    FissionGroups(FnvHashMap<Point, usize>, Vec<usize>),
}

impl StaticKTAM {
    pub fn determine_fission<C: Canvas>(
        &self,
        canvas: &C,
        possible_start_points: &[Point],
        now_empty: &[Point],
    ) -> FissionResult {
        // Optimizations for a single empty site.
        if now_empty.len() == 1 {
            let p = now_empty[0];

            let tn = unsafe { canvas.uv_n(p) } as usize;
            let tw = unsafe { canvas.uv_w(p) } as usize;
            let te = unsafe { canvas.uv_e(p) } as usize;
            let ts = unsafe { canvas.uv_s(p) } as usize;
            let tnw = unsafe { canvas.uv_nw(p) as usize };
            let tne = unsafe { canvas.uv_ne(p) as usize };
            let tsw = unsafe { canvas.uv_sw(p) as usize };
            let tse = unsafe { canvas.uv_se(p) as usize };

            let ri: u8 = (((tn != 0) as u8) << 7)
                + ((((self.energy_we[(tn, tne)] != 0.) & (self.energy_ns[(tne, te)] != 0.)) as u8)
                    << 6)
                + (((te != 0) as u8) << 5)
                + ((((self.energy_ns[(te, tse)] != 0.) & (self.energy_we[(ts, tse)] != 0.)) as u8)
                    << 4)
                + (((ts != 0) as u8) << 3)
                + ((((self.energy_we[(tsw, ts)] != 0.) & (self.energy_ns[(tw, tsw)] != 0.)) as u8)
                    << 2)
                + (((tw != 0) as u8) << 1)
                + (((self.energy_ns[(tnw, tw)] != 0.) & (self.energy_we[(tnw, tn)] != 0.)) as u8);

            if CONNECTED_RING[ri as usize] {
                return FissionResult::NoFission
            }

            if ri == 0 {
                println!("Unattached tile detaching!");
                return FissionResult::NoFission
            }
        }

        let start_points:Vec<&(usize, usize)> = (*possible_start_points).iter().filter(|x| unsafe {canvas.uv_p(**x)} != 0 ).collect();

        // Not a single site...
        let mut groups = GroupVec {
            ident: (0usize..=start_points.len()).collect(),
        };
        let now_empty_group = 0usize;
        let mut groupmap = FnvHashMap::<Point, usize>::default();
        for point in now_empty {
            groupmap.insert(*point, now_empty_group);
        }

        let mut queue = VecDeque::<Point>::new();

        // Put all the starting points in the queue, with different group numbers.
        // Group numbers start from 1 because 0 is the group of now-emptied sites.
        for (i, point) in start_points.iter().enumerate() {
            queue.push_back(**point);
            groupmap.insert(**point, i + 1);
        }

        while let Some(p) = queue.pop_front() {
            let t = unsafe { canvas.uv_p(p) } as usize;
            let pn = canvas.move_point_n(p);
            let tn = unsafe { canvas.uv_p(pn) } as usize;
            let pw = canvas.move_point_w(p);
            let tw = unsafe { canvas.uv_p(pw) } as usize;
            let pe = canvas.move_point_e(p);
            let te = unsafe { canvas.uv_p(pe) } as usize;
            let ps = canvas.move_point_s(p);
            let ts = unsafe { canvas.uv_p(ps) } as usize;

            let pg = groupmap[&p]; // Must have a group, because we're here!

            if (unsafe { *self.energy_ns.uget((tn, t)) } != 0.) {
                match groupmap.get(&pn) {
                    None => {
                        groupmap.insert(pn, groups.ident[*groupmap.get(&p).unwrap()]);
                        queue.push_back(pn);
                    }
                    Some(ng) => {
                        if *ng != 0 {
                            groups.merge_groups(pg, *ng);
                        }
                    }
                }
            }

            if (unsafe { *self.energy_we.uget((t, te)) } != 0.) {
                match groupmap.get(&pe) {
                    None => {
                        groupmap.insert(pe, groups.ident[*groupmap.get(&p).unwrap()]);
                        queue.push_back(pe);
                    }
                    Some(ng) => {
                        if *ng != 0 {
                            groups.merge_groups(pg, *ng);
                        }
                    }
                }
            }

            if (unsafe { *self.energy_ns.uget((t, ts)) } != 0.) {
                match groupmap.get(&ps) {
                    None => {
                        groupmap.insert(ps, groups.ident[*groupmap.get(&p).unwrap()]);
                        queue.push_back(ps);
                    }
                    Some(ng) => {
                        if *ng != 0 {
                            groups.merge_groups(pg, *ng);
                        }
                    }
                }
            }

            if (unsafe { *self.energy_we.uget((tw, t)) } != 0.) {
                match groupmap.get(&pw) {
                    None => {
                        groupmap.insert(pw, groups.ident[*groupmap.get(&p).unwrap()]);
                        queue.push_back(pw);
                    }
                    Some(ng) => {
                        if *ng != 0 {
                            groups.merge_groups(pg, *ng);
                        }
                    }
                }
            }

            // We break on *two* groups, because group 0 is the removed area,
            // and so there will be two groups (0 and something) if there
            // is one contiguous area.
            if groups.n_groups() <= 2 {
                return FissionResult::NoFission;
            }
        }

        FissionResult::FissionGroups(groupmap, groups.ident)
    }
}
