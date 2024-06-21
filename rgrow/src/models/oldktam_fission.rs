use super::fission_base::*;
use crate::state::State;
use crate::{canvas::PointSafe2, models::oldktam::OldKTAM};
use std::collections::VecDeque;

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

        let start_points: Vec<_> = (*possible_start_points)
            .iter()
            .filter(|x| canvas.tile_at_point(**x) != 0)
            .collect();

        let mut groupinfo = GroupInfo::new(&start_points, now_empty);

        let mut queue = VecDeque::new();

        // Put all the starting points in the queue.
        for &&point in start_points.iter() {
            queue.push_back(point);
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
