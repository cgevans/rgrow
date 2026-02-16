use std::borrow::BorrowMut;

use crate::state::StateEnum;
use enum_dispatch::enum_dispatch;
use fnv::FnvHashSet;
use ndarray::Array2;
use ndarray::ArrayView2;
use num_traits::Zero;
use rand::rng;
use rand::Rng;
use serde::Deserialize;
use serde::Serialize;

use crate::base::Point;

use crate::canvas::PointSafeHere;
use crate::units::{PerSecond, Rate};

// A RateStore stores event rates for points on a canvas, and allows a continuous-time Markov chain
// choice of a point based on those rates.  It makes no assumptions about relationships between the
// points, beyond the points being defined by two integer coordinates; eg, they do not need to be a
// rectilinear grid.
#[enum_dispatch]
pub trait RateStore {
    fn choose_point(&self) -> (Point, PerSecond);
    fn rate_at_point(&self, point: PointSafeHere) -> PerSecond;
    fn update_point(&mut self, point: PointSafeHere, new_rate: PerSecond);
    fn update_multiple(&mut self, to_update: &[(PointSafeHere, PerSecond)]);
    fn total_rate(&self) -> PerSecond;
    fn rate_array(&'_ self) -> ArrayView2<'_, PerSecond>;
}

pub trait CreateSizedRateStore {
    /// Create a RateStore capable of holding
    fn new_with_size(rows: usize, cols: usize) -> Self;
}

/// A RateStore for a 2D canvas, using a:
/// - A quadtree to store and choose rates.
/// - Square arrays in the quadtree.
/// - Linear rate storage.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct QuadTreeSquareArray<R: Rate>(pub Vec<Array2<R>>, pub R);

impl<R: Rate> CreateSizedRateStore for QuadTreeSquareArray<R> {
    fn new_with_size(rows: usize, cols: usize) -> Self {
        let p = f64::log2(rows.max(cols) as f64).ceil() as u32;

        let mut rates = Vec::<Array2<R>>::new();

        for i in (1..=p).rev() {
            rates.push(Array2::<R>::zeros((2usize.pow(i), 2usize.pow(i))))
        }

        Self(rates, R::zero())
    }
}

impl RateStore for QuadTreeSquareArray<PerSecond> {
    fn rate_at_point(&self, point: PointSafeHere) -> PerSecond {
        unsafe { *self.0[0].uget(point.0) }
    }

    fn choose_point(&self) -> (Point, PerSecond) {
        let mut threshold = self.1 * rng().random::<f64>();

        let mut x: usize = 0;
        let mut y: usize = 0;

        for r in self.0.iter().rev() {
            y *= 2;
            x *= 2;
            let mut v = unsafe { *r.uget((y, x)) };
            if threshold - v <= PerSecond::zero() {
                continue;
            } else {
                threshold -= v;
                x += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= PerSecond::zero() {
                continue;
            } else {
                threshold -= v;
                x -= 1;
                y += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= PerSecond::zero() {
                continue;
            } else {
                threshold -= v;
                x += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= PerSecond::zero() {
                continue;
            } else {
                panic!("Failure in quadtree position finding: remaining threshold {threshold:?}, ratetree array {r:?}.");
            }
        }

        ((y, x), threshold)
    }

    #[inline(always)]
    fn update_point(&mut self, point: PointSafeHere, new_rate: PerSecond) {
        let mut rtiter = self.0.iter_mut();
        let mut r_prev = rtiter.next().unwrap();

        let mut point = point.0;

        unsafe {
            *r_prev.uget_mut(point) = new_rate;
        }

        for r_next in rtiter {
            point = (point.0 / 2, point.1 / 2);
            qt_update_level(r_next, r_prev, point);
            r_prev = r_next;
        }
        self.1 = r_prev.sum();
    }

    #[inline(always)]
    fn update_multiple(&mut self, to_update: &[(PointSafeHere, PerSecond)]) {
        // Two code paths here: one for small N, using a sorted Vec,
        // and one for large N, using an FnvHashSet.

        if to_update.len() < 512 {
            self._update_multiple_small(to_update);
        } else if to_update.len() < self.0[0].len() / 16 {
            self._update_multiple_large(to_update);
        } else {
            self._update_multiple_all(to_update);
        }
    }

    fn total_rate(&self) -> PerSecond {
        self.1
    }

    fn rate_array(&self) -> ArrayView2<'_, PerSecond> {
        self.0.first().unwrap().view()
    }
}

impl<R: Rate> QuadTreeSquareArray<R> {
    pub fn _update_multiple_small(&mut self, to_update: &[(PointSafeHere, R)]) {
        let mut todo = Vec::<Point>::new();

        let mut rtiter = self.0.iter_mut();
        let mut r_prev = rtiter.next().unwrap();

        for (p, r) in to_update {
            r_prev[p.0] = *r;
            let n = (p.0 .0 / 2, p.0 .1 / 2);
            if todo.iter().rev().all(|x| n != *x) {
                todo.push(n);
            }
        }

        for r_next in rtiter {
            for p in todo.iter_mut() {
                qt_update_level(r_next, r_prev, *p);
                *p = (p.0 / 2, p.1 / 2);
            }
            todo.sort_unstable();
            todo.dedup();
            r_prev = r_next;
        }

        self.1 = r_prev.sum();
    }

    pub fn _update_multiple_large(&mut self, to_update: &[(PointSafeHere, R)]) {
        let mut h1 = FnvHashSet::<Point>::default();
        let mut h2 = FnvHashSet::<Point>::default();

        let mut rtiter = self.0.iter_mut();
        let mut r_prev = rtiter.next().unwrap();

        let mut todo = h1.borrow_mut();
        for (p, r) in to_update {
            r_prev[p.0] = *r;
            let n = (p.0 .0 / 2, p.0 .1 / 2);
            todo.insert(n);
        }

        let mut next_todo = h2.borrow_mut();
        for r_next in rtiter {
            for p in todo.drain() {
                qt_update_level(r_next, r_prev, p);
                next_todo.insert((p.0 / 2, p.1 / 2));
            }
            r_prev = r_next;
            (todo, next_todo) = (next_todo, todo);
        }

        self.1 = r_prev.sum();
    }

    pub fn _update_multiple_all(&mut self, to_update: &[(PointSafeHere, R)]) {
        let mut rtiter = self.0.iter_mut();
        let mut r_prev = rtiter.next().unwrap();

        for (p, r) in to_update {
            r_prev[p.0] = *r;
        }

        for r_next in rtiter {
            for p in r_next.indexed_iter_mut() {
                qt_update_level_val(p.1, r_prev, p.0);
            }
            r_prev = r_next;
        }

        self.1 = r_prev.sum();
    }
}

#[inline(always)]
fn qt_update_level<R: Rate>(rn: &mut Array2<R>, rt: &Array2<R>, np: Point) {
    qt_update_level_val(unsafe { rn.uget_mut(np) }, rt, np);
}

#[inline(always)]
fn qt_update_level_val<R: Rate>(rn: &mut R, rt: &Array2<R>, np: Point) {
    let ip = (np.0 * 2, np.1 * 2);

    unsafe {
        *rn = *rt.uget(ip)
            + *rt.uget((ip.0, ip.1 + 1))
            + *rt.uget((ip.0 + 1, ip.1))
            + *rt.uget((ip.0 + 1, ip.1 + 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_ratestore_qsta_update() -> anyhow::Result<()> {
        // Create a new RateStore
        let mut rs: QuadTreeSquareArray<PerSecond> = QuadTreeSquareArray::new_with_size(128, 128);
        let mut rs_large = rs.clone();
        let mut rs_single = rs.clone();
        let mut rs_all = rs.clone();

        let rng = rand::rng();
        let it = rng
            .sample_iter(rand::distr::Uniform::new(0.0, 1.0).unwrap())
            .map(PerSecond::new);

        let allchanges = (0..128usize)
            .flat_map(|x| (0..128usize).map(move |y| (x, y)))
            .zip(it)
            .map(|((x, y), r)| (PointSafeHere((x, y)), r))
            .collect::<Vec<_>>();

        rs._update_multiple_small(&allchanges);
        rs_large._update_multiple_large(&allchanges);

        assert_eq!(rs, rs_large);

        for (p, r) in allchanges.iter() {
            rs_single.update_point(*p, *r);
        }

        assert_eq!(rs, rs_single);

        rs_all._update_multiple_all(&allchanges);

        assert_eq!(rs, rs_all);

        Ok(())
    }
}
