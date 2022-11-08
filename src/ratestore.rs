use ndarray::Array2;
use rand::thread_rng;
use rand::Rng;

use crate::base::{Point, Rate};
use crate::canvas::PointSafeHere;
// A RateStore stores event rates for points on a canvas, and allows a continuous-time Markov chain
// choice of a point based on those rates.  It makes no assumptions about relationships between the
// points, beyond the points being defined by two integer coordinates; eg, they do not need to be a
// rectilinear grid.
pub trait RateStore {
    fn choose_point(&self) -> (Point, Rate);
    fn update_point(&mut self, point: Point, new_rate: Rate);
    fn update_multiple(&mut self, to_update: &[(PointSafeHere, Rate)]);
    fn total_rate(&self) -> Rate;
}

pub trait CreateSizedRateStore {
    /// Create a RateStore capable of holding
    fn new_with_size(rows: usize, cols: usize) -> Self;
}

/// A RateStore for a 2D canvas, using a:
/// - A quadtree to store and choose rates.
/// - Square arrays in the quadtree.
/// - Linear rate storage.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct QuadTreeSquareArray<R>(pub Vec<Array2<R>>, pub R);

impl CreateSizedRateStore for QuadTreeSquareArray<Rate> {
    fn new_with_size(rows: usize, cols: usize) -> Self {
        let p = f64::log2(rows.max(cols) as f64).ceil() as u32;

        let mut rates = Vec::<Array2<Rate>>::new();

        for i in (1..=p).rev() {
            rates.push(Array2::<Rate>::zeros((2usize.pow(i), 2usize.pow(i))))
        }

        Self(rates, 0.)
    }
}

impl RateStore for QuadTreeSquareArray<Rate> {
    fn choose_point(&self) -> (Point, Rate) {
        let mut threshold = self.1 * thread_rng().gen::<f64>();

        let mut x: usize = 0;
        let mut y: usize = 0;

        for r in self.0.iter().rev() {
            y *= 2;
            x *= 2;
            let mut v = unsafe { *r.uget((y, x)) };
            if threshold - v <= 0. {
                continue;
            } else {
                threshold -= v;
                x += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= 0. {
                continue;
            } else {
                threshold -= v;
                x -= 1;
                y += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= 0. {
                continue;
            } else {
                threshold -= v;
                x += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= 0. {
                continue;
            } else {
                panic!("Failure in quadtree position finding: remaining threshold {threshold:?}, ratetree array {r:?}.");
            }
        }

        ((y, x), threshold)
    }

    #[inline(always)]
    fn update_point(&mut self, mut point: Point, new_rate: Rate) {
        let mut rtiter = self.0.iter_mut();
        let mut r_prev = rtiter.next().unwrap();

        r_prev[point] = new_rate;

        for r_next in rtiter {
            point = (point.0 / 2, point.1 / 2);
            qt_update_level(r_next, r_prev, point);
            r_prev = r_next;
        }
        self.1 = r_prev.sum();
    }

    #[inline(always)]
    fn update_multiple(&mut self, to_update: &[(PointSafeHere, Rate)]) {
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

    fn total_rate(&self) -> Rate {
        self.1
    }
}

#[inline(always)]
fn qt_update_level(rn: &mut Array2<Rate>, rt: &Array2<Rate>, np: Point) {
    qt_update_level_val(unsafe { rn.uget_mut(np) }, rt, np);
}

#[inline(always)]
fn qt_update_level_val(rn: &mut f64, rt: &Array2<Rate>, np: Point) {
    let ip = (np.0 * 2, np.1 * 2);

    unsafe {
        *rn = *rt.uget(ip)
            + *rt.uget((ip.0, ip.1 + 1))
            + *rt.uget((ip.0 + 1, ip.1))
            + *rt.uget((ip.0 + 1, ip.1 + 1));
    }
}
