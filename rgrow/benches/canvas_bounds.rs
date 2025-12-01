use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::prelude::*;
use rgrow::{
    base::{GrowResult, NumTiles, Point, Tile},
    canvas::{Canvas, CanvasCreate, PointSafe2, PointSafeHere},
};
use std::hint::black_box;
use std::time::Duration;

// Test canvas implementation with bounds checking (no border)
#[derive(Clone, Debug)]
struct CanvasSquareBoundsCheck(Array2<Tile>);

unsafe impl Send for CanvasSquareBoundsCheck {}
unsafe impl Sync for CanvasSquareBoundsCheck {}

impl CanvasCreate for CanvasSquareBoundsCheck {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        Ok(Self(Array2::zeros(shape)))
    }

    fn from_array(arr: Array2<Tile>) -> GrowResult<Self> {
        Ok(Self(arr))
    }
}

impl Canvas for CanvasSquareBoundsCheck {
    #[inline(always)]
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.0.uget(p)
    }

    #[inline(always)]
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.0.uget_mut(p)
    }

    #[inline(always)]
    fn inbounds(&self, p: Point) -> bool {
        (p.0 < self.nrows()) & (p.1 < self.ncols())
    }

    #[inline(always)]
    fn u_move_point_n(&self, p: Point) -> Point {
        (p.0.wrapping_sub(1), p.1)
    }

    #[inline(always)]
    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0, p.1 + 1)
    }

    #[inline(always)]
    fn u_move_point_s(&self, p: Point) -> Point {
        (p.0 + 1, p.1)
    }

    #[inline(always)]
    fn u_move_point_w(&self, p: Point) -> Point {
        (p.0, p.1.wrapping_sub(1))
    }

    #[inline(always)]
    fn u_move_point_ne(&self, p: Point) -> Point {
        (p.0.wrapping_sub(1), p.1 + 1)
    }

    #[inline(always)]
    fn u_move_point_se(&self, p: Point) -> Point {
        (p.0 + 1, p.1 + 1)
    }

    #[inline(always)]
    fn u_move_point_sw(&self, p: Point) -> Point {
        (p.0 + 1, p.1.wrapping_sub(1))
    }

    #[inline(always)]
    fn u_move_point_nw(&self, p: Point) -> Point {
        (p.0.wrapping_sub(1), p.1.wrapping_sub(1))
    }

    #[inline(always)]
    fn u_move_point_nn(&self, p: Point) -> Point {
        (p.0.wrapping_sub(2), p.1)
    }

    #[inline(always)]
    fn u_move_point_ee(&self, p: Point) -> Point {
        (p.0, p.1 + 2)
    }

    #[inline(always)]
    fn u_move_point_ss(&self, p: Point) -> Point {
        (p.0 + 2, p.1)
    }

    #[inline(always)]
    fn u_move_point_ww(&self, p: Point) -> Point {
        (p.0, p.1.wrapping_sub(2))
    }

    #[inline(always)]
    fn calc_n_tiles(&self) -> NumTiles {
        self.0.fold(0, |x, y| x + u32::from(*y != 0))
    }

    fn raw_array(&self) -> ArrayView2<'_, Tile> {
        self.0.view()
    }

    fn raw_array_mut(&mut self) -> ArrayViewMut2<'_, Tile> {
        self.0.view_mut()
    }

    #[inline(always)]
    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    #[inline(always)]
    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.0
            .fold(0, |x, y| x + u32::from(should_be_counted[*y as usize]))
    }

    fn nrows_usable(&self) -> usize {
        self.0.nrows()
    }

    fn ncols_usable(&self) -> usize {
        self.0.ncols()
    }

    // Override tile_to_* methods to use bounds checking
    #[inline(always)]
    fn v_sh(&self, p: PointSafeHere) -> Tile {
        if self.inbounds(p.0) {
            unsafe { self.uv_p(p.0) }
        } else {
            0
        }
    }

    fn tile_to_n(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        if r > 0 {
            unsafe { self.uv_p((r - 1, c)) }
        } else {
            0
        }
    }
    fn tile_to_e(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        let ncols = self.ncols();
        if c < ncols - 1 {
            unsafe { self.uv_p((r, c + 1)) }
        } else {
            0
        }
    }
    fn tile_to_s(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        let nrows = self.nrows();
        if r < nrows - 1 {
            unsafe { self.uv_p((r + 1, c)) }
        } else {
            0
        }
    }
    fn tile_to_w(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        if c > 0 {
            unsafe { self.uv_p((r, c - 1)) }
        } else {
            0
        }
    }
    fn tile_to_ne(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        let ncols = self.ncols();
        if r > 0 && c < ncols - 1 {
            unsafe { self.uv_p((r - 1, c + 1)) }
        } else {
            0
        }
    }

    fn tile_to_se(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        let nrows = self.nrows();
        let ncols = self.ncols();
        if r < nrows - 1 && c < ncols - 1 {
            unsafe { self.uv_p((r + 1, c + 1)) }
        } else {
            0
        }
    }

    fn tile_to_sw(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        let nrows = self.nrows();
        if r < nrows - 1 && c > 0 {
            unsafe { self.uv_p((r + 1, c - 1)) }
        } else {
            0
        }
    }

    fn tile_to_nw(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        if r > 0 && c > 0 {
            unsafe { self.uv_p((r - 1, c - 1)) }
        } else {
            0
        }
    }

    fn tile_to_nn(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        if r > 1 {
            unsafe { self.uv_p((r - 2, c)) }
        } else {
            0
        }
    }

    fn tile_to_ee(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        let ncols = self.ncols();
        if c < ncols - 2 {
            unsafe { self.uv_p((r, c + 2)) }
        } else {
            0
        }
    }

    fn tile_to_ss(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        let nrows = self.nrows();
        if r < nrows - 2 {
            unsafe { self.uv_p((r + 2, c)) }
        } else {
            0
        }
    }

    fn tile_to_ww(&self, p: PointSafe2) -> Tile {
        let (r, c) = p.0;
        if c > 1 {
            unsafe { self.uv_p((r, c - 2)) }
        } else {
            0
        }
    }
}

fn setup_canvas_with_border(size: usize) -> rgrow::canvas::CanvasSquare {
    let mut canvas = rgrow::canvas::CanvasSquare::new_sized((size + 4, size + 4)).unwrap();
    // Fill with some test data
    for r in 2..(size + 2) {
        for c in 2..(size + 2) {
            let tile = ((r + c) % 10 + 1) as Tile;
            canvas.set_sa(&PointSafe2((r, c)), &tile);
        }
    }
    canvas
}

fn setup_canvas_with_bounds_check(size: usize) -> CanvasSquareBoundsCheck {
    let mut canvas = CanvasSquareBoundsCheck::new_sized((size, size)).unwrap();
    // Fill with same test data pattern
    for r in 0..size {
        for c in 0..size {
            let tile = ((r + c) % 10 + 1) as Tile;
            canvas.set_sa(&PointSafe2((r, c)), &tile);
        }
    }
    canvas
}

fn bench_neighbor_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_access");
    group.measurement_time(Duration::from_secs(5));

    for size in [100, 500] {
        let canvas_border = setup_canvas_with_border(size);
        let canvas_bounds = setup_canvas_with_bounds_check(size);

        // Test tile_to_n
        group.bench_with_input(
            BenchmarkId::new("tile_to_n_border", size),
            &(&canvas_border, size),
            |b, (canvas, size)| {
                let usable_size = size;
                b.iter(|| {
                    let mut sum = 0u32;
                    for r in 2..(usable_size + 2) {
                        for c in 2..(usable_size + 2) {
                            sum = sum.wrapping_add(
                                black_box(canvas.tile_to_n(PointSafe2((r, c)))) as u32
                            );
                        }
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tile_to_n_bounds_check", size),
            &(&canvas_bounds, size),
            |b, (canvas, size)| {
                b.iter(|| {
                    let mut sum = 0u32;
                    for r in 0..*size {
                        for c in 0..*size {
                            sum = sum.wrapping_add(
                                black_box(canvas.tile_to_n(PointSafe2((r, c)))) as u32
                            );
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Repeat to S (uses nrows)

        // Test tile_to_n
        group.bench_with_input(
            BenchmarkId::new("tile_to_s_border", size),
            &(&canvas_border, size),
            |b, (canvas, size)| {
                let usable_size = size;
                b.iter(|| {
                    let mut sum = 0u32;
                    for r in 2..(usable_size + 2) {
                        for c in 2..(usable_size + 2) {
                            sum = sum.wrapping_add(
                                black_box(canvas.tile_to_s(PointSafe2((r, c)))) as u32
                            );
                        }
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tile_to_s_bounds_check", size),
            &(&canvas_bounds, size),
            |b, (canvas, size)| {
                b.iter(|| {
                    let mut sum = 0u32;
                    for r in 0..*size {
                        for c in 0..*size {
                            sum = sum.wrapping_add(
                                black_box(canvas.tile_to_s(PointSafe2((r, c)))) as u32
                            );
                        }
                    }
                    black_box(sum)
                })
            },
        );

        // Test all four directions
        group.bench_with_input(
            BenchmarkId::new("tile_to_all_dirs_border", size),
            &(&canvas_border, size),
            |b, (canvas, size)| {
                let usable_size = size;
                b.iter(|| {
                    let mut sum = 0u32;
                    for r in 2..(usable_size + 2) {
                        for c in 2..(usable_size + 2) {
                            let p = PointSafe2((r, c));
                            sum = sum.wrapping_add(black_box(canvas.tile_to_n(p)) as u32);
                            sum = sum.wrapping_add(black_box(canvas.tile_to_e(p)) as u32);
                            sum = sum.wrapping_add(black_box(canvas.tile_to_s(p)) as u32);
                            sum = sum.wrapping_add(black_box(canvas.tile_to_w(p)) as u32);
                        }
                    }
                    black_box(sum)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tile_to_all_dirs_bounds_check", size),
            &(&canvas_bounds, size),
            |b, (canvas, size)| {
                b.iter(|| {
                    let mut sum = 0u32;
                    for r in 0..*size {
                        for c in 0..*size {
                            let p = PointSafe2((r, c));
                            sum = sum.wrapping_add(black_box(canvas.tile_to_n(p)) as u32);
                            sum = sum.wrapping_add(black_box(canvas.tile_to_e(p)) as u32);
                            sum = sum.wrapping_add(black_box(canvas.tile_to_s(p)) as u32);
                            sum = sum.wrapping_add(black_box(canvas.tile_to_w(p)) as u32);
                        }
                    }
                    black_box(sum)
                })
            },
        );
    }

    group.finish();
}

// fn bench_energy_calculation(c: &mut Criterion) {
//     let mut group = c.benchmark_group("energy_calculation");
//     group.measurement_time(Duration::from_secs(5));

//     // Create a simple KTAM system for energy calculations
//     let mut sys = rgrow::models::ktam::KTAM::new_sized(1, 1);
//     sys.tile_edges = array![[0, 0, 0, 0], [1, 1, 1, 1]];
//     sys.glue_strengths = array![0.0, 1.0];
//     sys.update_system();

//     for size in [100, 500] {
//         // Create states with proper canvas setup
//         use rgrow::state::StateWithCreate;
//         let mut state_border = <rgrow::state::QuadTreeState<
//             rgrow::canvas::CanvasSquare,
//             rgrow::state::NullStateTracker,
//         > as StateWithCreate>::empty_with_types((size + 4, size + 4), 10)
//         .unwrap();

//         let mut state_bounds = <rgrow::state::QuadTreeState<
//             CanvasSquareBoundsCheck,
//             rgrow::state::NullStateTracker,
//         > as StateWithCreate>::empty_with_types((size, size), 10)
//         .unwrap();

//         // Fill with test data
//         for r in 2..(size + 2) {
//             for c in 2..(size + 2) {
//                 let tile = ((r + c) % 10 + 1) as Tile;
//                 state_border.set_sa(&PointSafe2((r, c)), &tile);
//             }
//         }
//         for r in 0..size {
//             for c in 0..size {
//                 let tile = ((r + c) % 10 + 1) as Tile;
//                 state_bounds.set_sa(&PointSafe2((r, c)), &tile);
//             }
//         }

//         let sys_ref = &sys;
//         let state_border_ref = &state_border;
//         let state_bounds_ref = &state_bounds;

//         group.bench_with_input(
//             BenchmarkId::new("bond_energy_border", size),
//             &size,
//             |b, &size| {
//                 b.iter(|| {
//                     let mut sum = 0.0;
//                     for r in 2..(size + 2) {
//                         for c in 2..(size + 2) {
//                             let p = PointSafe2((r, c));
//                             let tile = state_border_ref.tile_at_point(p);
//                             if tile != 0 {
//                                 sum += black_box(
//                                     sys_ref.bond_energy_of_tile_type_at_point(state_border_ref, p, tile),
//                                 );
//                             }
//                         }
//                     }
//                     black_box(sum)
//                 })
//             },
//         );

//         group.bench_with_input(
//             BenchmarkId::new("bond_energy_bounds_check", size),
//             &size,
//             |b, &size| {
//                 b.iter(|| {
//                     let mut sum = 0.0;
//                     for r in 0..size {
//                         for c in 0..size {
//                             let p = PointSafe2((r, c));
//                             let tile = state_bounds_ref.tile_at_point(p);
//                             if tile != 0 {
//                                 sum += black_box(
//                                     sys_ref.bond_energy_of_tile_type_at_point(state_bounds_ref, p, tile),
//                                 );
//                             }
//                         }
//                     }
//                     black_box(sum)
//                 })
//             },
//         );
//     }

//     group.finish();
// }

criterion_group!(benches, bench_neighbor_access);
criterion_main!(benches);
