extern crate ndarray;
use rgrow::{State2DQT, StaticKTAM, StateEvolve, StateCreate, StateStatus};
use ndarray::prelude::*;

fn main() {

    let size = 512usize;

    let gs = arr1(&[0.0, 2.0, 1.0, 1.0]);

    let tc = arr1(&[0.00000e+00, 1.12535e-07, 1.12535e-07, 1.12535e-07, 1.12535e-07,
        1.12535e-07, 1.12535e-07, 1.12535e-07]);

    let te = arr2(&[[0, 0, 0, 0], 
        [0, 1, 1, 0], 
        [0, 1, 3, 1], 
        [1, 3, 1, 0],
        [2, 2, 2, 2],
        [3, 3, 3, 2],
        [2, 3, 3, 3],
        [3, 2, 2, 3]]);

    let gse = 8.1;

    let mut canvas = Array2::<usize>::zeros((512, 512));

    let internal = arr2(&[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 3, 7, 5, 7, 5, 7, 5, 7, 5],
    [0, 3, 6, 7, 4, 5, 6, 7, 4, 5],
    [0, 3, 7, 4, 4, 5, 7, 4, 4, 5],
    [0, 3, 6, 6, 6, 7, 4, 4, 4, 5],
    [0, 3, 7, 5, 7, 4, 4, 4, 4, 5],
    [0, 3, 6, 7, 4, 4, 4, 4, 4, 5],
    [0, 3, 7, 4, 4, 4, 4, 4, 4, 5],
    [0, 3, 6, 6, 6, 6, 6, 6, 6, 7]]);


    canvas.slice_mut(s![0..10,0..10]).assign(&internal);

    let mut sys = StaticKTAM::new(tc, te, gs, gse);

    let mut state = State2DQT::create(&canvas, &sys);

    state.evolve_in_size_range(2, 100000, 50000000);

    println!("{} tiles, {} events", state.ntiles(), state.total_events())
}
