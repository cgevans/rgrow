extern crate ndarray;
use rgrow::{State2DQT, StaticKTAM, StateEvolve, StateCreate, StateStatus};
use ndarray::prelude::*;
use std::{time::{Instant}};
use num_format::{Locale, ToFormattedString};

fn main() {

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

    let sys = StaticKTAM::new(tc, te, gs, gse);

    let mut state = State2DQT::create(&canvas, &sys);

    let now = Instant::now();

    state.evolve_in_size_range(&sys, 2, 100000, 50000000);

    let el = now.elapsed().as_secs_f64();

    let evps = ((state.total_events() as f64 / el).round() as u64).to_formatted_string(&Locale::en);

    let ev = state.total_events().to_formatted_string(&Locale::en);

    let nt = state.ntiles().to_formatted_string(&Locale::en);

    println!("{} tiles, {} events, {} secs, {} ev/sec", nt, ev, el, evps);
}
