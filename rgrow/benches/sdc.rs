use std::collections::HashMap;

use rgrow::{canvas::CanvasSquare, models::sdc1d::{RefOrPair, SDCParams, SDC}, state::{NullStateTracker, QuadTreeState, StateStatus, StateWithCreate}, system::{DynSystem, EvolveBounds, NeededUpdate, System}};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn make_params(c: &mut Criterion) -> SDCParams {
    let mut scaffold = vec![None; 2];
    scaffold.extend((1..20).map(|x| Some(format!("s{}*", x))));
    scaffold.extend((0..2).map(|x| None));

    let mut glue_dg_s = HashMap::new();
    for s in 1..20 {
        glue_dg_s.insert(RefOrPair::Ref(format!("s{}", s)), (-27.3, -0.486));
    }

    for s in "abcdefghijklmnopqrst".chars() {
        glue_dg_s.insert(RefOrPair::Ref(format!("{}", s)), (-15.36, -0.2366));
    }

    for s in "jklmnopqrst".chars() {
        glue_dg_s.insert(RefOrPair::Ref(format!("e{}", s)), (-15.36, -0.2366));
    }

    let params = SDCParams {
        tile_glues: [
            ["a", "s1", "b"],
            ["b*", "s2", "c"],
            ["c*", "s3", "d"],
            ["d*", "s4", "e"],
            ["e*", "s5", "f"],
            ["f*", "s6", "g"],
            ["g*", "s7", "h"],
            ["h*", "s8", "i"],
            ["i*", "s9", "j"],
            ["j*", "s10", "k"],
            ["k*", "s11", "l"],
            ["l*", "s12", "m"],
            ["m*", "s13", "n"],
            ["n*", "s14", "o"],
            ["o*", "s15", "p"],
            ["p*", "s16", "q"],
            ["q*", "s17", "r"],
            ["r*", "s18", "s"],
            ["s*", "s19", "t"],
            ["ej*", "s10", "ek"],
            ["ek*", "s11", "el"],
            ["el*", "s12", "em"],
            ["em*", "s13", "en"],
            ["en*", "s14", "eo"],
            ["eo*", "s15", "ep"],
            ["ep*", "s16", "eq"],
            ["eq*", "s17", "er"],
            ["er*", "s18", "es"],
            ["es*", "s19", "et"],
        ].map(|x| x.map(|y| Some(y.to_string())).to_vec()).to_vec(),
        tile_concentration: vec![1000.0e-9; 30],
        tile_names: (1..30).map(|x| Some(format!("tile{}", x))).collect(),
        tile_colors: (1..30).map(|x| None).collect(),
        scaffold: rgrow::models::sdc1d::SingleOrMultiScaffold::Single(scaffold),
        glue_dg_s,
        k_f: 1e6,
        k_n: 1e5,
        k_c: 1e4,
        temperature: 70.0,
    };

    params
}

fn bench_hold(c: &mut Criterion) {
    let params = make_params(c);
    let mut sdc = SDC::from_params(params);

    let mut state = QuadTreeState::<CanvasSquare, NullStateTracker>::empty((64, 24)).unwrap();
    let bounds = EvolveBounds::default().for_events(100);
    
    System::update_all(&mut sdc, &mut state, &NeededUpdate::All);

    // c.bench_function("evolve100_sdc_at_temp", |b| b.iter(|| {
    //     System::evolve(&sdc, &mut state, bounds).unwrap();
    //     assert!(state.n_tiles() > 0);
    // }));

    c.bench_function("unistep_sdc_at_temp", |b| b.iter(|| {
        System::take_single_step(&sdc, &mut state, 1e6);
    }));


    c.bench_function("update_sdc_temps", |b| b.iter(|| {
        sdc.change_temperature_to(50.0);
        System::update_all(&mut sdc, &mut state, &NeededUpdate::NonZero);
    }));

}

criterion_group!(benches, bench_hold);
criterion_main!(benches);
