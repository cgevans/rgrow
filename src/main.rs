#![feature(stmt_expr_attributes)]
extern crate ndarray;

use clap::Clap;

use rgrow::{ffs, parser::TileSet, parser_xgrow};

use rgrow::{canvas, state};

use serde_yaml;
use std::fs::File;

#[cfg(feature = "ui")]
use rgrow::ui::run_ktam_window;

#[derive(Clap)]
#[clap(version = "0.1.0", author = "Constantine Evans <cevans@costinet.org")]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Clap)]
enum SubCommand {
    //Run(EO),
    //RunSubs(EO),
    //Parse(PO),
    //RunAtam(PO),
    //Run(PO),
    NucRate(PO),
    //RunXgrow(PO),
    //FissionTest(EO)
}

#[derive(Clap)]
struct EO {}

#[derive(Clap)]
struct PO {
    input: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::parse();

    match opts.subcmd {
        // SubCommand::Run(po) =>
        // #[cfg(feature = "ui")]
        // {
        //     let file = match File::open(po.input) {
        //         Ok(f) => f,
        //         Err(e) => return Err(Box::new(rgrow::parser::ParserError::Io { source: e })),
        //     };
        //     let parsed: TileSet = serde_yaml::from_reader(file)?;
        //     run_ktam_window(parsed)
        // }
        SubCommand::NucRate(po) => nucrate(po.input),
        // SubCommand::RunXgrow(po) =>
        // #[cfg(feature = "ui")]
        // {
        //     let parsed = parser_xgrow::parse_xgrow(po.input)?;
        //     run_ktam_window(parsed)
        // }
    };

    Ok(())
}

fn nucrate(input: String) {
    let parsed: TileSet =
        serde_yaml::from_reader(File::open(input).expect("Input file not found."))
            .expect("Input file parse erorr.");

    let system = parsed.into_static_seeded_ktam::<state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>>();

    let ffsrun = ffs::FFSRun::create(system, 1000, 30, parsed.options.size, 1_000, 50_000, 3, 2);

    println!("Nuc rate: {:e}", ffsrun.nucleation_rate());
    println!("Forwards: {:?}", ffsrun.forward_vec());
}

// fn run_example() {
//     let gs = arr1(&[0.0, 2.0, 1.0, 1.0]);

//     let tc = arr1(&[0.00000e+00, 1., 1., 1., 1., 1., 1., 1.]);

//     let te = arr2(&[
//         [0, 0, 0, 0],
//         [0, 1, 1, 0],
//         [0, 1, 3, 1],
//         [1, 3, 1, 0],
//         [2, 2, 2, 2],
//         [3, 3, 3, 2],
//         [2, 3, 3, 3],
//         [3, 2, 2, 3],
//     ]);

//     let gse = 8.1;

//     let mut canvas = Array2::<Tile>::zeros((512, 512));

//     let internal = arr2(&[
//         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
//         [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
//         [0, 3, 7, 5, 7, 5, 7, 5, 7, 5],
//         [0, 3, 6, 7, 4, 5, 6, 7, 4, 5],
//         [0, 3, 7, 4, 4, 5, 7, 4, 4, 5],
//         [0, 3, 6, 6, 6, 7, 4, 4, 4, 5],
//         [0, 3, 7, 5, 7, 4, 4, 4, 4, 5],
//         [0, 3, 6, 7, 4, 4, 4, 4, 4, 5],
//         [0, 3, 7, 4, 4, 4, 4, 4, 4, 5],
//         [0, 3, 6, 6, 6, 6, 6, 6, 6, 7],
//     ]);

//     canvas.slice_mut(s![0..10, 0..10]).assign(&internal);

//     let mut sys = StaticKTAM::from_ktam(tc, te, gs, gse, 16., None, None, None, Some(FissionHandling::KeepLargest), None, None, None, None);

//     let mut state = QuadTreeState::<CanvasSquare, _, NullStateTracker>::from_canvas(&mut sys, canvas);

//     let now = Instant::now();

//     state.evolve_in_size_range_emax_cond(&mut sys, 2, 100000, 50_000_000);

//     let el = now.elapsed().as_secs_f64();

//     let evps = ((state.total_events() as f64 / el).round() as u64).to_formatted_string(&Locale::en);

//     let ev = state.total_events().to_formatted_string(&Locale::en);

//     let nt = state.ntiles().to_formatted_string(&Locale::en);

//     println!("{} tiles, {} events, {} secs, {} ev/sec", nt, ev, el, evps);
// }

// fn fission_test() {
//     let gs = arr1(&[0.0, 2.0, 1.0, 1.0]);

//     let tc = arr1(&[0.00000e+00, 1., 1., 1., 1., 1., 1., 1.]);

//     let te = arr2(&[
//         [0, 0, 0, 0],
//         [0, 1, 1, 0],
//         [0, 1, 3, 1],
//         [1, 3, 1, 0],
//         [2, 2, 2, 2],
//         [3, 3, 3, 2],
//         [2, 3, 3, 3],
//         [3, 2, 2, 3],
//     ]);

//     let gse = 8.1;

//     let mut canvas = Array2::<Tile>::zeros((512, 512));

//     let internal = arr2(&[
//         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
//         [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
//         [0, 3, 0, 0, 0, 0, 0, 0, 7, 5],
//         [0, 3, 0, 7, 4, 5, 6, 0, 4, 5],
//         [0, 3, 0, 4, 4, 5, 7, 0, 4, 5],
//         [0, 3, 6, 6, 6, 7, 4, 0, 4, 5],
//         [0, 3, 0, 5, 7, 4, 4, 0, 4, 5],
//         [0, 3, 0, 7, 4, 4, 4, 0, 0, 0],
//         [0, 3, 7, 4, 4, 4, 4, 4, 4, 5],
//         [0, 3, 6, 6, 6, 6, 6, 6, 6, 7],
//     ]);

//     canvas.slice_mut(s![0..10, 0..10]).assign(&internal);

//     let mut sys = StaticKTAM::from_ktam(tc, te, gs, gse, 16., None, None, None, Some(FissionHandling::KeepLargest), None, None, None, None);

//     let mut state = QuadTreeState::<CanvasSquare, _, NullStateTracker>::from_canvas(&mut sys, canvas);

//     let x = sys.determine_fission(&state.canvas, &[(4, 1), (5, 2), (6, 1)], &[(5,1)]);

//     println!("{:?}", x);
// }

// fn run_example_subs() {
//     let gs = arr1(&[0.0, 2.0, 1.0, 1.0]);

//     let tc = arr1(&[0.00000e+00, 1., 1., 1., 1., 1., 1., 1.]);

//     let te = arr2(&[
//         [0, 0, 0, 0],
//         [0, 1, 1, 0],
//         [0, 1, 3, 1],
//         [1, 3, 1, 0],
//         [2, 2, 2, 2],
//         [3, 3, 3, 2],
//         [2, 3, 3, 3],
//         [3, 2, 2, 3],
//     ]);

//     let gse = 8.1;

//     let mut canvas = Array2::<Tile>::zeros((512, 512));

//     let internal = arr2(&[
//         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
//         [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
//         [0, 3, 7, 5, 7, 5, 7, 5, 7, 5],
//         [0, 3, 6, 7, 4, 5, 6, 7, 4, 5],
//         [0, 3, 7, 4, 4, 5, 7, 4, 4, 5],
//         [0, 3, 6, 6, 6, 7, 4, 4, 4, 5],
//         [0, 3, 7, 5, 7, 4, 4, 4, 4, 5],
//         [0, 3, 6, 7, 4, 4, 4, 4, 4, 5],
//         [0, 3, 7, 4, 4, 4, 4, 4, 4, 5],
//         [0, 3, 6, 6, 6, 6, 6, 6, 6, 7],
//     ]);

//     canvas.slice_mut(s![0..10, 0..10]).assign(&internal);

//     let mut sys = StaticKTAM::from_ktam(tc, te, gs, gse, 16.0, None, None, None, Some(FissionHandling::KeepLargest), None, None, None, None);

//     let mut state = QuadTreeState::<CanvasSquare, _, TileSubsetTracker>::from_canvas(&mut sys, canvas);

//     let tracker = TileSubsetTracker::new(vec![2, 3]);

//     state.set_tracker(tracker);

//     let now = Instant::now();

//     let condition = |s: &QuadTreeState<CanvasSquare, _, TileSubsetTracker>, _events| s.tracker.num_in_subset > 200;
//     state.evolve_until_condition(&mut sys, &condition);

//     let el = now.elapsed().as_secs_f64();

//     let evps = ((state.total_events() as f64 / el).round() as u64).to_formatted_string(&Locale::en);

//     let ev = state.total_events().to_formatted_string(&Locale::en);

//     let nt = state.ntiles().to_formatted_string(&Locale::en);

//     println!("{} tiles, {} events, {} secs, {} ev/sec", nt, ev, el, evps);
// }
