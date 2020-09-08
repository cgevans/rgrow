extern crate ndarray;
use ndarray::prelude::*;
use num_format::{Locale, ToFormattedString};
use rgrow::{
    ffs, NullStateTracker, State2DQT, StateCreate, StateEvolve, StateStatus, StateTracked,
    StaticKTAM, TileSubsetTracker, Tile, Glue, System, CanvasSquare, Canvas, CanvasSize
};
use std::time::Instant;

use clap::Clap;

use rgrow::{parser::TileSet, StateStep};

use serde_yaml;
use std::fs::File;
use std::io::prelude::*;

use std::sync::{Arc, Mutex};

use std::{convert::TryInto, thread};

use rgrow::ffstest::ffstest;

#[derive(Clap)]
#[clap(version = "0.1.0", author = "Constantine Evans <cevans@costinet.org")]
struct Opts {
    #[clap(subcommand)]
    subcmd: SubCommand,
}

#[derive(Clap)]
enum SubCommand {
    Run(EO),
    RunSubs(EO),
    Parse(PO),
    RunAtam(PO),
    RunKtamWindow(PO),
    FfsTest(EO),
    NucRate(PO)}

#[derive(Clap)]
struct EO {}

#[derive(Clap)]
struct PO {
    input: String,
}

fn main() {
    let opts = Opts::parse();

    match opts.subcmd {
        SubCommand::Run(_) => run_example(),
        SubCommand::RunSubs(_) => run_example_subs(),
        SubCommand::Parse(po) => parse_example(po.input),
        SubCommand::RunAtam(po) => run_atam(po.input),
        SubCommand::RunKtamWindow(po) => run_ktam_window(po.input),
        SubCommand::FfsTest(_eo) => ffstest(),
        SubCommand::NucRate(po) => nucrate(po.input),
    }
}

fn nucrate(input: String) {
    let parsed: TileSet =
        serde_yaml::from_reader(File::open(input).expect("Input file not found."))
            .expect("Input file parse erorr.");

    let system = parsed.into_static_seeded_ktam();

    let ffsrun = ffs::FFSRun::create(&system, 1000, 30, parsed.options.size, 1_000, 50_000, 4, 2);

    println!("Nuc rate: {:?}", ffsrun.nucleation_rate());
    println!("Forwards: {:?}", ffsrun.forward_vec());
}

fn run_atam(input: String) {
    let file = File::open(input).unwrap();
    let parsed: TileSet = serde_yaml::from_reader(file).unwrap();

    let mut system = parsed.into_static_seeded_atam();
    let mut state = State2DQT::<_, NullStateTracker>::default(
        (parsed.options.size, parsed.options.size),
        &mut system,
    );

    println!("{:?}", state.canvas.canvas);
    println!("{:?}", state.rates[2]);

    //state.evolve_in_size_range(&mut system, 0, parsed.options.smax.unwrap(), 1_000_000);

    loop {
        state.take_step(&mut system).unwrap();
        println!("{:?}", state.canvas.canvas);
    }
}

fn parse_example(filename: String) {
    let file = File::open(filename).unwrap();

    let parsed: TileSet = serde_yaml::from_reader(file).unwrap();

    println!("{:?}", parsed);

    let (gm, ng) = parsed.number_glues().unwrap();

    let te = parsed.tile_edge_process(&gm);

    println!("{:?} {:?} {:?}", gm, ng, te);

    println!("{:?}", parsed.into_static_seeded_atam());
}

fn run_example() {
    let gs = arr1(&[0.0, 2.0, 1.0, 1.0]);

    let tc = arr1(&[0.00000e+00, 1., 1., 1., 1., 1., 1., 1.]);

    let te = arr2(&[
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 3, 1],
        [1, 3, 1, 0],
        [2, 2, 2, 2],
        [3, 3, 3, 2],
        [2, 3, 3, 3],
        [3, 2, 2, 3],
    ]);

    let gse = 8.1;

    let mut canvas = Array2::<Tile>::zeros((512, 512));

    let internal = arr2(&[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 3, 7, 5, 7, 5, 7, 5, 7, 5],
        [0, 3, 6, 7, 4, 5, 6, 7, 4, 5],
        [0, 3, 7, 4, 4, 5, 7, 4, 4, 5],
        [0, 3, 6, 6, 6, 7, 4, 4, 4, 5],
        [0, 3, 7, 5, 7, 4, 4, 4, 4, 5],
        [0, 3, 6, 7, 4, 4, 4, 4, 4, 5],
        [0, 3, 7, 4, 4, 4, 4, 4, 4, 5],
        [0, 3, 6, 6, 6, 6, 6, 6, 6, 7],
    ]);

    canvas.slice_mut(s![0..10, 0..10]).assign(&internal);

    let mut sys = StaticKTAM::from_ktam(tc, te, gs, gse, 16., None, None, None);

    let mut state = State2DQT::<_, NullStateTracker>::from_canvas(&mut sys, canvas);

    let now = Instant::now();

    state.evolve_in_size_range_emax_cond(&mut sys, 2, 100000, 50_000_000);

    let el = now.elapsed().as_secs_f64();

    let evps = ((state.total_events() as f64 / el).round() as u64).to_formatted_string(&Locale::en);

    let ev = state.total_events().to_formatted_string(&Locale::en);

    let nt = state.ntiles().to_formatted_string(&Locale::en);

    println!("{} tiles, {} events, {} secs, {} ev/sec", nt, ev, el, evps);
}

fn run_example_subs() {
    let gs = arr1(&[0.0, 2.0, 1.0, 1.0]);

    let tc = arr1(&[0.00000e+00, 1., 1., 1., 1., 1., 1., 1.]);

    let te = arr2(&[
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 3, 1],
        [1, 3, 1, 0],
        [2, 2, 2, 2],
        [3, 3, 3, 2],
        [2, 3, 3, 3],
        [3, 2, 2, 3],
    ]);

    let gse = 8.1;

    let mut canvas = Array2::<Tile>::zeros((512, 512));

    let internal = arr2(&[
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 3, 7, 5, 7, 5, 7, 5, 7, 5],
        [0, 3, 6, 7, 4, 5, 6, 7, 4, 5],
        [0, 3, 7, 4, 4, 5, 7, 4, 4, 5],
        [0, 3, 6, 6, 6, 7, 4, 4, 4, 5],
        [0, 3, 7, 5, 7, 4, 4, 4, 4, 5],
        [0, 3, 6, 7, 4, 4, 4, 4, 4, 5],
        [0, 3, 7, 4, 4, 4, 4, 4, 4, 5],
        [0, 3, 6, 6, 6, 6, 6, 6, 6, 7],
    ]);

    canvas.slice_mut(s![0..10, 0..10]).assign(&internal);

    let mut sys = StaticKTAM::from_ktam(tc, te, gs, gse, 16.0, None, None, None);

    let mut state = State2DQT::<_, TileSubsetTracker>::from_canvas(&mut sys, canvas);

    let tracker = TileSubsetTracker::new(vec![2, 3]);

    state.set_tracker(tracker);

    let now = Instant::now();

    let condition = |s: &State2DQT<_, TileSubsetTracker>, _events| s.tracker.num_in_subset > 200;
    state.evolve_until_condition(&mut sys, &condition);

    let el = now.elapsed().as_secs_f64();

    let evps = ((state.total_events() as f64 / el).round() as u64).to_formatted_string(&Locale::en);

    let ev = state.total_events().to_formatted_string(&Locale::en);

    let nt = state.ntiles().to_formatted_string(&Locale::en);

    println!("{} tiles, {} events, {} secs, {} ev/sec", nt, ev, el, evps);
}


use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};

trait Draw<S>
where
    S: System<CanvasSquare>,
{
    fn draw(&self, frame: &mut [u8]);
}

impl<S> Draw<S> for State2DQT<S, NullStateTracker>
where
    S: System<CanvasSquare>,
{
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let x = i % self.canvas.size();
            let y = i / self.canvas.size();

            let tv = unsafe { self.canvas.uv_p((x, y)) };

            pixel.copy_from_slice(
                &(if tv > 0 {
                    [(20 * tv).try_into().unwrap(), 50, 50, 0xff]
                } else {
                    [0, 0, 0, 0xff]
                }),
            );
        }
    }
}

fn run_ktam_window(input: String) {
    let file = File::open(input).unwrap();
    let parsed: TileSet = serde_yaml::from_reader(file).unwrap();

    let mut system = parsed.into_static_seeded_ktam();
    let mut state = State2DQT::<_, NullStateTracker>::default(
        (parsed.options.size, parsed.options.size),
        &mut system,
    );

    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(state.canvas.size() as f64, state.canvas.size() as f64);
        WindowBuilder::new()
            .with_title("rgrow!")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, 
            window_size.height, &window);
        Pixels::new(
            state.canvas.size().try_into().unwrap(),
            state.canvas.size().try_into().unwrap(),
            surface_texture,
        )
        .unwrap()
    };

    let proxy = event_loop.create_proxy();

    let warc = Arc::new(window);

    let ap = Arc::new(Mutex::new(pixels));
    let dp = Arc::clone(&ap);

    let _x = thread::spawn(move || {
        loop {
            for _ in 0..parsed.options.update_rate {
                state.take_step(&mut system);
            }
            if state.ntiles() > parsed.options.smax.unwrap_or(500).try_into().unwrap() {
                break
            }
            state.draw(ap.lock().unwrap().get_frame());
            proxy.send_event(warc.request_redraw()).unwrap();
        }
    });

    event_loop.run(move |event, _, control_flow| {
        if let Event::RedrawRequested(_) = event {
            dp.lock().unwrap().render().unwrap();
        }

        if input.update(&event) {
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                dp.lock().unwrap().resize(size.width, size.height);
            }

            // Update internal state and request a redraw            
        }
    });

}
