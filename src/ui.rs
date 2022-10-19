use std::convert::TryInto;

use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::simulation::EvolveBounds;

pub fn run_ktam_window(parsed: crate::parser::TileSet) {
    let mut sim = parsed.into_sim();

    let size1: usize;
    let size2: usize;

    match parsed.options.size {
        crate::parser::Size::Single(x) => {
            size1 = x;
            size2 = x;
        }
        crate::parser::Size::Pair((x, y)) => {
            size1 = x;
            size2 = y;
        }
    }

    let state_i = sim.add_state((size1, size2)).unwrap();

    let scaled = parsed.options.block;

    let state = sim.state_ref(state_i);

    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(
            (state.ncols() * scaled) as f64,
            (state.nrows() * scaled) as f64,
        );
        WindowBuilder::new()
            .with_title("rgrow!")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(
            (state.ncols() * scaled).try_into().unwrap(),
            (state.nrows() * scaled).try_into().unwrap(),
            surface_texture,
        )
        .unwrap()
    };

    let bounds = EvolveBounds {
        events: Some(parsed.options.update_rate),
        time: Some(1e20),
        size_min: Some(0),
        size_max: None,
    };

    event_loop.run(move |event, _, control_flow| {
        if let Event::RedrawRequested(_) = event {
            pixels.render().unwrap();
        }

        if input.update(&event) {
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                pixels.resize(size.width, size.height);
            }

            // Update internal state and request a redraw
        }

        sim.evolve(state_i, bounds);
        // match parsed.options.smax {
        //     Some(smax) => {if state.ntiles() > smax {break}}
        //     None => {}
        // };

        sim.draw(state_i, pixels.get_frame(), scaled);
        window.request_redraw();
    });
}
