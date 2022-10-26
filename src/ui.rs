use std::time::Duration;

use fltk::{app, prelude::*, window::Window};

use pixels::{Pixels, SurfaceTexture};

use crate::base::RgrowError;
use crate::simulation::Simulation;
use crate::system::EvolveBounds;
thread_local! {
    pub static APP: fltk::app::App = app::App::default()
}

pub fn run_window(parsed: &crate::tileset::TileSet) -> Result<Box<dyn Simulation>, RgrowError> {
    let mut sim = parsed.into_simulation()?;

    let size1: usize;
    let size2: usize;

    match parsed.options.size {
        crate::tileset::Size::Single(x) => {
            size1 = x;
            size2 = x;
        }
        crate::tileset::Size::Pair((x, y)) => {
            size1 = x;
            size2 = y;
        }
    }

    let state_i = sim.add_state((size1, size2))?;
    let mut state = sim.state_ref(state_i);

    let scale = match parsed.options.block {
        Some(i) => i,
        None => {
            let (w, h) = app::screen_size();
            (w / (size2 as f64)).min(h / (size1 as f64)).floor() as usize
        }
    };
    app::screen_size();

    //let app = app::App::default();
    let mut win = Window::default()
        .with_size(
            (scale * state.ncols()) as i32,
            ((scale * state.nrows()) + 30) as i32,
        )
        .with_label("rgrow!");

    win.make_resizable(true);

    // add a frame with a label at the bottom of the window
    let mut frame = fltk::frame::Frame::default()
        .with_size(win.pixel_w(), 30)
        .with_pos(0, win.pixel_h() - 30)
        .with_label("Hello");
    win.end();
    win.show();

    let mut win_width = win.pixel_w() as u32;
    let mut win_height = win.pixel_h() as u32;

    let surface_texture = SurfaceTexture::new(win_width, win_height - 30, &win);

    let (width, height) = sim.draw_size(0);

    let mut pixels = { Pixels::new(width, height, surface_texture)? };

    let bounds = EvolveBounds {
        events: None,
        time: None,
        size_min: None,
        size_max: None,
        wall_time: Some(Duration::from_millis(16)),
    };

    while app::wait() {
        // Check if window was resized
        if win.w() != win_width as i32 || win.h() != win_height as i32 {
            win_width = win.pixel_w() as u32;
            win_height = win.pixel_h() as u32;
            pixels.resize_surface(win_width, win_height - 30);
            frame.set_pos(0, (win_height - 30) as i32);
            frame.set_size(win_width as i32, 30);
        }

        sim.evolve(state_i, bounds)?;

        sim.draw(state_i, pixels.get_frame());
        pixels.render()?;
        state = sim.state_ref(state_i);

        // Update text with the simulation time, events, and tiles
        frame.set_label(&format!(
            "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}",
            state.time(),
            state.total_events(),
            state.ntiles()
        ));

        app::flush();
        app::awake();
    }
    Ok(sim)
}
