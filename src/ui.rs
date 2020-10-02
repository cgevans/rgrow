
use std::{convert::TryInto};

use ndarray::Array2;
use rand::SeedableRng;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::{canvas::Canvas, state::NullStateTracker, state::QuadTreeState, state::State, system::System, system::TileBondInfo};

use crate::state::{StateCreate};

trait Draw<S: State>
{
    fn draw(&self, state: &S, frame: &mut [u8], scaled: usize);
}

impl<S: State, Sy: System<S> + TileBondInfo> Draw<S> for Sy
{
    fn draw(&self, state: &S, frame: &mut [u8], scaled: usize) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let x = i % (state.nrows()*scaled);
            let y = i / (state.ncols()*scaled);

            let tv = unsafe { state.uv_p((y/scaled, x/scaled)) };

            pixel.copy_from_slice(
                &(if tv > 0 {
                    self.tile_color(tv)
                } else {
                    [0, 0, 0, 0x00]
                }),
            );
        }
    }
}

pub fn run_ktam_window(parsed: crate::parser::TileSet) {
    let mut system = parsed.into_static_seeded_ktam_p();
    let mut state = QuadTreeState::<_, NullStateTracker>::create_raw(
        Array2::zeros((parsed.options.size, parsed.options.size)),
    ).unwrap();

    for (p, t) in system.seed_locs() {
        // FIXME: for large seeds,
        // this could be faster by doing raw writes, then update_entire_state
        // but we would need to distinguish sizing.
        // Or maybe there is fancier way with a set?
        system.set_point(&mut state, p.0, t);
    }


    let scaled = parsed.options.block;

    let mut rng = rand::rngs::SmallRng::from_entropy();

    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new((state.ncols()*scaled) as f64, (state.nrows()*scaled) as f64);
        WindowBuilder::new()
            .with_title("rgrow!")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, 
            window_size.height, &window);
        Pixels::new(
            (state.ncols()*scaled).try_into().unwrap(),
            (state.nrows()*scaled).try_into().unwrap(),
            surface_texture,
        )
        .unwrap()
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

        for _ in 0..parsed.options.update_rate {
                system.state_step(&mut state, &mut rng, 1e20);
            
        }
        // match parsed.options.smax {
        //     Some(smax) => {if state.ntiles() > smax {break}}
        //     None => {}
        // };

            system.draw(&state, pixels.get_frame(), scaled);
        window.request_redraw();
    });



}