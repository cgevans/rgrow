
use std::{thread, sync::Arc, sync::Mutex, convert::TryInto};

use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::{Canvas, CanvasSquarable, CanvasSquare, NullStateTracker, QuadTreeState, StateStatus, StateStep, System, TileBondInfo, state::StateCreate};


trait Draw<C,S>
where
    S: System<C> + TileBondInfo,
    C: Canvas
{
    fn draw(&self, frame: &mut [u8], scaled: usize, system: &S);
}

impl<C,S> Draw<C,S> for QuadTreeState<C, S, NullStateTracker>
where
    S: System<C> + TileBondInfo,
    C: Canvas + CanvasSquarable
{
    fn draw(&self, frame: &mut [u8], scaled: usize, system: &S) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let x = i % (self.canvas.square_size()*scaled);
            let y = i / (self.canvas.square_size()*scaled);

            let tv = unsafe { self.canvas.uv_p((y/scaled, x/scaled)) };

            pixel.copy_from_slice(
                &(if tv > 0 {
                    system.tile_color(tv)
                } else {
                    [0, 0, 0, 0x00]
                }),
            );
        }
    }
}

pub fn run_ktam_window(parsed: crate::parser::TileSet) {
    let mut system = parsed.into_static_seeded_ktam_p();
    let mut state = QuadTreeState::<_, _, NullStateTracker>::default(
        (parsed.options.size, parsed.options.size),
        &mut system,
    );

    let scaled = parsed.options.block;

    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new((state.canvas.square_size()*scaled) as f64, (state.canvas.square_size()*scaled) as f64);
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
            (state.canvas.square_size()*scaled).try_into().unwrap(),
            (state.canvas.square_size()*scaled).try_into().unwrap(),
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
                state.take_step(&mut system).unwrap();
            
        }
        // match parsed.options.smax {
        //     Some(smax) => {if state.ntiles() > smax {break}}
        //     None => {}
        // };

            state.draw(pixels.get_frame(), scaled, &system);
        window.request_redraw();
    });



}