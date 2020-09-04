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

fn run_atam_window(input: String) {
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
