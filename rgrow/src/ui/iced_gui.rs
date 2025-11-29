use iced::widget::{column, text, Container};
use iced::{Alignment, Application, Command, Element, Length, Settings, Theme};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;

use crate::ui::ipc::{InitMessage, IpcMessage};

#[cfg(feature = "ui")]
use image;

pub struct RgrowGui {
    current_image: Option<(Vec<u8>, u32, u32)>,
    stats_text: String,
    width: u32,
    height: u32,
    scale: usize,
    receiver: Arc<Mutex<mpsc::Receiver<IpcMessage>>>,
}

#[derive(Debug, Clone)]
pub enum Message {
    IpcMessage(IpcMessage),
    WindowClosed,
    Tick,
}

impl Application for RgrowGui {
    type Executor = iced::executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = (Arc<Mutex<mpsc::Receiver<IpcMessage>>>, InitMessage);

    fn new(flags: Self::Flags) -> (Self, Command<Self::Message>) {
        let (receiver, init) = flags;
        let stats_text = format!(
            "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}\t Mismatches: {}",
            0.0, 0, 0, 0
        );

        let app = RgrowGui {
            current_image: None,
            stats_text,
            width: init.width,
            height: init.height,
            scale: init.block.unwrap_or(1),
            receiver,
        };

        let command = Command::batch(vec![
            Command::perform(async {}, |_| Message::IpcMessage(IpcMessage::Init(init))),
            Command::perform(async {}, |_| Message::Tick),
        ]);

        (app, command)
    }

    fn title(&self) -> String {
        "rgrow".to_string()
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match message {
            Message::IpcMessage(msg) => match msg {
                IpcMessage::Init(init) => {
                    self.width = init.width;
                    self.height = init.height;
                    self.scale = init.block.unwrap_or(1);
                }
                IpcMessage::Update(update) => {
                    self.scale = update.scale;
                    let pixel_count = update.frame_data.len() / 4;
                    let frame_width = (self.width * update.scale as u32) as usize;
                    let frame_height = if frame_width > 0 {
                        pixel_count / frame_width
                    } else {
                        (self.height * update.scale as u32) as usize
                    };
                    if pixel_count == frame_width * frame_height {
                        self.current_image =
                            Some((update.frame_data, frame_width as u32, frame_height as u32));
                    }
                    self.stats_text = format!(
                        "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}\t Mismatches: {}",
                        update.time, update.total_events, update.n_tiles, update.mismatches
                    );
                }
                IpcMessage::Close => {
                    return iced::window::close(iced::window::Id::MAIN);
                }
                _ => {}
            },
            Message::WindowClosed => {
                return iced::window::close(iced::window::Id::MAIN);
            }
            Message::Tick => {
                let receiver = self.receiver.lock().unwrap();
                if let Ok(msg) = receiver.try_recv() {
                    return Command::perform(async {}, move |_| Message::IpcMessage(msg));
                }
                return Command::none();
            }
        }

        Command::none()
    }

    fn view(&self) -> Element<Self::Message> {
        use iced::widget::canvas::{self, Canvas, Frame, Geometry, Path, Program};
        use iced::{mouse, Color, Point, Rectangle, Renderer as _, Theme};

        struct ImageCanvas {
            pixels: Vec<u8>,
            width: u32,
            height: u32,
        }

        impl<Message> Program<Message> for ImageCanvas {
            type State = ();

            fn draw(
                &self,
                _state: &Self::State,
                renderer: &iced::Renderer,
                _theme: &Theme,
                bounds: Rectangle,
                _cursor: mouse::Cursor,
            ) -> Vec<Geometry> {
                let mut frame = Frame::new(renderer, bounds.size());

                if self.pixels.len() >= (self.width * self.height * 4) as usize {
                    let img =
                        image::RgbaImage::from_raw(self.width, self.height, self.pixels.clone())
                            .expect("Invalid image dimensions");

                    let scale_x = bounds.width / self.width as f32;
                    let scale_y = bounds.height / self.height as f32;

                    for y in 0..self.height {
                        for x in 0..self.width {
                            let pixel = img.get_pixel(x, y);
                            let r = pixel[0] as f32 / 255.0;
                            let g = pixel[1] as f32 / 255.0;
                            let b = pixel[2] as f32 / 255.0;
                            let a = pixel[3] as f32 / 255.0;

                            if a > 0.0 {
                                let rect = Path::rectangle(
                                    Point::new(x as f32 * scale_x, y as f32 * scale_y),
                                    iced::Size::new(scale_x.max(1.0), scale_y.max(1.0)),
                                );
                                frame.fill(&rect, Color::from_rgba(r, g, b, a));
                            }
                        }
                    }
                }

                vec![frame.into_geometry()]
            }
        }

        let image_widget: Element<Self::Message> =
            if let Some((pixels, width, height)) = &self.current_image {
                Canvas::new(ImageCanvas {
                    pixels: pixels.clone(),
                    width: *width,
                    height: *height,
                })
                .width(Length::Fixed(*width as f32))
                .height(Length::Fixed(*height as f32))
                .into()
            } else {
                Container::new(text("Loading..."))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .into()
            };

        let stats = text(&self.stats_text).size(14);

        column![image_widget, stats]
            .spacing(5)
            .padding(10)
            .align_items(Alignment::Center)
            .into()
    }

    fn subscription(&self) -> iced::Subscription<Self::Message> {
        iced::event::listen_with(|_event, _status| Some(Message::Tick))
    }
}

pub fn run_gui(
    receiver: Arc<Mutex<mpsc::Receiver<IpcMessage>>>,
    init: InitMessage,
) -> iced::Result {
    let mut settings = Settings::with_flags((receiver, init.clone()));
    settings.window.size = iced::Size::new(
        (init.width * init.block.unwrap_or(1) as u32) as f32,
        ((init.height * init.block.unwrap_or(1) as u32) + 30) as f32,
    );
    settings.window.resizable = true;
    RgrowGui::run(settings)
}
