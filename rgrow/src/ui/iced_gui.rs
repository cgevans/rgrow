use iced::widget::{column, container, image, text};
use iced::{window, Element, Length, Size, Subscription, Task, Theme};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use crate::ui::ipc::{InitMessage, IpcMessage};

pub struct RgrowGui {
    current_image: Option<image::Handle>,
    stats_text: String,
    width: u32,
    height: u32,
    scale: usize,
    receiver: Arc<Mutex<mpsc::Receiver<IpcMessage>>>,
}

#[derive(Debug, Clone)]
pub enum Message {
    IpcMessage(IpcMessage),
    Tick,
    CloseWindow,
}

impl RgrowGui {
    fn new(receiver: Arc<Mutex<mpsc::Receiver<IpcMessage>>>, init: InitMessage) -> Self {
        RgrowGui {
            current_image: None,
            stats_text: format!(
                "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}\t Mismatches: {}",
                0.0, 0, 0, 0
            ),
            width: init.width,
            height: init.height,
            scale: init.block.unwrap_or(1),
            receiver,
        }
    }

    fn title(&self) -> String {
        "rgrow".to_string()
    }

    fn update(&mut self, message: Message) -> Task<Message> {
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
                        self.current_image = Some(image::Handle::from_rgba(
                            frame_width as u32,
                            frame_height as u32,
                            update.frame_data,
                        ));
                    }
                    self.stats_text = format!(
                        "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}\t Mismatches: {}",
                        update.time, update.total_events, update.n_tiles, update.mismatches
                    );
                }
                IpcMessage::Close => {
                    return window::get_latest().and_then(window::close);
                }
                _ => {}
            },
            Message::Tick => {
                let receiver = self.receiver.lock().unwrap();
                if let Ok(msg) = receiver.try_recv() {
                    return Task::done(Message::IpcMessage(msg));
                }
            }
            Message::CloseWindow => {
                return window::get_latest().and_then(window::close);
            }
        }

        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let image_widget: Element<Message> = if let Some(handle) = &self.current_image {
            image(handle.clone())
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            container(text("Loading..."))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
        };

        let stats = text(&self.stats_text).size(14);

        column![image_widget, stats].spacing(5).padding(10).into()
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::run(|| {
            iced::stream::channel(100, |output| async move {
                use iced::futures::SinkExt;
                let mut output = output;
                loop {
                    let _ = output.send(Message::Tick).await;
                    tokio::time::sleep(Duration::from_millis(16)).await;
                }
            })
        })
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }
}

pub fn run_gui(
    receiver: Arc<Mutex<mpsc::Receiver<IpcMessage>>>,
    init: InitMessage,
) -> iced::Result {
    let window_width = (init.width * init.block.unwrap_or(1) as u32) as f32;
    let window_height = ((init.height * init.block.unwrap_or(1) as u32) + 30) as f32;

    iced::application(RgrowGui::title, RgrowGui::update, RgrowGui::view)
        .subscription(RgrowGui::subscription)
        .theme(RgrowGui::theme)
        .window(window::Settings {
            size: Size::new(window_width, window_height),
            resizable: true,
            ..Default::default()
        })
        .run_with(move || (RgrowGui::new(receiver, init), Task::none()))
}
