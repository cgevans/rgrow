use iced::widget::{column, container, image, text};
use iced::{window, Element, Length, Size, Subscription, Task, Theme};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::ui::ipc::InitMessage;

fn debug_enabled() -> bool {
    std::env::var("RGROW_DEBUG_PERF").is_ok()
}

/// Message type for GUI updates (with frame data already read from shared memory)
#[derive(Debug, Clone)]
pub enum GuiMessage {
    Update {
        frame_data: Vec<u8>,
        frame_width: u32,
        frame_height: u32,
        time: f64,
        total_events: u64,
        n_tiles: u32,
        mismatches: u32,
    },
    Close,
}

pub struct RgrowGui {
    current_image: Option<image::Handle>,
    stats_text: String,
    receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>,
}

#[derive(Debug, Clone)]
pub enum Message {
    GuiMessage(GuiMessage),
    Tick,
    CloseWindow,
}

impl RgrowGui {
    fn new(receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>, _init: InitMessage) -> Self {
        RgrowGui {
            current_image: None,
            stats_text: format!(
                "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}\t Mismatches: {}",
                0.0, 0, 0, 0
            ),
            receiver,
        }
    }

    fn title(&self) -> String {
        "rgrow".to_string()
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::GuiMessage(msg) => match msg {
                GuiMessage::Update {
                    frame_data,
                    frame_width,
                    frame_height,
                    time,
                    total_events,
                    n_tiles,
                    mismatches,
                } => {
                    let t0 = Instant::now();
                    let t1 = Instant::now();
                    self.current_image = Some(image::Handle::from_rgba(
                        frame_width,
                        frame_height,
                        frame_data,
                    ));
                    if debug_enabled() {
                        eprintln!(
                            "[GUI] image handle creation: {:?} ({}x{} = {} bytes)",
                            t1.elapsed(),
                            frame_width,
                            frame_height,
                            frame_width * frame_height * 4
                        );
                    }
                    self.stats_text = format!(
                        "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}\t Mismatches: {}",
                        time, total_events, n_tiles, mismatches
                    );
                    if debug_enabled() {
                        eprintln!("[GUI] total update processing: {:?}", t0.elapsed());
                    }
                }
                GuiMessage::Close => {
                    return window::get_latest().and_then(window::close);
                }
            },
            Message::Tick => {
                let t0 = Instant::now();
                let receiver = self.receiver.lock().unwrap();
                match receiver.try_recv() {
                    Ok(msg) => {
                        if debug_enabled() {
                            eprintln!("[GUI] recv from channel: {:?}", t0.elapsed());
                        }
                        return Task::done(Message::GuiMessage(msg));
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        // No message yet, that's fine
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        if debug_enabled() {
                            eprintln!("[GUI] Channel disconnected!");
                        }
                    }
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
    receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>,
    init: InitMessage,
) -> iced::Result {
    let window_width = (init.width * init.block.unwrap_or(1) as u32) as f32;
    let window_height = ((init.height * init.block.unwrap_or(1) as u32) + 30) as f32;

    // Ensure minimum window size of 800x600
    let window_width = window_width.max(800.0);
    let window_height = window_height.max(600.0);

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
