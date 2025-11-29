use iced::widget::{button, column, container, image, row, text, text_input};
use iced::{window, Color, Element, Length, Size, Subscription, Task, Theme};
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::ui::ipc::{ControlMessage, InitMessage};

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
    control_sender: mpsc::Sender<ControlMessage>,
    paused: bool,
    events_per_step: String,
    max_events_per_sec: String,
    timescale: String,
}

#[derive(Debug, Clone)]
pub enum Message {
    GuiMessage(GuiMessage),
    Tick,
    CloseWindow,
    TogglePause,
    Step,
    UpdateEventsPerStep(String),
    UpdateMaxEventsPerSec(String),
    UpdateTimescale(String),
    ApplyMaxEventsPerSec,
    ApplyTimescale,
}

impl RgrowGui {
    fn new(
        receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>,
        control_sender: mpsc::Sender<ControlMessage>,
        _init: InitMessage,
    ) -> Self {
        RgrowGui {
            current_image: None,
            stats_text: format!(
                "Time: {:0.4e}  Events: {:0.4e}  Tiles: {}  Mismatches: {}",
                0.0, 0, 0, 0
            ),
            receiver,
            control_sender,
            paused: false,
            events_per_step: "1000".to_string(),
            max_events_per_sec: "".to_string(),
            timescale: "".to_string(),
        }
    }

    fn title(&self) -> String {
        "rgrow".to_string()
    }

    fn send_control(&self, msg: ControlMessage) {
        let _ = self.control_sender.send(msg);
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
                        "Time: {:0.4e}  Events: {:0.4e}  Tiles: {}  Mismatches: {}",
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
                    Err(mpsc::TryRecvError::Empty) => {}
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
            Message::TogglePause => {
                self.paused = !self.paused;
                if self.paused {
                    self.send_control(ControlMessage::Pause);
                } else {
                    self.send_control(ControlMessage::Resume);
                }
            }
            Message::Step => {
                let events = self.events_per_step.parse().unwrap_or(1000);
                self.paused = false;
                self.send_control(ControlMessage::Step { events });
            }
            Message::UpdateEventsPerStep(s) => {
                self.events_per_step = s;
            }
            Message::UpdateMaxEventsPerSec(s) => {
                self.max_events_per_sec = s;
            }
            Message::UpdateTimescale(s) => {
                self.timescale = s;
            }
            Message::ApplyMaxEventsPerSec => {
                let value = if self.max_events_per_sec.is_empty() {
                    None
                } else {
                    self.max_events_per_sec.parse().ok()
                };
                self.send_control(ControlMessage::SetMaxEventsPerSec(value));
            }
            Message::ApplyTimescale => {
                let value = if self.timescale.is_empty() {
                    None
                } else {
                    self.timescale.parse().ok()
                };
                self.send_control(ControlMessage::SetTimescale(value));
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

        let pause_text = if self.paused { "Resume" } else { "Pause" };
        let pause_button = button(text(pause_text).size(14))
            .on_press(Message::TogglePause)
            .padding([5, 15]);

        let step_button = button(text("Step").size(14))
            .on_press(Message::Step)
            .padding([5, 15]);

        let events_per_step_input = text_input("1000", &self.events_per_step)
            .on_input(Message::UpdateEventsPerStep)
            .width(80)
            .size(14);

        let control_row1 = row![
            pause_button,
            step_button,
            text("Events/step:").size(14),
            events_per_step_input,
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let max_eps_input = text_input("unlimited", &self.max_events_per_sec)
            .on_input(Message::UpdateMaxEventsPerSec)
            .on_submit(Message::ApplyMaxEventsPerSec)
            .width(100)
            .size(14);

        let apply_eps_button = button(text("Apply").size(12))
            .on_press(Message::ApplyMaxEventsPerSec)
            .padding([3, 8]);

        let timescale_input = text_input("unlimited", &self.timescale)
            .on_input(Message::UpdateTimescale)
            .on_submit(Message::ApplyTimescale)
            .width(100)
            .size(14);

        let apply_ts_button = button(text("Apply").size(12))
            .on_press(Message::ApplyTimescale)
            .padding([3, 8]);

        let control_row2 = row![
            text("Max events/sec:").size(14),
            max_eps_input,
            apply_eps_button,
            text("Timescale:").size(14),
            timescale_input,
            apply_ts_button,
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let stats = text(&self.stats_text)
            .size(14)
            .color(Color::from_rgb(0.7, 0.7, 0.7));

        column![image_widget, control_row1, control_row2, stats]
            .spacing(8)
            .padding(10)
            .into()
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
    control_sender: mpsc::Sender<ControlMessage>,
    init: InitMessage,
) -> iced::Result {
    let window_width = (init.width * init.block.unwrap_or(1) as u32) as f32;
    let window_height = ((init.height * init.block.unwrap_or(1) as u32) + 100) as f32;

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
        .run_with(move || (RgrowGui::new(receiver, control_sender, init), Task::none()))
}
