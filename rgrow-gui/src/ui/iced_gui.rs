use iced::task::Handle;
use iced::widget::{button, column, container, image, row, text, text_input};
use iced::{window, Color, Element, Length, Size, Subscription, Task, Theme};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use rgrow_ipc::{ControlMessage, InitMessage, ParameterInfo};

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
        energy: f64,
    },
    Close,
}

#[derive(Clone)]
pub struct ParameterState {
    pub input_value: String,
    pub current_value: f64,
    pub increment: f64,
    pub info: ParameterInfo,
}

pub struct RgrowGui {
    current_image: Option<image::Handle>,
    stats_text: String,
    receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>,
    control_sender: mpsc::Sender<ControlMessage>,
    #[cfg_attr(test, allow(dead_code))]
    pub paused: bool,
    pub events_per_step: String,
    pub max_events_per_sec: String,
    pub timescale: String,
    pub model_name: String,
    pub parameters: HashMap<String, ParameterState>,
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
    UpdateParameter { name: String, value: String },
    ApplyParameter { name: String },
    IncrementParameter { name: String },
    DecrementParameter { name: String },
    UpdateIncrement { name: String, increment: String },
}

impl RgrowGui {
    pub fn new(
        receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>,
        control_sender: mpsc::Sender<ControlMessage>,
        init: InitMessage,
    ) -> Self {
        let paused = init.start_paused;
        if paused {
            let _ = control_sender.send(ControlMessage::Pause);
        }
        let mut parameters = HashMap::new();
        for param_info in init.parameters {
            let input_value = format!("{:.3}", param_info.current_value);
            parameters.insert(
                param_info.name.clone(),
                ParameterState {
                    input_value,
                    current_value: param_info.current_value,
                    increment: param_info.default_increment,
                    info: param_info,
                },
            );
        }
        let max_events_per_sec = init
            .initial_max_events_per_sec
            .map(|v| v.to_string())
            .unwrap_or_default();
        let timescale = init
            .initial_timescale
            .map(|v| v.to_string())
            .unwrap_or_default();

        if let Some(max_eps) = init.initial_max_events_per_sec {
            let _ = control_sender.send(ControlMessage::SetMaxEventsPerSec(Some(max_eps)));
        }
        if let Some(ts) = init.initial_timescale {
            let _ = control_sender.send(ControlMessage::SetTimescale(Some(ts)));
        }
        RgrowGui {
            current_image: None,
            stats_text: format!(
                "Time: {:0.4e}  Events: {:0.4e}  Tiles: {}  Mismatches: {}  Energy: {:0.4e}",
                0.0, 0, 0, 0, 0.0
            ),
            receiver,
            control_sender,
            paused,
            events_per_step: "1000".to_string(),
            max_events_per_sec,
            timescale,
            model_name: init.model_name,
            parameters,
        }
    }

    fn title(&self) -> String {
        format!("rgrow - {}", self.model_name)
    }

    fn send_control(&self, msg: ControlMessage) {
        let _ = self.control_sender.send(msg);
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
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
                    energy,
                } => {
                    let t0 = Instant::now();
                    let t1 = Instant::now();
                    let handle = image::Handle::from_rgba(frame_width, frame_height, frame_data);
                    self.current_image = Some(handle);
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
                        "Time: {:0.4e}  Events: {:0.4e}  Tiles: {}  Mismatches: {}  Energy: {:0.4e}",
                        time, total_events, n_tiles, mismatches, energy
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
            Message::UpdateParameter { name, value } => {
                if let Some(param) = self.parameters.get_mut(&name) {
                    param.input_value = value;
                }
            }
            Message::ApplyParameter { name } => {
                if let Some(param) = self.parameters.get_mut(&name) {
                    if let Ok(new_value) = param.input_value.parse::<f64>() {
                        let clamped_value = if let Some(min) = param.info.min_value {
                            new_value.max(min)
                        } else {
                            new_value
                        };
                        let clamped_value = if let Some(max) = param.info.max_value {
                            clamped_value.min(max)
                        } else {
                            clamped_value
                        };
                        param.current_value = clamped_value;
                        param.input_value = format!("{:.3}", clamped_value);
                        self.send_control(ControlMessage::SetParameter {
                            name: name.clone(),
                            value: clamped_value,
                        });
                    }
                }
            }
            Message::IncrementParameter { name } => {
                if let Some(param) = self.parameters.get_mut(&name) {
                    let new_value = param.current_value + param.increment;
                    let clamped_value = if let Some(max) = param.info.max_value {
                        new_value.min(max)
                    } else {
                        new_value
                    };
                    param.current_value = clamped_value;
                    param.input_value = format!("{:.3}", clamped_value);
                    self.send_control(ControlMessage::SetParameter {
                        name: name.clone(),
                        value: clamped_value,
                    });
                }
            }
            Message::DecrementParameter { name } => {
                if let Some(param) = self.parameters.get_mut(&name) {
                    let new_value = param.current_value - param.increment;
                    let clamped_value = if let Some(min) = param.info.min_value {
                        new_value.max(min)
                    } else {
                        new_value
                    };
                    param.current_value = clamped_value;
                    param.input_value = format!("{:.3}", clamped_value);
                    self.send_control(ControlMessage::SetParameter {
                        name: name.clone(),
                        value: clamped_value,
                    });
                }
            }
            Message::UpdateIncrement { name, increment } => {
                if let Some(param) = self.parameters.get_mut(&name) {
                    if let Ok(inc) = increment.parse::<f64>() {
                        param.increment = inc.max(0.0);
                    }
                }
            }
        }

        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let image_widget: Element<Message> = if let Some(handle) = &self.current_image {
            image::viewer(handle.clone())
                // The image can get blurry when you zoom in, this helps with that
                .filter_method(image::FilterMethod::Nearest)
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

        let mut controls = vec![image_widget, control_row1.into(), control_row2.into()];

        let mut param_names: Vec<String> = self.parameters.keys().cloned().collect();
        param_names.sort();

        for param_name in param_names {
            if let Some(param) = self.parameters.get(&param_name) {
                let name_clone1 = param_name.clone();
                let name_clone2 = param_name.clone();
                let name_clone3 = param_name.clone();
                let name_clone4 = param_name.clone();
                let name_clone5 = param_name.clone();
                let name_clone6 = param_name.clone();

                let label_text = if param.info.units.is_empty() {
                    format!("{}:", param.info.name)
                } else {
                    format!("{} ({}):", param.info.name, param.info.units)
                };

                let value_input = text_input("Value", &param.input_value)
                    .on_input(move |s| Message::UpdateParameter {
                        name: name_clone1.clone(),
                        value: s,
                    })
                    .on_submit(Message::ApplyParameter {
                        name: name_clone2.clone(),
                    })
                    .width(100)
                    .size(14);

                let increment_input = text_input("Increment", &format!("{:.3}", param.increment))
                    .on_input(move |s| Message::UpdateIncrement {
                        name: name_clone3.clone(),
                        increment: s,
                    })
                    .width(80)
                    .size(12);

                let apply_button = button(text("Apply").size(12))
                    .on_press(Message::ApplyParameter {
                        name: name_clone4.clone(),
                    })
                    .padding([3, 8]);

                let inc_button = button(text("+").size(12))
                    .on_press(Message::IncrementParameter {
                        name: name_clone5.clone(),
                    })
                    .padding([3, 8]);

                let dec_button = button(text("-").size(12))
                    .on_press(Message::DecrementParameter {
                        name: name_clone6.clone(),
                    })
                    .padding([3, 8]);

                let param_row = row![
                    text(label_text).size(14),
                    value_input,
                    inc_button,
                    dec_button,
                    text("Increment:").size(12),
                    increment_input,
                    apply_button,
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center);

                controls.push(param_row.into());
            }
        }

        let stats = text(&self.stats_text)
            .size(14)
            .color(Color::from_rgb(0.7, 0.7, 0.7));

        controls.push(stats.into());

        column(controls).spacing(8).padding(10).into()
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
