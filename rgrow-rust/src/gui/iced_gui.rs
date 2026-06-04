use iced::widget::canvas::{self, Event, Frame, Geometry, Text};
use iced::widget::{
    button, checkbox, column, container, image, row, scrollable, stack, text, text_input, Canvas,
    Space,
};
use iced::{
    mouse, window, Color, ContentFit, Element, Length, Point, Rectangle, Renderer, Size,
    Subscription, Task, Theme,
};
use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::ui::ipc::{ControlMessage, InitMessage, ModelSnapshot, ParameterInfo};

fn debug_enabled() -> bool {
    std::env::var("RGROW_DEBUG_PERF").is_ok()
}

/// Cap on the number of rows rendered in each panel. Tile alphabets can reach
/// ~10k entries; rebuilding that many widgets every frame would stutter, so
/// we render a prefix and tell the user how many were hidden.
const MAX_PANEL_ROWS: usize = 256;

/// Format a numeric value for display / for matching in the propagate
/// popover (so "same displayed value" is well-defined).
fn fmt_val(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else if v.abs() < 1e-3 || v.abs() >= 1e6 {
        format!("{v:.4e}")
    } else {
        format!("{v:.5}")
    }
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
        /// Tile-id grid (row-major over `raw_array`), present only when
        /// inspection is enabled.
        grid: Option<Vec<u32>>,
        grid_rows: u32,
        grid_cols: u32,
        subcell_px: u32,
        tile_shape_diamond: bool,
        scale: u32,
        overlay_linear: bool,
    },
    /// Full editable-model snapshot for rebuilding the panels.
    Snapshot(Box<ModelSnapshot>),
    Close,
}

#[derive(Clone)]
pub struct ParameterState {
    pub input_value: String,
    pub current_value: f64,
    pub increment: f64,
    pub info: ParameterInfo,
}

/// Which kind of numeric field a propagate popover is operating on.
#[derive(Debug, Clone, Copy)]
pub enum PropagateKind {
    TileConc,
    GlueDg,
    Blocker,
}

#[derive(Clone)]
struct PropagateState {
    kind: PropagateKind,
    source: f64,
    label: String,
    input: String,
}

pub struct RgrowGui {
    current_image: Option<image::Handle>,
    stats_text: String,
    fps: f32,
    last_frame_at: Option<Instant>,
    receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>,
    control_sender: mpsc::Sender<ControlMessage>,
    #[cfg_attr(test, allow(dead_code))]
    pub paused: bool,
    pub events_per_step: String,
    pub max_events_per_sec: String,
    pub timescale: String,
    pub model_name: String,
    pub parameters: HashMap<String, ParameterState>,
    pub show_mismatches: bool,
    pub show_tile_names: bool,

    // Editable-model snapshot + per-frame inspection data.
    snapshot: Option<ModelSnapshot>,
    // The tile-id grid and geometry are consumed by the canvas overlay
    // (tile-name labels / hover); refreshed as they arrive each frame.
    tile_grid: Option<Vec<u32>>,
    grid_rows: u32,
    grid_cols: u32,
    subcell_px: u32,
    tile_shape_diamond: bool,
    last_scale: u32,
    overlay_linear: bool,
    tile_id_shift: u32,
    frame_px: (u32, u32),
    tile_name_by_id: HashMap<u32, String>,
    hover_cell: Option<(u32, u32)>,
    place_tile_id: Option<u32>,

    // Edit input buffers, re-seeded from the snapshot after each edit.
    tile_conc_inputs: HashMap<u32, String>,
    glue_dg_inputs: HashMap<(u32, u32), String>,
    glue_ds_inputs: HashMap<(u32, u32), String>,
    blocker_inputs: HashMap<u32, String>,
    propagate: Option<PropagateState>,
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
    UpdateParameter {
        name: String,
        value: String,
    },
    ApplyParameter {
        name: String,
    },
    IncrementParameter {
        name: String,
    },
    DecrementParameter {
        name: String,
    },
    UpdateIncrement {
        name: String,
        increment: String,
    },
    ToggleShowMismatches(bool),
    ToggleShowTileNames(bool),
    UpdateTileConc {
        id: u32,
        value: String,
    },
    ApplyTileConc {
        id: u32,
    },
    UpdateGlueDg {
        a: u32,
        b: u32,
        value: String,
    },
    UpdateGlueDs {
        a: u32,
        b: u32,
        value: String,
    },
    ApplyGlue {
        a: u32,
        b: u32,
    },
    UpdateBlocker {
        glue_id: u32,
        value: String,
    },
    ApplyBlocker {
        glue_id: u32,
    },
    OpenPropagate {
        kind: PropagateKind,
        source: f64,
        label: String,
    },
    UpdatePropagate(String),
    ApplyPropagate,
    CancelPropagate,
    CanvasHover(Option<(u32, u32)>),
    CanvasClickAt {
        px: u32,
        py: u32,
    },
    SelectPaintTile(Option<u32>),
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
            fps: 0.0,
            last_frame_at: None,
            receiver,
            control_sender,
            paused,
            events_per_step: "1000".to_string(),
            max_events_per_sec,
            timescale,
            model_name: init.model_name,
            parameters,
            show_mismatches: true,
            show_tile_names: false,
            snapshot: None,
            tile_grid: None,
            grid_rows: 0,
            grid_cols: 0,
            subcell_px: 0,
            tile_shape_diamond: false,
            last_scale: 1,
            overlay_linear: false,
            tile_id_shift: 0,
            frame_px: (0, 0),
            tile_name_by_id: HashMap::new(),
            hover_cell: None,
            place_tile_id: None,
            tile_conc_inputs: HashMap::new(),
            glue_dg_inputs: HashMap::new(),
            glue_ds_inputs: HashMap::new(),
            blocker_inputs: HashMap::new(),
            propagate: None,
        }
    }

    fn title(&self) -> String {
        format!("rgrow - {}", self.model_name)
    }

    fn send_control(&self, msg: ControlMessage) {
        let _ = self.control_sender.send(msg);
    }

    /// Re-seed edit input buffers from a fresh snapshot so displayed values
    /// reflect what the simulator actually applied (edits can cascade).
    fn seed_inputs_from_snapshot(&mut self, snap: &ModelSnapshot) {
        self.tile_id_shift = snap.tile_id_shift;
        self.tile_conc_inputs.clear();
        self.tile_name_by_id.clear();
        for t in &snap.tiles {
            if !t.name.is_empty() {
                self.tile_name_by_id.insert(t.id, t.name.clone());
            }
            if let Some(c) = t.concentration {
                self.tile_conc_inputs.insert(t.id, fmt_val(c));
            }
        }
        self.glue_dg_inputs.clear();
        self.glue_ds_inputs.clear();
        for gi in &snap.interactions {
            self.glue_dg_inputs.insert((gi.a, gi.b), fmt_val(gi.dg));
            if let Some(ds) = gi.ds {
                self.glue_ds_inputs.insert((gi.a, gi.b), fmt_val(ds));
            }
        }
        self.blocker_inputs.clear();
        for b in &snap.blockers {
            self.blocker_inputs
                .insert(b.glue_id, fmt_val(b.concentration));
        }
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
                    grid,
                    grid_rows,
                    grid_cols,
                    subcell_px,
                    tile_shape_diamond,
                    scale,
                    overlay_linear,
                } => {
                    let handle = image::Handle::from_rgba(frame_width, frame_height, frame_data);
                    self.current_image = Some(handle);
                    self.frame_px = (frame_width, frame_height);
                    if grid.is_some() {
                        self.tile_grid = grid;
                    }
                    self.grid_rows = grid_rows;
                    self.grid_cols = grid_cols;
                    self.subcell_px = subcell_px;
                    self.tile_shape_diamond = tile_shape_diamond;
                    self.last_scale = scale;
                    self.overlay_linear = overlay_linear;

                    let now = Instant::now();
                    if let Some(prev) = self.last_frame_at {
                        let dt = now.duration_since(prev).as_secs_f32();
                        if dt > 0.0 {
                            let inst = 1.0 / dt;
                            self.fps = if self.fps == 0.0 {
                                inst
                            } else {
                                self.fps * 0.9 + inst * 0.1
                            };
                        }
                    }
                    self.last_frame_at = Some(now);

                    self.stats_text = format!(
                        "Time: {time:0.4e}  Events: {total_events:0.4e}  Tiles: {n_tiles}  \
                         Mismatches: {mismatches}  Energy: {energy:0.4e}"
                    );
                }
                GuiMessage::Snapshot(snap) => {
                    self.seed_inputs_from_snapshot(&snap);
                    self.snapshot = Some(*snap);
                }
                GuiMessage::Close => {
                    return window::get_latest().and_then(window::close);
                }
            },
            Message::Tick => {
                let receiver = self.receiver.lock().unwrap();
                match receiver.try_recv() {
                    Ok(msg) => {
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
                        param.input_value = format!("{clamped_value:.3}");
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
                    param.input_value = format!("{clamped_value:.3}");
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
                    param.input_value = format!("{clamped_value:.3}");
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
            Message::ToggleShowMismatches(v) => {
                self.show_mismatches = v;
                self.send_control(ControlMessage::SetShowMismatches(v));
            }
            Message::ToggleShowTileNames(v) => {
                self.show_tile_names = v;
                self.send_control(ControlMessage::SetInspection(v));
                if !v {
                    // Stop pushing/using the now-stale grid.
                    self.tile_grid = None;
                    self.hover_cell = None;
                }
            }
            Message::UpdateTileConc { id, value } => {
                self.tile_conc_inputs.insert(id, value);
            }
            Message::ApplyTileConc { id } => {
                if let Some(s) = self.tile_conc_inputs.get(&id) {
                    if let Ok(value) = s.parse::<f64>() {
                        self.send_control(ControlMessage::SetTileConcentration { id, value });
                    }
                }
            }
            Message::UpdateGlueDg { a, b, value } => {
                self.glue_dg_inputs.insert((a, b), value);
            }
            Message::UpdateGlueDs { a, b, value } => {
                self.glue_ds_inputs.insert((a, b), value);
            }
            Message::ApplyGlue { a, b } => {
                let has_ds = self
                    .snapshot
                    .as_ref()
                    .map(|s| s.schema.has_ds)
                    .unwrap_or(false);
                let dg = self
                    .glue_dg_inputs
                    .get(&(a, b))
                    .and_then(|s| s.parse::<f64>().ok());
                let ds = if has_ds {
                    self.glue_ds_inputs
                        .get(&(a, b))
                        .and_then(|s| s.parse::<f64>().ok())
                } else {
                    None
                };
                if let Some(dg) = dg {
                    self.send_control(ControlMessage::SetGlueInteraction { a, b, dg, ds });
                }
            }
            Message::UpdateBlocker { glue_id, value } => {
                self.blocker_inputs.insert(glue_id, value);
            }
            Message::ApplyBlocker { glue_id } => {
                if let Some(s) = self.blocker_inputs.get(&glue_id) {
                    if let Ok(value) = s.parse::<f64>() {
                        self.send_control(ControlMessage::SetBlockerConcentration {
                            glue_id,
                            value,
                        });
                    }
                }
            }
            Message::OpenPropagate {
                kind,
                source,
                label,
            } => {
                self.propagate = Some(PropagateState {
                    kind,
                    source,
                    label,
                    input: fmt_val(source),
                });
            }
            Message::UpdatePropagate(s) => {
                if let Some(p) = self.propagate.as_mut() {
                    p.input = s;
                }
            }
            Message::CancelPropagate => {
                self.propagate = None;
            }
            Message::CanvasHover(cell) => {
                self.hover_cell = cell;
            }
            Message::CanvasClickAt { px, py } => {
                if let Some(tile) = self.place_tile_id {
                    self.send_control(ControlMessage::SetPointAtPixel {
                        px,
                        py,
                        scale: self.last_scale,
                        tile,
                    });
                }
            }
            Message::SelectPaintTile(id) => {
                self.place_tile_id = id;
            }
            Message::ApplyPropagate => {
                if let Some(p) = self.propagate.take() {
                    if let (Ok(new_val), Some(snap)) =
                        (p.input.parse::<f64>(), self.snapshot.as_ref())
                    {
                        let key = fmt_val(p.source);
                        match p.kind {
                            PropagateKind::TileConc => {
                                for t in &snap.tiles {
                                    if let Some(c) = t.concentration {
                                        if fmt_val(c) == key {
                                            self.send_control(
                                                ControlMessage::SetTileConcentration {
                                                    id: t.id,
                                                    value: new_val,
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                            PropagateKind::GlueDg => {
                                for gi in &snap.interactions {
                                    if fmt_val(gi.dg) == key {
                                        self.send_control(ControlMessage::SetGlueInteraction {
                                            a: gi.a,
                                            b: gi.b,
                                            dg: new_val,
                                            ds: gi.ds,
                                        });
                                    }
                                }
                            }
                            PropagateKind::Blocker => {
                                for b in &snap.blockers {
                                    if fmt_val(b.concentration) == key {
                                        self.send_control(
                                            ControlMessage::SetBlockerConcentration {
                                                glue_id: b.glue_id,
                                                value: new_val,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Task::none()
    }

    fn left_column(&self) -> Element<'_, Message> {
        let image_widget: Element<Message> = if let Some(handle) = &self.current_image {
            let img = image(handle.clone())
                .filter_method(image::FilterMethod::Nearest)
                .content_fit(ContentFit::Contain)
                .width(Length::Fill)
                .height(Length::Fill);
            let overlay = Canvas::new(Overlay {
                frame_w: self.frame_px.0 as f32,
                frame_h: self.frame_px.1 as f32,
                subcell: self.subcell_px as f32,
                show_names: self.show_tile_names,
                linear: self.overlay_linear,
                id_shift: self.tile_id_shift,
                grid: self.tile_grid.as_deref(),
                grid_cols: self.grid_cols,
                grid_rows: self.grid_rows,
                names: &self.tile_name_by_id,
                hover: self.hover_cell,
            })
            .width(Length::Fill)
            .height(Length::Fill);
            stack![img, overlay].into()
        } else {
            container(text("Loading..."))
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
        };

        let pause_text = if self.paused { "Resume" } else { "Pause" };
        let control_row1 = row![
            button(text(pause_text).size(14))
                .on_press(Message::TogglePause)
                .padding([5, 15]),
            button(text("Step").size(14))
                .on_press(Message::Step)
                .padding([5, 15]),
            text("Events/step:").size(14),
            text_input("1000", &self.events_per_step)
                .on_input(Message::UpdateEventsPerStep)
                .width(80)
                .size(14),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let control_row2 = row![
            text("Max events/sec:").size(14),
            text_input("unlimited", &self.max_events_per_sec)
                .on_input(Message::UpdateMaxEventsPerSec)
                .on_submit(Message::ApplyMaxEventsPerSec)
                .width(100)
                .size(14),
            button(text("Apply").size(12))
                .on_press(Message::ApplyMaxEventsPerSec)
                .padding([3, 8]),
            text("Timescale:").size(14),
            text_input("unlimited", &self.timescale)
                .on_input(Message::UpdateTimescale)
                .on_submit(Message::ApplyTimescale)
                .width(100)
                .size(14),
            button(text("Apply").size(12))
                .on_press(Message::ApplyTimescale)
                .padding([3, 8]),
        ]
        .spacing(10)
        .align_y(iced::Alignment::Center);

        let toggles = row![
            checkbox("Show mismatches", self.show_mismatches)
                .on_toggle(Message::ToggleShowMismatches)
                .size(16)
                .text_size(14),
            checkbox("Show tile names", self.show_tile_names)
                .on_toggle(Message::ToggleShowTileNames)
                .size(16)
                .text_size(14),
        ]
        .spacing(16)
        .align_y(iced::Alignment::Center);

        // Paint-tile indicator: lets the user place tiles by clicking the canvas.
        let paint_row: Element<Message> = if let Some(id) = self.place_tile_id {
            let nm = self.tile_name_by_id.get(&id).cloned().unwrap_or_default();
            let label = if id == 0 {
                "Painting: erase".to_string()
            } else {
                format!("Painting: tile {id} {nm}")
            };
            row![
                text(label).size(13),
                button(text("erase").size(12))
                    .on_press(Message::SelectPaintTile(Some(0)))
                    .padding([2, 8]),
                button(text("stop").size(12))
                    .on_press(Message::SelectPaintTile(None))
                    .padding([2, 8]),
            ]
            .spacing(8)
            .align_y(iced::Alignment::Center)
            .into()
        } else {
            text("Tip: click a tile's ✎ in the panel to paint it onto the canvas")
                .size(12)
                .color(Color::from_rgb(0.6, 0.6, 0.6))
                .into()
        };

        let hover_text = if let (true, Some((col, row)), Some(grid)) = (
            self.show_tile_names,
            self.hover_cell,
            self.tile_grid.as_ref(),
        ) {
            let idx = (row * self.grid_cols + col) as usize;
            let raw = grid.get(idx).copied().unwrap_or(0);
            let base = raw >> self.tile_id_shift;
            let name = self.tile_name_by_id.get(&base).cloned().unwrap_or_default();
            format!("Cell ({col}, {row}): tile {base} {name}")
        } else {
            String::new()
        };

        let mut controls: Vec<Element<Message>> = vec![
            image_widget,
            control_row1.into(),
            control_row2.into(),
            toggles.into(),
            paint_row,
            text(hover_text)
                .size(12)
                .color(Color::from_rgb(0.6, 0.6, 0.6))
                .into(),
        ];

        let mut param_names: Vec<String> = self.parameters.keys().cloned().collect();
        param_names.sort();
        for param_name in param_names {
            if let Some(param) = self.parameters.get(&param_name) {
                let n1 = param_name.clone();
                let n2 = param_name.clone();
                let n3 = param_name.clone();
                let n4 = param_name.clone();
                let n5 = param_name.clone();
                let n6 = param_name.clone();

                let label_text = if param.info.units.is_empty() {
                    format!("{}:", param.info.name)
                } else {
                    format!("{} ({}):", param.info.name, param.info.units)
                };

                let param_row = row![
                    text(label_text).size(14),
                    text_input("Value", &param.input_value)
                        .on_input(move |s| Message::UpdateParameter {
                            name: n1.clone(),
                            value: s,
                        })
                        .on_submit(Message::ApplyParameter { name: n2.clone() })
                        .width(100)
                        .size(14),
                    button(text("+").size(12))
                        .on_press(Message::IncrementParameter { name: n5.clone() })
                        .padding([3, 8]),
                    button(text("-").size(12))
                        .on_press(Message::DecrementParameter { name: n6.clone() })
                        .padding([3, 8]),
                    text("Increment:").size(12),
                    text_input("Increment", &format!("{:.3}", param.increment))
                        .on_input(move |s| Message::UpdateIncrement {
                            name: n3.clone(),
                            increment: s,
                        })
                        .width(80)
                        .size(12),
                    button(text("Apply").size(12))
                        .on_press(Message::ApplyParameter { name: n4.clone() })
                        .padding([3, 8]),
                ]
                .spacing(10)
                .align_y(iced::Alignment::Center);

                controls.push(param_row.into());
            }
        }

        let stats = text(format!(
            "{}  |  {}  |  FPS: {:.0}",
            self.model_name, self.stats_text, self.fps
        ))
        .size(13)
        .color(Color::from_rgb(0.7, 0.7, 0.7));
        controls.push(stats.into());

        column(controls).spacing(8).padding(10).into()
    }

    fn right_panel(&self) -> Element<'_, Message> {
        let Some(snap) = &self.snapshot else {
            return container(text("Loading model…").size(14))
                .padding(10)
                .into();
        };

        let mut sections: Vec<Element<Message>> = Vec::new();

        // ── Tileset panel ────────────────────────────────────────────────
        sections.push(text("Tiles").size(16).into());
        for t in snap.tiles.iter().take(MAX_PANEL_ROWS) {
            let tid = t.id;
            let mut cells: Vec<Element<Message>> = vec![
                swatch(t.color),
                button(text("✎").size(11))
                    .on_press(Message::SelectPaintTile(Some(tid)))
                    .padding([1, 5])
                    .into(),
                text(format!("{}", t.id)).size(12).width(28).into(),
                text(if t.name.is_empty() {
                    "(unnamed)".to_string()
                } else {
                    t.name.clone()
                })
                .size(12)
                .width(110)
                .into(),
            ];
            if let Some(conc) = t.concentration {
                if snap.editable.tile_concentration {
                    let id = t.id;
                    let value = self
                        .tile_conc_inputs
                        .get(&id)
                        .cloned()
                        .unwrap_or_else(|| fmt_val(conc));
                    cells.push(
                        text_input("conc", &value)
                            .on_input(move |s| Message::UpdateTileConc { id, value: s })
                            .on_submit(Message::ApplyTileConc { id })
                            .width(90)
                            .size(12)
                            .into(),
                    );
                    cells.push(propagate_button(Message::OpenPropagate {
                        kind: PropagateKind::TileConc,
                        source: conc,
                        label: format!("Set all tile concentrations = {}", fmt_val(conc)),
                    }));
                } else {
                    cells.push(text(fmt_val(conc)).size(12).width(90).into());
                }
            }
            if let Some(free) = t.free_concentration {
                cells.push(
                    text(format!("free {}", fmt_val(free)))
                        .size(11)
                        .color(Color::from_rgb(0.6, 0.6, 0.6))
                        .into(),
                );
            }
            sections.push(
                row(cells)
                    .spacing(8)
                    .align_y(iced::Alignment::Center)
                    .into(),
            );
        }
        if snap.tiles.len() > MAX_PANEL_ROWS {
            sections.push(
                text(format!(
                    "(showing {} of {} tiles)",
                    MAX_PANEL_ROWS,
                    snap.tiles.len()
                ))
                .size(11)
                .color(Color::from_rgb(0.6, 0.6, 0.6))
                .into(),
            );
        }

        // ── Glue interactions panel ──────────────────────────────────────
        if !snap.interactions.is_empty() {
            sections.push(Space::with_height(8).into());
            let header = if snap.schema.has_ds {
                format!(
                    "Glue interactions ({} / {})",
                    snap.schema.label_dg,
                    snap.schema.label_ds.clone().unwrap_or_default()
                )
            } else {
                format!("Glue interactions ({})", snap.schema.label_dg)
            };
            sections.push(text(header).size(16).into());
            for (i, gi) in snap.interactions.iter().enumerate() {
                if i >= MAX_PANEL_ROWS {
                    sections.push(
                        text(format!(
                            "(showing {} of {} interactions)",
                            MAX_PANEL_ROWS,
                            snap.interactions.len()
                        ))
                        .size(11)
                        .color(Color::from_rgb(0.6, 0.6, 0.6))
                        .into(),
                    );
                    break;
                }
                let pair_label = if gi.matching {
                    format!("{} (self)", gi.a_name)
                } else {
                    format!("{} × {}", gi.a_name, gi.b_name)
                };
                let (a, b) = (gi.a, gi.b);
                let dg_val = self
                    .glue_dg_inputs
                    .get(&(a, b))
                    .cloned()
                    .unwrap_or_else(|| fmt_val(gi.dg));
                let mut cells: Vec<Element<Message>> = vec![
                    text(pair_label).size(12).width(150).into(),
                    text_input("ΔG", &dg_val)
                        .on_input(move |s| Message::UpdateGlueDg { a, b, value: s })
                        .on_submit(Message::ApplyGlue { a, b })
                        .width(90)
                        .size(12)
                        .into(),
                ];
                if snap.schema.has_ds {
                    let ds_val = self
                        .glue_ds_inputs
                        .get(&(a, b))
                        .cloned()
                        .unwrap_or_else(|| gi.ds.map(fmt_val).unwrap_or_default());
                    cells.push(
                        text_input("ΔS", &ds_val)
                            .on_input(move |s| Message::UpdateGlueDs { a, b, value: s })
                            .on_submit(Message::ApplyGlue { a, b })
                            .width(90)
                            .size(12)
                            .into(),
                    );
                }
                cells.push(propagate_button(Message::OpenPropagate {
                    kind: PropagateKind::GlueDg,
                    source: gi.dg,
                    label: format!("Set all {} = {}", snap.schema.label_dg, fmt_val(gi.dg)),
                }));
                sections.push(
                    row(cells)
                        .spacing(8)
                        .align_y(iced::Alignment::Center)
                        .into(),
                );
            }
        }

        // ── Blocker panel (KBlock) ───────────────────────────────────────
        if snap.editable.blocker && !snap.blockers.is_empty() {
            sections.push(Space::with_height(8).into());
            sections.push(text("Blockers").size(16).into());
            for b in snap.blockers.iter().take(MAX_PANEL_ROWS) {
                let glue_id = b.glue_id;
                let value = self
                    .blocker_inputs
                    .get(&glue_id)
                    .cloned()
                    .unwrap_or_else(|| fmt_val(b.concentration));
                let cells: Vec<Element<Message>> = vec![
                    text(b.glue_name.clone()).size(12).width(110).into(),
                    text_input("conc", &value)
                        .on_input(move |s| Message::UpdateBlocker { glue_id, value: s })
                        .on_submit(Message::ApplyBlocker { glue_id })
                        .width(90)
                        .size(12)
                        .into(),
                    text(format!("free {}", fmt_val(b.free_concentration)))
                        .size(11)
                        .color(Color::from_rgb(0.6, 0.6, 0.6))
                        .into(),
                    propagate_button(Message::OpenPropagate {
                        kind: PropagateKind::Blocker,
                        source: b.concentration,
                        label: format!(
                            "Set all blocker concentrations = {}",
                            fmt_val(b.concentration)
                        ),
                    }),
                ];
                sections.push(
                    row(cells)
                        .spacing(8)
                        .align_y(iced::Alignment::Center)
                        .into(),
                );
            }
            if snap.blockers.len() > MAX_PANEL_ROWS {
                sections.push(
                    text(format!(
                        "(showing {} of {} blockers)",
                        MAX_PANEL_ROWS,
                        snap.blockers.len()
                    ))
                    .size(11)
                    .color(Color::from_rgb(0.6, 0.6, 0.6))
                    .into(),
                );
            }
        }

        scrollable(column(sections).spacing(4).padding(10))
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn view(&self) -> Element<'_, Message> {
        let base = row![
            container(self.left_column())
                .width(Length::FillPortion(2))
                .height(Length::Fill),
            container(self.right_panel())
                .width(Length::FillPortion(1))
                .height(Length::Fill),
        ];

        if let Some(p) = &self.propagate {
            let popover = container(
                column![
                    text(p.label.clone()).size(14),
                    text_input("new value", &p.input)
                        .on_input(Message::UpdatePropagate)
                        .on_submit(Message::ApplyPropagate)
                        .width(160)
                        .size(14),
                    row![
                        button(text("Apply").size(13))
                            .on_press(Message::ApplyPropagate)
                            .padding([4, 12]),
                        button(text("Cancel").size(13))
                            .on_press(Message::CancelPropagate)
                            .padding([4, 12]),
                    ]
                    .spacing(10),
                ]
                .spacing(10)
                .padding(16),
            )
            .style(|_theme| container::background(Color::from_rgb(0.13, 0.13, 0.16)))
            .padding(4);

            let overlay = container(popover)
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x(Length::Fill)
                .center_y(Length::Fill);

            stack![base, overlay].into()
        } else {
            base.into()
        }
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

/// A small fixed-size color swatch for the tileset panel.
fn swatch(color: [u8; 4]) -> Element<'static, Message> {
    let col = Color::from_rgba8(color[0], color[1], color[2], color[3] as f32 / 255.0);
    container(Space::new(16, 16))
        .style(move |_theme| container::background(col))
        .into()
}

/// The "↪" propagate button used next to editable numeric cells.
fn propagate_button(msg: Message) -> Element<'static, Message> {
    button(text("↪").size(12))
        .on_press(msg)
        .padding([2, 6])
        .into()
}

#[derive(Default)]
pub struct OverlayState {
    last_cell: Option<(u32, u32)>,
}

/// Canvas overlay drawn on top of the frame image: tile-name labels and a
/// hover highlight, plus cursor→cell / cursor→pixel mapping for inspection
/// and click-to-place. Uses the same `ContentFit::Contain` transform as the
/// image below it so overlays line up with the rendered frame.
struct Overlay<'a> {
    frame_w: f32,
    frame_h: f32,
    subcell: f32,
    show_names: bool,
    /// Positioned overlays (labels / hover) are only correct on plain-square
    /// canvases; disabled otherwise (the sheared / staggered tube canvases).
    linear: bool,
    /// Right-shift to decode a raw grid value to a base tile id for names.
    id_shift: u32,
    grid: Option<&'a [u32]>,
    grid_cols: u32,
    grid_rows: u32,
    names: &'a HashMap<u32, String>,
    hover: Option<(u32, u32)>,
}

impl Overlay<'_> {
    /// `Contain` fit: `(scale, offset_x, offset_y)` mapping frame-pixel space
    /// into `bounds`.
    fn fit(&self, bounds: Rectangle) -> (f32, f32, f32) {
        if self.frame_w <= 0.0 || self.frame_h <= 0.0 {
            return (1.0, 0.0, 0.0);
        }
        let s = (bounds.width / self.frame_w).min(bounds.height / self.frame_h);
        let ox = (bounds.width - self.frame_w * s) / 2.0;
        let oy = (bounds.height - self.frame_h * s) / 2.0;
        (s, ox, oy)
    }

    fn cursor_to_frame(&self, bounds: Rectangle, cursor: mouse::Cursor) -> Option<(f32, f32)> {
        let p = cursor.position_in(bounds)?;
        let (s, ox, oy) = self.fit(bounds);
        if s <= 0.0 {
            return None;
        }
        let fx = (p.x - ox) / s;
        let fy = (p.y - oy) / s;
        if fx < 0.0 || fy < 0.0 || fx >= self.frame_w || fy >= self.frame_h {
            return None;
        }
        Some((fx, fy))
    }

    fn frame_to_cell(&self, fx: f32, fy: f32) -> Option<(u32, u32)> {
        // Only plain-square canvases have a linear storage<->pixel map; for
        // the tube canvases the cursor->cell inverse would be wrong, so we
        // report no cell (click-to-place still works via the sim's inverse).
        if !self.linear || self.subcell <= 0.0 {
            return None;
        }
        let col = (fx / self.subcell) as u32;
        let row = (fy / self.subcell) as u32;
        if col < self.grid_cols && row < self.grid_rows {
            Some((col, row))
        } else {
            None
        }
    }
}

impl canvas::Program<Message> for Overlay<'_> {
    type State = OverlayState;

    fn update(
        &self,
        state: &mut OverlayState,
        event: Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (canvas::event::Status, Option<Message>) {
        match event {
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                let cell = self
                    .cursor_to_frame(bounds, cursor)
                    .and_then(|(fx, fy)| self.frame_to_cell(fx, fy));
                if cell != state.last_cell {
                    state.last_cell = cell;
                    (
                        canvas::event::Status::Captured,
                        Some(Message::CanvasHover(cell)),
                    )
                } else {
                    (canvas::event::Status::Ignored, None)
                }
            }
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                match self.cursor_to_frame(bounds, cursor) {
                    Some((fx, fy)) => (
                        canvas::event::Status::Captured,
                        Some(Message::CanvasClickAt {
                            px: fx as u32,
                            py: fy as u32,
                        }),
                    ),
                    None => (canvas::event::Status::Ignored, None),
                }
            }
            _ => (canvas::event::Status::Ignored, None),
        }
    }

    fn draw(
        &self,
        _state: &OverlayState,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());
        let (s, ox, oy) = self.fit(bounds);
        let disp_cell = self.subcell * s;

        if let Some((col, row)) = self.hover {
            frame.fill_rectangle(
                Point::new(ox + col as f32 * disp_cell, oy + row as f32 * disp_cell),
                Size::new(disp_cell, disp_cell),
                Color::from_rgba(1.0, 1.0, 1.0, 0.25),
            );
        }

        // Labels are only correct on linear canvases and only legible (and
        // cheap) when cells are large enough.
        if self.show_names && self.linear && disp_cell >= 12.0 {
            if let Some(grid) = self.grid {
                for row in 0..self.grid_rows {
                    for col in 0..self.grid_cols {
                        let idx = (row * self.grid_cols + col) as usize;
                        let Some(&raw) = grid.get(idx) else { continue };
                        if raw == 0 {
                            continue;
                        }
                        let Some(name) = self.names.get(&(raw >> self.id_shift)) else {
                            continue;
                        };
                        frame.fill_text(Text {
                            content: name.clone(),
                            position: Point::new(
                                ox + (col as f32 + 0.5) * disp_cell,
                                oy + (row as f32 + 0.5) * disp_cell,
                            ),
                            color: Color::WHITE,
                            size: iced::Pixels((disp_cell * 0.32).clamp(8.0, 16.0)),
                            horizontal_alignment: iced::alignment::Horizontal::Center,
                            vertical_alignment: iced::alignment::Vertical::Center,
                            ..Text::default()
                        });
                    }
                }
            }
        }

        vec![frame.into_geometry()]
    }
}

pub fn run_gui(
    receiver: Arc<Mutex<mpsc::Receiver<GuiMessage>>>,
    control_sender: mpsc::Sender<ControlMessage>,
    init: InitMessage,
) -> iced::Result {
    let canvas_width = (init.width * init.block.unwrap_or(1) as u32) as f32;
    let window_width = (canvas_width + 380.0).max(1100.0);
    let window_height = ((init.height * init.block.unwrap_or(1) as u32) + 160) as f32;
    let window_height = window_height.max(640.0);

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
