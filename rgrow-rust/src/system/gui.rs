use std::time::{Duration, Instant};

use crate::base::RgrowError;
use crate::canvas::TileShape;
use crate::painter::render_frame;
use crate::state::State;
use crate::ui::find_gui_command;

use super::core::System;
use super::types::*;

/// Build the full editable-model snapshot from trait calls on the concrete
/// model. Mirrors the assembly the wasm `Sim::tile_set` / `glue_list` do,
/// but stays generic so it runs against `S: System` in the sim loop.
fn build_model_snapshot<S: System>(sys: &S, model_name: &str) -> crate::ui::ipc::ModelSnapshot {
    use crate::base::Tile;
    use crate::ui::ipc::{GlueSnapshot, ModelSnapshot, TileSnapshot};

    let names = sys.tile_names();
    let colors = sys.tile_colors();
    let n = names.len().min(colors.len());
    let editable = sys.editable_features();
    let schema = sys.interaction_schema();
    let concs = sys.tile_concentrations();
    let free_concs = sys.free_tile_concentrations();

    // `bond_names` is `todo!()` on models without glue data, so only touch
    // it for models that expose glue editing.
    let glue_names: &[String] = if editable.glue_interaction || editable.tile_edge_glue {
        sys.bond_names()
    } else {
        &[]
    };

    let mut tiles = Vec::with_capacity(n.saturating_sub(1));
    for id in 1..n {
        let edges = sys.tile_edge_glues(id as Tile);
        let edge_glues = edges.map(|g| {
            g.and_then(|gid| glue_names.get(gid).cloned())
                .filter(|s| !s.is_empty())
        });
        let edge_glue_ids = edges.map(|g| g.map(|gid| gid as u32));
        tiles.push(TileSnapshot {
            id: id as u32,
            name: names.get(id).cloned().unwrap_or_default(),
            color: colors.get(id).copied().unwrap_or([0, 0, 0, 0]),
            tri_colors: sys.panel_tri_colors(id),
            concentration: concs.as_ref().and_then(|v| v.get(id).copied()),
            free_concentration: free_concs.as_ref().and_then(|v| v.get(id).copied()),
            edge_glues,
            edge_glue_ids,
        });
    }

    let mut glues = Vec::new();
    for (id, name) in glue_names.iter().enumerate().skip(1) {
        if name.is_empty() {
            continue;
        }
        glues.push(GlueSnapshot {
            id: id as u32,
            name: name.clone(),
        });
    }

    ModelSnapshot {
        model_name: model_name.to_string(),
        editable,
        schema,
        tiles,
        glues,
        interactions: sys.glue_interactions(),
        blockers: sys.blocker_list(),
        tile_id_shift: sys.canvas_id_shift(),
    }
}

pub(super) fn evolve_in_window_impl<S: System, St: State>(
    sys: &mut S,
    state: &mut St,
    block: Option<usize>,
    start_paused: bool,
    mut bounds: EvolveBounds,
    initial_timescale: Option<f64>,
    initial_max_events_per_sec: Option<u64>,
) -> Result<EvolveOutcome, RgrowError> {
    use crate::ui::ipc::{ControlMessage, InitMessage, UpdateNotification};
    use crate::ui::ipc_server::IpcClient;
    use std::process::{Command, Stdio};

    let debug_perf = std::env::var("RGROW_DEBUG_PERF").is_ok();

    let (width, height) = state.draw_size();
    let tile_colors_vec = sys.tile_colors().clone();

    let scale = block.unwrap_or(12);

    let socket_path = std::env::temp_dir().join(format!("rgrow-gui-{}.sock", std::process::id()));
    let socket_path_str = socket_path.to_string_lossy().to_string();

    // Try to find rgrow GUI command
    let (gui_exe, extra_args) = find_gui_command().ok_or_else(|| {
        RgrowError::IO(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "rgrow GUI binary (version {}) not found. The GUI functionality requires an rgrow binary built with the 'gui' feature.\n\nFor Rust installations, ensure rgrow is built with GUI support:\n  cargo build --package rgrow --features gui\n\nFor Python installations, ensure the rgrow binary is available on PATH.",
                env!("CARGO_PKG_VERSION")
            )
        ))
    })?;

    let mut gui_process = Command::new(&gui_exe)
        .args(&extra_args)
        .arg(&socket_path_str)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| {
            RgrowError::IO(std::io::Error::other(format!(
                "Failed to spawn GUI process: {}. Make sure rgrow is built with the 'gui' feature.",
                e
            )))
        })?;

    std::thread::sleep(Duration::from_millis(100));

    let mut ipc_client = IpcClient::connect(&socket_path).map_err(|e| {
        RgrowError::IO(std::io::Error::other(format!(
            "Failed to connect to GUI: {}",
            e
        )))
    })?;

    let shm_size = (width * height * scale as u32 * scale as u32 * 4) as usize;
    #[cfg(all(unix, not(target_os = "macos")))]
    let shm_path = format!("/dev/shm/rgrow-frame-{}", std::process::id());
    #[cfg(any(windows, target_os = "macos"))]
    let shm_path = std::env::temp_dir()
        .join(format!("rgrow-frame-{}", std::process::id()))
        .to_string_lossy()
        .to_string();

    // Second shm region for the per-frame tile-id grid (`u32` per cell over
    // `raw_array`). The canvas is fixed-size, so this never needs resizing.
    let grid_shm_size = state.raw_array().len() * 4;
    #[cfg(all(unix, not(target_os = "macos")))]
    let grid_shm_path = format!("/dev/shm/rgrow-grid-{}", std::process::id());
    #[cfg(any(windows, target_os = "macos"))]
    let grid_shm_path = std::env::temp_dir()
        .join(format!("rgrow-grid-{}", std::process::id()))
        .to_string_lossy()
        .to_string();

    let has_temperature = sys.get_param("temperature").is_ok();
    let model_name = S::extract_model_name(&sys.system_info());
    let initial_temperature = if has_temperature {
        sys.get_param("temperature")
            .ok()
            .and_then(|v| v.downcast_ref::<f64>().copied())
    } else {
        None
    };

    let mut parameters = sys.list_parameters();
    for param in &mut parameters {
        if let Ok(value) = sys.get_param(&param.name) {
            if let Some(f64_value) = value.downcast_ref::<f64>() {
                param.current_value = *f64_value;
            }
        }
    }

    let init_msg = InitMessage {
        width,
        height,
        tile_colors: tile_colors_vec.clone(),
        block,
        shm_path: shm_path.clone(),
        shm_size,
        grid_shm_path: grid_shm_path.clone(),
        grid_shm_size,
        start_paused,
        model_name: model_name.clone(),
        has_temperature,
        initial_temperature,
        parameters,
        initial_timescale,
        initial_max_events_per_sec,
    };

    ipc_client.send_init(&init_msg).map_err(|e| {
        RgrowError::IO(std::io::Error::other(format!(
            "Failed to send init message: {}",
            e
        )))
    })?;

    // Wait for GUI to signal it's ready (up to 10 seconds)
    ipc_client
        .wait_for_ready(Duration::from_secs(10))
        .map_err(|e| {
            RgrowError::IO(std::io::Error::other(format!(
                "GUI failed to become ready: {}",
                e
            )))
        })?;

    // Push the initial editable-model snapshot so the GUI can build panels.
    let _ = ipc_client.send_snapshot(&build_model_snapshot(sys, &model_name));

    // Control state
    let mut paused = start_paused;
    let mut remaining_step_events: Option<u64> = None;
    let mut max_events_per_sec: Option<u64> = initial_max_events_per_sec;
    let mut timescale: Option<f64> = initial_timescale;
    let mut show_mismatches = true;
    let mut inspection = false;
    let mut snapshot_dirty = false;

    let mut evres: EvolveOutcome = EvolveOutcome::ReachedZeroRate;
    let mut frame_buffer = vec![0u8; shm_size];
    let mut last_frame_time = Instant::now();
    let mut events_this_second: u64 = 0;
    let mut second_start = Instant::now();

    loop {
        // Process control messages
        while let Some(ctrl) = ipc_client.try_recv_control() {
            if debug_perf {
                eprintln!("[Sim] Received control message: {:?}", ctrl);
            }
            match ctrl {
                ControlMessage::Pause => {
                    paused = true;
                    remaining_step_events = None;
                }
                ControlMessage::Resume => {
                    paused = false;
                    remaining_step_events = None;
                }
                ControlMessage::Step { events } => {
                    paused = false;
                    remaining_step_events = Some(events);
                }
                ControlMessage::SetMaxEventsPerSec(max) => {
                    max_events_per_sec = max;
                }
                ControlMessage::SetTimescale(ts) => {
                    timescale = ts;
                }
                ControlMessage::SetTemperature(temp) => {
                    if let Ok(needed) = sys.set_param("temperature", Box::new(temp)) {
                        sys.update_state(state, &needed);
                    }
                }
                ControlMessage::SetParameter { name, value } => {
                    if let Ok(needed) = sys.set_param(&name, Box::new(value)) {
                        sys.update_state(state, &needed);
                    }
                }
                ControlMessage::SetShowMismatches(v) => {
                    show_mismatches = v;
                }
                ControlMessage::SetInspection(v) => {
                    inspection = v;
                }
                ControlMessage::RequestSnapshot => {
                    snapshot_dirty = true;
                }
                ControlMessage::SetTileConcentration { id, value } => {
                    if let Ok(needed) = sys.set_tile_concentration(id as usize, value) {
                        sys.update_state(state, &needed);
                        snapshot_dirty = true;
                    }
                }
                ControlMessage::SetTileEdgeGlue { id, side, glue_id } => {
                    if let Ok(needed) = sys.set_tile_edge_glue(
                        id as usize,
                        side as usize,
                        glue_id.map(|g| g as usize),
                    ) {
                        sys.update_state(state, &needed);
                        snapshot_dirty = true;
                    }
                }
                ControlMessage::SetGlueInteraction { a, b, dg, ds } => {
                    if let Ok(needed) = sys.set_glue_interaction(a as usize, b as usize, dg, ds) {
                        sys.update_state(state, &needed);
                        snapshot_dirty = true;
                    }
                }
                ControlMessage::SetBlockerConcentration { glue_id, value } => {
                    if let Ok(needed) = sys.set_blocker_concentration(glue_id as usize, value) {
                        sys.update_state(state, &needed);
                        snapshot_dirty = true;
                    }
                }
                ControlMessage::SetPoint { x, y, tile } => {
                    // The GUI sends a base id; encode it to the canvas value.
                    let raw = tile << sys.canvas_id_shift();
                    let _ = sys.set_point(state, (y as usize, x as usize), raw);
                }
                ControlMessage::SetPointAtPixel {
                    px,
                    py,
                    scale: pscale,
                    tile,
                } => {
                    let raw = tile << sys.canvas_id_shift();
                    if let Some(p) = state.pixel_to_storage(px, py, pscale) {
                        sys.set_safe_point(state, p, raw);
                    }
                }
                ControlMessage::LoadXgrowSeed {
                    text,
                    offset_i,
                    offset_j,
                } => {
                    if let Ok(grid) = crate::parser_xgrow::parse_xgrow_seed(&text) {
                        let flake_size = grid.len();
                        if flake_size > 0 {
                            let (cw, ch) = state.draw_size();
                            let canvas_w = cw as i32;
                            let canvas_h = ch as i32;
                            let translate_i =
                                (canvas_h - flake_size as i32) / 2 + offset_i.unwrap_or(0);
                            let translate_j =
                                (canvas_w - flake_size as i32) / 2 + offset_j.unwrap_or(0);
                            {
                                let mut arr = state.raw_array_mut();
                                for (i, grow) in grid.iter().enumerate() {
                                    for (j, &t) in grow.iter().enumerate() {
                                        let y = translate_i + i as i32;
                                        let x = translate_j + j as i32;
                                        if y < 0 || y >= canvas_h || x < 0 || x >= canvas_w {
                                            continue;
                                        }
                                        arr[[y as usize, x as usize]] = t;
                                    }
                                }
                            }
                            sys.update_state(state, &NeededUpdate::All);
                        }
                    }
                }
            }
        }

        // Re-send the model snapshot after any edit so derived values
        // (free concentrations, etc.) refresh in the GUI panels.
        if snapshot_dirty {
            let _ = ipc_client.send_snapshot(&build_model_snapshot(sys, &model_name));
            snapshot_dirty = false;
        }

        // Reset events counter each second
        if second_start.elapsed() >= Duration::from_secs(1) {
            events_this_second = 0;
            second_start = Instant::now();
        }

        // Determine if we should run simulation this frame
        let should_run = !paused || remaining_step_events.is_some();

        if should_run {
            // Calculate bounds based on speed settings
            let events_before = state.total_events();

            if let Some(ts) = timescale {
                // Timescale mode: run for (real_elapsed * timescale) simulation time
                let real_elapsed = last_frame_time.elapsed().as_secs_f64();
                let target_sim_time = real_elapsed * ts;
                bounds.for_time = Some(target_sim_time);
                bounds.for_wall_time = None;
                bounds.for_events = remaining_step_events;
            } else if let Some(ref mut step_events) = remaining_step_events {
                // Step mode: run for specified events
                bounds.for_events = Some(*step_events);
                bounds.for_wall_time = Some(Duration::from_millis(16));
                bounds.for_time = None;
            } else {
                // Normal mode
                bounds.for_wall_time = Some(Duration::from_millis(16));
                bounds.for_events = None;
                bounds.for_time = None;
            }

            // Check events per second limit
            if let Some(max_eps) = max_events_per_sec {
                if events_this_second >= max_eps {
                    // Already hit limit this second, skip evolution
                    std::thread::sleep(Duration::from_millis(10));
                } else {
                    let remaining_allowed = max_eps - events_this_second;
                    if let Some(ref mut be) = bounds.for_events {
                        *be = (*be).min(remaining_allowed);
                    } else {
                        bounds.for_events = Some(remaining_allowed);
                    }
                    evres = sys.evolve(state, bounds)?;
                }
            } else {
                evres = sys.evolve(state, bounds)?;
            }

            let events_this_frame = state.total_events() - events_before;
            events_this_second += events_this_frame;

            // Update step counter
            if let Some(ref mut step_events) = remaining_step_events {
                if events_this_frame >= *step_events {
                    remaining_step_events = None;
                    paused = true;
                } else {
                    *step_events -= events_this_frame;
                }
            }
        }

        last_frame_time = Instant::now();

        // Draw frame via the shared painter so the desktop GUI and the
        // wasm bindings stay in sync.
        let frame_width = (width * scale as u32) as usize;
        let frame_height = (height * scale as u32) as usize;
        frame_buffer.resize(frame_width * frame_height * 4, 0);

        let pixel_frame = &mut frame_buffer[..];
        let stats = render_frame(sys, state, scale, show_mismatches, pixel_frame);

        // Geometry (always) and the tile-id grid (only while inspection is
        // on, to avoid the per-frame copy when nothing reads it).
        let grid_rows;
        let grid_cols;
        let subcell_px;
        let tile_shape_diamond;
        let frame_subcells_w;
        let frame_subcells_h;
        let grid_bytes: Option<Vec<u8>>;
        {
            let arr = state.raw_array();
            grid_rows = arr.nrows() as u32;
            grid_cols = arr.ncols() as u32;
            subcell_px = state.tile_size_px(scale as u32);
            tile_shape_diamond = matches!(state.tile_shape(), TileShape::Diamond);
            let (fw, fh) = state.frame_size_subcells();
            frame_subcells_w = fw;
            frame_subcells_h = fh;
            grid_bytes = if inspection {
                Some(arr.iter().flat_map(|&t| t.to_le_bytes()).collect())
            } else {
                None
            };
        }
        // Plain-square canvases map storage (row, col) -> (col*scale,
        // row*scale); the sheared/staggered tube canvases do not, so the GUI
        // must not draw positioned overlays for them.
        let overlay_linear =
            !tile_shape_diamond && frame_subcells_w == grid_cols && frame_subcells_h == grid_rows;

        let notification = UpdateNotification {
            frame_width: stats.frame_width,
            frame_height: stats.frame_height,
            time: stats.time,
            total_events: stats.total_events,
            n_tiles: stats.n_tiles,
            mismatches: stats.mismatch_count,
            energy: stats.energy,
            scale,
            data_len: stats.data_len,
            grid_included: grid_bytes.is_some(),
            grid_rows,
            grid_cols,
            grid_data_len: grid_bytes.as_ref().map_or(0, |g| g.len()),
            subcell_px,
            tile_shape_diamond,
            frame_subcells_w,
            frame_subcells_h,
            overlay_linear,
        };

        let t_send = Instant::now();
        if ipc_client
            .send_frame(pixel_frame, grid_bytes.as_deref(), notification)
            .is_err()
        {
            break;
        }
        let t_send_elapsed = t_send.elapsed();

        if debug_perf {
            eprintln!(
                "[IPC-send] shm write + notify: {:?}, size: {} bytes",
                t_send_elapsed,
                frame_buffer.len()
            );
        }

        std::thread::sleep(Duration::from_millis(16));

        // Only break on terminal conditions if not paused
        // Continue running for: wall time limit, time limit, events limit, zero rate
        // These are all normal "frame complete" conditions
        if !paused && remaining_step_events.is_none() {
            match evres {
                EvolveOutcome::ReachedWallTimeMax => {}
                EvolveOutcome::ReachedTimeMax => {}
                EvolveOutcome::ReachedEventsMax => {}
                EvolveOutcome::ReachedZeroRate => {}
                _ => {
                    break;
                }
            }
        }
    }

    let _ = ipc_client.send_close();
    let _ = gui_process.wait();
    let _ = std::fs::remove_file(&socket_path);

    Ok(evres)
}
