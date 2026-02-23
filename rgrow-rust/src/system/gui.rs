use std::time::{Duration, Instant};

use crate::base::RgrowError;
use crate::canvas::PointSafeHere;
use crate::state::State;

use crate::ui::find_gui_command;

use super::core::System;
use super::types::*;

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

    let socket_path =
        std::env::temp_dir().join(format!("rgrow-gui-{}.sock", std::process::id()));
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
        start_paused,
        model_name,
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

    // Control state
    let mut paused = start_paused;
    let mut remaining_step_events: Option<u64> = None;
    let mut max_events_per_sec: Option<u64> = initial_max_events_per_sec;
    let mut timescale: Option<f64> = initial_timescale;
    let mut show_mismatches = true;

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
            }
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

        // Draw frame
        let edge_size = scale / 10;
        let _tile_size = scale - 2 * edge_size;
        let frame_width = (width * scale as u32) as usize;
        let frame_height = (height * scale as u32) as usize;
        frame_buffer.resize(frame_width * frame_height * 4, 0);

        let pixel_frame = &mut frame_buffer[..];

        for ((y, x), &tileid) in state.raw_array().indexed_iter() {
            let sprite = sys.tile_pixels(tileid, scale);
            state.draw_sprite(pixel_frame, sprite, PointSafeHere((y, x)));
        }

        // Draw thin outlines around non-empty tiles to distinguish them
        if scale >= 12 {
            use crate::painter::draw_rect;
            let outline_color = [0u8, 0, 0, 255];
            for ((y, x), &tileid) in state.raw_array().indexed_iter() {
                if tileid == 0 {
                    continue;
                }
                let tile_x = x * scale;
                let tile_y = y * scale;
                draw_rect(pixel_frame, tile_x, tile_x + scale, tile_y, tile_y + 1, outline_color, frame_width);
                draw_rect(pixel_frame, tile_x, tile_x + scale, tile_y + scale - 1, tile_y + scale, outline_color, frame_width);
                draw_rect(pixel_frame, tile_x, tile_x + 1, tile_y, tile_y + scale, outline_color, frame_width);
                draw_rect(pixel_frame, tile_x + scale - 1, tile_x + scale, tile_y, tile_y + scale, outline_color, frame_width);
            }
        }

        // Draw blocker rectangles protruding outside tile edges
        {
            use crate::painter::draw_rect;
            let depth = (scale / 3).max(2); // how far the rectangle protrudes outward
            let half_len = (scale / 3).max(2); // half-length along the edge
            let blocker_color = [140, 140, 140, 255];
            for ((y, x), &tileid) in state.raw_array().indexed_iter() {
                let mask = sys.tile_blocker_mask(tileid);
                if mask == 0 {
                    continue;
                }
                let tile_x = x * scale;
                let tile_y = y * scale;
                let mid_x = tile_x + scale / 2;
                let mid_y = tile_y + scale / 2;
                // North blocker: rectangle above tile
                if mask & 0b0001 != 0 {
                    draw_rect(
                        pixel_frame,
                        mid_x.saturating_sub(half_len),
                        mid_x + half_len,
                        tile_y.saturating_sub(depth),
                        tile_y,
                        blocker_color,
                        frame_width,
                    );
                }
                // East blocker: rectangle to the right
                if mask & 0b0010 != 0 {
                    let right = tile_x + scale;
                    draw_rect(
                        pixel_frame,
                        right,
                        (right + depth).min(frame_width),
                        mid_y.saturating_sub(half_len),
                        mid_y + half_len,
                        blocker_color,
                        frame_width,
                    );
                }
                // South blocker: rectangle below tile
                if mask & 0b0100 != 0 {
                    let bottom = tile_y + scale;
                    draw_rect(
                        pixel_frame,
                        mid_x.saturating_sub(half_len),
                        mid_x + half_len,
                        bottom,
                        (bottom + depth).min(frame_height),
                        blocker_color,
                        frame_width,
                    );
                }
                // West blocker: rectangle to the left
                if mask & 0b1000 != 0 {
                    draw_rect(
                        pixel_frame,
                        tile_x.saturating_sub(depth),
                        tile_x,
                        mid_y.saturating_sub(half_len),
                        mid_y + half_len,
                        blocker_color,
                        frame_width,
                    );
                }
            }
        }

        // Compute mismatch locations and derive count
        let (mismatch_count, mismatch_locs) = if show_mismatches {
            let locs = sys.calc_mismatch_locations(state);
            let count: u32 = locs
                .iter()
                .map(|x| ((x & 0b01) + ((x & 0b10) >> 1)) as u32)
                .sum();
            (count, Some(locs))
        } else {
            (sys.calc_mismatches(state) as u32, None)
        };

        // Draw mismatch markers if enabled
        if let Some(ref locs) = mismatch_locs {
            use crate::painter::draw_rect;
            // `thick`: pixels straddling each side of the boundary
            // `long`: half-length along the edge (from tile center outward)
            let thick = (scale / 4).max(1);
            let long = (scale / 3).max(1);
            let color = [255, 0, 0, 255]; // red
            for ((y, x), &mm) in locs.indexed_iter() {
                if mm == 0 {
                    continue;
                }
                // S mismatch: horizontal bar straddling bottom edge
                if mm & 0b0010 != 0 {
                    let edge_y = y * scale + scale;
                    let mid_x = x * scale + scale / 2;
                    draw_rect(
                        pixel_frame,
                        mid_x.saturating_sub(long),
                        mid_x + long,
                        edge_y.saturating_sub(thick),
                        edge_y + thick,
                        color,
                        frame_width,
                    );
                }
                // W mismatch: vertical bar straddling left edge
                if mm & 0b0001 != 0 {
                    let edge_x = x * scale;
                    let mid_y = y * scale + scale / 2;
                    draw_rect(
                        pixel_frame,
                        edge_x.saturating_sub(thick),
                        edge_x + thick,
                        mid_y.saturating_sub(long),
                        mid_y + long,
                        color,
                        frame_width,
                    );
                }
            }
        }

        let notification = UpdateNotification {
            frame_width: frame_width as u32,
            frame_height: frame_height as u32,
            time: state.time().into(),
            total_events: state.total_events(),
            n_tiles: state.n_tiles(),
            mismatches: mismatch_count,
            energy: state.energy(),
            scale,
            data_len: pixel_frame.len(),
        };

        let t_send = Instant::now();
        if ipc_client.send_frame(pixel_frame, notification).is_err() {
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

