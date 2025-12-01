// UI module is now in lib.rs for testing
use rgrow_gui::ui;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use crate::ui::iced_gui;
    use crate::ui::shm_reader::ShmReader;
    #[cfg(windows)]
    use named_pipe::{PipeClient, PipeOptions, PipeServer};
    use rgrow_ipc::{ControlMessage, IpcMessage};
    use std::env;
    use std::io::{Read, Write};
    #[cfg(unix)]
    use std::os::unix::net::UnixListener;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        println!("rgrow-gui {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if args.len() < 2 {
        eprintln!("Usage: rgrow-gui <socket_path>");
        std::process::exit(1);
    }

    let socket_path = &args[1];

    #[cfg(unix)]
    let listener = UnixListener::bind(socket_path)?;

    #[cfg(windows)]
    let listener = {
        let pipe_name = format!(
            r"\\.\pipe\{}",
            socket_path.replace('/', "_").replace('\\', "_")
        );
        PipeOptions::new(pipe_name.as_str())?.single()?
    };

    // Channel for GUI updates (IPC -> GUI)
    let (update_sender, update_receiver) = mpsc::channel();
    let update_receiver = Arc::new(Mutex::new(update_receiver));

    // Channel for control messages (GUI -> IPC)
    let (control_sender, control_receiver) = mpsc::channel::<ControlMessage>();
    let control_receiver = Arc::new(Mutex::new(control_receiver));

    let socket_path_clone = socket_path.to_string();

    #[cfg(unix)]
    let stream = {
        if let Ok((stream, _)) = listener.accept() {
            Some(Arc::new(Mutex::new(stream)))
        } else {
            None
        }
    };

    #[cfg(windows)]
    let stream = {
        match listener.wait() {
            Ok(stream) => Some(Arc::new(Mutex::<PipeClient>::new(stream))),
            Err(_) => None,
        }
    };

    if let Some(stream) = stream {
        let mut init_buffer = vec![0u8; 1024 * 1024];
        let mut len_bytes = [0u8; 8];
        {
            let mut stream_guard = stream.lock().unwrap();
            stream_guard.read_exact(&mut len_bytes)?;
        }
        let len = u64::from_le_bytes(len_bytes) as usize;
        if len > init_buffer.len() {
            init_buffer.resize(len, 0);
        }
        {
            let mut stream_guard = stream.lock().unwrap();
            stream_guard.read_exact(&mut init_buffer[..len])?;
        }
        let init_msg: IpcMessage = bincode::deserialize(&init_buffer[..len])?;

        if let IpcMessage::Init(init) = init_msg {
            let shm_path = init.shm_path.clone();
            let shm_size = init.shm_size;

            // Send Ready signal immediately - before spawning reader thread
            {
                let msg = IpcMessage::Ready;
                let serialized = bincode::serialize(&msg)?;
                let len = serialized.len() as u64;
                let mut stream_guard = stream.lock().unwrap();
                stream_guard.write_all(&len.to_le_bytes())?;
                stream_guard.write_all(&serialized)?;
                stream_guard.flush()?;
            }

            let stream_for_read = stream.clone();
            let stream_for_control = stream.clone();
            let update_sender_clone = update_sender.clone();
            let control_receiver_clone = control_receiver.clone();

            // Thread to read updates from simulation
            thread::spawn(move || {
                let debug = std::env::var("RGROW_DEBUG_PERF").is_ok();
                if debug {
                    eprintln!(
                        "[IPC-thread] Starting, shm_path={}, shm_size={}",
                        shm_path, shm_size
                    );
                }
                let mut buffer = vec![0u8; 1024 * 64];

                let shm_reader = match ShmReader::open(&shm_path, shm_size) {
                    Ok(r) => {
                        if debug {
                            eprintln!("[IPC-thread] Shared memory opened successfully");
                        }
                        r
                    }
                    Err(e) => {
                        eprintln!("[IPC-thread] Failed to open shared memory: {}", e);
                        return;
                    }
                };

                loop {
                    let t0 = Instant::now();
                    let mut len_bytes = [0u8; 8];
                    let mut stream_guard = stream_for_read.lock().unwrap();
                    if stream_guard.read_exact(&mut len_bytes).is_err() {
                        break;
                    }
                    let len = u64::from_le_bytes(len_bytes) as usize;
                    if len > buffer.len() {
                        buffer.resize(len, 0);
                    }
                    if stream_guard.read_exact(&mut buffer[..len]).is_err() {
                        break;
                    }
                    drop(stream_guard);
                    let t_read = t0.elapsed();

                    let t1 = Instant::now();
                    match bincode::deserialize::<IpcMessage>(&buffer[..len]) {
                        Ok(msg) => {
                            let t_deser = t1.elapsed();

                            let msg_to_send = match &msg {
                                IpcMessage::Update(notif) => {
                                    let t_shm = Instant::now();
                                    let frame_data = shm_reader.read(notif.data_len).to_vec();
                                    let t_shm_elapsed = t_shm.elapsed();

                                    if debug {
                                        eprintln!(
                                            "[IPC-recv] read: {:?}, deser: {:?}, shm read: {:?}, size: {} bytes",
                                            t_read, t_deser, t_shm_elapsed, notif.data_len
                                        );
                                    }

                                    iced_gui::GuiMessage::Update {
                                        frame_data,
                                        frame_width: notif.frame_width,
                                        frame_height: notif.frame_height,
                                        time: notif.time,
                                        total_events: notif.total_events,
                                        n_tiles: notif.n_tiles,
                                        mismatches: notif.mismatches,
                                        energy: notif.energy,
                                    }
                                }
                                IpcMessage::Close => iced_gui::GuiMessage::Close,
                                IpcMessage::Init(_) => continue,
                                IpcMessage::Resize(_) => continue,
                                IpcMessage::Ready => continue,
                            };

                            let is_close = matches!(&msg, IpcMessage::Close);
                            if debug {
                                eprintln!(
                                    "[IPC-thread] Sending message to GUI channel, is_close={}",
                                    is_close
                                );
                            }
                            if update_sender_clone.send(msg_to_send).is_err() {
                                if debug {
                                    eprintln!("[IPC-thread] Failed to send to channel");
                                }
                                break;
                            }
                            if is_close {
                                break;
                            }
                        }
                        Err(e) => {
                            if debug {
                                eprintln!("[IPC-thread] Deserialize error: {}", e);
                            }
                            break;
                        }
                    }
                }
                #[cfg(unix)]
                {
                    let _ = std::fs::remove_file(&socket_path_clone);
                }
            });

            // Thread to forward control messages from GUI to simulation
            thread::spawn(move || {
                let debug = std::env::var("RGROW_DEBUG_PERF").is_ok();
                loop {
                    let control_msg = {
                        let control_recv = control_receiver_clone.lock().unwrap();
                        match control_recv.try_recv() {
                            Ok(msg) => Some(msg),
                            Err(mpsc::TryRecvError::Empty) => {
                                drop(control_recv);
                                std::thread::sleep(Duration::from_millis(10));
                                continue;
                            }
                            Err(mpsc::TryRecvError::Disconnected) => {
                                break;
                            }
                        }
                    };

                    if let Some(control_msg) = control_msg {
                        if debug {
                            eprintln!(
                                "[Control-thread] Sending control message: {:?}",
                                control_msg
                            );
                        }
                        if let Ok(serialized) = bincode::serialize(&control_msg) {
                            let len = serialized.len() as u64;
                            let mut stream_guard = stream_for_control.lock().unwrap();
                            if stream_guard.write_all(&len.to_le_bytes()).is_err() {
                                break;
                            }
                            if stream_guard.write_all(&serialized).is_err() {
                                break;
                            }
                            let _ = stream_guard.flush();
                        }
                    }
                }
            });

            iced_gui::run_gui(update_receiver, control_sender, init)?;
        }
    }

    Ok(())
}
