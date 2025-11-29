#[cfg(feature = "ui")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rgrow::ui::iced_gui;
    use rgrow::ui::ipc::IpcMessage;
    use rgrow::ui::ipc_server::ShmReader;
    use std::env;
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Instant;

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rgrow-gui <socket_path>");
        std::process::exit(1);
    }

    let socket_path = &args[1];
    let listener = UnixListener::bind(socket_path)?;

    let (sender, receiver) = mpsc::channel();
    let receiver = Arc::new(Mutex::new(receiver));

    let socket_path_clone = socket_path.to_string();

    if let Ok((stream, _)) = listener.accept() {
        let stream = Arc::new(Mutex::new(stream));

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
            // This avoids deadlock where reader holds lock while blocking on read
            {
                let msg = IpcMessage::Ready;
                let serialized = bincode::serialize(&msg)?;
                let len = serialized.len() as u64;
                let mut stream_guard = stream.lock().unwrap();
                stream_guard.write_all(&len.to_le_bytes())?;
                stream_guard.write_all(&serialized)?;
                stream_guard.flush()?;
            }

            let stream_clone = stream.clone();
            let sender_clone = sender.clone();

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
                    let mut stream_guard = stream_clone.lock().unwrap();
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
                            if sender_clone.send(msg_to_send).is_err() {
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
                let _ = std::fs::remove_file(&socket_path_clone);
            });

            iced_gui::run_gui(receiver, init)?;
        }
    }

    Ok(())
}

#[cfg(not(feature = "ui"))]
fn main() {
    eprintln!("UI feature not enabled");
    std::process::exit(1);
}
