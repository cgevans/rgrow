#[cfg(feature = "ui")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rgrow::ui::iced_gui;
    use rgrow::ui::ipc::IpcMessage;
    use std::env;
    use std::io::Read;
    use std::os::unix::net::UnixListener;
    use std::sync::mpsc;
    use std::sync::{Arc, Mutex};
    use std::thread;

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
        let init_msg = bincode::deserialize(&init_buffer[..len])?;

        if let IpcMessage::Init(init) = init_msg {
            let stream_clone = stream.clone();
            let sender_clone = sender.clone();

            thread::spawn(move || {
                let mut buffer = vec![0u8; 1024 * 1024 * 10];
                loop {
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

                    match bincode::deserialize::<IpcMessage>(&buffer[..len]) {
                        Ok(msg) => {
                            let is_close = matches!(&msg, IpcMessage::Close);
                            if sender_clone.send(msg).is_err() {
                                break;
                            }
                            if is_close {
                                break;
                            }
                        }
                        Err(_) => break,
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
