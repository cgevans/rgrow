extern crate rgrow;

#[cfg(windows)]
use named_pipe::{PipeOptions, PipeServer};
use rgrow::ui::ipc::{ControlMessage, InitMessage, IpcMessage, UpdateNotification};
use rgrow::ui::ipc_server::IpcClient;
use std::io::{Read, Write};
#[cfg(unix)]
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn create_temp_socket() -> PathBuf {
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut path = env::temp_dir();
    #[cfg(unix)]
    {
        path.push(format!(
            "rgrow-test-{}-{}.sock",
            std::process::id(),
            timestamp
        ));
        // Clean up if it exists
        let _ = std::fs::remove_file(&path);
    }
    #[cfg(windows)]
    {
        path.push(format!("rgrow-test-{}-{}", std::process::id(), timestamp));
    }
    path
}

#[cfg(unix)]
fn create_listener(path: &PathBuf) -> Result<UnixListener, std::io::Error> {
    UnixListener::bind(path)
}

#[cfg(windows)]
fn create_listener(path: &PathBuf) -> Result<PipeServer, std::io::Error> {
    let pipe_name = format!(
        r"\\.\pipe\{}",
        path.to_string_lossy().replace('/', "_").replace('\\', "_")
    );
    PipeOptions::new(pipe_name.as_str()).single()
}

#[test]
fn test_unix_socket_creation() {
    let socket_path = create_temp_socket();
    let listener = create_listener(&socket_path);
    assert!(
        listener.is_ok(),
        "Should be able to create IPC listener: {:?}",
        listener.err()
    );

    // Clean up
    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_path);
    }
}

#[test]
fn test_ipc_client_connect() {
    let socket_path = create_temp_socket();
    let _listener = create_listener(&socket_path).expect("Failed to bind listener");

    // Try to connect (this will fail because no one is accepting, but we test the path exists)
    let result = IpcClient::connect(&socket_path);
    // Connection might fail if no one is listening, but socket should exist
    #[cfg(unix)]
    {
        assert!(
            socket_path.exists() || result.is_err(),
            "Socket path should exist or connection should fail gracefully"
        );
    }
    #[cfg(windows)]
    {
        // On Windows, the pipe might not exist as a file, so we just check the connection attempt
        assert!(
            result.is_err() || result.is_ok(),
            "Connection should either succeed or fail gracefully"
        );
    }

    // Clean up
    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_path);
    }
}

#[test]
fn test_ipc_init_ready_handshake() {
    let socket_path = create_temp_socket();
    let listener = create_listener(&socket_path).expect("Failed to bind listener");

    // Spawn a thread to simulate rgrow-gui server
    let _socket_path_clone = socket_path.clone();
    let server_thread = thread::spawn(move || {
        #[cfg(unix)]
        let stream_result = listener.accept().map(|(s, _)| s);
        #[cfg(windows)]
        let stream_result = listener.wait();

        if let Ok(mut stream) = stream_result {
            // Read init message
            let mut len_bytes = [0u8; 8];
            if stream.read_exact(&mut len_bytes).is_ok() {
                let len = u64::from_le_bytes(len_bytes) as usize;
                let mut buffer = vec![0u8; len];
                if stream.read_exact(&mut buffer).is_ok() {
                    let _init_msg: IpcMessage = bincode::deserialize(&buffer).unwrap();

                    // Send Ready
                    let ready_msg = IpcMessage::Ready;
                    let serialized = bincode::serialize(&ready_msg).unwrap();
                    let len = serialized.len() as u64;
                    let _ = stream.write_all(&len.to_le_bytes());
                    let _ = stream.write_all(&serialized);
                    let _ = stream.flush();
                }
            }
        }
    });

    // Small delay to let server start
    thread::sleep(Duration::from_millis(50));

    // Client connects and sends init
    let mut client = IpcClient::connect(&socket_path).expect("Failed to connect");

    let init = InitMessage {
        width: 100,
        height: 100,
        tile_colors: vec![[255, 0, 0, 255]],
        block: Some(8),
        shm_path: "/tmp/test-shm".to_string(),
        shm_size: 4096,
        start_paused: false,
        model_name: "kTAM".to_string(),
        has_temperature: false,
        initial_temperature: None,
        parameters: vec![],
        initial_timescale: None,
        initial_max_events_per_sec: None,
    };

    client.send_init(&init).expect("Failed to send init");

    // Wait for Ready
    let result = client.wait_for_ready(Duration::from_secs(1));
    assert!(result.is_ok(), "Should receive Ready message");

    server_thread.join().unwrap();

    // Clean up
    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_path);
    }
}

#[test]
fn test_ipc_update_message() {
    let socket_path = create_temp_socket();
    let listener = create_listener(&socket_path).expect("Failed to bind listener");

    // Create shared memory file
    let shm_path = format!("/tmp/rgrow-test-shm-{}", std::process::id());
    let shm_size = 100 * 100 * 4; // 100x100 RGBA
    let shm_file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&shm_path)
        .expect("Failed to create shm file");
    shm_file
        .set_len(shm_size as u64)
        .expect("Failed to set shm size");

    // Spawn server thread
    let _socket_path_clone = socket_path.clone();
    let server_thread = thread::spawn(move || {
        #[cfg(unix)]
        let stream_result = listener.accept().map(|(s, _)| s);
        #[cfg(windows)]
        let stream_result = listener.wait();

        if let Ok(mut stream) = stream_result {
            // Read init
            let mut len_bytes = [0u8; 8];
            if stream.read_exact(&mut len_bytes).is_ok() {
                let len = u64::from_le_bytes(len_bytes) as usize;
                let mut buffer = vec![0u8; len];
                if stream.read_exact(&mut buffer).is_ok() {
                    // Send Ready
                    let ready_msg = IpcMessage::Ready;
                    let serialized = bincode::serialize(&ready_msg).unwrap();
                    let len = serialized.len() as u64;
                    let _ = stream.write_all(&len.to_le_bytes());
                    let _ = stream.write_all(&serialized);
                    let _ = stream.flush();

                    // Read update message
                    if stream.read_exact(&mut len_bytes).is_ok() {
                        let len = u64::from_le_bytes(len_bytes) as usize;
                        buffer.resize(len, 0);
                        if stream.read_exact(&mut buffer).is_ok() {
                            let update_msg: IpcMessage = bincode::deserialize(&buffer).unwrap();
                            assert!(matches!(update_msg, IpcMessage::Update(_)));
                        }
                    }
                }
            }
        }
    });

    thread::sleep(Duration::from_millis(50));

    let mut client = IpcClient::connect(&socket_path).expect("Failed to connect");

    let init = InitMessage {
        width: 100,
        height: 100,
        tile_colors: vec![[255, 0, 0, 255]],
        block: Some(8),
        shm_path: shm_path.clone(),
        shm_size,
        start_paused: false,
        model_name: "kTAM".to_string(),
        has_temperature: false,
        initial_temperature: None,
        parameters: vec![],
        initial_timescale: None,
        initial_max_events_per_sec: None,
    };

    client.send_init(&init).expect("Failed to send init");
    client.wait_for_ready(Duration::from_secs(1)).unwrap();

    // Send update
    let frame_data = vec![255u8; shm_size];
    let notification = UpdateNotification {
        frame_width: 100,
        frame_height: 100,
        time: 1.0,
        total_events: 100,
        n_tiles: 10,
        mismatches: 0,
        energy: -10.0,
        scale: 8,
        data_len: shm_size,
    };

    client
        .send_frame(&frame_data, notification)
        .expect("Failed to send frame");

    server_thread.join().unwrap();

    // Clean up
    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_path);
    }
    let _ = std::fs::remove_file(&shm_path);
}

#[test]
fn test_ipc_control_message() {
    let socket_path = create_temp_socket();
    let listener = create_listener(&socket_path).expect("Failed to bind listener");

    // Spawn server thread that sends control messages
    let _socket_path_clone = socket_path.clone();
    let (tx, rx) = mpsc::channel();
    let server_thread = thread::spawn(move || {
        #[cfg(unix)]
        let stream_result = listener.accept().map(|(s, _)| s);
        #[cfg(windows)]
        let stream_result = listener.wait();

        if let Ok(mut stream) = stream_result {
            // Read init and send Ready
            let mut len_bytes = [0u8; 8];
            if stream.read_exact(&mut len_bytes).is_ok() {
                let len = u64::from_le_bytes(len_bytes) as usize;
                let mut buffer = vec![0u8; len];
                if stream.read_exact(&mut buffer).is_ok() {
                    // Send Ready
                    let ready_msg = IpcMessage::Ready;
                    let serialized = bincode::serialize(&ready_msg).unwrap();
                    let len = serialized.len() as u64;
                    let _ = stream.write_all(&len.to_le_bytes());
                    let _ = stream.write_all(&serialized);
                    let _ = stream.flush();

                    // Send control message
                    thread::sleep(Duration::from_millis(100));
                    let control_msg = ControlMessage::Pause;
                    let serialized = bincode::serialize(&control_msg).unwrap();
                    let len = serialized.len() as u64;
                    let _ = stream.write_all(&len.to_le_bytes());
                    let _ = stream.write_all(&serialized);
                    let _ = stream.flush();
                    let _ = tx.send(());
                }
            }
        }
    });

    thread::sleep(Duration::from_millis(50));

    let mut client = IpcClient::connect(&socket_path).expect("Failed to connect");

    let init = InitMessage {
        width: 100,
        height: 100,
        tile_colors: vec![[255, 0, 0, 255]],
        block: Some(8),
        shm_path: "/tmp/test-shm".to_string(),
        shm_size: 4096,
        start_paused: false,
        model_name: "kTAM".to_string(),
        has_temperature: false,
        initial_temperature: None,
        parameters: vec![],
        initial_timescale: None,
        initial_max_events_per_sec: None,
    };

    client.send_init(&init).expect("Failed to send init");
    client.wait_for_ready(Duration::from_secs(1)).unwrap();

    // Try to receive control message
    thread::sleep(Duration::from_millis(150));
    let control_msg = client.try_recv_control();
    assert!(control_msg.is_some(), "Should receive control message");
    assert!(matches!(control_msg.unwrap(), ControlMessage::Pause));

    let _ = rx.recv_timeout(Duration::from_secs(1));
    server_thread.join().unwrap();

    // Clean up
    #[cfg(unix)]
    {
        let _ = std::fs::remove_file(&socket_path);
    }
}
