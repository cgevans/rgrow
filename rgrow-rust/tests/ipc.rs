extern crate rgrow;

use rgrow::system::ParameterInfo;
use rgrow::ui::ipc::{ControlMessage, InitMessage, IpcMessage, ResizeMessage, UpdateNotification};

#[test]
fn test_ipc_message_serialization() {
    // Test all IpcMessage variants
    let messages = vec![
        IpcMessage::Ready,
        IpcMessage::Close,
        IpcMessage::Init(InitMessage {
            width: 100,
            height: 200,
            tile_colors: vec![[255, 0, 0, 255], [0, 255, 0, 255]],
            block: Some(8),
            shm_path: "/dev/shm/test".to_string(),
            shm_size: 1024 * 1024,
            start_paused: true,
            model_name: "kTAM".to_string(),
            has_temperature: true,
            initial_temperature: Some(25.0),
            parameters: vec![],
            initial_timescale: None,
            initial_max_events_per_sec: None,
        }),
        IpcMessage::Update(UpdateNotification {
            frame_width: 100,
            frame_height: 200,
            time: 1.5,
            total_events: 1000,
            n_tiles: 50,
            mismatches: 2,
            energy: -10.5,
            scale: 8,
            data_len: 80000,
        }),
        IpcMessage::Resize(ResizeMessage {
            width: 200,
            height: 300,
        }),
    ];

    for msg in messages {
        let serialized = bincode::serialize(&msg).expect("Failed to serialize");
        let deserialized: IpcMessage =
            bincode::deserialize(&serialized).expect("Failed to deserialize");
        assert_eq!(format!("{:?}", msg), format!("{:?}", deserialized));
    }
}

#[test]
fn test_control_message_serialization() {
    let messages = vec![
        ControlMessage::Pause,
        ControlMessage::Resume,
        ControlMessage::Step { events: 100 },
        ControlMessage::SetMaxEventsPerSec(Some(1000)),
        ControlMessage::SetMaxEventsPerSec(None),
        ControlMessage::SetTimescale(Some(2.5)),
        ControlMessage::SetTimescale(None),
        ControlMessage::SetTemperature(25.0),
        ControlMessage::SetParameter {
            name: "temperature".to_string(),
            value: 30.0,
        },
    ];

    for msg in messages {
        let serialized = bincode::serialize(&msg).expect("Failed to serialize");
        let deserialized: ControlMessage =
            bincode::deserialize(&serialized).expect("Failed to deserialize");
        assert_eq!(format!("{:?}", msg), format!("{:?}", deserialized));
    }
}

#[test]
fn test_init_message_with_parameters() {
    let init = InitMessage {
        width: 64,
        height: 64,
        tile_colors: vec![[0, 0, 0, 255], [255, 255, 255, 255]],
        block: None,
        shm_path: "/tmp/test-shm".to_string(),
        shm_size: 4096,
        start_paused: false,
        model_name: "aTAM".to_string(),
        has_temperature: false,
        initial_temperature: None,
        parameters: vec![
            ParameterInfo {
                name: "g_se".to_string(),
                current_value: 8.0,
                default_increment: 0.1,
                min_value: Some(0.0),
                max_value: Some(20.0),
                units: "kcal/mol".to_string(),
                description: Some("Seed energy".to_string()),
            },
            ParameterInfo {
                name: "g_mc".to_string(),
                current_value: 16.0,
                default_increment: 0.5,
                min_value: None,
                max_value: None,
                units: "kcal/mol".to_string(),
                description: None,
            },
        ],
        initial_timescale: None,
        initial_max_events_per_sec: None,
    };

    let serialized = bincode::serialize(&init).expect("Failed to serialize");
    let deserialized: InitMessage =
        bincode::deserialize(&serialized).expect("Failed to deserialize");

    assert_eq!(init.width, deserialized.width);
    assert_eq!(init.height, deserialized.height);
    assert_eq!(init.parameters.len(), deserialized.parameters.len());
    assert_eq!(init.parameters[0].name, deserialized.parameters[0].name);
    assert_eq!(
        init.parameters[0].current_value,
        deserialized.parameters[0].current_value
    );
}
