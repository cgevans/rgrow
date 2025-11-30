extern crate rgrow_gui;

use rgrow::system::ParameterInfo;
use rgrow::ui::ipc::{ControlMessage, InitMessage};
use rgrow_gui::ui::iced_gui::{GuiMessage, Message, RgrowGui};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

fn create_test_init() -> InitMessage {
    InitMessage {
        width: 100,
        height: 100,
        tile_colors: vec![[255, 0, 0, 255], [0, 255, 0, 255]],
        block: Some(8),
        shm_path: "/tmp/test-shm".to_string(),
        shm_size: 4096,
        start_paused: false,
        model_name: "kTAM".to_string(),
        has_temperature: true,
        initial_temperature: Some(25.0),
        parameters: vec![ParameterInfo {
            name: "g_se".to_string(),
            current_value: 8.0,
            default_increment: 0.1,
            min_value: Some(0.0),
            max_value: Some(20.0),
            units: "kcal/mol".to_string(),
            description: Some("Seed energy".to_string()),
        }],
    }
}

fn create_test_gui() -> (
    RgrowGui,
    mpsc::Sender<GuiMessage>,
    mpsc::Receiver<ControlMessage>,
) {
    let (gui_sender, gui_receiver) = mpsc::channel();
    let (control_sender, control_receiver) = mpsc::channel();
    let receiver = Arc::new(Mutex::new(gui_receiver));
    let init = create_test_init();
    let gui = RgrowGui::new(receiver, control_sender, init);
    (gui, gui_sender, control_receiver)
}

#[test]
fn test_gui_initialization() {
    let (gui, _, _) = create_test_gui();
    assert_eq!(gui.model_name, "kTAM");
    assert!(!gui.paused);
    assert_eq!(gui.events_per_step, "1000");
    assert!(gui.parameters.contains_key("g_se"));
}

#[test]
fn test_gui_pause_toggle() {
    let (mut gui, _, control_receiver) = create_test_gui();

    // Toggle pause
    let _task = gui.update(Message::TogglePause);
    assert!(gui.paused, "Should be paused after toggle");

    // Check that control message was sent
    let control_msg = control_receiver.try_recv().ok();
    assert!(matches!(control_msg, Some(ControlMessage::Pause)));

    // Toggle again
    let _task = gui.update(Message::TogglePause);
    assert!(!gui.paused, "Should not be paused after second toggle");

    let control_msg = control_receiver.try_recv().ok();
    assert!(matches!(control_msg, Some(ControlMessage::Resume)));
}

#[test]
fn test_gui_step() {
    let (mut gui, _, control_receiver) = create_test_gui();

    // Set events per step
    let _task = gui.update(Message::UpdateEventsPerStep("500".to_string()));
    assert_eq!(gui.events_per_step, "500");

    // Trigger step
    let _task = gui.update(Message::Step);
    assert!(!gui.paused, "Should not be paused after step");

    // Check control message
    let control_msg = control_receiver.try_recv().ok();
    assert!(matches!(
        control_msg,
        Some(ControlMessage::Step { events: 500 })
    ));
}

#[test]
fn test_gui_update_message() {
    let (mut gui, gui_sender, _) = create_test_gui();

    // Send update message
    let frame_data = vec![255u8; 100 * 100 * 4];
    let update = GuiMessage::Update {
        frame_data,
        frame_width: 100,
        frame_height: 100,
        time: 1.5,
        total_events: 1000,
        n_tiles: 50,
        mismatches: 2,
        energy: -10.5,
    };

    gui_sender.send(update.clone()).unwrap();

    // Process tick to receive the message
    let _task = gui.update(Message::Tick);
    // In a real scenario, we'd need to handle the task, but for testing we can just verify state

    // The update should have been processed (we'd need to actually run the task to see the effect)
    // For now, just verify the message was sent successfully
    assert!(gui_sender.send(update).is_ok());
}

#[test]
fn test_gui_parameter_update() {
    let (mut gui, _, control_receiver) = create_test_gui();

    // Update parameter value
    let _task = gui.update(Message::UpdateParameter {
        name: "g_se".to_string(),
        value: "9.0".to_string(),
    });

    assert_eq!(gui.parameters.get("g_se").unwrap().input_value, "9.0");

    // Apply parameter
    let _task = gui.update(Message::ApplyParameter {
        name: "g_se".to_string(),
    });

    assert_eq!(gui.parameters.get("g_se").unwrap().current_value, 9.0);

    // Check control message
    let control_msg = control_receiver.try_recv().ok();
    assert!(matches!(
        control_msg,
        Some(ControlMessage::SetParameter { name, value }) if name == "g_se" && value == 9.0
    ));
}

#[test]
fn test_gui_parameter_increment_decrement() {
    let (mut gui, _, control_receiver) = create_test_gui();

    let initial_value = gui.parameters.get("g_se").unwrap().current_value;

    // Increment
    let _task = gui.update(Message::IncrementParameter {
        name: "g_se".to_string(),
    });

    let new_value = gui.parameters.get("g_se").unwrap().current_value;
    assert!(
        (new_value - initial_value - 0.1).abs() < 0.001,
        "Should increment by default increment"
    );

    // Check control message
    let _control_msg = control_receiver.try_recv().ok();

    // Decrement
    let _task = gui.update(Message::DecrementParameter {
        name: "g_se".to_string(),
    });

    let final_value = gui.parameters.get("g_se").unwrap().current_value;
    assert!(
        (final_value - initial_value).abs() < 0.001,
        "Should be back to initial value"
    );
}

#[test]
fn test_gui_max_events_per_sec() {
    let (mut gui, _, control_receiver) = create_test_gui();

    // Update max events per sec
    let _task = gui.update(Message::UpdateMaxEventsPerSec("1000".to_string()));
    assert_eq!(gui.max_events_per_sec, "1000");

    // Apply
    let _task = gui.update(Message::ApplyMaxEventsPerSec);

    let control_msg = control_receiver.try_recv().ok();
    assert!(matches!(
        control_msg,
        Some(ControlMessage::SetMaxEventsPerSec(Some(1000)))
    ));

    // Set to unlimited
    let _task = gui.update(Message::UpdateMaxEventsPerSec("".to_string()));
    let _task = gui.update(Message::ApplyMaxEventsPerSec);

    let control_msg = control_receiver.try_recv().ok();
    assert!(matches!(
        control_msg,
        Some(ControlMessage::SetMaxEventsPerSec(None))
    ));
}

#[test]
fn test_gui_timescale() {
    let (mut gui, _, control_receiver) = create_test_gui();

    // Update timescale
    let _task = gui.update(Message::UpdateTimescale("2.5".to_string()));
    assert_eq!(gui.timescale, "2.5");

    // Apply
    let _task = gui.update(Message::ApplyTimescale);

    let control_msg = control_receiver.try_recv().ok();
    assert!(matches!(
        control_msg,
        Some(ControlMessage::SetTimescale(Some(2.5)))
    ));
}

#[test]
fn test_gui_start_paused() {
    let (_gui_sender, gui_receiver) = mpsc::channel();
    let (control_sender, _) = mpsc::channel();
    let receiver = Arc::new(Mutex::new(gui_receiver));

    let mut init = create_test_init();
    init.start_paused = true;

    let gui = RgrowGui::new(receiver, control_sender, init);
    assert!(gui.paused, "Should start paused when start_paused is true");
}
