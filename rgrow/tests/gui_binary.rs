extern crate rgrow;

use rgrow::system;
use std::process::Command;

#[test]
fn test_find_gui_binary_in_target_dir() {
    // Try to find the binary in the target directory (where it would be after building)
    let exe_path = std::env::current_exe().expect("Failed to get current exe path");
    let target_dir = exe_path
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent());

    if let Some(target_dir) = target_dir {
        let gui_exe = target_dir.join("rgrow-gui");
        #[cfg(windows)]
        let gui_exe = target_dir.join("rgrow-gui.exe");

        // Only test if the binary actually exists (i.e., if rgrow-gui was built)
        if gui_exe.exists() {
            // Build rgrow-gui first to ensure it exists
            let build_status = Command::new("cargo")
                .args(["build", "--package", "rgrow-gui"])
                .status();

            if build_status.is_ok() && build_status.unwrap().success() {
                let found = system::find_gui_binary();
                assert!(
                    found.is_some(),
                    "Should find rgrow-gui binary in target directory"
                );
                let found_path = found.unwrap();
                assert!(found_path.exists(), "Found binary path should exist");
            }
        }
    }
}

#[test]
fn test_find_gui_binary_version_check() {
    // Test that version checking works
    // First, try to build rgrow-gui
    let build_status = Command::new("cargo")
        .args(["build", "--package", "rgrow-gui"])
        .status();

    if build_status.is_ok() && build_status.unwrap().success() {
        if let Some(gui_path) = system::find_gui_binary() {
            // Test that the binary returns the correct version
            let output = Command::new(&gui_path).arg("--version").output();

            if let Ok(output) = output {
                let version_str = String::from_utf8_lossy(&output.stdout);
                let expected_version = env!("CARGO_PKG_VERSION");
                assert!(
                    version_str.contains(expected_version),
                    "Version output should contain expected version. Got: {}, Expected: {}",
                    version_str,
                    expected_version
                );
            }
        }
    }
}

#[test]
fn test_find_gui_binary_not_found_handling() {
    // This test verifies that the function handles the case when binary is not found
    // We can't easily test this without mocking, but we can at least verify
    // the function doesn't panic
    let _result = system::find_gui_binary();
    // If binary is not found, result should be None, which is fine
    // The function should not panic in any case
}

#[test]
fn test_gui_binary_executable() {
    // Test that if we find the binary, it's actually executable
    if let Some(gui_path) = system::find_gui_binary() {
        // Test --version flag
        let output = Command::new(&gui_path).arg("--version").output();

        assert!(
            output.is_ok(),
            "rgrow-gui binary should be executable and respond to --version"
        );

        let output = output.unwrap();
        assert!(
            output.status.success(),
            "rgrow-gui --version should succeed"
        );

        // Test that it requires socket path argument
        let output = Command::new(&gui_path).output();

        if let Ok(output) = output {
            // Should exit with error code when no arguments provided
            assert!(
                !output.status.success(),
                "rgrow-gui should fail when no socket path is provided"
            );
        }
    }
}
