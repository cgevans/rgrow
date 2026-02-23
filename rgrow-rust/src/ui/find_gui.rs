/// Find the GUI command to spawn. Returns `(executable_path, extra_args)`.
///
/// Search order:
/// 1. Self-spawn (non-Python build with gui feature): use current_exe() + ["gui-subprocess"]
/// 2. External `rgrow` binary: search package dir (Python), RGROW_PACKAGE_DIR, PATH, exe dir,
///    version-checked with `rgrow gui-subprocess --version`.
/// 3. Legacy `rgrow-gui` binary fallback: PATH, package dir (Python), exe dir.
///    Returns `(path, vec![])` since `rgrow-gui` takes socket_path directly.
#[cfg_attr(test, allow(dead_code))]
#[cfg_attr(not(test), allow(dead_code))]
pub fn find_gui_command() -> Option<(std::path::PathBuf, Vec<String>)> {
    use std::process::Command;

    const EXPECTED_VERSION: &str = env!("CARGO_PKG_VERSION");

    // Helper to check version of an external rgrow binary's gui-subprocess subcommand
    let check_version = |path: &std::path::Path| -> bool {
        if let Ok(output) = Command::new(path)
            .args(["gui-subprocess", "--version"])
            .output()
        {
            if let Ok(version_str) = String::from_utf8(output.stdout) {
                let version = version_str.split_whitespace().last().unwrap_or("");
                return version.trim() == EXPECTED_VERSION;
            }
        }
        false
    };

    // 1. Self-spawn (non-Python + gui feature): current_exe() + ["gui-subprocess"]
    //    No version check needed since it's the same binary.
    #[cfg(all(feature = "gui", not(feature = "python")))]
    {
        if let Ok(exe_path) = std::env::current_exe() {
            return Some((exe_path, vec!["gui-subprocess".to_string()]));
        }
    }

    // 2. Check package directory (for Python installations where rgrow binary might be bundled)
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;

        if let Ok(package_dir) = Python::attach(|py| -> PyResult<String> {
            let importlib = PyModule::import(py, "importlib.util")?;
            let spec = importlib.call_method1("find_spec", ("rgrow",))?;
            let origin = spec.getattr("origin")?;

            if origin.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not find rgrow package",
                ));
            }

            let origin_str = origin.extract::<String>()?;
            let path = std::path::PathBuf::from(origin_str);

            if let Some(parent) = path.parent() {
                Ok(parent.to_string_lossy().to_string())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not determine package directory",
                ))
            }
        }) {
            let rgrow_exe = std::path::PathBuf::from(&package_dir).join("rgrow");
            #[cfg(windows)]
            let rgrow_exe = std::path::PathBuf::from(&package_dir).join("rgrow.exe");

            if rgrow_exe.exists() && check_version(&rgrow_exe) {
                return Some((rgrow_exe, vec!["gui-subprocess".to_string()]));
            }
        }
    }

    // 3. Check environment variable (set by Python if available)
    if let Ok(package_dir) = std::env::var("RGROW_PACKAGE_DIR") {
        let rgrow_exe = std::path::PathBuf::from(&package_dir).join("rgrow");
        #[cfg(windows)]
        let rgrow_exe = std::path::PathBuf::from(&package_dir).join("rgrow.exe");

        if rgrow_exe.exists() && check_version(&rgrow_exe) {
            return Some((rgrow_exe, vec!["gui-subprocess".to_string()]));
        }
    }

    // 4. Check PATH for rgrow
    if let Ok(path) = which::which("rgrow") {
        if check_version(&path) {
            return Some((path, vec!["gui-subprocess".to_string()]));
        } else {
            eprintln!(
                "Warning: Found rgrow on PATH but version mismatch. Expected version {}",
                env!("CARGO_PKG_VERSION")
            );
        }
    }

    // 5. Check in the same directory as the current executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let rgrow_exe = exe_dir.join("rgrow");
            #[cfg(windows)]
            let rgrow_exe = exe_dir.join("rgrow.exe");

            if rgrow_exe.exists() && check_version(&rgrow_exe) {
                return Some((rgrow_exe, vec!["gui-subprocess".to_string()]));
            }
        }
    }

    // Backward compatibility fallbacks: search for legacy `rgrow-gui` binary.
    // The old `rgrow-gui` binary takes socket_path directly (no extra args).

    // 6. Check PATH for rgrow-gui
    if let Ok(path) = which::which("rgrow-gui") {
        return Some((path, vec![]));
    }

    // 7. Check package directory for rgrow-gui (Python installations)
    #[cfg(feature = "python")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyModule;

        if let Ok(package_dir) = Python::attach(|py| -> PyResult<String> {
            let importlib = PyModule::import(py, "importlib.util")?;
            let spec = importlib.call_method1("find_spec", ("rgrow",))?;
            let origin = spec.getattr("origin")?;

            if origin.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not find rgrow package",
                ));
            }

            let origin_str = origin.extract::<String>()?;
            let path = std::path::PathBuf::from(origin_str);

            if let Some(parent) = path.parent() {
                Ok(parent.to_string_lossy().to_string())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not determine package directory",
                ))
            }
        }) {
            let gui_exe = std::path::PathBuf::from(&package_dir).join("rgrow-gui");
            #[cfg(windows)]
            let gui_exe = std::path::PathBuf::from(&package_dir).join("rgrow-gui.exe");

            if gui_exe.exists() {
                return Some((gui_exe, vec![]));
            }
        }
    }

    // 8. Check in the same directory as the current executable for rgrow-gui
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let gui_exe = exe_dir.join("rgrow-gui");
            #[cfg(windows)]
            let gui_exe = exe_dir.join("rgrow-gui.exe");

            if gui_exe.exists() {
                return Some((gui_exe, vec![]));
            }
        }
    }

    None
}
