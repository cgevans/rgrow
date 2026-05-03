pub mod ipc;

#[cfg(not(target_arch = "wasm32"))]
mod find_gui;
#[cfg(not(target_arch = "wasm32"))]
pub mod ipc_server;

#[cfg(not(target_arch = "wasm32"))]
pub use find_gui::find_gui_command;
