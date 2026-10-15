//! Agent TUI — terminal user interface for the agent.
//!
//! Design and patterns modeled on [dsh-tui](https://github.com/rbelem/dsh-tui)
//! (MIT): ratatui + crossterm, composer at the bottom, chat pane above,
//! theme from a single `Theme` struct. DSH-specific coupling (wire, gateway,
//! client, catalog) removed; agent drives the store from `AgentEvent`.

pub mod chat;
pub mod composer;
pub mod store;
pub mod theme;

pub use chat::render_chat;
pub use composer::{Composer, ComposerWidget};
pub use store::{ChatEntry, ChatStore};
pub use theme::Theme;
