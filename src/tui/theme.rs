//! Theme — colors and styles for the agent TUI.
//! Design modeled on dsh-tui's `theme::Theme` (MIT).

#![allow(dead_code)]

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub accent: Color,
    pub accent_dimmed: Color,
    pub panel_bg: Color,
    pub panel_border: Color,
    pub text: Color,
    pub text_dimmed: Color,
    pub threat: Color,
    pub success: Color,
    pub tool: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            accent: Color::Cyan,
            accent_dimmed: Color::DarkGray,
            panel_bg: Color::Black,
            panel_border: Color::DarkGray,
            text: Color::White,
            text_dimmed: Color::Gray,
            threat: Color::Red,
            success: Color::Green,
            tool: Color::Yellow,
        }
    }
}

// style helpers (parity with dsh-tui's ui/style.rs)
pub fn panel_fill(t: &Theme) -> Style {
    Style::default().bg(t.panel_bg)
}
pub fn border(t: &Theme) -> Style {
    Style::default().fg(t.panel_border)
}
pub fn border_focused(t: &Theme) -> Style {
    Style::default().fg(t.accent).add_modifier(Modifier::BOLD)
}
pub fn hint(t: &Theme) -> Style {
    Style::default().fg(t.text_dimmed)
}
pub fn accent(t: &Theme) -> Style {
    Style::default().fg(t.accent)
}
pub fn text(t: &Theme) -> Style {
    Style::default().fg(t.text)
}
pub fn success(t: &Theme) -> Style {
    Style::default().fg(t.success)
}
pub fn tool(t: &Theme) -> Style {
    Style::default().fg(t.tool)
}
pub fn threat(t: &Theme) -> Style {
    Style::default().fg(t.threat)
}
