//! Composer — prompt input line at the bottom of the agent TUI.
//! Original from dsh-tui `ui/composer.rs` (MIT); reduced to agent needs
//! (no DSH catalog/i18n — slash commands replaced by a simple handler).

#![allow(dead_code)]

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Widget};
use unicode_width::UnicodeWidthStr;

use crate::tui::theme::{self, Theme};

/// A compact prompt input. Owns the buffer text + cursor position.
#[derive(Debug, Default)]
pub struct Composer {
    pub text: String,
    pub cursor: usize, // byte index into text
    pub focused: bool,
}

impl Composer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a character at the cursor.
    pub fn insert_char(&mut self, c: char) {
        let idx = self.byte_index();
        self.text.insert(idx, c);
        self.cursor += c.len_utf8();
    }

    /// Insert a string at the cursor.
    pub fn insert_str(&mut self, s: &str) {
        let idx = self.byte_index();
        self.text.insert_str(idx, s);
        self.cursor += s.len();
    }

    pub fn remove_char_before(&mut self) {
        if self.cursor > 0 {
            let idx = self.byte_index();
            if let Some((ch_idx, _)) = self.text[..idx].char_indices().next_back() {
                self.text.remove(ch_idx);
                self.cursor = ch_idx;
            }
        }
    }

    pub fn remove_char_at(&mut self) {
        let idx = self.byte_index();
        if idx < self.text.len() {
            let next = self.text[idx..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| i)
                .unwrap_or(self.text.len() - idx);
            self.text.drain(idx..idx + next);
        }
    }

    pub fn move_left(&mut self) {
        let idx = self.byte_index();
        if idx > 0 {
            if let Some((ch_idx, _)) = self.text[..idx].char_indices().next_back() {
                self.cursor = ch_idx;
            }
        }
    }

    pub fn move_right(&mut self) {
        let idx = self.byte_index();
        if idx < self.text.len() {
            let next = self.text[idx..]
                .char_indices()
                .nth(1)
                .map(|(i, _)| i)
                .unwrap_or(self.text.len() - idx);
            self.cursor = idx + next;
        }
    }

    pub fn move_home(&mut self) {
        self.cursor = 0;
    }
    pub fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
    }

    /// Byte index from character-index cursor (cursor stores char positions).
    fn byte_index(&self) -> usize {
        self.cursor.min(self.text.len())
    }

    /// Visible cursor column for ratatui (accounts for UTF-8 width).
    pub fn cursor_position(&self) -> (u16, u16) {
        let before = &self.text[..self.byte_index()];
        let width = UnicodeWidthStr::width(before) as u16;
        (width, 0)
    }
}

/// Widget rendering for the composer.
pub struct ComposerWidget<'a> {
    pub composer: &'a Composer,
    pub theme: &'a Theme,
}

impl<'a> Widget for ComposerWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // panel background
        buf.set_style(area, theme::panel_fill(self.theme));
        let border_style = if self.composer.focused {
            theme::border_focused(self.theme)
        } else {
            theme::border(self.theme)
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(" prompt ");
        let inner = block.inner(area);
        block.render(area, buf);

        // placeholder when empty
        let prompt_span = Span::styled("> ", Style::default().fg(self.theme.accent));
        let display: Line = if self.composer.text.is_empty() {
            Line::from(vec![
                prompt_span,
                Span::styled(
                    "Ask the agent…",
                    Style::default().fg(self.theme.text_dimmed),
                ),
            ])
        } else {
            Line::from(vec![
                prompt_span,
                Span::styled(&self.composer.text, Style::default().fg(self.theme.text)),
            ])
        };
        let p = Paragraph::new(display);
        p.render(inner, buf);
    }
}
