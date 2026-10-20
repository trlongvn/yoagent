//! Chat pane -- renders the conversation list.
//! Pattern modeled on dsh-tui `render/chat_view.rs` (MIT), simplified for agent.

#![allow(dead_code)]

use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Widget, Wrap};

use crate::tui::store::ChatEntry;
use crate::tui::theme::{self, Theme};

/// Render the chat history into a bordered pane.
pub fn render_chat<'a>(entries: &'a [ChatEntry], theme: &'a Theme, area: Rect, buf: &mut Buffer) {
    buf.set_style(area, theme::panel_fill(theme));
    let block = Block::default()
        .borders(Borders::ALL)
        .title(" conversation ")
        .border_style(Style::default().fg(theme.panel_border));
    let inner = block.inner(area);
    block.render(area, buf);

    if entries.is_empty() {
        let placeholder = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "Welcome! Type a message to talk to the agent.",
                Style::default().fg(theme.text_dimmed),
            )),
            Line::from(Span::styled(
                "> /quit to exit  > Ctrl+L to reset  > Esc to clear input",
                Style::default().fg(theme.text_dimmed),
            )),
        ]))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: false });
        placeholder.render(inner, buf);
        return;
    }

    // Build lines: each entry = role header (colored) + body text.
    let mut lines: Vec<Line> = Vec::new();
    for entry in entries {
        let (label, label_style, body_style): (&str, Style, Style) = match entry {
            ChatEntry::User { .. } => (
                "> you",
                theme::accent(theme).add_modifier(Modifier::BOLD),
                theme::text(theme),
            ),
            ChatEntry::Assistant { .. } => (
                ". agent",
                theme::success(theme).add_modifier(Modifier::BOLD),
                theme::text(theme),
            ),
            ChatEntry::Tool { .. } => (
                "o tool",
                theme::tool(theme).add_modifier(Modifier::BOLD),
                theme::hint(theme),
            ),
            ChatEntry::Info { .. } => ("...", theme::hint(theme), theme::hint(theme)),
        };

        // Header
        lines.push(Line::from(vec![Span::styled(
            label.to_string(),
            label_style,
        )]));
        // Body
        let body = match entry {
            ChatEntry::User { text } => text.clone(),
            ChatEntry::Assistant { text } => text.clone(),
            ChatEntry::Tool { name, status } => format!("{name} - {status}"),
            ChatEntry::Info { text } => text.clone(),
        };
        for body_line in body.split('\n') {
            lines.push(Line::from(vec![Span::styled(
                body_line.to_string(),
                body_style,
            )]));
        }
        lines.push(Line::from(""));
    }

    let paragraph = Paragraph::new(Text::from(lines)).wrap(Wrap { trim: false });
    paragraph.render(inner, buf);
}
