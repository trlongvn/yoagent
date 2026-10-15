//! TUI store — messages + session state for rendering.
//! Adapted from dsh-tui `store/session.rs` concept (MIT).

#![allow(dead_code)]

use std::collections::VecDeque;

/// A single chat entry in the TUI (agent-agnostic view model).
#[derive(Debug, Clone)]
pub enum ChatEntry {
    User { text: String },
    Assistant { text: String },
    Tool { name: String, status: String },
    Info { text: String },
}

impl ChatEntry {
    pub fn role_label(&self) -> &'static str {
        match self {
            ChatEntry::User { .. } => "you",
            ChatEntry::Assistant { .. } => "agent",
            ChatEntry::Tool { .. } => "tool",
            ChatEntry::Info { .. } => "info",
        }
    }
}

/// Rolling chat history (bounded to avoid unbounded memory).
#[derive(Debug, Default)]
pub struct ChatStore {
    pub entries: VecDeque<ChatEntry>,
    max_entries: usize,
}

impl ChatStore {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
        }
    }

    pub fn push(&mut self, entry: ChatEntry) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    pub fn last_mut(&mut self) -> Option<&mut ChatEntry> {
        self.entries.back_mut()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
