//! agent-tui -- full-screen terminal UI for the agent.
//!
//! Design/pattern from dsh-tui (MIT): ratatui + crossterm.
//! Shows conversation pane + composer; streams agent events into the chat.
//!
//! Usage:
//!   cargo run --bin agent-tui --features memory
//!   (model config from AGENT_CONFIG / --config, same as agent-cli)

use std::io::{self, Write};
#[cfg(feature = "memory")]
use std::sync::Arc;
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, LeaveAlternateScreen};

use agent::config::load_config_raw;
use agent::core::Agent;
use agent::tools::default_tools;
use agent::tui::{render_chat, ChatEntry, ChatStore, Composer, ComposerWidget, Theme};
use agent::*;

#[cfg(feature = "memory")]
use agent::memory_integration::UtekeMemory;

struct TuiApp {
    store: ChatStore,
    composer: Composer,
    theme: Theme,
    streaming: bool,
    status: String,
    quit: bool,
}

impl TuiApp {
    fn new() -> Self {
        Self {
            store: ChatStore::new(200),
            composer: Composer::new(),
            theme: Theme::default(),
            streaming: false,
            status: "ready".into(),
            quit: false,
        }
    }

    fn submit(&mut self) -> Option<String> {
        let text = self.composer.text.trim().to_string();
        if text.is_empty() {
            return None;
        }
        self.composer.clear();
        self.store.push(ChatEntry::User { text: text.clone() });
        Some(text)
    }

    fn push_agent_text(&mut self, delta: &str) {
        match self.store.last_mut() {
            Some(ChatEntry::Assistant { text }) => {
                text.push_str(delta);
            }
            _ => {
                self.store.push(ChatEntry::Assistant {
                    text: delta.to_string(),
                });
            }
        }
    }

    fn begin_tool(&mut self, name: &str) {
        self.store.push(ChatEntry::Tool {
            name: name.to_string(),
            status: "running...".to_string(),
        });
        self.status = format!("tool: {name}");
        self.streaming = false;
    }

    fn end_tool(&mut self, name: &str, ok: bool) {
        self.status = if ok {
            format!("OK {name}")
        } else {
            format!("ERR {name}")
        };
        if let Some(ChatEntry::Tool { status, .. }) = self.store.last_mut() {
            *status = if ok { "ok".into() } else { "error".into() };
        }
        self.streaming = false;
    }

    fn info(&mut self, text: &str) {
        self.store.push(ChatEntry::Info {
            text: text.to_string(),
        });
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- config ---
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--config" && i + 1 < args.len() {
            unsafe { std::env::set_var("AGENT_CONFIG", &args[i + 1]) };
            i += 2;
        } else {
            i += 1;
        }
    }
    let raw = load_config_raw()?;
    let model_cfg = raw.clone().into_model_config();
    println!("model: {} @ {}", model_cfg.id, model_cfg.base_url);

    // optional memory
    #[cfg(feature = "memory")]
    let memory: Option<Arc<UtekeMemory>> = match UtekeMemory::from_config(&raw.memory) {
        Ok(m) => {
            println!("memory: uteke ({})", raw.memory.backend);
            Some(Arc::new(m))
        }
        Err(e) => {
            eprintln!("memory disabled: {e}");
            None
        }
    };
    #[cfg(not(feature = "memory"))]
    let memory: Option<std::marker::PhantomData<()>> = None;
    let _mem = memory;

    let mut agent = Agent::from_config(model_cfg)
        .with_system_prompt("You are a helpful assistant with tools. Use them when needed; answer directly otherwise.")
        .with_tools(default_tools());

    #[cfg(feature = "memory")]
    if let Some(m) = &_mem {
        // Route the blocking Uteke backend through the bounded worker so a
        // slow or unavailable embedder can never stall the TUI's render loop,
        // and so recall, the write hook, and the two tools share one queue.
        let top_k = 4;
        let svc = agent::memory_integration::MemoryService::new(
            m.clone(),
            64,
            std::time::Duration::from_secs(5),
        );
        // Default scope, so TUI memory shares the same store the CLI and the
        // same profile would read.
        agent = agent::memory_integration::install_memory_hooks(agent, svc.clone(), top_k);
        agent = agent.with_tools(vec![
            Box::new(agent::memory_integration::MemorySaveTool {
                service: svc.clone(),
            }),
            Box::new(agent::memory_integration::MemoryManageTool { service: svc }),
        ]);
    }

    // --- TUI setup: ratatui::init() does enable_raw_mode + EnterAlternateScreen ---
    let mut terminal = ratatui::init();

    // restore terminal on panic so raw mode never leaks
    let panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        let _ = io::stdout().flush();
        panic_hook(info);
    }));

    let mut app = TuiApp::new();
    app.info("Type a message to start. Ctrl+C quits, Ctrl+L clears.");

    let tick = std::time::Instant::now();
    loop {
        // draw
        terminal.draw(|frame| {
            let area = frame.area();
            let (chat_area, composer_area) = split_layout(area);
            let entries: Vec<ChatEntry> = app.store.entries.iter().cloned().collect();
            render_chat(&entries, &app.theme, chat_area, frame.buffer_mut());

            let status_area = ratatui::layout::Rect {
                x: composer_area.x + 2,
                y: composer_area.y.saturating_sub(1),
                width: composer_area.width.saturating_sub(4),
                height: 1,
            };
            frame.render_widget(
                ratatui::widgets::Paragraph::new(ratatui::text::Line::from(
                    ratatui::text::Span::styled(
                        format!("  {}  ", app.status),
                        ratatui::style::Style::default().fg(app.theme.text_dimmed),
                    ),
                )),
                status_area,
            );
            app.composer.focused = true;
            frame.render_widget(
                ComposerWidget {
                    composer: &app.composer,
                    theme: &app.theme,
                },
                composer_area,
            );
            frame.set_cursor_position((
                composer_area.x + 2 + app.composer.cursor_position().0,
                composer_area.y + 1,
            ));
        })?;

        // handle events (non-blocking so streaming stays responsive)
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(k) = event::read()? {
                handle_key(&mut app, k, &mut agent).await?;
            }
        }
        let _ = tick; // keep tick for future animations
        if app.quit {
            break;
        }
    }

    // teardown
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
    let _ = disable_raw_mode();
    ratatui::restore();
    Ok(())
}

fn split_layout(area: ratatui::layout::Rect) -> (ratatui::layout::Rect, ratatui::layout::Rect) {
    let layout = ratatui::layout::Layout::vertical([
        ratatui::layout::Constraint::Min(3),
        ratatui::layout::Constraint::Length(3),
    ])
    .split(area);
    (layout[0], layout[1])
}

/// Handle a key event; for Enter, runs the agent and streams events.
async fn handle_key(
    app: &mut TuiApp,
    k: KeyEvent,
    agent: &mut Agent,
) -> Result<(), Box<dyn std::error::Error>> {
    // Ctrl+L clear must come before the general Char(c) arm.
    if k.code == KeyCode::Char('l') && k.modifiers.contains(KeyModifiers::CONTROL) {
        app.store.entries.clear();
        agent.clear_messages();
        app.status = "cleared".into();
        return Ok(());
    }
    match k.code {
        KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
            app.quit = true;
        }
        KeyCode::Enter => {
            if let Some(query) = app.submit() {
                let mut rx = agent.prompt(&query).await;
                app.status = "running...".into();
                app.streaming = true;
                while let Some(ev) = rx.recv().await {
                    match ev {
                        AgentEvent::MessageUpdate {
                            delta: StreamDelta::Text { delta },
                            ..
                        } => {
                            app.push_agent_text(&delta);
                        }
                        AgentEvent::ToolExecutionStart { tool_name, .. } => {
                            app.begin_tool(&tool_name);
                        }
                        AgentEvent::ToolExecutionEnd { tool_name, .. } => {
                            app.end_tool(&tool_name, true);
                        }
                        AgentEvent::AgentEnd { .. } => {
                            app.status = "done".into();
                            app.streaming = false;
                            break;
                        }
                        _ => {}
                    }
                }
                // Restore agent state (messages, tools) before the next turn
                agent.finish().await;
            }
        }
        KeyCode::Char(c) => {
            app.composer.insert_char(c);
        }
        KeyCode::Backspace => app.composer.remove_char_before(),
        KeyCode::Delete => app.composer.remove_char_at(),
        KeyCode::Left => app.composer.move_left(),
        KeyCode::Right => app.composer.move_right(),
        KeyCode::Home => app.composer.move_home(),
        KeyCode::End => app.composer.move_end(),
        KeyCode::Esc => {
            app.composer.clear();
        }
        _ => {}
    }
    Ok(())
}
