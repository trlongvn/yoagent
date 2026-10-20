//! Dev-only: dump crossterm events to diagnose PTY key handling.
use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use std::io::{self, Write};
use std::time::Duration;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    enable_raw_mode().unwrap();
    println!("probe: press keys; Ctrl+C to quit");
    loop {
        if event::poll(Duration::from_millis(100)).unwrap() {
            if let Event::Key(k) = event::read().unwrap() {
                let code = format!("{:?}", k.code);
                println!("KEY: code={} mods={:?}", code, k.modifiers);
                io::stdout().flush().unwrap();
                if k.code == KeyCode::Char('c')
                    && k.modifiers
                        .contains(crossterm::event::KeyModifiers::CONTROL)
                {
                    break;
                }
            }
        }
    }
    disable_raw_mode().unwrap();
    println!();
    println!("done");
}
