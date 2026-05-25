#![allow(dead_code)]

use std::fmt::Write as _;

use ratatui::crossterm::event::{KeyCode, KeyModifiers};

#[derive(Debug, Clone, Copy)]
pub struct Bind(pub (KeyModifiers, KeyCode));

impl std::fmt::Display for Bind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut first = true;

        for modifier in self.0.0.iter() {
            if !first {
                f.write_str("+")?;
            }

            first = false;

            match modifier {
                KeyModifiers::SHIFT => f.write_str("Shift")?,
                KeyModifiers::CONTROL => f.write_str("Ctrl")?,
                #[cfg(target_os = "macos")]
                KeyModifiers::ALT => f.write_str("Option")?,
                #[cfg(not(target_os = "macos"))]
                KeyModifiers::ALT => f.write_str("Alt")?,
                #[cfg(target_os = "macos")]
                KeyModifiers::SUPER => f.write_str("Cmd")?,
                #[cfg(target_os = "windows")]
                KeyModifiers::SUPER => f.write_str("Win")?,
                #[cfg(not(any(target_os = "macos", target_os = "windows")))]
                KeyModifiers::SUPER => f.write_str("Super")?,
                KeyModifiers::HYPER => f.write_str("Hyper")?,
                KeyModifiers::META => f.write_str("Meta")?,
                _ => unreachable!(),
            }
        }

        if !first {
            f.write_str("+")?;
        }

        match self.0.1 {
            KeyCode::Char(c) => f.write_char(c.to_ascii_uppercase()),
            _ => write!(f, "{}", self.0.1),
        }
    }
}

pub const UP: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Up);
pub const DOWN: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Down);
pub const LEFT: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Left);
pub const RIGHT: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Right);
pub const ENTER: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Enter);
pub const ESC: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Esc);
pub const TAB: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Tab);
pub const SHIFT_TAB: (KeyModifiers, KeyCode) = (KeyModifiers::SHIFT, KeyCode::BackTab);

pub const EOF: (KeyModifiers, KeyCode) = (KeyModifiers::CONTROL, KeyCode::Char('d'));
pub const REDRAW: (KeyModifiers, KeyCode) = (KeyModifiers::CONTROL, KeyCode::Char('r'));
pub const OPEN_SCREEN: (KeyModifiers, KeyCode) = (KeyModifiers::CONTROL, KeyCode::Char('o'));
pub const UPDATE: (KeyModifiers, KeyCode) = (KeyModifiers::CONTROL, KeyCode::Char('u'));
pub const NEW: (KeyModifiers, KeyCode) = (KeyModifiers::CONTROL, KeyCode::Char('n'));
pub const SAVE: (KeyModifiers, KeyCode) = (KeyModifiers::CONTROL, KeyCode::Char('s'));
pub const QUIT: (KeyModifiers, KeyCode) = ESC;
pub const SELECT: (KeyModifiers, KeyCode) = ENTER;

pub const FOCUS_TEAMS: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Char('t'));
pub const FOCUS_PICKS: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Char('e'));
pub const FOCUS_PARAMETERS: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Char('p'));
pub const FOCUS_REPORT: (KeyModifiers, KeyCode) = (KeyModifiers::NONE, KeyCode::Char('r'));
pub const FOCUS_NEXT: (KeyModifiers, KeyCode) = TAB;
pub const FOCUS_PREV: (KeyModifiers, KeyCode) = SHIFT_TAB;
