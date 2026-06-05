use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::{Alignment, Constraint, Layout, Rect},
    style::Style,
    widgets::{Block, BorderType, Clear, Padding, Paragraph, Wrap},
};

use crate::app::tui::{Context, Msg, State, Task, Update, binds, framework::Entity};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastKind {
    Info,
    Error,
}

#[derive(Debug, Clone)]
pub struct ToastMessage {
    pub kind: ToastKind,
    pub text: String,
}

impl ToastMessage {
    fn visible(self, now: &Instant) -> VisibleToast {
        let lengths = self
            .text
            .lines()
            .map(|line| line.chars().count() as u16)
            .collect();

        VisibleToast {
            message: self,
            shown_at: *now,
            lengths,
        }
    }
}

#[derive(Debug, Clone)]
struct VisibleToast {
    message: ToastMessage,
    shown_at: Instant,
    lengths: Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct Toast {
    current: Option<VisibleToast>,
    queue: VecDeque<ToastMessage>,
    timeout: Duration,
}

impl Toast {
    pub const fn new() -> Self {
        Self {
            current: None,
            queue: VecDeque::new(),
            timeout: Duration::from_secs(5),
        }
    }

    pub fn push(&mut self, message: ToastMessage, now: &Instant) {
        if self.current.is_some() {
            self.queue.push_back(message);
        } else {
            self.current = Some(message.visible(now));
        }
    }

    const fn style(kind: ToastKind) -> Style {
        match kind {
            ToastKind::Info => Style::new().blue().bold(),
            ToastKind::Error => Style::new().red().bold(),
        }
    }

    const fn title(kind: ToastKind) -> &'static str {
        match kind {
            ToastKind::Info => "Info",
            ToastKind::Error => "Error",
        }
    }
}

impl Entity<Update, Task, State> for Toast {
    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        if (modifiers, code) == binds::DISMISS_TOAST && self.current.is_some() {
            self.current = self.queue.pop_front().map(|t| t.visible(cx.tick()));
            Some(Msg::Redraw)
        } else {
            None
        }
    }

    fn on_tick(&mut self, cx: &mut Context) {
        if let Some(current) = &self.current
            && cx.tick().saturating_duration_since(current.shown_at) > self.timeout
        {
            self.current = self.queue.pop_front().map(|t| t.visible(cx.tick()));
            cx.redraw();
        }
    }

    fn render(&mut self, _cx: &Context, frame: &mut Frame, area: Rect) {
        let Some(current) = &self.current else {
            return;
        };

        let [_, col, _] = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Max(60),
            Constraint::Fill(1),
        ])
        .areas(area);

        if col.width < 8 {
            return;
        }

        let inner_width = col.width - 4;
        let lines = current
            .lengths
            .iter()
            .map(|&length| length.max(1).div_ceil(inner_width))
            .sum::<u16>()
            .max(1);

        let [_, cell, _] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(lines + 2),
            Constraint::Length(2),
        ])
        .areas(col);

        let style = Self::style(current.message.kind);

        frame.render_widget(Clear, cell);
        frame.render_widget(
            Paragraph::new(current.message.text.as_str())
                .wrap(Wrap::default())
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .border_style(style)
                        .title_style(style)
                        .title_top(Self::title(current.message.kind))
                        .title_bottom(format!("Dismiss [{}]", binds::Bind(binds::DISMISS_TOAST)))
                        .title_alignment(Alignment::Center)
                        .padding(Padding::horizontal(1)),
                ),
            cell,
        );
    }
}
