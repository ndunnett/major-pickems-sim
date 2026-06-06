use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::{Margin, Rect},
    style::Style,
    widgets::{
        Block, BorderType, Paragraph, ScrollDirection, Scrollbar, ScrollbarOrientation,
        ScrollbarState, Wrap,
    },
};

use crate::app::tui::{Context, Id, Msg, State, Task, Update, binds, framework::Entity};

#[derive(Debug, Clone)]
pub struct ReportPane {
    content: String,
    scrollbar: ScrollbarState,
    len_set: bool,
}

impl ReportPane {
    pub const fn new() -> Self {
        Self {
            content: String::new(),
            scrollbar: ScrollbarState::new(0),
            len_set: false,
        }
    }

    fn style(cx: &Context) -> Style {
        if matches!(cx.report_focus, Id::Report) && !cx.modal_open {
            Style::new().blue().bold()
        } else {
            Style::new().bold()
        }
    }
}

impl Entity<Update, Task, State> for ReportPane {
    fn on_resize(&mut self, _cx: &mut Context, _width: u16, _height: u16) -> Option<Msg> {
        self.len_set = false;
        None
    }

    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::UP => {
                self.scrollbar.scroll(ScrollDirection::Backward);
                Some(Msg::Redraw)
            }
            binds::DOWN => {
                self.scrollbar.scroll(ScrollDirection::Forward);
                Some(Msg::Redraw)
            }
            _ => None,
        }
    }

    fn update(&mut self, _cx: &mut Context, msg: Update) {
        match msg {
            Update::RefreshFull => {
                self.content.clear();
                self.scrollbar = self.scrollbar.content_length(0);
                self.len_set = false;
            }
            Update::ReportContent(content) => {
                self.content = content;
                self.scrollbar = self.scrollbar.content_length(0);
                self.len_set = false;
            }
            _ => {}
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        if !self.len_set {
            let cols = area.width as usize;
            let mut lines = 1_usize;
            let mut start = 0;
            let mut last_space = 0;

            for (i, ch) in self.content.trim().chars().enumerate() {
                if ch.is_ascii_whitespace() {
                    last_space = i;
                }

                if ch == '\n' || i - start > cols {
                    lines += 1;
                    start = last_space;
                }
            }

            lines = lines.saturating_sub(area.height as usize) + 3;
            self.scrollbar = self.scrollbar.content_length(lines);

            if self.scrollbar.get_position() > lines {
                self.scrollbar.last();
            }

            self.len_set = true;
        }

        frame.render_widget(
            Paragraph::new(self.content.trim())
                .wrap(Wrap::default())
                .block(
                    Block::bordered()
                        .border_type(BorderType::Rounded)
                        .border_style(Self::style(cx))
                        .title_style(Self::style(cx))
                        .title(format!("Report [{}]", binds::Bind(binds::FOCUS_REPORT))),
                )
                .scroll((self.scrollbar.get_position() as u16, 0)),
            area,
        );

        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight),
            area.inner(Margin::new(0, 1)),
            &mut self.scrollbar,
        );
    }
}
