use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::{Constraint, Layout, Position, Rect},
    style::Style,
    widgets::{Block, BorderType, Paragraph, Wrap},
};

use crate::app::tui::{Context, Msg, State, Task, Update, binds, framework::Entity};

type SubmitFn<T> = Box<dyn Fn(&mut Context, T, bool)>;
type ValidateFn<T> = Box<dyn Fn(&Context, &str) -> Result<T, String>>;

pub struct TextField<T> {
    title: String,
    buffer: Vec<char>,
    index: usize,
    scroll: u16,
    initial: Option<Result<T, String>>,
    validated: Option<Result<T, String>>,
    validator: ValidateFn<T>,
    submit: SubmitFn<T>,
}

impl<T: PartialEq + Clone> TextField<T> {
    pub fn new(
        cx: &Context,
        title: impl Into<String>,
        initial_buffer: Vec<char>,
        validator_fn: impl Fn(&Context, &str) -> Result<T, String> + 'static,
        submit_fn: impl Fn(&mut Context, T, bool) + 'static,
    ) -> Self {
        let index = initial_buffer.len();

        let mut field = Self {
            title: title.into(),
            buffer: initial_buffer,
            index,
            scroll: 0,
            initial: None,
            validated: None,
            validator: Box::new(validator_fn),
            submit: Box::new(submit_fn),
        };

        field.validate(cx);

        if matches!(field.validated, Some(Ok(..))) {
            field.initial = field.validated.clone();
        }

        field
    }

    fn insert(&mut self, cx: &Context, c: char) {
        self.buffer.insert(self.index, c);
        self.index += 1;
        self.validate(cx);
    }

    fn backspace(&mut self, cx: &Context) {
        if self.index > 0 {
            self.index -= 1;
            self.buffer.remove(self.index);
        }

        self.validate(cx);
    }

    fn delete(&mut self, cx: &Context) {
        if self.index < self.buffer.len() {
            self.buffer.remove(self.index);
        }

        self.validate(cx);
    }

    const fn left(&mut self) {
        self.index = self.index.saturating_sub(1);
    }

    fn right(&mut self) {
        self.index = self.buffer.len().min(self.index + 1);
    }

    fn validate(&mut self, cx: &Context) {
        let string = self.buffer.iter().collect::<String>();
        self.validated = Some((self.validator)(cx, &string));
    }

    fn is_valid(&self) -> bool {
        self.validated.as_ref().is_some_and(Result::is_ok)
    }

    fn is_changed(&self) -> bool {
        self.initial != self.validated
    }

    fn is_not_valid(&self) -> bool {
        self.validated.as_ref().is_some_and(Result::is_err)
    }

    fn style(&self) -> Style {
        if self.is_changed() && self.is_valid() {
            Style::new().green()
        } else if self.is_not_valid() {
            Style::new().red()
        } else {
            Style::new()
        }
    }
}

impl<T: PartialEq + Clone> Entity<Update, Task, State> for TextField<T> {
    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => self.insert(cx, c),
            binds::BACKSPACE => self.backspace(cx),
            binds::DELETE => self.delete(cx),
            binds::SELECT => {
                let changed = self.is_changed();

                return if let Some(Ok(t)) = self.validated.take_if(|r| r.is_ok()) {
                    (self.submit)(cx, t, changed);
                    Some(Msg::Update(Update::CloseModal))
                } else if self.validated.is_none() {
                    Some(Msg::Update(Update::CloseModal))
                } else {
                    None
                };
            }
            binds::LEFT => self.left(),
            binds::RIGHT => self.right(),
            _ => return None,
        }

        Some(Msg::Redraw)
    }

    fn render(&mut self, _cx: &Context, frame: &mut Frame, area: Rect) {
        let [block_area, message_area] =
            Layout::vertical([Constraint::Length(3), Constraint::Fill(1)]).areas(area);

        let style = self.style();

        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(style)
            .title(self.title.clone());

        let text_area = block.inner(block_area);

        let cursor_offset = {
            let index = self.index as u16;

            if index < self.scroll {
                self.scroll = index;
            } else if index >= self.scroll.saturating_add(text_area.width) {
                self.scroll = index.saturating_add(1).saturating_sub(text_area.width);
            }

            Some(index.saturating_sub(self.scroll))
        };

        frame.render_widget(block, block_area);
        frame.render_widget(
            Paragraph::new(self.buffer.iter().collect::<String>()).scroll((0, self.scroll)),
            text_area,
        );

        if let Some(cursor_offset) = cursor_offset {
            frame.set_cursor_position(Position::new(
                text_area.x.saturating_add(cursor_offset),
                text_area.y,
            ));
        }

        if let Some(Err(e)) = &self.validated {
            frame.render_widget(
                Paragraph::new(e.as_str())
                    .wrap(Wrap::default())
                    .centered()
                    .style(style),
                message_area,
            );
        }
    }
}

impl<T> std::fmt::Debug for TextField<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextField")
            .field("title", &self.title)
            .field("buffer", &self.buffer)
            .field("index", &self.index)
            .field("scroll", &self.scroll)
            .field("initial", &"...")
            .field("validated", &"...")
            .field("validator", &"...")
            .field("submit", &"...")
            .finish()
    }
}
