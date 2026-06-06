use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::Rect,
    style::Style,
    widgets::{Block, BorderType, List, ListState},
};

use crate::app::tui::{Context, Msg, State, Task, Update, binds, framework::Entity};

type SubmitFn<T> = Box<dyn Fn(&mut Context, T, bool)>;

pub struct SelectField<T> {
    title: String,
    values: Vec<T>,
    state: ListState,
    initial: Option<usize>,
    submit: SubmitFn<T>,
}

impl<T: PartialEq> SelectField<T> {
    pub fn new(
        title: impl Into<String>,
        values: Vec<T>,
        initial: &T,
        submit_fn: impl Fn(&mut Context, T, bool) + 'static,
    ) -> Self {
        let initial = values.iter().position(|v| v == initial);

        Self {
            title: title.into(),
            values,
            state: ListState::default().with_selected(initial),
            initial,
            submit: Box::new(submit_fn),
        }
    }

    fn is_changed(&self) -> bool {
        self.state.selected() != self.initial
    }
}

impl<T> Entity<Update, Task, State> for SelectField<T>
where
    T: AsRef<str> + PartialEq,
{
    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::SELECT if let Some(i) = self.state.selected() => {
                let changed = self.is_changed();
                (self.submit)(cx, self.values.swap_remove(i), changed);
                return Some(Msg::Update(Update::CloseModal));
            }
            binds::UP => self.state.select_previous(),
            binds::DOWN => self.state.select_next(),
            _ => return None,
        }

        Some(Msg::Redraw)
    }

    fn render(&mut self, _cx: &Context, frame: &mut Frame, area: Rect) {
        let items = self.values.iter().map(AsRef::as_ref);

        let list = List::new(items)
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title(self.title.as_str()),
            )
            .highlight_style(Style::new().reversed())
            .highlight_symbol(">")
            .scroll_padding(2);

        frame.render_stateful_widget(list, area, &mut self.state);
    }
}

impl<T> std::fmt::Debug for SelectField<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectField")
            .field("title", &self.title)
            .field("values", &"Vec(...)")
            .field("state", &self.state)
            .field("initial", &self.initial)
            .field("submit", &"...")
            .finish()
    }
}
