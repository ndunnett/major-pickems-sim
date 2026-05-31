use std::str::FromStr;

use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers},
    layout::{Constraint, Layout, Position, Rect},
    style::Style,
    text::{Line, Text},
    widgets::{Block, BorderType, Clear, Padding, Paragraph, Wrap},
};

use pickems::datatypes::{Iterations, Name, Rating, Seed, Sigma};

use crate::app::tui::{Context, Msg, Notify, State, Task, Update, binds, framework::Entity};

#[derive(Debug)]
pub struct InputModal {
    field: InputField,
}

impl InputModal {
    pub fn iterations(cx: &Context) -> Self {
        let buffer = cx.iterations.to_string().chars().collect::<Vec<_>>();
        let index = buffer.len();

        Self {
            field: InputField::Iterations(TextField {
                title: String::from("Iterations"),
                buffer,
                index,
                validated: None,
                submit: Box::new(|cx, iterations| {
                    cx.iterations = iterations;
                    cx.update(Update::DataOrParams);
                }),
            }),
        }
    }

    pub fn sigma(cx: &Context) -> Self {
        let buffer = cx.sigma.to_string().chars().collect::<Vec<_>>();
        let index = buffer.len();

        Self {
            field: InputField::Sigma(TextField {
                title: String::from("Sigma"),
                buffer,
                index,
                validated: None,
                submit: Box::new(|cx, sigma| {
                    cx.sigma = sigma;
                    cx.update(Update::DataOrParams);
                }),
            }),
        }
    }

    pub fn rating(cx: &Context, name: Name) -> Self {
        let initial = cx
            .teams
            .as_ref()
            .map_or_else(|| Rating::new(0), |teams| teams[&name].rating);

        let buffer = initial.to_string().chars().collect::<Vec<_>>();
        let index = buffer.len();

        Self {
            field: InputField::Rating(TextField {
                title: format!("{name} Rating"),
                buffer,
                index,
                validated: None,
                submit: Box::new(move |cx, new_rating| {
                    if let Some(teams) = &mut cx.teams
                        && let Some(team) = teams.get_mut(&name)
                    {
                        team.rating = new_rating;
                        cx.update(Update::DataOrParams);
                    }
                }),
            }),
        }
    }

    pub fn name(initial_name: Name) -> Self {
        let buffer = initial_name.to_string().chars().collect::<Vec<_>>();
        let index = buffer.len();

        Self {
            field: InputField::Name(TextField {
                title: format!("{initial_name} Name"),
                buffer,
                index,
                validated: None,
                submit: Box::new(move |cx, new_name| {
                    if let Some(teams) = &mut cx.teams
                        && !teams.contains_key(&new_name)
                        && let Some(team) = teams.remove(&initial_name)
                    {
                        teams.insert(new_name, team);
                        cx.update(Update::DataOrParams);
                    }
                }),
            }),
        }
    }

    pub fn seed(cx: &Context, name: Name) -> Self {
        let initial = cx
            .teams
            .as_ref()
            .map_or_else(|| Seed::new(1), |teams| teams[&name].seed);

        let buffer = initial.to_string().chars().collect::<Vec<_>>();
        let index = buffer.len();

        Self {
            field: InputField::Seed(TextField {
                title: format!("{name} Seed"),
                buffer,
                index,
                validated: None,
                submit: Box::new(move |cx, new_seed| {
                    if let Some(teams) = &mut cx.teams {
                        let old_seed = teams[&name].seed;

                        if old_seed < new_seed {
                            for team in teams.values_mut() {
                                if team.seed >= old_seed && team.seed <= new_seed {
                                    team.seed = team.seed.wrapping_decrement();
                                }
                            }
                        } else if old_seed > new_seed {
                            for team in teams.values_mut() {
                                if team.seed <= old_seed && team.seed >= new_seed {
                                    team.seed = team.seed.wrapping_increment();
                                }
                            }
                        }

                        if let Some(team) = teams.get_mut(&name) {
                            team.seed = new_seed;
                        }

                        cx.update(Update::DataOrParams);
                    }
                }),
            }),
        }
    }
}

impl Entity<Update, Notify, Task, State> for InputModal {
    fn dispatch_event(&mut self, cx: &mut Context, event: &Event) -> Option<Msg> {
        self.field
            .dispatch_event(cx, event)
            .map_or_else(|| self.handle_event(cx, event), Some)
    }

    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::ESC => Some(Msg::Update(Update::CloseModal)),
            _ => None,
        }
    }

    fn notify(&mut self, _cx: &mut Context, _msg: Notify) {}

    fn update(&mut self, _cx: &mut Context, _msg: Update) {}

    fn render(&mut self, cx: &Context, frame: &mut Frame, _area: Rect) {
        let [_, column, _] = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Length(40),
            Constraint::Fill(1),
        ])
        .areas(frame.area());

        let [_, cell, _] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(9),
            Constraint::Fill(1),
        ])
        .areas(column);

        let block = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().bold().blue())
            .padding(Padding::horizontal(1));

        let field_area = block.inner(cell);
        frame.render_widget(Clear, cell);
        frame.render_widget(block, cell);
        self.field.render(cx, frame, field_area);
    }
}

#[allow(dead_code)]
#[derive(Debug)]
enum InputField {
    Iterations(TextField<Iterations>),
    Sigma(TextField<Sigma>),
    Name(TextField<Name>),
    Rating(TextField<Rating>),
    Seed(TextField<Seed>),
    Select(SelectField),
}

impl Entity<Update, Notify, Task, State> for InputField {
    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match self {
            Self::Select(select) => select.on_key_press(cx, code, modifiers),
            Self::Iterations(text) => text.on_key_press(cx, code, modifiers),
            Self::Name(text) => text.on_key_press(cx, code, modifiers),
            Self::Rating(text) => text.on_key_press(cx, code, modifiers),
            Self::Seed(text) => text.on_key_press(cx, code, modifiers),
            Self::Sigma(text) => text.on_key_press(cx, code, modifiers),
        }
    }

    fn notify(&mut self, cx: &mut Context, msg: Notify) {
        match self {
            Self::Select(select) => select.notify(cx, msg),
            Self::Iterations(text) => text.notify(cx, msg),
            Self::Name(text) => text.notify(cx, msg),
            Self::Rating(text) => text.notify(cx, msg),
            Self::Seed(text) => text.notify(cx, msg),
            Self::Sigma(text) => text.notify(cx, msg),
        }
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        match self {
            Self::Select(select) => select.update(cx, msg),
            Self::Iterations(text) => text.update(cx, msg),
            Self::Name(text) => text.update(cx, msg),
            Self::Rating(text) => text.update(cx, msg),
            Self::Seed(text) => text.update(cx, msg),
            Self::Sigma(text) => text.update(cx, msg),
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        match self {
            Self::Select(select) => select.render(cx, frame, area),
            Self::Iterations(text) => text.render(cx, frame, area),
            Self::Name(text) => text.render(cx, frame, area),
            Self::Rating(text) => text.render(cx, frame, area),
            Self::Seed(text) => text.render(cx, frame, area),
            Self::Sigma(text) => text.render(cx, frame, area),
        }
    }
}

type SubmitFn<T> = Box<dyn Fn(&mut Context, T)>;

struct TextField<T> {
    title: String,
    buffer: Vec<char>,
    index: usize,
    validated: Option<Result<T, String>>,
    submit: SubmitFn<T>,
}

impl<T> TextField<T>
where
    T: FromStr,
    <T as FromStr>::Err: ToString,
{
    fn insert(&mut self, c: char) {
        self.buffer.insert(self.index, c);
        self.index += 1;
        self.validate();
    }

    fn backspace(&mut self) {
        if self.index > 0 {
            self.index -= 1;
            self.buffer.remove(self.index);
        }

        self.validate();
    }

    fn delete(&mut self) {
        if self.index + 1 < self.buffer.len() {
            self.buffer.remove(self.index);
        }

        self.validate();
    }

    const fn left(&mut self) {
        self.index = self.index.saturating_sub(1);
    }

    fn right(&mut self) {
        self.index = self.buffer.len().min(self.index + 1);
    }

    fn validate(&mut self) {
        let string = self.buffer.iter().collect::<String>();
        self.validated = Some(T::from_str(&string).map_err(|e| e.to_string()));
    }

    fn style(&self) -> Style {
        if self.validated.as_ref().is_some_and(Result::is_ok) {
            Style::new().green()
        } else if self.validated.as_ref().is_some_and(Result::is_err) {
            Style::new().red()
        } else {
            Style::new()
        }
    }
}

impl<T> Entity<Update, Notify, Task, State> for TextField<T>
where
    T: FromStr,
    <T as FromStr>::Err: ToString,
{
    fn on_load(&mut self, _cx: &mut Context) {
        self.validate();
    }

    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            (KeyModifiers::NONE | KeyModifiers::SHIFT, KeyCode::Char(c)) => self.insert(c),
            binds::BACKSPACE => self.backspace(),
            binds::DELETE => self.delete(),
            binds::ENTER => {
                return if let Some(Ok(t)) = self.validated.take_if(|r| r.is_ok()) {
                    (self.submit)(cx, t);
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

    fn notify(&mut self, _cx: &mut Context, _msg: Notify) {}

    fn update(&mut self, _cx: &mut Context, _msg: Update) {}

    fn render(&mut self, _cx: &Context, frame: &mut Frame, area: Rect) {
        let [block_area, message_area, bottom] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Fill(1),
            Constraint::Length(1),
        ])
        .areas(area);

        let style = self.style();

        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(style)
            .title(self.title.as_str());

        let text = Text::from(self.buffer.iter().collect::<String>());
        let text_area = block.inner(block_area);
        let cursor_position = Position::new(text_area.x + self.index as u16, text_area.y);

        let accept = Line::from(format!("Accept [{}]", binds::Bind(binds::ENTER)))
            .left_aligned()
            .style(Style::new().blue().bold());

        let cancel = Line::from(format!("Cancel [{}]", binds::Bind(binds::ESC)))
            .right_aligned()
            .style(Style::new().blue().bold());

        frame.render_widget(block, block_area);
        frame.render_widget(text, text_area);
        frame.set_cursor_position(cursor_position);

        if let Some(Err(e)) = &self.validated {
            frame.render_widget(
                Paragraph::new(e.as_str())
                    .wrap(Wrap::default())
                    .centered()
                    .style(style),
                message_area,
            );
        }

        frame.render_widget(accept, bottom);
        frame.render_widget(cancel, bottom);
    }
}

impl<T> std::fmt::Debug for TextField<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextField")
            .field("buffer", &self.buffer)
            .field("index", &self.index)
            .field("validated", &"...")
            .field("submit", &"...")
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone)]
struct SelectField {
    // list: ListState,
}

// impl<S> SelectField<S> {
//     fn new() -> Self {
//         Self {
//             list: ListState::default(),
//             marker: PhantomData,
//         }
//     }
// }

impl Entity<Update, Notify, Task, State> for SelectField {
    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        #[allow(clippy::match_single_binding)]
        match (modifiers, code) {
            _ => None,
        }
    }

    fn notify(&mut self, _cx: &mut Context, _msg: Notify) {}

    fn update(&mut self, _cx: &mut Context, _msg: Update) {}

    fn render(&mut self, _cx: &Context, _frame: &mut Frame, _area: Rect) {}
}
