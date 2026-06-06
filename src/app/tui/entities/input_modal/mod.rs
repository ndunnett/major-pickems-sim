use std::path::PathBuf;

use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::Line,
    widgets::{Block, BorderType, Clear, Padding},
};

use pickems::datatypes::{Iterations, Name, Rating, Seed, Sigma};

use crate::app::tui::{
    Context, Msg, PicksMode, ReportType, State, Task, Update, binds, framework::Entity,
};

mod select_field;
mod text_field;

use select_field::SelectField;
use text_field::TextField;

#[derive(Debug)]
pub struct InputModal {
    field: InputField,
}

impl InputModal {
    const fn new(field: InputField) -> Self {
        Self { field }
    }

    pub fn save(cx: &Context) -> Self {
        Self::new(InputField::SavePath(TextField::new(
            cx,
            "Save As",
            cx.opened.as_ref().map_or_else(Vec::new, |opened| {
                opened.to_string_lossy().chars().collect::<Vec<_>>()
            }),
            |_, input| {
                if input.trim().is_empty() {
                    Err(String::from("filename cannot be empty"))
                } else {
                    Ok(PathBuf::from(input))
                }
            },
            |cx, path, _| {
                let Some(teams) = &cx.teams else {
                    cx.update(Update::ErrorToast(
                        "Failed to save data: no teams data in context".to_string(),
                    ));
                    return;
                };

                if let Err(e) = teams.write_toml(path.clone()) {
                    cx.update(Update::ErrorToast(format!("Failed to save data: {e}")));
                } else {
                    cx.update(Update::DataSaved(path));
                }
            },
        )))
    }

    pub fn iterations(cx: &Context) -> Self {
        Self::new(InputField::Iterations(TextField::new(
            cx,
            "Iterations",
            cx.iterations.to_string().chars().collect::<Vec<_>>(),
            |_, input| input.parse::<Iterations>().map_err(|e| e.to_string()),
            |cx, iterations, is_changed| {
                if is_changed {
                    cx.iterations = iterations;
                    cx.update(Update::RefreshParameters);
                }
            },
        )))
    }

    pub fn sigma(cx: &Context) -> Self {
        Self::new(InputField::Sigma(TextField::new(
            cx,
            "Sigma",
            cx.sigma.to_string().chars().collect::<Vec<_>>(),
            |_, input| input.parse::<Sigma>().map_err(|e| e.to_string()),
            |cx, sigma, is_changed| {
                if is_changed {
                    cx.sigma = sigma;
                    cx.update(Update::RefreshParameters);
                }
            },
        )))
    }

    pub fn rating(cx: &Context, name: Name) -> Self {
        Self::new(InputField::Rating(TextField::new(
            cx,
            format!("{name} Rating"),
            cx.teams
                .as_ref()
                .map_or_else(|| Rating::new(1000), |teams| teams[&name].rating)
                .to_string()
                .chars()
                .collect::<Vec<_>>(),
            |_, input| input.parse::<Rating>().map_err(|e| e.to_string()),
            move |cx, new_rating, is_changed| {
                if is_changed
                    && let Some(teams) = &mut cx.teams
                    && let Some(team) = teams.get_mut(&name)
                {
                    team.rating = new_rating;
                    cx.update(Update::RefreshTeamValues);
                }
            },
        )))
    }

    pub fn name(cx: &Context, initial_name: &Name) -> Self {
        let buffer = initial_name.to_string().chars().collect::<Vec<_>>();
        let validator_initial_name = initial_name.clone();
        let submit_initial_name = initial_name.clone();

        Self::new(InputField::Name(TextField::new(
            cx,
            format!("{initial_name} Name"),
            buffer,
            move |cx, input| {
                let new_name = Name::try_new(input).map_err(|e| e.to_string())?;

                if let Some(teams) = &cx.teams
                    && teams
                        .keys()
                        .any(|name| name != &validator_initial_name && name == &new_name)
                {
                    return Err(String::from("team name already exists"));
                }

                Ok(new_name)
            },
            move |cx, new_name, _| {
                if let Some(teams) = &mut cx.teams
                    && new_name.as_str() != submit_initial_name.as_str()
                    && let Some(team) = teams.remove(&submit_initial_name)
                {
                    teams.insert(new_name, team);
                    cx.update(Update::RefreshFull);
                }
            },
        )))
    }

    pub fn seed(cx: &Context, name: Name) -> Self {
        Self::new(InputField::Seed(TextField::new(
            cx,
            format!("{name} Seed"),
            cx.teams
                .as_ref()
                .map_or_else(|| Seed::new(1), |teams| teams[&name].seed)
                .to_string()
                .chars()
                .collect::<Vec<_>>(),
            |_, input| input.parse::<Seed>().map_err(|e| e.to_string()),
            move |cx, new_seed, is_changed| {
                if is_changed
                    && let Some(teams) = &mut cx.teams
                    && let Some(old_seed) = teams.get(&name).map(|t| t.seed)
                {
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

                    cx.update(Update::RefreshTeamValues);
                }
            },
        )))
    }

    pub fn report(cx: &Context) -> Self {
        Self::new(InputField::Report(SelectField::new(
            "Report",
            vec![
                ReportType::Basic,
                ReportType::Strength,
                ReportType::Picks,
                ReportType::All,
            ],
            &cx.report_type,
            |cx, report_type, is_changed| {
                if is_changed {
                    cx.report_type = report_type;
                    cx.update(Update::RefreshReport);
                }
            },
        )))
    }

    pub fn picks_mode(cx: &Context) -> Self {
        Self::new(InputField::PicksMode(SelectField::new(
            "Picks Mode",
            vec![PicksMode::Auto, PicksMode::Manual],
            &cx.picks_mode,
            |cx, picks_mode, is_changed| {
                if is_changed {
                    cx.update(Update::PicksMode(picks_mode));
                }
            },
        )))
    }

    pub fn pick_select(cx: &Context, index: usize) -> Self {
        let none_selected = Name::new("-");
        let current = cx.picks[index].as_ref().unwrap_or(&none_selected);

        Self::new(InputField::PickSelect(SelectField::new(
            match index {
                0..2 => "Select 3-0 pick",
                2..8 => "Select 3-1 or 3-2 pick",
                8..10 => "Select 0-3 pick",
                _ => unreachable!(),
            },
            cx.teams.as_ref().map_or_else(Vec::new, |teams| {
                teams
                    .keys()
                    .filter(|&key| {
                        !cx.picks.iter().any(|name| {
                            name.as_ref()
                                .is_some_and(|name| name == key && name != current)
                        })
                    })
                    .cloned()
                    .collect()
            }),
            current,
            move |cx, name, is_changed| {
                if is_changed {
                    cx.update(Update::SetPick { index, name });
                }
            },
        )))
    }
}

impl Entity<Update, Task, State> for InputModal {
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

    fn render(&mut self, cx: &Context, frame: &mut Frame, _area: Rect) {
        let [_, column, _] = Layout::horizontal([
            Constraint::Fill(1),
            Constraint::Length(40),
            Constraint::Fill(1),
        ])
        .areas(frame.area());

        let [_, cell, _] = Layout::vertical([
            Constraint::Fill(1),
            Constraint::Length(11),
            Constraint::Fill(1),
        ])
        .areas(column);

        let block = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().bold().blue())
            .padding(Padding::horizontal(1));

        let inner = block.inner(cell);

        let [field_area, bind_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);

        let accept = Line::from(format!("Accept [{}]", binds::Bind(binds::ENTER)))
            .left_aligned()
            .style(Style::new().blue().bold());

        let cancel = Line::from(format!("Cancel [{}]", binds::Bind(binds::ESC)))
            .right_aligned()
            .style(Style::new().blue().bold());

        frame.render_widget(Clear, cell);
        frame.render_widget(block, cell);
        self.field.render(cx, frame, field_area);
        frame.render_widget(accept, bind_area);
        frame.render_widget(cancel, bind_area);
    }
}

#[derive(Debug)]
enum InputField {
    SavePath(TextField<PathBuf>),
    Iterations(TextField<Iterations>),
    Sigma(TextField<Sigma>),
    Name(TextField<Name>),
    Rating(TextField<Rating>),
    Seed(TextField<Seed>),
    Report(SelectField<ReportType>),
    PicksMode(SelectField<PicksMode>),
    PickSelect(SelectField<Name>),
}

impl Entity<Update, Task, State> for InputField {
    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match self {
            Self::SavePath(text) => text.on_key_press(cx, code, modifiers),
            Self::Iterations(text) => text.on_key_press(cx, code, modifiers),
            Self::Name(text) => text.on_key_press(cx, code, modifiers),
            Self::Rating(text) => text.on_key_press(cx, code, modifiers),
            Self::Seed(text) => text.on_key_press(cx, code, modifiers),
            Self::Sigma(text) => text.on_key_press(cx, code, modifiers),
            Self::Report(select) => select.on_key_press(cx, code, modifiers),
            Self::PicksMode(select) => select.on_key_press(cx, code, modifiers),
            Self::PickSelect(select) => select.on_key_press(cx, code, modifiers),
        }
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        match self {
            Self::SavePath(text) => text.update(cx, msg),
            Self::Iterations(text) => text.update(cx, msg),
            Self::Name(text) => text.update(cx, msg),
            Self::Rating(text) => text.update(cx, msg),
            Self::Seed(text) => text.update(cx, msg),
            Self::Sigma(text) => text.update(cx, msg),
            Self::Report(select) => select.update(cx, msg),
            Self::PicksMode(select) => select.update(cx, msg),
            Self::PickSelect(select) => select.update(cx, msg),
        }
    }

    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        match self {
            Self::SavePath(text) => text.render(cx, frame, area),
            Self::Iterations(text) => text.render(cx, frame, area),
            Self::Name(text) => text.render(cx, frame, area),
            Self::Rating(text) => text.render(cx, frame, area),
            Self::Seed(text) => text.render(cx, frame, area),
            Self::Sigma(text) => text.render(cx, frame, area),
            Self::Report(select) => select.render(cx, frame, area),
            Self::PicksMode(select) => select.render(cx, frame, area),
            Self::PickSelect(select) => select.render(cx, frame, area),
        }
    }
}
