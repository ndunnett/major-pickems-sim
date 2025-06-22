use anyhow::anyhow;
use itertools::Itertools;
use ratatui::{
    Frame,
    crossterm::event,
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    text::Text,
    widgets::{
        Block, BorderType, Borders, Cell, HighlightSpacing, Paragraph, Row, Table, TableState,
    },
};
use tui_textarea::{Input, Key, TextArea};

use crate::data::Team;

/// Footer content to render when in browse mode.
const BROWSE_FOOTER: [&str; 2] = [
    "(Esc) quit | (Ctrl + S) save | (Ctrl + N) add team | (Del) remove team",
    "(↑) up | (↓) down | (←) left | (→) right | (Enter) edit field",
];

/// Footer content to render when in edit mode.
const EDITOR_FOOTER: &str = "(Esc) cancel edit | (Enter) commit edit";

/// Wizard to create input data, using ratatui.
pub struct Wizard<'a> {
    save: bool,
    cancel: bool,
    teams: Vec<Team>,
    problems: Vec<String>,
    state: TableState,
    editor: Option<TextArea<'a>>,
}

impl Wizard<'_> {
    /// Run an instance of the wizard to completion, handling ratatui and returning a vector of teams.
    pub fn run() -> anyhow::Result<Option<Vec<Team>>> {
        let mut wizard = Self {
            save: false,
            cancel: false,
            teams: Vec::with_capacity(16),
            problems: Vec::with_capacity(4),
            state: TableState::default().with_selected(0),
            editor: None,
        };

        wizard.add_new_team();
        let mut terminal = ratatui::init();

        while !(wizard.cancel || wizard.save) {
            if let Err(e) = {
                terminal.draw(|frame| wizard.render(frame))?;
                wizard.handle_crossterm_events()?;
                Ok(())
            } {
                ratatui::restore();
                return Err(e);
            }
        }

        ratatui::restore();

        if wizard.save {
            Ok(Some(wizard.teams))
        } else {
            Ok(None)
        }
    }

    /// Handle crossterm events and dispatch key inputs.
    fn handle_crossterm_events(&mut self) -> anyhow::Result<()> {
        if let Some(editor) = self.editor.as_mut() {
            match event::read()?.into() {
                Input {
                    key: Key::Char('c') | Key::Char('C'),
                    ctrl: true,
                    ..
                } => self.cancel = true,
                Input { key: Key::Esc, .. } => self.editor = None,
                Input {
                    key: Key::Enter, ..
                } => self.commit_edit(),
                input => {
                    if editor.input(input) {
                        self.validate_editor();
                    }
                }
            }
        } else {
            match event::read()?.into() {
                Input { key: Key::Esc, .. }
                | Input {
                    key: Key::Char('c') | Key::Char('C'),
                    ctrl: true,
                    ..
                } => self.cancel = true,
                Input {
                    key: Key::Char('s') | Key::Char('S'),
                    ctrl: true,
                    ..
                } => {
                    if self.problems.is_empty() {
                        self.save = true;
                    }
                }
                Input {
                    key: Key::Char('n') | Key::Char('N'),
                    ctrl: true,
                    ..
                } => self.add_new_team(),
                Input {
                    key: Key::Delete, ..
                } => self.remove_selected_team(),
                Input {
                    key: Key::Enter, ..
                } => self.start_edit(),
                Input {
                    key: Key::Down | Key::Char('j'),
                    ..
                } => self.next_row(),
                Input {
                    key: Key::Up | Key::Char('k'),
                    ..
                } => self.previous_row(),
                Input {
                    key: Key::Right | Key::Char('l'),
                    ..
                } => self.next_column(),
                Input {
                    key: Key::Left | Key::Char('h'),
                    ..
                } => self.previous_column(),
                _ => {}
            }
        }

        Ok(())
    }

    /// Attempt to parse current editor contents into a valid team.
    fn parse_edit(&self) -> anyhow::Result<Team> {
        if let Some((row, col)) = self.state.selected_cell() {
            let mut team = self.teams[row].clone();

            if let Some(editor) = &self.editor {
                match col {
                    0 => {
                        let seed = editor.lines()[0].parse::<u8>()?;

                        if !(1..=16).contains(&seed) {
                            Err(anyhow!("invalid seed, must be within range 1-16"))
                        } else {
                            team.seed = seed;
                            Ok(team)
                        }
                    }
                    1 => {
                        if self
                            .teams
                            .iter()
                            .map(|team| &team.name)
                            .contains(&editor.lines()[0])
                        {
                            Err(anyhow!("name already exists, must be unique"))
                        } else {
                            team.name = editor.lines()[0].clone();
                            Ok(team)
                        }
                    }
                    2 => {
                        let rating = editor.lines()[0].parse::<i16>()?;

                        if rating > 0 {
                            team.rating = rating;
                            Ok(team)
                        } else {
                            Err(anyhow!("rating must be positive"))
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                Err(anyhow!("failed to access editor (cancel edit and retry)"))
            }
        } else {
            Err(anyhow!("failed to select team (cancel edit and retry)"))
        }
    }

    /// Validate that current editor contents will produce a team.
    fn validate_editor(&mut self) -> bool {
        let parse = self.parse_edit();

        if let Some(editor) = &mut self.editor {
            if let Err(err) = parse {
                editor.set_block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(Color::LightRed)
                        .title(format!("ERROR: {err}")),
                );
                false
            } else {
                editor.set_block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_type(BorderType::Rounded)
                        .border_style(Color::LightGreen)
                        .title("OK"),
                );
                true
            }
        } else {
            false
        }
    }

    /// Start edit mode on the currently selected cell.
    fn start_edit(&mut self) {
        if let Some((_, col)) = self.state.selected_cell() {
            let mut textarea = TextArea::default();

            let placeholder = match col {
                0 => "Enter a valid seed (1-16)",
                1 => "Enter the team name",
                2 => "Enter the global ranking points for the team",
                _ => unreachable!(),
            };

            textarea.set_placeholder_text(placeholder);
            self.editor = Some(textarea);
            self.validate_editor();
        }
    }

    /// Commit current editor contents and exit editor if it is valid.
    fn commit_edit(&mut self) {
        if let Ok(team) = self.parse_edit() {
            if let Some(index) = self.state.selected() {
                self.teams[index] = team;
            } else {
                self.teams.push(team);
            }

            self.on_teams_change();
            self.editor = None;
        }
    }

    /// Add a new team to the table.
    fn add_new_team(&mut self) {
        let seeds = self.teams.iter().map(|team| team.seed).collect::<Vec<_>>();

        let seed = (1..=16)
            .find(|n| !seeds.contains(n))
            .unwrap_or_else(|| seeds.last().map(|seed| seed + 1).unwrap_or(1));

        self.teams.push(Team {
            name: format!("Team {seed}"),
            seed,
            rating: 0,
        });

        self.on_teams_change();

        self.state.select(
            self.teams
                .iter()
                .enumerate()
                .find_map(|(i, team)| if team.seed == seed { Some(i) } else { None }),
        );
    }

    /// Remove the team at the currently selected row.
    fn remove_selected_team(&mut self) {
        if let Some(i) = self.state.selected() {
            self.teams.remove(i);

            if i >= self.teams.len() {
                self.state.select(self.teams.len().checked_sub(1));
            }
        }

        self.on_teams_change();
    }

    /// Resort teams and check for problems. Call when the teams vec is changed.
    fn on_teams_change(&mut self) {
        self.teams.sort_by(|a, b| a.seed.cmp(&b.seed));
        self.problems.clear();

        if self.teams.len() < 16 {
            self.problems
                .push(format!("Not enough teams ({}/16)", self.teams.len()));
        }

        let seeds = self.teams.iter().map(|team| team.seed).collect::<Vec<_>>();

        for seed in seeds.iter() {
            if !(1..=16).contains(seed) {
                self.problems.push(format!("Invalid seed ({seed})"));
            }
        }

        for i in 1..=16 {
            if seeds.iter().filter(|&&seed| seed == i).count() > 1 {
                self.problems.push(format!("Duplicate seed ({i})"));
            }
        }

        let names = self
            .teams
            .iter()
            .map(|team| team.name.as_str())
            .collect::<Vec<_>>();

        for name_a in names.iter().unique() {
            if names.iter().filter(|&name_b| name_a == name_b).count() > 1 {
                self.problems.push(format!("Duplicate name ({name_a})"));
            }
        }

        for team in self.teams.iter() {
            if team.rating == 0 {
                self.problems.push(format!("Rating not set ({team})"));
            }
        }
    }

    /// Move the cursor to the next row.
    fn next_row(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i >= self.teams.len() - 1 {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };

        self.state.select(Some(i));
    }

    /// Move the cursor to the previous row.
    fn previous_row(&mut self) {
        let i = match self.state.selected() {
            Some(i) => {
                if i == 0 {
                    self.teams.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };

        self.state.select(Some(i));
    }

    /// Move the cursor to the next column.
    fn next_column(&mut self) {
        self.state.select_next_column();
    }

    /// Move the cursor to the previous column.
    fn previous_column(&mut self) {
        self.state.select_previous_column();
    }

    /// Render the user interface.
    fn render(&mut self, frame: &mut Frame) {
        let rects = Layout::vertical([
            Constraint::Min(17),
            Constraint::Length(if self.problems.is_empty() {
                1
            } else {
                self.problems.len() as u16 + 2
            }),
            Constraint::Length(if self.editor.is_some() { 3 } else { 0 }),
            Constraint::Length(if self.editor.is_some() { 3 } else { 4 }),
        ])
        .split(frame.area());

        self.render_table(frame, rects[0]);
        self.render_problems(frame, rects[1]);
        self.render_editor(frame, rects[2]);
        self.render_footer(frame, rects[3]);
    }

    /// Render the table displaying the teams.
    fn render_table(&mut self, frame: &mut Frame, area: Rect) {
        let header_style = Style::default();

        let selected_row_style = if self.editor.is_some() {
            Style::default()
        } else {
            Style::default().add_modifier(Modifier::REVERSED)
        };

        let selected_cell_style = if self.editor.is_some() {
            Style::default()
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED)
                .add_modifier(Modifier::REVERSED)
        } else {
            Style::default()
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED)
        };

        let header = ["Seed", "Team", "Rating"]
            .into_iter()
            .map(Cell::from)
            .collect::<Row>()
            .style(header_style)
            .height(1);

        let rows = self.teams.iter().map(|data| {
            let item = [
                format!(" {}.", data.seed),
                data.name.clone(),
                format!("{}", data.rating),
            ];

            item.into_iter()
                .map(|content| Cell::from(Text::from(content)))
                .collect::<Row>()
                .height(1)
        });

        let t = Table::new(
            rows,
            [
                Constraint::Length(7),
                Constraint::Min(20),
                Constraint::Min(8),
            ],
        )
        .header(header)
        .row_highlight_style(selected_row_style)
        .cell_highlight_style(selected_cell_style)
        .highlight_spacing(HighlightSpacing::Always);

        frame.render_stateful_widget(t, area, &mut self.state);
    }

    /// Render any detected problems.
    fn render_problems(&self, frame: &mut Frame, area: Rect) {
        let problems = if self.problems.is_empty() {
            Paragraph::new(Text::from("No problems")).light_green()
        } else {
            Paragraph::new(Text::from_iter(self.problems.iter().map(|s| s.as_str()))).block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Color::LightRed)
                    .title("Problems"),
            )
        };

        frame.render_widget(problems, area);
    }

    /// Render the editor when in edit mode.
    fn render_editor(&self, frame: &mut Frame, area: Rect) {
        if let Some(editor) = &self.editor {
            frame.render_widget(editor, area);
        }
    }

    /// Render the footer showing common keybinds.
    fn render_footer(&self, frame: &mut Frame, area: Rect) {
        let footer = if self.editor.is_some() {
            Paragraph::new(Text::from(EDITOR_FOOTER))
                .centered()
                .block(Block::bordered().border_type(BorderType::Rounded))
        } else {
            Paragraph::new(Text::from_iter(BROWSE_FOOTER))
                .centered()
                .block(Block::bordered().border_type(BorderType::Rounded))
        };

        frame.render_widget(footer, area);
    }
}
