use std::{
    fs, io,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyModifiers},
    layout::Rect,
    style::Style,
    widgets::{Block, BorderType, List, ListState, Paragraph, Wrap},
};

use crate::app::tui::{Context, Msg, State, Task, Update, binds, framework::Entity};

#[derive(Debug, Clone)]
pub struct FilePicker {
    state: ListState,
    items: Vec<PathBuf>,
    scratch: Vec<PathBuf>,
    display_path: PathBuf,
    error: Option<String>,
    next_refresh: Instant,
}

impl FilePicker {
    const REFRESH_INTERVAL: Duration = Duration::from_secs(1);
    const RETRY_DELAY: Duration = Duration::from_secs(5);

    pub fn new() -> Self {
        Self {
            state: ListState::default(),
            items: Vec::with_capacity(8),
            scratch: Vec::new(),
            display_path: PathBuf::new(),
            error: None,
            next_refresh: Instant::now(),
        }
    }

    fn list(path: &Path, items: &mut Vec<PathBuf>) -> io::Result<()> {
        items.clear();

        for entry in fs::read_dir(path)? {
            let file = entry?.path();
            if file
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
            {
                items.push(file);
            }
        }

        items.sort_unstable_by(|a, b| b.cmp(a));
        Ok(())
    }

    fn refresh(&mut self, path: &Path, now: Instant) -> bool {
        let display_path = path.canonicalize().unwrap_or_else(|_| path.to_owned());
        let path_changed = display_path != self.display_path;
        self.display_path = display_path;

        match Self::list(&self.display_path, &mut self.scratch) {
            Ok(()) => {
                self.next_refresh = now + Self::REFRESH_INTERVAL;
                let recovered = self.error.take().is_some();

                if self.scratch == self.items {
                    return path_changed || recovered;
                }

                let old_index = self.state.selected().unwrap_or_default();
                let selection = self
                    .items
                    .get(old_index)
                    .and_then(|selected| self.scratch.iter().position(|item| item == selected))
                    .or_else(|| {
                        self.scratch
                            .len()
                            .checked_sub(1)
                            .map(|last| old_index.min(last))
                    });
                self.state.select(selection);
                std::mem::swap(&mut self.items, &mut self.scratch);
                true
            }
            Err(error) => {
                self.next_refresh = now + Self::RETRY_DELAY;
                let message = format!("Failed to list {}: {error}", self.display_path.display());
                let changed = self.error.as_ref() != Some(&message);
                self.items.clear();
                self.state.select(None);
                self.error = Some(message);
                path_changed || changed
            }
        }
    }
}

impl Entity<Update, Task, State> for FilePicker {
    fn on_key_press(
        &mut self,
        _cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::SELECT
                if let Some(path) = self.state.selected().and_then(|i| self.items.get(i)) =>
            {
                Some(Msg::Update(Update::LoadDataFile(path.clone())))
            }
            binds::UP => {
                self.state.select_previous();
                Some(Msg::Redraw)
            }
            binds::DOWN => {
                self.state.select_next();
                Some(Msg::Redraw)
            }
            _ => None,
        }
    }

    fn on_tick(&mut self, cx: &mut Context) {
        if *cx.tick() >= self.next_refresh && self.refresh(&cx.path, *cx.tick()) {
            cx.redraw();
        }
    }

    fn next_deadline(&self) -> Option<Instant> {
        Some(self.next_refresh)
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        if matches!(msg, Update::LoadFileList) {
            self.refresh(&cx.path, *cx.tick());
        }
    }

    fn render(&mut self, _cx: &Context, frame: &mut Frame, area: Rect) {
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .title_style(Style::new().bold())
            .title(self.display_path.to_string_lossy());

        if let Some(error) = &self.error {
            frame.render_widget(
                Paragraph::new(error.as_str())
                    .style(Style::new().red())
                    .wrap(Wrap::default())
                    .block(block),
                area,
            );
            return;
        }

        let items = self
            .items
            .iter()
            .filter_map(|path| path.file_name())
            .map(|path| path.to_string_lossy());

        frame.render_stateful_widget(
            List::new(items)
                .block(block)
                .highlight_style(Style::new().reversed())
                .highlight_symbol(">")
                .repeat_highlight_symbol(true),
            area,
            &mut self.state,
        );
    }
}
