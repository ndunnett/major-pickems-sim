use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::Style,
    text::Span,
};

use crate::app::tui::{
    Context, Msg, Notify, State, Task, Update, binds, entities::FilePicker, framework::Entity,
};

pub struct OpenScreen {
    file_picker: FilePicker,
}

impl OpenScreen {
    pub fn new() -> Self {
        Self {
            file_picker: FilePicker::new(),
        }
    }
}

impl Entity<Update, Notify, Task, State> for OpenScreen {
    fn dispatch_event(&mut self, cx: &mut Context, event: &Event) -> Option<Msg> {
        self.file_picker
            .dispatch_event(cx, event)
            .map_or_else(|| self.handle_event(cx, event), Some)
    }

    fn on_key_press(
        &mut self,
        cx: &mut Context,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg> {
        match (modifiers, code) {
            binds::SELECT => Some(Msg::Notify(Notify::Select)),
            binds::UPDATE => Some(Msg::SpawnTask(Task::UpdateData {
                path: cx.path.clone(),
            })),
            _ => None,
        }
    }

    fn notify(&mut self, cx: &mut Context, msg: Notify) {
        self.file_picker.notify(cx, msg);
    }

    fn update(&mut self, cx: &mut Context, msg: Update) {
        self.file_picker.update(cx, msg);
    }

    #[inline]
    fn render(&mut self, cx: &Context, frame: &mut Frame, area: Rect) {
        let [window, keybar] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(area);

        self.file_picker.render(cx, frame, window);

        let open = Span::from(format!("Open [{}]", binds::Bind(binds::SELECT)))
            .style(Style::new().blue().bold());

        let update = Span::from(format!("Update [{}]", binds::Bind(binds::UPDATE)))
            .style(Style::new().blue().bold());

        let quit = Span::from(format!("Quit [{}]", binds::Bind(binds::QUIT)))
            .style(Style::new().blue().bold());

        let [open_area, update_area, _, quit_area] = Layout::horizontal([
            Constraint::Length(open.width() as u16),
            Constraint::Length(update.width() as u16),
            Constraint::Fill(1),
            Constraint::Length(quit.width() as u16),
        ])
        .spacing(2)
        .areas(keybar);

        frame.render_widget(open, open_area);
        frame.render_widget(update, update_area);
        frame.render_widget(quit, quit_area);
    }
}
