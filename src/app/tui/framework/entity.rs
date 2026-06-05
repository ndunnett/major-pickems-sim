use ratatui::{
    Frame,
    crossterm::event::{Event, KeyCode, KeyModifiers, MouseEvent},
    layout::Rect,
};

use super::{Context, Msg};

pub trait Entity<U, T, S> {
    #[inline]
    fn dispatch_event(&mut self, cx: &mut Context<U, T, S>, event: &Event) -> Option<Msg<U, T>> {
        self.handle_event(cx, event)
    }

    fn handle_event(&mut self, cx: &mut Context<U, T, S>, event: &Event) -> Option<Msg<U, T>> {
        match event {
            Event::FocusGained => self.on_focus_gained(cx),
            Event::FocusLost => self.on_focus_lost(cx),
            Event::Key(event) if event.is_press() => {
                self.on_key_press(cx, event.code, event.modifiers)
            }
            Event::Key(_) => None,
            Event::Mouse(event) => self.on_mouse_event(cx, *event),
            Event::Paste(value) => self.on_paste(cx, value),
            Event::Resize(w, h) => self.on_resize(cx, *w, *h),
        }
    }

    #[allow(unused_variables)]
    #[inline]
    fn on_focus_gained(&mut self, cx: &mut Context<U, T, S>) -> Option<Msg<U, T>> {
        None
    }

    #[allow(unused_variables)]
    #[inline]
    fn on_focus_lost(&mut self, cx: &mut Context<U, T, S>) -> Option<Msg<U, T>> {
        None
    }

    #[allow(unused_variables)]
    #[inline]
    fn on_key_press(
        &mut self,
        cx: &mut Context<U, T, S>,
        code: KeyCode,
        modifiers: KeyModifiers,
    ) -> Option<Msg<U, T>> {
        None
    }

    #[allow(unused_variables)]
    #[inline]
    fn on_mouse_event(
        &mut self,
        cx: &mut Context<U, T, S>,
        event: MouseEvent,
    ) -> Option<Msg<U, T>> {
        None
    }

    #[allow(unused_variables)]
    #[inline]
    fn on_paste(&mut self, cx: &mut Context<U, T, S>, value: &str) -> Option<Msg<U, T>> {
        None
    }

    #[allow(unused_variables)]
    #[inline]
    fn on_resize(
        &mut self,
        cx: &mut Context<U, T, S>,
        width: u16,
        height: u16,
    ) -> Option<Msg<U, T>> {
        None
    }

    #[allow(unused_variables)]
    #[inline]
    fn on_load(&mut self, cx: &mut Context<U, T, S>) {}

    #[allow(unused_variables)]
    #[inline]
    fn on_quit(&mut self, cx: &mut Context<U, T, S>) {}

    #[allow(unused_variables)]
    #[inline]
    fn update(&mut self, cx: &mut Context<U, T, S>, msg: U) {}

    #[allow(unused_variables)]
    #[inline]
    fn render(&mut self, cx: &Context<U, T, S>, frame: &mut Frame, area: Rect) {}
}
