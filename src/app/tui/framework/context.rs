use std::{
    collections::VecDeque,
    fmt::Debug,
    ops::{Deref, DerefMut},
    time::Instant,
};

use super::Msg;

pub struct Context<U, N, T, S> {
    state: S,
    messages: VecDeque<Msg<U, N, T>>,
    pub(super) tick: Instant,
    pub(super) should_exit: bool,
    pub(super) should_redraw: bool,
}

impl<U, N, T, S> Context<U, N, T, S> {
    pub fn new(initial_state: impl FnOnce() -> S) -> Self {
        Self {
            state: initial_state(),
            messages: VecDeque::new(),
            tick: Instant::now(),
            should_exit: false,
            should_redraw: true,
        }
    }

    #[inline]
    pub const fn tick(&self) -> &Instant {
        &self.tick
    }

    #[inline]
    pub fn queue_message(&mut self, msg: Msg<U, N, T>) {
        self.messages.push_back(msg);
    }

    #[inline]
    pub(super) fn pop_message(&mut self) -> Option<Msg<U, N, T>> {
        self.messages.pop_front()
    }

    #[inline]
    pub fn notify(&mut self, msg: N) {
        self.queue_message(Msg::Notify(msg));
    }

    #[inline]
    pub fn update(&mut self, msg: U) {
        self.queue_message(Msg::Update(msg));
    }

    #[inline]
    pub fn task(&mut self, msg: T) {
        self.queue_message(Msg::SpawnTask(msg));
    }

    #[inline]
    pub fn redraw(&mut self) {
        self.queue_message(Msg::Redraw);
    }

    #[inline]
    pub fn quit(&mut self) {
        self.queue_message(Msg::Quit);
    }
}

impl<U, N, T, S> Deref for Context<U, N, T, S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<U, N, T, S> DerefMut for Context<U, N, T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl<U: Debug, N: Debug, T: Debug, S: Debug> Debug for Context<U, N, T, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("state", &self.state)
            .field("messages", &self.messages)
            .field("tick", &self.tick)
            .field("should_exit", &self.should_exit)
            .field("should_redraw", &self.should_redraw)
            .finish()
    }
}
