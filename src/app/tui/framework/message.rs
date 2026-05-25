use std::fmt::Debug;

pub enum Msg<U, N, T> {
    Quit,
    Redraw,
    Update(U),
    Notify(N),
    SpawnTask(T),
}

impl<U: Debug, N: Debug, T: Debug> Debug for Msg<U, N, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quit => write!(f, "Quit"),
            Self::Redraw => write!(f, "Redraw"),
            Self::Update(update) => f.debug_tuple("Update").field(update).finish(),
            Self::Notify(notify) => f.debug_tuple("Notify").field(notify).finish(),
            Self::SpawnTask(task) => f.debug_tuple("SpawnTask").field(task).finish(),
        }
    }
}

#[allow(clippy::expl_impl_clone_on_copy)]
impl<U: Clone, N: Clone, T: Clone> Clone for Msg<U, N, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Redraw => Self::Redraw,
            Self::Quit => Self::Quit,
            Self::Update(update) => Self::Update(update.clone()),
            Self::Notify(notify) => Self::Notify(notify.clone()),
            Self::SpawnTask(task) => Self::SpawnTask(task.clone()),
        }
    }
}

impl<U: Copy, N: Copy, T: Copy> Copy for Msg<U, N, T> {}
