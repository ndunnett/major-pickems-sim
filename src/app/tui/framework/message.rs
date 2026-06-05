use std::fmt::Debug;

pub enum Msg<U, T> {
    Quit,
    Redraw,
    Update(U),
    SpawnTask(T),
}

impl<U: Debug, T: Debug> Debug for Msg<U, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Quit => write!(f, "Quit"),
            Self::Redraw => write!(f, "Redraw"),
            Self::Update(update) => f.debug_tuple("Update").field(update).finish(),
            Self::SpawnTask(task) => f.debug_tuple("SpawnTask").field(task).finish(),
        }
    }
}

#[allow(clippy::expl_impl_clone_on_copy)]
impl<U: Clone, T: Clone> Clone for Msg<U, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Redraw => Self::Redraw,
            Self::Quit => Self::Quit,
            Self::Update(update) => Self::Update(update.clone()),
            Self::SpawnTask(task) => Self::SpawnTask(task.clone()),
        }
    }
}

impl<U: Copy, T: Copy> Copy for Msg<U, T> {}
