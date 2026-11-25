use std::{
    fmt::Debug,
    io::{self, Write},
    ops::{Deref, DerefMut},
    sync::mpsc,
    time::Instant,
};

use ratatui::{
    CompletedFrame, Frame, Terminal,
    backend::CrosstermBackend,
    crossterm::{
        cursor,
        event::{Event, read as read_event},
        execute, terminal,
    },
    layout::Rect,
};

use super::Msg;

pub enum Wake<U, T> {
    Input(io::Result<Event>),
    Message(Msg<U, T>),
}

pub struct Engine<U, T, W: Write> {
    terminal: TerminalBackend<W>,
    receiver: mpsc::Receiver<Wake<U, T>>,
    task_sender: mpsc::Sender<T>,
}

impl<U, T, W> Engine<U, T, W>
where
    U: Send + Sync + 'static,
    T: Send + Sync + 'static,
    W: Write,
{
    pub fn new<F>(task_handler: F, writer: W) -> anyhow::Result<Self>
    where
        F: Fn(T) -> Option<Msg<U, T>> + Send + Sync + 'static,
    {
        let terminal = TerminalBackend::new(writer)?;
        let (sender, receiver) = mpsc::channel();
        let input_sender = sender.clone();

        std::thread::spawn(move || {
            loop {
                let input = read_event();
                let failed = input.is_err();

                if input_sender.send(Wake::Input(input)).is_err() || failed {
                    return;
                }
            }
        });

        let (task_sender, task_receiver) = mpsc::channel();

        std::thread::spawn(move || {
            while let Ok(task) = task_receiver.recv() {
                if let Some(msg) = task_handler(task)
                    && sender.send(Wake::Message(msg)).is_err()
                {
                    return;
                }
            }
        });

        Ok(Self {
            terminal,
            receiver,
            task_sender,
        })
    }

    #[inline]
    pub fn draw<F>(&mut self, render_callback: F) -> anyhow::Result<CompletedFrame<'_>>
    where
        F: FnOnce(&mut Frame),
    {
        Ok(self.terminal.draw(render_callback)?)
    }

    #[inline]
    pub fn resize(&mut self, area: Rect) -> anyhow::Result<()> {
        Ok(self.terminal.resize(area)?)
    }

    #[inline]
    pub fn send_task(&self, t: T) -> anyhow::Result<()> {
        Ok(self.task_sender.send(t)?)
    }

    pub fn wait(&self, deadline: Option<Instant>) -> anyhow::Result<Option<Wake<U, T>>> {
        let Some(deadline) = deadline else {
            return Ok(Some(self.receiver.recv()?));
        };

        let timeout = deadline.saturating_duration_since(Instant::now());

        match self.receiver.recv_timeout(timeout) {
            Ok(msg) => Ok(Some(msg)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(e) => Err(e)?,
        }
    }
}

impl<U: Debug, T: Debug, W: Write + Debug> Debug for Engine<U, T, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("terminal", &self.terminal)
            .field("receiver", &self.receiver)
            .field("task_sender", &self.task_sender)
            .finish()
    }
}

struct TerminalRawMode;

impl TerminalRawMode {
    #[inline]
    pub fn enter() -> anyhow::Result<Self> {
        terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for TerminalRawMode {
    #[inline]
    fn drop(&mut self) {
        _ = terminal::disable_raw_mode();
    }
}

struct TerminalBackend<W: Write> {
    #[allow(unused)]
    raw_mode: TerminalRawMode,
    terminal: Terminal<CrosstermBackend<W>>,
}

impl<W: Write> TerminalBackend<W> {
    fn new(writer: W) -> anyhow::Result<Self> {
        let raw_mode = TerminalRawMode::enter()?;
        let mut terminal = Terminal::new(CrosstermBackend::new(writer))?;

        execute!(
            terminal.backend_mut(),
            terminal::EnterAlternateScreen,
            cursor::Hide
        )?;

        Ok(Self { raw_mode, terminal })
    }
}

impl<W: Write> Drop for TerminalBackend<W> {
    fn drop(&mut self) {
        _ = execute!(
            self.terminal.backend_mut(),
            terminal::LeaveAlternateScreen,
            cursor::Show
        );
    }
}

impl<W: Write + Debug> Debug for TerminalBackend<W> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.terminal.fmt(f)
    }
}

impl<W: Write> Deref for TerminalBackend<W> {
    type Target = Terminal<CrosstermBackend<W>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.terminal
    }
}

impl<W: Write> DerefMut for TerminalBackend<W> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.terminal
    }
}
