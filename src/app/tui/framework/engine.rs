use std::{
    fmt::Debug,
    io::Write,
    ops::{Deref, DerefMut},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::JoinHandle,
    time::Duration,
};

use ratatui::{
    CompletedFrame, Frame, Terminal,
    backend::CrosstermBackend,
    crossterm::{
        cursor,
        event::{Event, poll as poll_event, read as read_event},
        execute, terminal,
    },
    layout::Rect,
};

use super::Msg;

const POLL_TIME: Duration = Duration::from_millis(100);

pub struct Engine<U, T, W: Write> {
    terminal: TerminalBackend<W>,
    event_receiver: mpsc::Receiver<Event>,
    task_sender: mpsc::Sender<T>,
    msg_receiver: mpsc::Receiver<Msg<U, T>>,
    shutdown_signal: Arc<AtomicBool>,
    event_handler: Option<JoinHandle<()>>,
    task_handler: Option<JoinHandle<()>>,
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
        let shutdown_signal = Arc::new(AtomicBool::new(false));
        let event_shutdown = shutdown_signal.clone();
        let (event_sender, event_receiver) = mpsc::channel();

        let event_handler = Some(std::thread::spawn(move || {
            loop {
                let Ok(event_available) = poll_event(POLL_TIME) else {
                    return;
                };

                if event_available {
                    let Ok(event) = read_event() else {
                        return;
                    };

                    if event_sender.send(event).is_err() {
                        return;
                    }
                }

                if event_shutdown.load(Ordering::Relaxed) {
                    return;
                }
            }
        }));

        let task_shutdown = shutdown_signal.clone();
        let (task_sender, task_receiver) = mpsc::channel();
        let (msg_sender, msg_receiver) = mpsc::channel();

        let task_handler = Some(std::thread::spawn(move || {
            loop {
                match task_receiver.recv_timeout(POLL_TIME) {
                    Ok(task) => {
                        if let Some(msg) = task_handler(task)
                            && msg_sender.send(msg).is_err()
                        {
                            return;
                        }
                    }
                    Err(e) => match e {
                        mpsc::RecvTimeoutError::Timeout => {}
                        mpsc::RecvTimeoutError::Disconnected => return,
                    },
                }

                if task_shutdown.load(Ordering::Relaxed) {
                    return;
                }
            }
        }));

        Ok(Self {
            terminal,
            event_receiver,
            task_sender,
            msg_receiver,
            shutdown_signal,
            event_handler,
            task_handler,
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

    pub fn receive_event(&self) -> anyhow::Result<Option<Event>> {
        match self.event_receiver.try_recv() {
            Ok(event) => Ok(Some(event)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(e) => Err(e)?,
        }
    }

    pub fn receive_msg(&self) -> anyhow::Result<Option<Msg<U, T>>> {
        match self.msg_receiver.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(e) => Err(e)?,
        }
    }
}

impl<U, T, W: Write> Drop for Engine<U, T, W> {
    fn drop(&mut self) {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        if let Some(handle) = self.event_handler.take() {
            _ = handle.join();
        }

        if let Some(handle) = self.task_handler.take() {
            _ = handle.join();
        }
    }
}

impl<U: Debug, T: Debug, W: Write + Debug> Debug for Engine<U, T, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Engine")
            .field("terminal", &self.terminal)
            .field("event_receiver", &self.event_receiver)
            .field("task_sender", &self.task_sender)
            .field("msg_receiver", &self.msg_receiver)
            .field("shutdown_signal", &self.shutdown_signal)
            .field("event_handler", &self.event_handler)
            .field("task_handler", &self.task_handler)
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
