use std::{
    io::Write,
    thread::sleep,
    time::{Duration, Instant},
};

use ratatui::{crossterm::event::Event, layout::Rect};

use super::{Context, Engine, Entity, Msg};

pub trait Root<U, T, S>
where
    Self: Entity<U, T, S> + Sized + 'static,
    U: Send + Sync + 'static,
    T: Send + Sync + 'static,
    S: 'static,
{
    const MAX_FPS: u64;
    const TICK_TIME: Duration = Duration::from_nanos(1_000_000_000 / Self::MAX_FPS);

    fn handle_task(task: T) -> Option<Msg<U, T>>;

    fn run<W: Write>(mut self, initial_state: impl FnOnce() -> S, writer: W) -> anyhow::Result<()> {
        let mut engine = Engine::new(Self::handle_task, writer)?;
        let mut cx = Context::new(initial_state);
        self.on_load(&mut cx);

        while !cx.should_exit {
            if cx.should_redraw {
                engine.draw(|frame| self.render(&cx, frame, frame.area()))?;
                cx.should_redraw = false;
            }

            sleep(Self::TICK_TIME.saturating_sub(cx.tick.elapsed()));
            cx.tick = Instant::now();
            self.on_tick(&mut cx);

            while let Some(msg) = engine.receive_msg()? {
                cx.queue_message(msg);
            }

            while let Some(event) = engine.receive_event()? {
                if let Event::Resize(w, h) = event {
                    engine.resize(Rect::new(0, 0, w, h))?;
                    cx.should_redraw = true;
                }

                if let Some(msg) = self.dispatch_event(&mut cx, &event) {
                    cx.queue_message(msg);
                }
            }

            while let Some(msg) = cx.pop_message() {
                match msg {
                    Msg::Quit => cx.should_exit = true,
                    Msg::Redraw => cx.should_redraw = true,
                    Msg::Update(update) => {
                        self.update(&mut cx, update);
                        cx.should_redraw = true;
                    }
                    Msg::SpawnTask(task) => {
                        engine.send_task(task)?;
                    }
                }
            }
        }

        self.on_quit(&mut cx);
        Ok(())
    }
}
