use std::{io::Write, time::Instant};

use ratatui::{crossterm::event::Event, layout::Rect};

use super::{Context, Engine, Entity, Msg, Wake};

pub trait Root<U, T, S>
where
    Self: Entity<U, T, S> + Sized + 'static,
    U: Send + Sync + 'static,
    T: Send + Sync + 'static,
    S: 'static,
{
    fn handle_task(task: T) -> Option<Msg<U, T>>;

    fn run<W: Write>(mut self, initial_state: impl FnOnce() -> S, writer: W) -> anyhow::Result<()> {
        let mut engine = Engine::new(Self::handle_task, writer)?;
        let mut cx = Context::new(initial_state);
        self.on_load(&mut cx);

        loop {
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

            if cx.should_exit {
                break;
            }

            if cx.should_redraw {
                engine.draw(|frame| self.render(&cx, frame, frame.area()))?;
                cx.should_redraw = false;
            }

            let deadline = self.next_deadline();
            let wake = engine.wait(deadline)?;
            cx.tick = Instant::now();

            if deadline.is_some_and(|instant| instant <= cx.tick) {
                self.on_tick(&mut cx);
            }

            match wake {
                Some(Wake::Input(input)) => {
                    let event = input?;

                    if let Event::Resize(w, h) = event {
                        engine.resize(Rect::new(0, 0, w, h))?;
                        cx.should_redraw = true;
                    }

                    if let Some(msg) = self.dispatch_event(&mut cx, &event) {
                        cx.queue_message(msg);
                    }
                }
                Some(Wake::Message(msg)) => cx.queue_message(msg),
                None => {}
            }
        }

        self.on_quit(&mut cx);
        Ok(())
    }
}
