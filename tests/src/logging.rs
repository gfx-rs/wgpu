use std::sync::{mpsc, Mutex};

use log::{Level, LevelFilter, SetLoggerError};

static CAPTURE: Mutex<Option<mpsc::Sender<(Level, String)>>> = Mutex::new(None);

fn dispatch(output: fern::Output) -> fern::Dispatch {
    fern::Dispatch::new().chain(output).chain(
        fern::Dispatch::new()
            .level(LevelFilter::Info)
            .filter(|_| CAPTURE.lock().unwrap().is_some())
            .chain(fern::Output::call(|record| {
                let sender = CAPTURE.lock().unwrap().clone();
                if let Some(sender) = sender {
                    let _ = sender.send((record.level(), record.args().to_string()));
                }
            })),
    )
}

pub(crate) fn init() -> Result<(), SetLoggerError> {
    #[cfg(not(target_arch = "wasm32"))]
    let (output, level) = {
        let logger = env_logger::Builder::from_default_env().build();
        let level = logger.filter();
        (
            fern::Output::from(Box::new(logger) as Box<dyn log::Log>),
            level,
        )
    };
    #[cfg(target_arch = "wasm32")]
    let (output, level) = (
        fern::Dispatch::new()
            .filter(|metadata| metadata.level() <= log::max_level())
            .chain(fern::Output::call(console_log::log))
            .into(),
        LevelFilter::Info,
    );
    let (_, logger) = dispatch(output).into_log();
    log::set_boxed_logger(logger)?;
    log::set_max_level(level);
    Ok(())
}

/// A log capture that restores the previous log level when dropped.
pub struct LogCapture {
    /// Messages captured at Info level and higher.
    pub messages: mpsc::Receiver<(Level, String)>,
    previous_level: LevelFilter,
}

impl Drop for LogCapture {
    fn drop(&mut self) {
        *CAPTURE.lock().unwrap() = None;
        log::set_max_level(self.previous_level);
    }
}

/// Capture messages at Info level and higher through the test logger.
/// Only one capture can be active at a time.
pub fn capture_logs() -> LogCapture {
    let (sender, messages) = mpsc::channel();
    let mut capture = CAPTURE.lock().unwrap();
    assert!(capture.is_none(), "A log capture is already active");
    *capture = Some(sender);
    let previous_level = log::max_level();
    log::set_max_level(previous_level.max(LevelFilter::Info));
    LogCapture {
        messages,
        previous_level,
    }
}

#[test]
fn capture_preserves_output_filter() {
    let (sender, output) = mpsc::channel();
    let (_, logger) = dispatch(
        fern::Dispatch::new()
            .level(LevelFilter::Error)
            .chain(sender)
            .into(),
    )
    .into_log();
    log::set_boxed_logger(logger).unwrap();
    log::set_max_level(LevelFilter::Error);

    assert!(!log::log_enabled!(Level::Info));
    let mut evaluated = false;
    log::info!("{}", {
        evaluated = true;
        "disabled"
    });
    assert!(!evaluated);

    let capture = capture_logs();
    assert!(log::log_enabled!(Level::Info));
    log::info!("captured");
    assert_eq!(
        capture.messages.try_recv().unwrap(),
        (Level::Info, "captured".into())
    );
    assert!(output.try_recv().is_err());

    log::error!("visible");
    assert_eq!(
        capture.messages.try_recv().unwrap(),
        (Level::Error, "visible".into())
    );
    assert_eq!(output.try_recv().unwrap().trim(), "visible");

    drop(capture);
    assert_eq!(log::max_level(), LevelFilter::Error);
    assert!(!log::log_enabled!(Level::Info));
}
