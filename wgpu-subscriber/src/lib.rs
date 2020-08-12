pub use chrome::*;
pub use fmt_layer::*;
use std::path::Path;
use tracing_subscriber::{layer::SubscriberExt as _, EnvFilter};

mod chrome;
mod fmt_layer;

/// Set up the "standard" logger.
///
/// This is fairly inflexible, but a good default to start with. If you need more customization,
/// take what this function does and implement it however you need.
///
/// If this function is called, you should **not** set up a log-based logger like env_logger
/// or fern. This will result in duplicate messages.
///
/// # Args
///
/// - `chrome_tracing_path` if set to `Some`, will create a trace compatible with chrome://tracing
///   at that location.
pub fn initialize_default_subscriber(chrome_trace_path: Option<&Path>) {
    let chrome_tracing_layer_opt =
        chrome_trace_path.map(|path| ChromeTracingLayer::with_file(path).unwrap());

    // Tracing currently doesn't support type erasure with layer composition
    if let Some(chrome_tracing_layer) = chrome_tracing_layer_opt {
        tracing::subscriber::set_global_default(
            tracing_subscriber::Registry::default()
                .with(chrome_tracing_layer)
                .with(FmtLayer::new())
                .with(EnvFilter::from_default_env()),
        )
        .unwrap();
    } else {
        tracing::subscriber::set_global_default(
            tracing_subscriber::Registry::default()
                .with(FmtLayer::new())
                .with(EnvFilter::from_default_env()),
        )
        .unwrap();
    }

    tracing_log::LogTracer::init().unwrap();
}

thread_local! {
    static CURRENT_THREAD_ID: usize = thread_id::get();
}
