use std::{
    borrow::Cow,
    fmt, io,
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};
use tracing::{
    field::{Field, Visit},
    span, Event, Metadata, Subscriber,
};
use tracing_subscriber::{
    layer::{Context, SubscriberExt},
    registry::LookupSpan,
    EnvFilter, Layer,
};

/// Set up the "standard" logger.
///
/// This is fairly inflexible, but a good default to start with. If you need more customization,
/// take what this function does and implement it however you need.
///
/// # Args
///
/// - `chrome_tracing_path` if set to `Some`, will create a trace compatible with chrome://tracing
///   at that location.  
pub fn initialize_default_subscriber(chrome_trace_path: Option<impl AsRef<Path>>) {
    let chrome_tracing_layer_opt =
        chrome_trace_path.map(|path| ChromeTracingLayer::with_file(path).unwrap());

    // Tracing currently doesn't support type erasure with layer composition
    if let Some(chrome_tracing_layer) = chrome_tracing_layer_opt {
        tracing::subscriber::set_global_default(
            tracing_subscriber::Registry::default()
                .with(chrome_tracing_layer)
                .with(tracing_subscriber::fmt::Layer::new())
                .with(EnvFilter::from_default_env()),
        )
        .unwrap();
    } else {
        tracing::subscriber::set_global_default(
            tracing_subscriber::Registry::default()
                .with(tracing_subscriber::fmt::Layer::new())
                .with(EnvFilter::from_default_env()),
        )
        .unwrap();
    }
}

static CURRENT_PROCESS_ID: once_cell::sync::Lazy<u32> =
    once_cell::sync::Lazy::new(|| std::process::id());

thread_local! {
    static CURRENT_THREAD_ID: usize = thread_id::get();
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum EventType {
    Begin,
    Event,
    End,
}

/// A layer to add to a [`tracing_subscriber::Registry`] to output to a chrome
/// trace.
///
/// If you want an easy "set and forget" method of installing this and normal
/// tracing logging, call [`initialize_default_subscriber`].
#[derive(Debug, Clone)]
pub struct ChromeTracingLayer {
    file: Arc<Mutex<std::fs::File>>,
    start_time: Instant,
}

impl ChromeTracingLayer {
    /// Create a trace which outputs to the given file. The file will be cleared if it exits.
    pub fn with_file(file: impl AsRef<Path>) -> io::Result<Self> {
        std::fs::File::create(file).map(|mut file| {
            writeln!(file, "[").unwrap();
            ChromeTracingLayer {
                file: Arc::new(Mutex::new(file)),
                start_time: Instant::now(),
            }
        })
    }

    fn write_event(
        &self,
        mut fields: Option<EventVisitor>,
        metadata: &'static Metadata<'static>,
        event_type: EventType,
    ) {
        if let Some(EventVisitor { trace: false, .. }) = fields {
            return;
        }

        let current_time = Instant::now();

        let diff = current_time - self.start_time;
        let diff_in_us = diff.as_micros();

        let event_type_str = match event_type {
            EventType::Begin => "B",
            EventType::Event => "i",
            EventType::End => "E",
        };

        let instant_scope = if event_type == EventType::Event {
            r#","s": "p""#
        } else {
            ""
        };

        let name = match event_type {
            EventType::Event => fields
                .as_mut()
                .and_then(|fields| fields.message.take())
                .map_or_else(
                    || {
                        let name = metadata.name();
                        // The default name for events has a path in it, filter paths only on windows
                        if cfg!(target_os = "windows") {
                            Cow::Owned(name.replace("\\", "\\\\"))
                        } else {
                            Cow::Borrowed(name)
                        }
                    },
                    Cow::from,
                ),
            EventType::Begin | EventType::End => Cow::Borrowed(metadata.name()),
        };

        let category = fields
            .as_mut()
            .and_then(|fields| fields.category.take())
            .map_or_else(|| Cow::Borrowed("trace"), Cow::from);

        let mut file = self.file.lock().unwrap();
        writeln!(
            file,
            r#"{{ "name": "{}", "cat": "{}", "ph": "{}", "ts": {}, "pid": {}, "tid": {} {} }},"#,
            name,
            category,
            event_type_str,
            diff_in_us,
            *CURRENT_PROCESS_ID,
            CURRENT_THREAD_ID.with(|v| *v),
            instant_scope,
        )
        .unwrap();
    }
}

impl Drop for ChromeTracingLayer {
    fn drop(&mut self) {
        println!("drop");
        let mut file = self.file.lock().unwrap();
        writeln!(file, "]").unwrap();
        file.flush().unwrap();
    }
}

#[derive(Debug, Default)]
struct EventVisitor {
    message: Option<String>,
    category: Option<String>,
    trace: bool,
}

impl Visit for EventVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        match field.name() {
            "trace" => self.trace = value,
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        match field.name() {
            "message" => self.message = Some(format!("{:?}", value)),
            "category" => self.category = Some(format!("{:?}", value)),
            _ => {}
        }
    }
}

impl<S> Layer<S> for ChromeTracingLayer
where
    S: Subscriber + for<'span> LookupSpan<'span>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut event_visitor = EventVisitor::default();
        event.record(&mut event_visitor);

        self.write_event(Some(event_visitor), event.metadata(), EventType::Event);
    }

    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();
        self.write_event(None, span.metadata(), EventType::Begin);
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if std::thread::panicking() {
            return;
        }

        let span = ctx.span(id).unwrap();
        self.write_event(None, span.metadata(), EventType::End);
    }
}
