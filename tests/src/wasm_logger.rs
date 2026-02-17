#![cfg(target_arch = "wasm32")]

use log::{Log, Metadata, Record};

use crate::wasm::{send_message_to_runner, MessageKind};

pub static LOGGER: WasmLogger = WasmLogger {};

pub struct WasmLogger {}

impl Log for WasmLogger {
    #[inline]
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let message = record.args().to_string();
        match record.level() {
            log::Level::Error => send_message_to_runner(MessageKind::LogError, message),
            log::Level::Warn => send_message_to_runner(MessageKind::LogWarn, message),
            log::Level::Info => send_message_to_runner(MessageKind::LogInfo, message),
            log::Level::Debug => send_message_to_runner(MessageKind::LogDebug, message),
            log::Level::Trace => send_message_to_runner(MessageKind::LogTrace, message),
        };
    }

    fn flush(&self) {}
}
