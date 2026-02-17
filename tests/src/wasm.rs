#![cfg(target_arch = "wasm32")]

use crate::{execute_test, GpuTestInitializer};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(inline_js = "
  export function is_test_runner() {
    let url = new URL(window.location.href);

    return url.searchParams.get(`testrunner`) == `true`;
  }

  export function report(params) {
    if (is_test_runner()) {
      let url = new URL(`/report`, window.location.href);
      url.search = new URLSearchParams(params).toString();
      fetch(url.toString());
    } else {
      console.log(params);
    }
  }

  export function send_message_to_runner_js(kind, message) {
    report({ kind, message });
  }
")]

extern "C" {
    #[wasm_bindgen()]
    fn send_message_to_runner_js(kind: String, message: String);
}

pub fn send_message_to_runner(kind: MessageKind, message: String) {
    send_message_to_runner_js(kind.to_string(), message);
}

pub enum MessageKind {
    Success,
    Failure,
    LogError,
    LogWarn,
    LogInfo,
    LogDebug,
    LogTrace,
}

impl MessageKind {
    pub fn to_string(&self) -> String {
        match self {
            Self::Success => "Success".to_string(),
            Self::Failure => "Failure".to_string(),
            Self::LogError => "LogError".to_string(),
            Self::LogWarn => "LogWarn".to_string(),
            Self::LogInfo => "LogInfo".to_string(),
            Self::LogDebug => "LogDebug".to_string(),
            Self::LogTrace => "LogTrace".to_string(),
        }
    }
}

pub fn main(initializers: Vec<GpuTestInitializer>) {
    std::panic::set_hook(Box::new(|e| {
        send_message_to_runner(MessageKind::LogError, format!("{}", e));
    }));

    wasm_bindgen_futures::spawn_local(async {
        for initializer in initializers {
            let test = initializer();
            execute_test(None, test, None).await;
        }

        send_message_to_runner(MessageKind::Success, "Tests complete".to_string());
    });
}
