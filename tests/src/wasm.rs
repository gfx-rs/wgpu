#![cfg(target_arch = "wasm32")]
use crate::{
    execute_test,
    init::{init_logger, WebDisplayHandle},
    initialize_html_canvas,
    report::{AdapterReport, GpuReport},
    GpuTestInitializer,
};

use exhaust::Exhaust;
use wasm_bindgen::prelude::wasm_bindgen;
use wgpu::{Backends, TextureFormat};

#[wasm_bindgen(inline_js = "
  export function test_success() {
    window.sessionStorage.test_success = `true`;
  }

  export function test_failure(message) {
    window.sessionStorage.test_failure = message;
    console.error(message);
  }

  export function gpu_report(report) {
    window.sessionStorage.gpu_report = report;
    console.log(report);
  }
")]

extern "C" {
    #[wasm_bindgen()]
    fn test_success();

    #[wasm_bindgen()]
    fn test_failure(message: String);

    #[wasm_bindgen()]
    fn gpu_report(report: String);
}

#[wasm_bindgen]
pub async fn run_gpu_report() {
    std::panic::set_hook(Box::new(|e| {
        test_failure(format!("{}", e));
    }));

    init_logger();

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        flags: wgpu::InstanceFlags::debugging(),
        backends: Backends::from_comma_list("gles"),
        ..wgpu::InstanceDescriptor::new_with_display_handle(Box::new(WebDisplayHandle))
    });

    let canvas = initialize_html_canvas();

    let surface = Some(
        instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .expect("could not create surface from canvas"),
    );

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: surface.as_ref(),
            ..Default::default()
        })
        .await
        .unwrap();

    log::info!("{:?}", adapter);

    let features = adapter.features();
    let limits = adapter.limits();
    let downlevel_caps = adapter.get_downlevel_capabilities();
    let texture_format_features = TextureFormat::exhaust()
        .map(|format| (format, adapter.get_texture_format_features(format)))
        .collect();

    let report = AdapterReport {
        info: adapter.get_info(),
        features,
        limits,
        downlevel_caps,
        texture_format_features,
    };

    let report = GpuReport {
        devices: vec![report],
    };

    gpu_report(serde_json::to_string_pretty(&report).expect("Failed to generate gpu report"));
}

pub fn main(initializers: Vec<GpuTestInitializer>, test_name: String) {
    std::panic::set_hook(Box::new(|e| {
        test_failure(format!("{}", e));
    }));

    wasm_bindgen_futures::spawn_local(async move {
        let mut found_test = false;
        for initializer in initializers {
            let test = initializer();
            if test.name == test_name {
                found_test = true;
                execute_test(None, test, None).await;
            }
        }

        if !found_test {
            panic!("Can't find test with this name");
        }

        test_success();
    });
}
