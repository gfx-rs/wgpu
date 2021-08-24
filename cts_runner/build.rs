// Copyright 2018-2020 the Deno authors. All rights reserved. MIT license.

use deno_core::JsRuntime;
use deno_core::RuntimeOptions;
use deno_web::BlobStore;

use std::env;
use std::path::PathBuf;

fn main() {
    let ctsr_extension = deno_core::Extension::builder()
        .js(deno_core::include_js_files!(
          prefix "deno:wgpu_cts_runner",
          "src/bootstrap.js",
        ))
        .build();

    let o = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let snapshot_path = o.join("CTSR_SNAPSHOT.bin");
    let options = RuntimeOptions {
        will_snapshot: true,
        extensions: vec![
            deno_webidl::init(),
            deno_console::init(),
            deno_url::init(),
            deno_web::init(BlobStore::default(), None),
            deno_timers::init::<deno_timers::NoTimersPermission>(),
            deno_webgpu::init(true),
            ctsr_extension,
        ],
        ..Default::default()
    };
    let mut isolate = JsRuntime::new(options);

    let snapshot = isolate.snapshot();
    let snapshot_slice: &[u8] = &*snapshot;
    println!("Snapshot size: {}", snapshot_slice.len());
    std::fs::write(&snapshot_path, snapshot_slice).unwrap();
    println!("Snapshot written to: {} ", snapshot_path.display());
}
