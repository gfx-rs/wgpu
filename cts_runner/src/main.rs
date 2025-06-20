#![cfg_attr(target_arch = "wasm32", no_main)]
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::{
    env, fmt,
    io::{Read, Write},
    rc::Rc,
};

use deno_core::anyhow::anyhow;
use deno_core::error::AnyError;
use deno_core::op2;
use deno_core::resolve_url_or_path;
use deno_core::serde_json::json;
use deno_core::v8;
use deno_core::JsRuntime;
use deno_core::RuntimeOptions;
use deno_web::BlobStore;
use termcolor::Ansi;
use termcolor::Color::Red;
use termcolor::ColorSpec;
use termcolor::WriteColor;

pub async fn run() -> Result<(), AnyError> {
    let mut args_iter = env::args();
    let _ = args_iter.next();
    let url = args_iter
        .next()
        .ok_or_else(|| anyhow!("missing specifier in first command line argument"))?;
    let specifier = resolve_url_or_path(&url, &env::current_dir()?)?;

    let mut feature_checker = deno_core::FeatureChecker::default();
    feature_checker.enable_feature(deno_webgpu::UNSTABLE_FEATURE_NAME);

    let options = RuntimeOptions {
        module_loader: Some(Rc::new(deno_core::FsModuleLoader)),
        extensions: vec![
            deno_webidl::deno_webidl::init_ops_and_esm(),
            deno_console::deno_console::init_ops_and_esm(),
            deno_url::deno_url::init_ops_and_esm(),
            deno_web::deno_web::init_ops_and_esm::<Permissions>(
                Arc::new(BlobStore::default()),
                None,
            ),
            deno_webgpu::deno_webgpu::init_ops_and_esm(),
            cts_runner::init_ops_and_esm(),
        ],
        feature_checker: Some(Arc::new(feature_checker)),
        ..Default::default()
    };
    let mut js_runtime = JsRuntime::new(options);
    let args = args_iter.collect::<Vec<String>>();
    let cfg = json!({"args": args, "cwd": env::current_dir().unwrap().to_string_lossy() });

    {
        let context = js_runtime.main_context();
        let scope = &mut js_runtime.handle_scope();
        let context_local = v8::Local::new(scope, context);
        let global_obj = context_local.global(scope);
        let bootstrap_str = v8::String::new(scope, "bootstrap").unwrap();
        let bootstrap_fn = global_obj.get(scope, bootstrap_str.into()).unwrap();
        let bootstrap_fn = v8::Local::<v8::Function>::try_from(bootstrap_fn).unwrap();

        let options_v8 = deno_core::serde_v8::to_v8(scope, cfg).unwrap();
        let undefined = v8::undefined(scope);
        bootstrap_fn
            .call(scope, undefined.into(), &[options_v8])
            .unwrap();
    }

    let mod_id = js_runtime.load_main_es_module(&specifier).await?;
    let result = js_runtime.mod_evaluate(mod_id);
    js_runtime.run_event_loop(Default::default()).await?;
    result.await?;

    Ok(())
}

deno_core::extension!(
    cts_runner,
    deps = [deno_webidl, deno_web],
    ops = [op_exit, op_read_file_sync, op_write_file_sync],
    esm_entry_point = "ext:cts_runner/src/bootstrap.js",
    esm = ["src/bootstrap.js"],
    state = |state| {
        state.put(Permissions {});
    }
);

#[op2(fast)]
fn op_exit(code: i32) {
    std::process::exit(code)
}

#[op2]
#[buffer]
fn op_read_file_sync(#[string] path: &str) -> Result<Vec<u8>, std::io::Error> {
    let path = std::path::Path::new(path);
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

#[op2(fast)]
fn op_write_file_sync(#[string] path: &str, #[buffer] buf: &[u8]) -> Result<(), std::io::Error> {
    let path = std::path::Path::new(path);
    let mut file = std::fs::File::create(path)?;
    file.write_all(buf)?;
    Ok(())
}

pub fn unwrap_or_exit<T>(result: Result<T, AnyError>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{}: {:?}", red_bold("error"), error);
            std::process::exit(1);
        }
    }
}

fn style<S: AsRef<str>>(s: S, colorspec: ColorSpec) -> impl fmt::Display {
    let mut v = Vec::new();
    let mut ansi_writer = Ansi::new(&mut v);
    ansi_writer.set_color(&colorspec).unwrap();
    ansi_writer.write_all(s.as_ref().as_bytes()).unwrap();
    ansi_writer.reset().unwrap();
    String::from_utf8_lossy(&v).into_owned()
}

fn red_bold<S: AsRef<str>>(s: S) -> impl fmt::Display {
    let mut style_spec = ColorSpec::new();
    style_spec.set_fg(Some(Red)).set_bold(true);
    style(s, style_spec)
}

// NOP permissions
struct Permissions;

impl deno_web::TimersPermission for Permissions {
    fn allow_hrtime(&mut self) -> bool {
        false
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    env_logger::init();
    unwrap_or_exit(run().await)
}
