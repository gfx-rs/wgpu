#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::needless_borrowed_reference)]

use criterion::*;
use std::{fs, path::PathBuf, slice};

fn gather_inputs(folder: &str, extension: &str) -> Vec<Box<[u8]>> {
    let mut list = Vec::new();
    let read_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(folder)
        .read_dir()
        .unwrap();
    for file_entry in read_dir {
        match file_entry {
            Ok(entry) => match entry.path().extension() {
                Some(ostr) if ostr == extension => {
                    let input = fs::read(entry.path()).unwrap_or_default();
                    list.push(input.into_boxed_slice());
                }
                _ => continue,
            },
            Err(e) => {
                log::warn!("Skipping file: {:?}", e);
                continue;
            }
        }
    }
    list
}

fn parse_glsl(stage: naga::ShaderStage, inputs: &[Box<[u8]>]) {
    let mut parser = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage,
        defines: Default::default(),
    };
    for input in inputs.iter() {
        let string = std::str::from_utf8(input).unwrap();
        parser.parse(&options, string).unwrap();
    }
}

fn frontends(c: &mut Criterion) {
    let mut group = c.benchmark_group("front");
    #[cfg(all(feature = "wgsl-in", feature = "serialize", feature = "deserialize"))]
    group.bench_function("bin", |b| {
        let inputs_wgsl = gather_inputs("tests/in", "wgsl");
        let mut frontend = naga::front::wgsl::Frontend::new();
        let inputs_bin = inputs_wgsl
            .iter()
            .map(|input| {
                let string = std::str::from_utf8(input).unwrap();
                let module = frontend.parse(string).unwrap();
                bincode::serialize(&module).unwrap()
            })
            .collect::<Vec<_>>();
        b.iter(move || {
            for input in inputs_bin.iter() {
                bincode::deserialize::<naga::Module>(input).unwrap();
            }
        });
    });
    #[cfg(feature = "wgsl-in")]
    group.bench_function("wgsl", |b| {
        let inputs_wgsl = gather_inputs("tests/in", "wgsl");
        let inputs = inputs_wgsl
            .iter()
            .map(|input| std::str::from_utf8(input).unwrap())
            .collect::<Vec<_>>();
        let mut frontend = naga::front::wgsl::Frontend::new();
        b.iter(move || {
            for &input in inputs.iter() {
                frontend.parse(input).unwrap();
            }
        });
    });
    #[cfg(feature = "spv-in")]
    group.bench_function("spv", |b| {
        let inputs = gather_inputs("tests/in/spv", "spv");
        b.iter(move || {
            let options = naga::front::spv::Options::default();
            for input in inputs.iter() {
                let spv =
                    unsafe { slice::from_raw_parts(input.as_ptr() as *const u32, input.len() / 4) };
                let parser = naga::front::spv::Frontend::new(spv.iter().cloned(), &options);
                parser.parse().unwrap();
            }
        });
    });
    #[cfg(feature = "glsl-in")]
    group.bench_function("glsl", |b| {
        let vert = gather_inputs("tests/in/glsl", "vert");
        b.iter(move || parse_glsl(naga::ShaderStage::Vertex, &vert));
        let frag = gather_inputs("tests/in/glsl", "frag");
        b.iter(move || parse_glsl(naga::ShaderStage::Vertex, &frag));
        //TODO: hangs for some reason!
        //let comp = gather_inputs("tests/in/glsl", "comp");
        //b.iter(move || parse_glsl(naga::ShaderStage::Compute, &comp));
    });
}

#[cfg(feature = "wgsl-in")]
fn gather_modules() -> Vec<naga::Module> {
    let inputs = gather_inputs("tests/in", "wgsl");
    let mut frontend = naga::front::wgsl::Frontend::new();
    inputs
        .iter()
        .map(|input| {
            let string = std::str::from_utf8(input).unwrap();
            frontend.parse(string).unwrap()
        })
        .collect()
}
#[cfg(not(feature = "wgsl-in"))]
fn gather_modules() -> Vec<naga::Module> {
    Vec::new()
}

fn validation(c: &mut Criterion) {
    let inputs = gather_modules();
    let mut group = c.benchmark_group("valid");
    group.bench_function("safe", |b| {
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        b.iter(|| {
            for input in inputs.iter() {
                validator.validate(input).unwrap();
            }
        });
    });
    group.bench_function("unsafe", |b| {
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::empty(),
            naga::valid::Capabilities::all(),
        );
        b.iter(|| {
            for input in inputs.iter() {
                validator.validate(input).unwrap();
            }
        });
    });
}

fn backends(c: &mut Criterion) {
    let inputs = {
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::empty(),
            naga::valid::Capabilities::default(),
        );
        let input_modules = gather_modules();
        input_modules
            .into_iter()
            .flat_map(|module| validator.validate(&module).ok().map(|info| (module, info)))
            .collect::<Vec<_>>()
    };

    let mut group = c.benchmark_group("back");
    #[cfg(feature = "wgsl-out")]
    group.bench_function("wgsl", |b| {
        b.iter(|| {
            let mut string = String::new();
            let flags = naga::back::wgsl::WriterFlags::empty();
            for &(ref module, ref info) in inputs.iter() {
                let mut writer = naga::back::wgsl::Writer::new(&mut string, flags);
                writer.write(module, info).unwrap();
                string.clear();
            }
        });
    });

    #[cfg(feature = "spv-out")]
    group.bench_function("spv", |b| {
        b.iter(|| {
            let mut data = Vec::new();
            let options = naga::back::spv::Options::default();
            for &(ref module, ref info) in inputs.iter() {
                let mut writer = naga::back::spv::Writer::new(&options).unwrap();
                writer.write(module, info, None, &None, &mut data).unwrap();
                data.clear();
            }
        });
    });
    #[cfg(feature = "spv-out")]
    group.bench_function("spv-separate", |b| {
        b.iter(|| {
            let mut data = Vec::new();
            let options = naga::back::spv::Options::default();
            for &(ref module, ref info) in inputs.iter() {
                let mut writer = naga::back::spv::Writer::new(&options).unwrap();
                for ep in module.entry_points.iter() {
                    let pipeline_options = naga::back::spv::PipelineOptions {
                        shader_stage: ep.stage,
                        entry_point: ep.name.clone(),
                    };
                    writer
                        .write(module, info, Some(&pipeline_options), &None, &mut data)
                        .unwrap();
                    data.clear();
                }
            }
        });
    });

    #[cfg(feature = "msl-out")]
    group.bench_function("msl", |b| {
        b.iter(|| {
            let mut string = String::new();
            let options = naga::back::msl::Options::default();
            for &(ref module, ref info) in inputs.iter() {
                let pipeline_options = naga::back::msl::PipelineOptions::default();
                let mut writer = naga::back::msl::Writer::new(&mut string);
                writer
                    .write(module, info, &options, &pipeline_options)
                    .unwrap();
                string.clear();
            }
        });
    });

    #[cfg(feature = "hlsl-out")]
    group.bench_function("hlsl", |b| {
        b.iter(|| {
            let options = naga::back::hlsl::Options::default();
            let mut string = String::new();
            for &(ref module, ref info) in inputs.iter() {
                let mut writer = naga::back::hlsl::Writer::new(&mut string, &options);
                let _ = writer.write(module, info); // may fail on unimplemented things
                string.clear();
            }
        });
    });

    #[cfg(feature = "glsl-out")]
    group.bench_function("glsl-separate", |b| {
        b.iter(|| {
            let mut string = String::new();
            let options = naga::back::glsl::Options {
                version: naga::back::glsl::Version::new_gles(320),
                writer_flags: naga::back::glsl::WriterFlags::empty(),
                binding_map: Default::default(),
                zero_initialize_workgroup_memory: true,
            };
            for &(ref module, ref info) in inputs.iter() {
                for ep in module.entry_points.iter() {
                    let pipeline_options = naga::back::glsl::PipelineOptions {
                        shader_stage: ep.stage,
                        entry_point: ep.name.clone(),
                        multiview: None,
                    };

                    // might be `Err` if missing features
                    if let Ok(mut writer) = naga::back::glsl::Writer::new(
                        &mut string,
                        module,
                        info,
                        &options,
                        &pipeline_options,
                        naga::proc::BoundsCheckPolicies::default(),
                    ) {
                        let _ = writer.write(); // might be `Err` if unsupported
                    }

                    string.clear();
                }
            }
        });
    });
}

criterion_group!(criterion, frontends, validation, backends,);
criterion_main!(criterion);
