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
                Some(ostr) if &*ostr == extension => {
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
    let mut parser = naga::front::glsl::Parser::default();
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
    #[cfg(all(feature = "serialize", feature = "deserialize"))]
    group.bench_function("bin", |b| {
        let inputs_wgsl = gather_inputs("tests/in", "wgsl");
        let mut parser = naga::front::wgsl::Parser::new();
        let inputs_bin = inputs_wgsl
            .iter()
            .map(|input| {
                let string = std::str::from_utf8(input).unwrap();
                let module = parser.parse(string).unwrap();
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
        let inputs = gather_inputs("tests/in", "wgsl");
        let mut parser = naga::front::wgsl::Parser::new();
        b.iter(move || {
            for input in inputs.iter() {
                let string = std::str::from_utf8(input).unwrap();
                parser.parse(string).unwrap();
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
                let parser = naga::front::spv::Parser::new(spv.iter().cloned(), &options);
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

fn validation(c: &mut Criterion) {
    #[cfg(feature = "wgsl-in")]
    let inputs = {
        let inputs_wgsl = gather_inputs("tests/in", "wgsl");
        let mut parser = naga::front::wgsl::Parser::new();
        inputs_wgsl
            .iter()
            .map(|input| {
                let string = std::str::from_utf8(input).unwrap();
                parser.parse(string).unwrap()
            })
            .collect::<Vec<_>>()
    };
    #[cfg(not(feature = "wgsl-in"))]
    let inputs: Vec<Box<[u8]>> = Vec::new();

    let mut group = c.benchmark_group("valid");
    #[cfg(feature = "validate")]
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
    #[cfg(feature = "validate")]
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

fn backends(_c: &mut Criterion) {}

criterion_group!(criterion, frontends, validation, backends,);
criterion_main!(criterion);
