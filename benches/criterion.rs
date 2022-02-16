use criterion::*;
use std::{fs, path::PathBuf};

fn gather_inputs(folder: &str, extension: &str) -> Vec<String> {
    let mut list = Vec::new();
    let read_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(folder)
        .read_dir()
        .unwrap();
    for file_entry in read_dir {
        match file_entry {
            Ok(entry) => match entry.path().extension() {
                Some(ostr) if &*ostr == extension => {
                    let input = fs::read_to_string(entry.path()).unwrap_or_default();
                    list.push(input);
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

fn frontends(c: &mut Criterion) {
    let mut group = c.benchmark_group("tests/in");
    group.bench_function("wgsl", |b| {
        let inputs = gather_inputs("tests/in", "wgsl");
        let mut parser = naga::front::wgsl::Parser::new();
        b.iter(move || {
            for input in inputs.iter() {
                parser.parse(input).unwrap();
            }
        });
    });
}

fn validation(_c: &mut Criterion) {}

fn backends(_c: &mut Criterion) {}

criterion_group!(criterion, frontends, validation, backends,);
criterion_main!(criterion);
