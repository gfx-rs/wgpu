//! Tester for WebGPU
//!  It enumerates the available backends on the system,
//!  and run the tests through them.
//!
//!  Test requirements:
//!    - all IDs have the backend `Noop`
//!    - all expected buffers have `MAP_READ` usage
//!    - last action is `Submit`
//!    - no swapchain use

#![cfg(not(target_arch = "wasm32"))]

extern crate wgpu_core as wgc;
extern crate wgpu_types as wgt;

use player::Player;
use std::{
    fs::{read_to_string, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    slice,
    sync::Arc,
};
use wgc::command::PointerReferences;

#[derive(serde::Deserialize)]
enum ExpectedData {
    Raw(Vec<u8>),
    U64(Vec<u64>),
    File(String, usize),
}

impl ExpectedData {
    fn len(&self) -> usize {
        match self {
            ExpectedData::Raw(vec) => vec.len(),
            ExpectedData::U64(vec) => vec.len() * size_of::<u64>(),
            ExpectedData::File(_, size) => *size,
        }
    }
}

#[derive(serde::Deserialize)]
struct Expectation {
    name: String,
    buffer: wgc::id::PointerId<wgc::id::markers::Buffer>,
    offset: wgt::BufferAddress,
    data: ExpectedData,
}

#[derive(serde::Deserialize)]
struct Test<'a> {
    features: wgt::Features,
    expectations: Vec<Expectation>,
    actions: Vec<wgc::device::trace::Action<'a, PointerReferences>>,
}

fn map_callback(status: Result<(), wgc::resource::BufferAccessError>) {
    if let Err(e) = status {
        panic!("Buffer map error: {e}");
    }
}

impl Test<'_> {
    fn load(path: PathBuf, backend: wgt::Backend) -> Self {
        let backend_name = match backend {
            wgt::Backend::Vulkan => "Vulkan",
            wgt::Backend::Metal => "Metal",
            wgt::Backend::Dx12 => "Dx12",
            wgt::Backend::Gl => "Gl",
            _ => unreachable!(),
        };
        let string = read_to_string(&path).unwrap().replace("Noop", backend_name);
        ron::de::from_str(&string).unwrap_or_else(|e| panic!("{path:?}:{} {}", e.span, e.code))
    }

    fn run(
        self,
        dir: &Path,
        instance_desc: &wgt::InstanceDescriptor,
        adapter: Arc<wgc::instance::Adapter>,
    ) {
        let (device, queue) = adapter
            .create_device_and_queue(
                &wgt::DeviceDescriptor {
                    label: None,
                    required_features: self.features,
                    required_limits: wgt::Limits::default(),
                    experimental_features: unsafe { wgt::ExperimentalFeatures::enabled() },
                    memory_hints: wgt::MemoryHints::default(),
                    trace: wgt::Trace::Off,
                },
                instance_desc.flags,
            )
            .unwrap();

        let mut player = Player::default();

        println!("\t\t\tRunning...");
        for action in self.actions {
            player.process(&device, &queue, action, dir);
        }
        println!("\t\t\tMapping...");
        for expect in &self.expectations {
            player
                .resolve_buffer_id(expect.buffer)
                .map_async(
                    expect.offset,
                    Some(expect.data.len() as u64),
                    wgc::resource::BufferMapOperation {
                        host: wgc::device::HostMap::Read,
                        callback: Some(Box::new(map_callback)),
                    },
                )
                .unwrap();
        }

        println!("\t\t\tWaiting...");
        device
            .poll(wgt::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(1)), // Tests really shouldn't need longer than that!
            })
            .unwrap();

        for expect in self.expectations {
            println!("\t\t\tChecking {}", expect.name);
            let (ptr, size) = player
                .resolve_buffer_id(expect.buffer)
                .get_mapped_range(expect.offset, Some(expect.data.len() as wgt::BufferAddress))
                .unwrap();
            let contents = unsafe { slice::from_raw_parts(ptr.as_ptr(), size as usize) };
            let expected_data = match expect.data {
                ExpectedData::Raw(vec) => vec,
                ExpectedData::File(name, size) => {
                    let mut bin = vec![0; size];
                    let mut file = File::open(dir.join(name)).unwrap();
                    file.seek(SeekFrom::Start(expect.offset)).unwrap();
                    file.read_exact(&mut bin[..]).unwrap();

                    bin
                }
                ExpectedData::U64(vec) => vec
                    .into_iter()
                    .flat_map(|u| u.to_ne_bytes().to_vec())
                    .collect::<Vec<u8>>(),
            };

            if &expected_data[..] != contents {
                panic!(
                    "Test expectation is not met!\nBuffer content was:\n{contents:?}\nbut expected:\n{expected_data:?}"
                );
            }
        }
    }
}

#[derive(serde::Deserialize)]
struct Corpus {
    backends: wgt::Backends,
    tests: Vec<String>,
}

const BACKENDS: &[wgt::Backend] = &[
    wgt::Backend::Vulkan,
    wgt::Backend::Metal,
    wgt::Backend::Dx12,
    wgt::Backend::Gl,
];

impl Corpus {
    fn run_from(path: PathBuf) {
        println!("Corpus {path:?}");
        let dir = path.parent().unwrap();
        let corpus: Corpus = ron::de::from_reader(File::open(&path).unwrap()).unwrap();

        for &backend in BACKENDS {
            if !corpus.backends.contains(backend.into()) {
                continue;
            }
            for test_path in &corpus.tests {
                println!("\t\tTest '{test_path:?}'");

                let instance_desc = wgt::InstanceDescriptor::from_env_or_default();
                let instance = wgc::instance::Instance::new("test", &instance_desc, None);
                let adapter = match instance.request_adapter(
                    &wgt::RequestAdapterOptions {
                        power_preference: wgt::PowerPreference::None,
                        force_fallback_adapter: false,
                        compatible_surface: None,
                    },
                    wgt::Backends::from(backend),
                ) {
                    Ok(adapter) => Arc::new(adapter),
                    Err(_) => continue,
                };

                println!("\tBackend {backend:?}");
                let supported_features = adapter.features();
                let downlevel_caps = adapter.downlevel_capabilities();

                let test = Test::load(dir.join(test_path), backend);
                if !supported_features.contains(test.features) {
                    println!(
                        "\t\tSkipped due to missing features {:?}",
                        test.features - supported_features
                    );
                    continue;
                }
                if !downlevel_caps
                    .flags
                    .contains(wgt::DownlevelFlags::COMPUTE_SHADERS)
                {
                    println!("\t\tSkipped due to missing compute shader capability");
                    continue;
                }
                test.run(dir, &instance_desc, adapter);
            }
        }
    }
}

#[cfg_attr(miri, ignore)]
#[test]
fn test_api() {
    env_logger::init();

    Corpus::run_from(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/player/data/all.ron"))
}
