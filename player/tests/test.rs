/*! Tester for WebGPU
 *  It enumerates the available backends on the system,
 *  and run the tests through them.
 *
 *  Test requirements:
 *    - all IDs have the backend `Empty`
 *    - all expected buffers have `MAP_READ` usage
 *    - last action is `Submit`
 *    - no swapchain use
!*/
#![cfg(not(target_arch = "wasm32"))]

use player::GlobalPlay;
use std::{
    fs::{read_to_string, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    slice,
};

#[derive(serde::Deserialize)]
struct RawId {
    index: u32,
    epoch: u32,
}

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
            ExpectedData::U64(vec) => vec.len() * std::mem::size_of::<u64>(),
            ExpectedData::File(_, size) => *size,
        }
    }
}

#[derive(serde::Deserialize)]
struct Expectation {
    name: String,
    buffer: RawId,
    offset: wgt::BufferAddress,
    data: ExpectedData,
}

struct Test<'a> {
    features: wgt::Features,
    expectations: Vec<Expectation>,
    actions: Vec<wgc::device::trace::Action<'a>>,
}

fn map_callback(status: Result<(), wgc::resource::BufferAccessError>) {
    if let Err(e) = status {
        panic!("Buffer map error: {}", e);
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
        let string = read_to_string(path).unwrap().replace("Empty", backend_name);

        #[derive(serde::Deserialize)]
        struct SerializedTest<'a> {
            features: Vec<String>,
            expectations: Vec<Expectation>,
            actions: Vec<wgc::device::trace::Action<'a>>,
        }
        let SerializedTest {
            features,
            expectations,
            actions,
        } = ron::de::from_str(&string).unwrap();
        let features = features
            .iter()
            .map(|feature| {
                wgt::Features::from_name(feature)
                    .unwrap_or_else(|| panic!("Invalid feature flag {}", feature))
            })
            .fold(wgt::Features::empty(), |a, b| a | b);
        Test {
            features,
            expectations,
            actions,
        }
    }

    fn run(
        self,
        dir: &Path,
        global: &wgc::global::Global,
        adapter: wgc::id::AdapterId,
        test_num: u32,
    ) {
        let backend = adapter.backend();
        let device_id = wgc::id::Id::zip(test_num, 0, backend);
        let (_, _, error) = wgc::gfx_select!(adapter => global.adapter_request_device(
            adapter,
            &wgt::DeviceDescriptor {
                label: None,
                required_features: self.features,
                required_limits: wgt::Limits::default(),
            },
            None,
            Some(device_id),
            Some(device_id.into_queue_id())
        ));
        if let Some(e) = error {
            panic!("{:?}", e);
        }

        let mut command_buffer_id_manager = wgc::identity::IdentityManager::new();
        println!("\t\t\tRunning...");
        for action in self.actions {
            wgc::gfx_select!(device_id => global.process(device_id, action, dir, &mut command_buffer_id_manager));
        }
        println!("\t\t\tMapping...");
        for expect in &self.expectations {
            let buffer = wgc::id::Id::zip(expect.buffer.index, expect.buffer.epoch, backend);
            wgc::gfx_select!(device_id => global.buffer_map_async(
                buffer,
                expect.offset,
                Some(expect.data.len() as u64),
                wgc::resource::BufferMapOperation {
                    host: wgc::device::HostMap::Read,
                    callback: Some(wgc::resource::BufferMapCallback::from_rust(
                        Box::new(map_callback)
                    )),
                }
            ))
            .unwrap();
        }

        println!("\t\t\tWaiting...");
        wgc::gfx_select!(device_id => global.device_poll(device_id, wgt::Maintain::wait()))
            .unwrap();

        for expect in self.expectations {
            println!("\t\t\tChecking {}", expect.name);
            let buffer = wgc::id::Id::zip(expect.buffer.index, expect.buffer.epoch, backend);
            let (ptr, size) =
                wgc::gfx_select!(device_id => global.buffer_get_mapped_range(buffer, expect.offset, Some(expect.data.len() as wgt::BufferAddress)))
                    .unwrap();
            let contents = unsafe { slice::from_raw_parts(ptr, size as usize) };
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

            #[allow(unknown_lints, clippy::if_then_panic)]
            if &expected_data[..] != contents {
                panic!(
                    "Test expectation is not met!\nBuffer content was:\n{:?}\nbut expected:\n{:?}",
                    contents, expected_data
                );
            }
        }

        wgc::gfx_select!(device_id => global.clear_backend(()));
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
        println!("Corpus {:?}", path);
        let dir = path.parent().unwrap();
        let corpus: Corpus = ron::de::from_reader(File::open(&path).unwrap()).unwrap();

        let global = wgc::global::Global::new(
            "test",
            wgt::InstanceDescriptor {
                backends: corpus.backends,
                flags: wgt::InstanceFlags::debugging(),
                dx12_shader_compiler: wgt::Dx12Compiler::Fxc,
                gles_minor_version: wgt::Gles3MinorVersion::default(),
            },
        );
        for &backend in BACKENDS {
            if !corpus.backends.contains(backend.into()) {
                continue;
            }
            let adapter = match global.request_adapter(
                &wgc::instance::RequestAdapterOptions {
                    power_preference: wgt::PowerPreference::None,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                },
                wgc::instance::AdapterInputs::IdSet(&[wgc::id::Id::zip(0, 0, backend)]),
            ) {
                Ok(adapter) => adapter,
                Err(_) => continue,
            };

            println!("\tBackend {:?}", backend);
            let supported_features =
                wgc::gfx_select!(adapter => global.adapter_features(adapter)).unwrap();
            let downlevel_caps =
                wgc::gfx_select!(adapter => global.adapter_downlevel_capabilities(adapter))
                    .unwrap();
            let mut test_num = 0;
            for test_path in &corpus.tests {
                println!("\t\tTest '{:?}'", test_path);
                let test = Test::load(dir.join(test_path), adapter.backend());
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
                test.run(dir, &global, adapter, test_num);
                test_num += 1;
            }
        }
    }
}

#[test]
fn test_api() {
    env_logger::init();

    Corpus::run_from(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/all.ron"))
}
