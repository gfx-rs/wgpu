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

use player::{GlobalPlay, IdentityPassThroughFactory};
use std::{
    borrow::Borrow,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    ptr, slice,
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

#[derive(serde::Deserialize)]
struct Test<'a> {
    features: wgt::Features,
    expectations: Vec<Expectation>,
    #[serde(borrow)]
    actions: Vec<wgc::device::trace::Action<'a>>,
}

extern "C" fn map_callback(status: wgc::resource::BufferMapAsyncStatus, _user_data: *mut u8) {
    match status {
        wgc::resource::BufferMapAsyncStatus::Success => (),
        _ => panic!("Unable to map"),
    }
}

impl<'a> Test<'a> {
    fn load(path: PathBuf, backend: wgt::Backend, string: &'a mut String) -> Self {
        let backend_name = match backend {
            wgt::Backend::Vulkan => "Vulkan",
            wgt::Backend::Metal => "Metal",
            wgt::Backend::Dx12 => "Dx12",
            wgt::Backend::Dx11 => "Dx11",
            wgt::Backend::Gl => "Gl",
            _ => unreachable!(),
        };
        File::open(path).unwrap().read_to_string(string).unwrap();
        *string = string.replace("Empty", backend_name);
        ron::de::from_str(&*string).unwrap()
    }

    fn run(
        self,
        dir: &Path,
        global: &wgc::hub::Global<IdentityPassThroughFactory>,
        adapter: wgc::id::AdapterId,
        // test_num: u32,
    ) {
        let backend = adapter.backend();
        // let device = wgc::id::TypedId::zip(test_num, 0, backend);
        let device = wgc::gfx_select2!(Box adapter => wgc::hub::Global::<IdentityPassThroughFactory>::adapter_request_device(
            *adapter,
            &wgt::DeviceDescriptor {
                label: None,
                features: self.features | wgt::Features::MAPPABLE_PRIMARY_BUFFERS,
                limits: wgt::Limits::default(),
            },
            None,
            // device
        )).unwrap();
        /* if let Some(e) = error {
            panic!("{:?}", e);
        } */

        let mut command_buffer_id_manager = wgc::hub::IdentityManager::default();
        println!("\t\t\tRunning...");
        let mut trace_cache = wgc::id::IdMap::default();
        let mut cache = wgc::id::IdCache2::default();
        let device = device.borrow();
        let actions = global.init_actions(self.actions);
        for action in actions.into_iter().rev() {
            wgc::gfx_select2!(&Arc device =>
                              global.process(device, action, dir, &mut trace_cache, &mut cache, &mut command_buffer_id_manager));
        }
        println!("\t\t\tMapping...");
        for expect in &self.expectations {
            // let buffer = cached.get(expect.buffer.index).unwrap();
            let buffer: wgc::id::BufferId = wgc::id::TypedId::zip(expect.buffer.index, expect.buffer.epoch, backend);
            wgc::gfx_select!(buffer => global.buffer_map_async(
                buffer,
                expect.offset .. expect.offset+expect.data.len() as wgt::BufferAddress,
                wgc::resource::BufferMapOperation {
                    host: wgc::device::HostMap::Read,
                    callback: map_callback,
                    user_data: ptr::null_mut(),
                }
            ))
            .unwrap();
        }

        println!("\t\t\tWaiting...");
        wgc::gfx_select2!(&Arc device => global.device_poll(device, true)).unwrap();

        for expect in self.expectations {
            println!("\t\t\tChecking {}", expect.name);
            let buffer = wgc::id::TypedId::zip(expect.buffer.index, expect.buffer.epoch, backend);
            let (ptr, size) =
                wgc::gfx_select!(device => global.buffer_get_mapped_range(buffer, expect.offset, Some(expect.data.len() as wgt::BufferAddress)))
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

            if &expected_data[..] != contents {
                panic!(
                    "Test expectation is not met!\nBuffer content was:\n{:?}\nbut expected:\n{:?}",
                    contents, expected_data
                );
            }
        }

        wgc::gfx_select!(device => global.clear_backend(()));
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
    wgt::Backend::Dx11,
    wgt::Backend::Gl,
];

impl Corpus {
    fn run_from(path: PathBuf) {
        println!("Corpus {:?}", path);
        let dir = path.parent().unwrap();
        let corpus: Corpus = ron::de::from_reader(File::open(&path).unwrap()).unwrap();

        let global = wgc::hub::Global::new("test", IdentityPassThroughFactory, corpus.backends);
        let mut string = String::new();
        for &backend in BACKENDS {
            if !corpus.backends.contains(backend.into()) {
                continue;
            }
            for test_path in &corpus.tests {
                let adapter = match global.request_adapter(
                    &wgc::instance::RequestAdapterOptions {
                        power_preference: wgt::PowerPreference::LowPower,
                        compatible_surface: None,
                    },
                    /*wgc::instance::AdapterInputs::IdSet(
                        &[wgc::id::TypedId::zip(0, 0, backend)],
                        |id| id.backend(),
                    )*/backend.into(),
                ) {
                    Ok(adapter) => adapter,
                    Err(_) => continue,
                };

                println!("\tBackend {:?}", backend);
                let adapter_ = &adapter;
                let supported_features =
                    wgc::gfx_select2!(&Box adapter_ => wgc::hub::Global::<IdentityPassThroughFactory>::adapter_features(&adapter_));
                // let mut test_num = 0;
                println!("\t\tTest '{:?}'", test_path);
                let test = Test::load(dir.join(test_path), adapter.backend(), &mut string);
                if !supported_features.contains(test.features) {
                    println!(
                        "\t\tSkipped due to missing features {:?}",
                        test.features - supported_features
                    );
                    string.clear();
                    continue;
                }
                test.run(dir, &global, adapter/*, test_num*/);
                // test_num += 1;
                string.clear();
            }
        }
    }
}

#[test]
fn test_api() {
    env_logger::init();

    Corpus::run_from(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/all.ron"))
}
