/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/*! Tester for WebGPU
 *  It enumerates the available backends on the system,
 *  and run the tests through them.
 *
 *  Test requirements:
 *    - all IDs have the backend `Empty`
 *    - all expected buffers have `MAP_READ` usage
 *    - last action is `Submit`
!*/

use player::{gfx_select, GlobalPlay, IdentityPassThroughFactory};
use std::{
    fs::{read_to_string, File},
    path::{Path, PathBuf},
    ptr, slice,
};

#[derive(serde::Deserialize)]
struct RawId {
    index: u32,
    epoch: u32,
}

#[derive(serde::Deserialize)]
struct Expectation {
    name: String,
    buffer: RawId,
    offset: wgt::BufferAddress,
    data: Vec<u8>,
}

#[derive(serde::Deserialize)]
struct Test {
    path: String,
    features: wgt::Features,
    expectations: Vec<Expectation>,
}

#[derive(serde::Deserialize)]
struct Corpus {
    backends: wgt::BackendBit,
    tests: Vec<Test>,
}

const BACKENDS: &[wgt::Backend] = &[
    wgt::Backend::Vulkan,
    wgt::Backend::Metal,
    wgt::Backend::Dx12,
    wgt::Backend::Dx11,
    wgt::Backend::Gl,
];

extern "C" fn map_callback(status: wgc::resource::BufferMapAsyncStatus, _user_data: *mut u8) {
    match status {
        wgc::resource::BufferMapAsyncStatus::Success => (),
        _ => panic!("Unable to map"),
    }
}

fn run_test(
    test: Test,
    dir: &Path,
    global: &wgc::hub::Global<IdentityPassThroughFactory>,
    adapter: wgc::id::AdapterId,
) {
    println!("\t\tTest '{:?}'", test.path);
    let device = gfx_select!(adapter => global.adapter_request_device(
        adapter,
        &wgt::DeviceDescriptor {
            features: test.features | wgt::Features::MAPPABLE_PRIMARY_BUFFERS,
            limits: wgt::Limits::default(),
            shader_validation: true,
        },
        None,
        wgc::id::TypedId::zip(1, 0, adapter.backend())
    ))
    .unwrap();

    let backend = device.backend();
    let backend_name = match backend {
        wgt::Backend::Vulkan => "Vulkan",
        wgt::Backend::Metal => "Metal",
        wgt::Backend::Dx12 => "Dx12",
        wgt::Backend::Dx11 => "Dx11",
        wgt::Backend::Gl => "Gl",
        _ => unreachable!(),
    };
    let string = read_to_string(dir.join(test.path))
        .unwrap()
        .replace("Empty", backend_name);
    let actions: Vec<wgc::device::trace::Action> = ron::de::from_str(&string).unwrap();

    let mut command_buffer_id_manager = wgc::hub::IdentityManager::default();
    println!("\t\t\tRunning...");
    for action in actions {
        gfx_select!(device => global.process(device, action, dir, &mut command_buffer_id_manager));
    }
    println!("\t\t\tMapping...");
    for expect in &test.expectations {
        let buffer = wgc::id::TypedId::zip(expect.buffer.index, expect.buffer.epoch, backend);
        gfx_select!(device => global.buffer_map_async(
            buffer,
            expect.offset .. expect.offset+expect.data.len() as wgt::BufferAddress,
            wgc::resource::BufferMapOperation {
                host: wgc::device::HostMap::Read,
                callback: map_callback,
                user_data: ptr::null_mut(),
            }
        ));
    }

    println!("\t\t\tWaiting...");
    gfx_select!(device => global.device_poll(device, true));

    for expect in test.expectations {
        println!("\t\t\tChecking {}", expect.name);
        let buffer = wgc::id::TypedId::zip(expect.buffer.index, expect.buffer.epoch, backend);
        let ptr =
            gfx_select!(device => global.buffer_get_mapped_range(buffer, expect.offset, None));
        let contents = unsafe { slice::from_raw_parts(ptr, expect.data.len()) };
        assert_eq!(&expect.data[..], contents);
    }
}

fn run_corpus(path: PathBuf) {
    println!("Corpus {:?}", path);
    let mut corpus: Corpus = ron::de::from_reader(File::open(&path).unwrap()).unwrap();

    let global = wgc::hub::Global::new("test", IdentityPassThroughFactory, corpus.backends);
    for &backend in BACKENDS {
        if !corpus.backends.contains(backend.into()) {
            continue;
        }
        let adapter = match global.pick_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: wgt::PowerPreference::Default,
                compatible_surface: None,
            },
            wgc::instance::AdapterInputs::IdSet(&[wgc::id::TypedId::zip(0, 0, backend)], |id| {
                id.backend()
            }),
        ) {
            Some(adapter) => adapter,
            None => continue,
        };

        println!("\tBackend {:?}", backend);
        let supported_features = gfx_select!(adapter => global.adapter_features(adapter));
        for test in corpus.tests.drain(..) {
            if !supported_features.contains(test.features) {
                println!(
                    "\t\tSkipped due to missing features {:?}",
                    test.features - supported_features
                );
                continue;
            }
            run_test(test, path.parent().unwrap(), &global, adapter);
        }
    }
}

#[test]
fn test_api() {
    run_corpus(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/all.ron"))
}
