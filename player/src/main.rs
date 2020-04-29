/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use wgc::device::trace;

use std::{
    fmt::Debug,
    fs::File,
    marker::PhantomData,
    path::{Path, PathBuf},
};
use winit::{event_loop::EventLoop, window::WindowBuilder};

macro_rules! gfx_select {
    ($id:expr => $global:ident.$method:ident( $($param:expr),+ )) => {
        match $id.backend() {
            #[cfg(not(any(target_os = "ios", target_os = "macos")))]
            wgt::Backend::Vulkan => $global.$method::<wgc::backend::Vulkan>( $($param),+ ),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            wgt::Backend::Metal => $global.$method::<wgc::backend::Metal>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx12 => $global.$method::<wgc::backend::Dx12>( $($param),+ ),
            #[cfg(windows)]
            wgt::Backend::Dx11 => $global.$method::<wgc::backend::Dx11>( $($param),+ ),
            _ => unreachable!()
        }
    };
}

#[derive(Debug)]
struct IdentityPassThrough<I>(PhantomData<I>);

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandler<I> for IdentityPassThrough<I> {
    type Input = I;
    fn process(&self, id: I, backend: wgt::Backend) -> I {
        let (index, epoch, _backend) = id.unzip();
        I::zip(index, epoch, backend)
    }
    fn free(&self, _id: I) {}
}

struct IdentityPassThroughFactory;

impl<I: Clone + Debug + wgc::id::TypedId> wgc::hub::IdentityHandlerFactory<I>
    for IdentityPassThroughFactory
{
    type Filter = IdentityPassThrough<I>;
    fn spawn(&self, _min_index: u32) -> Self::Filter {
        IdentityPassThrough(PhantomData)
    }
}
impl wgc::hub::GlobalIdentityHandlerFactory for IdentityPassThroughFactory {}

fn main() {
    env_logger::init();

    let folder = match std::env::args().nth(1) {
        Some(arg) if Path::new(&arg).is_dir() => PathBuf::from(arg),
        _ => panic!("Provide the folder path as the parameter"),
    };

    let file = File::open(folder.join(trace::FILE_NAME)).unwrap();
    let actions: Vec<trace::Action> = ron::de::from_reader(file).unwrap();

    let event_loop = EventLoop::new();
    let mut builder = WindowBuilder::new();
    builder = builder.with_title("wgpu player");
    let window = builder.build(&event_loop).unwrap();

    let global = wgc::hub::Global::new("player", IdentityPassThroughFactory);
    let mut surface_id_manager = wgc::hub::IdentityManager::from_index(1);
    let mut adapter_id_manager = wgc::hub::IdentityManager::default();
    let mut device_id_manager = wgc::hub::IdentityManager::default();

    let (_size, surface) = {
        let size = window.inner_size();
        let id = surface_id_manager.alloc(wgt::Backend::Empty);
        let surface = global.instance_create_surface(&window, id);
        (size, surface)
    };

    let adapter = global
        .pick_adapter(
            &wgc::instance::RequestAdapterOptions {
                power_preference: wgt::PowerPreference::Default,
                compatible_surface: Some(surface),
            },
            wgc::instance::AdapterInputs::IdSet(
                &vec![
                    adapter_id_manager.alloc(wgt::Backend::Vulkan),
                    adapter_id_manager.alloc(wgt::Backend::Dx12),
                    adapter_id_manager.alloc(wgt::Backend::Metal),
                ],
                |id| wgc::id::TypedId::unzip(*id).2,
            ),
        )
        .unwrap();

    let limits = match actions.first() {
        Some(&trace::Action::Init { ref limits }) => limits.clone(),
        other => panic!("Unsupported first action: {:?}", other),
    };

    let device = gfx_select!(adapter => global.adapter_request_device(
        adapter,
        &wgt::DeviceDescriptor {
            extensions: wgt::Extensions {
                anisotropic_filtering: false,
            },
            limits,
        },
        device_id_manager.alloc(wgt::Backend::Vulkan)
    ));

    for action in actions.into_iter().skip(1) {
        match action {
            _ => {}
        }
    }
}
