use std::{
    ffi::c_void,
    fs::File,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    os::unix::fs::MetadataExt,
};

use wgpu::{Backends, Extent3d, Features, TextureDimension, TextureFormat, TextureUses};
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

const SIZE: u32 = 64;

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([GLES_IMPORT, VULKAN_IMPORT]);
}

#[apply(gpu_test!)]
static GLES_IMPORT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default().skip(FailureCase::backend(Backends::all() - Backends::GL)),
    )
    .run_sync(|ctx| {
        use wgpu::hal::{self, Device as _};

        let device = unsafe { ctx.device.as_hal::<hal::api::Gles>() }.expect("GLES device");
        let Some(device_id) = device.get_dmabuf_device_id().expect("DRM device query") else {
            eprintln!("Skipping GLES DMA-BUF import: no DRM render node for this adapter");
            return;
        };
        let formats = device.get_dmabuf_formats().expect("GLES DMA-BUF formats");
        assert!(
            !formats.is_empty(),
            "GLES render node has no importable formats"
        );

        let candidates: Vec<_> = formats
            .iter()
            .flat_map(|format| {
                format.modifiers.iter().map(|modifier| Candidate {
                    fourcc: format.fourcc,
                    format: format.texture_format,
                    modifier: modifier.modifier,
                })
            })
            .collect();
        let (candidate, buffer) = allocate(device_id, &candidates)
            .expect("GBM could not allocate any advertised GLES format/modifier");
        let desc = descriptor(candidate.format);
        let texture = unsafe {
            device.texture_from_dmabuf_fd(
                buffer.fd,
                &desc,
                candidate.fourcc,
                Some(candidate.modifier),
                buffer.stride,
                buffer.offset,
            )
        }
        .expect("GLES DMA-BUF import");
        unsafe { device.destroy_texture(texture) };
    });

#[apply(gpu_test!)]
static VULKAN_IMPORT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
            .skip(FailureCase::backend(Backends::all() - Backends::VULKAN)),
    )
    .run_sync(|ctx| {
        use wgpu::hal::{self, Device as _};

        let device = unsafe { ctx.device.as_hal::<hal::api::Vulkan>() }.expect("Vulkan device");
        let Some(device_id) = device.get_dmabuf_device_id().expect("DRM device query") else {
            eprintln!("Skipping Vulkan DMA-BUF import: no DRM render node for this adapter");
            return;
        };
        // Vulkan queries wgpu formats; these have unambiguous DRM FourCC mappings.
        let requested = [TextureFormat::Bgra8Unorm, TextureFormat::Rgba8Unorm];
        let formats = device
            .get_dmabuf_formats(&requested, TextureUses::RESOURCE)
            .expect("Vulkan DMA-BUF formats");
        assert!(
            !formats.is_empty(),
            "Vulkan render node has no importable formats"
        );

        let candidates: Vec<_> = formats
            .iter()
            .flat_map(|format| {
                let fourcc = match format.format {
                    TextureFormat::Bgra8Unorm => u32::from_le_bytes(*b"AR24"),
                    TextureFormat::Rgba8Unorm => u32::from_le_bytes(*b"AB24"),
                    _ => unreachable!(),
                };
                format.modifiers.iter().map(move |modifier| Candidate {
                    fourcc,
                    format: format.format,
                    modifier: modifier.modifier,
                })
            })
            .collect();
        let (candidate, buffer) = allocate(device_id, &candidates)
            .expect("GBM could not allocate any advertised Vulkan format/modifier");
        let desc = descriptor(candidate.format);
        let texture = unsafe {
            device.texture_from_dmabuf_fd(
                buffer.fd,
                &desc,
                candidate.modifier,
                buffer.stride,
                buffer.offset,
            )
        }
        .expect("Vulkan DMA-BUF import");
        unsafe { device.destroy_texture(texture) };
    });

#[derive(Clone, Copy)]
struct Candidate {
    fourcc: u32,
    format: TextureFormat,
    modifier: u64,
}

struct Buffer {
    fd: OwnedFd,
    stride: u64,
    offset: u64,
}

fn descriptor(format: TextureFormat) -> wgpu::hal::TextureDescriptor<'static> {
    wgpu::hal::TextureDescriptor {
        label: Some("DMA-BUF import test"),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUses::RESOURCE,
        memory_flags: wgpu::hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    }
}

fn allocate(device_id: u64, candidates: &[Candidate]) -> Option<(Candidate, Buffer)> {
    use libloading::Library;

    let node = std::fs::read_dir("/dev/dri")
        .ok()?
        .flatten()
        .find_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            if !name.starts_with("renderD") || entry.metadata().ok()?.rdev() != device_id {
                return None;
            }
            File::options()
                .read(true)
                .write(true)
                .open(entry.path())
                .ok()
        })?;

    // GBM creates a real DMA-BUF on the same DRM node as the wgpu device.
    // Runtime loading avoids a link-time libgbm dependency for the test suite.
    let library = unsafe { Library::new("libgbm.so.1") }.ok()?;
    unsafe {
        let create_device = *library
            .get::<unsafe extern "C" fn(i32) -> *mut c_void>(b"gbm_create_device")
            .ok()?;
        let destroy_device = *library
            .get::<unsafe extern "C" fn(*mut c_void)>(b"gbm_device_destroy")
            .ok()?;
        let create_bo =
            *library
                .get::<unsafe extern "C" fn(
                    *mut c_void,
                    u32,
                    u32,
                    u32,
                    *const u64,
                    u32,
                    u32,
                ) -> *mut c_void>(b"gbm_bo_create_with_modifiers2")
                .ok()?;
        let destroy_bo = *library
            .get::<unsafe extern "C" fn(*mut c_void)>(b"gbm_bo_destroy")
            .ok()?;
        let get_fd = *library
            .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"gbm_bo_get_fd")
            .ok()?;
        let get_stride = *library
            .get::<unsafe extern "C" fn(*mut c_void) -> u32>(b"gbm_bo_get_stride")
            .ok()?;
        let get_offset = *library
            .get::<unsafe extern "C" fn(*mut c_void, i32) -> u32>(b"gbm_bo_get_offset")
            .ok()?;
        let get_modifier = *library
            .get::<unsafe extern "C" fn(*mut c_void) -> u64>(b"gbm_bo_get_modifier")
            .ok()?;
        let get_plane_count = *library
            .get::<unsafe extern "C" fn(*mut c_void) -> i32>(b"gbm_bo_get_plane_count")
            .ok()?;

        let gbm = create_device(node.as_raw_fd());
        if gbm.is_null() {
            return None;
        }
        let mut result = None;
        for &candidate in candidates {
            let bo = create_bo(gbm, SIZE, SIZE, candidate.fourcc, &candidate.modifier, 1, 0);
            if bo.is_null() {
                continue;
            }
            if get_plane_count(bo) == 1 && get_modifier(bo) == candidate.modifier {
                let fd = get_fd(bo);
                if fd >= 0 {
                    // gbm_bo_get_fd duplicates the allocation's FD; it survives BO destruction.
                    result = Some((
                        candidate,
                        Buffer {
                            fd: OwnedFd::from_raw_fd(fd),
                            stride: get_stride(bo).into(),
                            offset: get_offset(bo, 0).into(),
                        },
                    ));
                }
            }
            destroy_bo(bo);
            if result.is_some() {
                break;
            }
        }
        destroy_device(gbm);
        result
    }
}
