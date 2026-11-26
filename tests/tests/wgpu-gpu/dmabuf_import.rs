//! Tests for `Device::import_dmabuf_texture`: linear DMA-buf planes made with
//! udmabuf from a memfd, imported, copied out and compared byte for byte; the
//! producer's writes between submissions seen; and each refusal, with the fd
//! closed once.

use wgpu_test::GpuTestInitializer;

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    #[cfg(target_os = "linux")]
    linux::all_tests(vec);
    #[cfg(not(target_os = "linux"))]
    let _ = vec;
}

#[cfg(target_os = "linux")]
mod linux {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

    use wgpu::{DmabufTextureDescriptor, ImportDmabufError, TextureFormat};
    use wgpu_test::{
        apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
    };

    pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
        vec.extend([
            DMABUF_NV12_PLANES_READ_BACK,
            DMABUF_YUYV_READS_BACK,
            DMABUF_CONTENTS_FOLLOW_THE_PRODUCER,
            DMABUF_SAMPLED_TWICE_THE_SAME_WAY,
            DMABUF_REFUSES_A_FILE_THAT_IS_NOT_A_DMABUF,
            DMABUF_REFUSES_A_PLANE_PAST_THE_END,
            DMABUF_REFUSES_A_PITCH_SHORTER_THAN_A_ROW,
            DMABUF_REFUSES_AN_UNLISTED_FORMAT,
            DMABUF_NV12_WITH_A_PADDED_PITCH_READS_BACK,
        ]);
    }

    fn params() -> TestParameters {
        TestParameters::default()
    }

    /// Whether this device can import DMA-bufs and this machine can make
    /// them. Where the device can't, the refusal it answers with is checked:
    /// `NotVulkan` off Vulkan, `MissingExtension` naming the first extension
    /// the driver lacks. Where `/dev/udmabuf` can't be opened, the test can
    /// make no DMA-buf and says so.
    fn ready(ctx: &TestingContext) -> bool {
        let plane = desc(TextureFormat::Rgba8Unorm, 16, 16, 0, 64);
        let any_fd = || std::fs::File::open("/proc/self/stat").unwrap().into();
        if ctx.adapter_info.backend != wgpu::Backend::Vulkan {
            let error = ctx
                .device
                .import_dmabuf_texture(any_fd(), &plane)
                .unwrap_err();
            assert!(matches!(error, ImportDmabufError::NotVulkan), "{error:?}");
            return false;
        }
        let missing = unsafe { ctx.device.as_hal::<wgpu::hal::api::Vulkan>() }
            .and_then(|device| device.missing_dmabuf_extension());
        if let Some(extension) = missing {
            let error = ctx
                .device
                .import_dmabuf_texture(any_fd(), &plane)
                .unwrap_err();
            assert!(
                matches!(error, ImportDmabufError::MissingExtension(name) if name == extension),
                "{error:?}"
            );
            return false;
        }
        if std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
            .is_err()
        {
            log::warn!("/dev/udmabuf can't be opened, so no DMA-buf can be made; skipped");
            return false;
        }
        true
    }

    /// A memfd holding `bytes`, rounded up to whole pages and sealed against
    /// shrinking, and a udmabuf of all of it.
    fn udmabuf(bytes: &[u8]) -> (OwnedFd, OwnedFd) {
        let page = 4096;
        let size = bytes.len().div_ceil(page) * page;
        unsafe {
            let memfd = libc::memfd_create(
                c"wgpu-dmabuf-test".as_ptr(),
                libc::MFD_ALLOW_SEALING | libc::MFD_CLOEXEC,
            );
            assert!(
                memfd >= 0,
                "memfd_create: {}",
                std::io::Error::last_os_error()
            );
            let memfd = OwnedFd::from_raw_fd(memfd);
            assert_eq!(libc::ftruncate(memfd.as_raw_fd(), size as libc::off_t), 0);
            write_at(&memfd, bytes);
            assert_eq!(
                libc::fcntl(memfd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK),
                0
            );

            #[repr(C)]
            struct UdmabufCreate {
                memfd: u32,
                flags: u32,
                offset: u64,
                size: u64,
            }
            // _IOW('u', 0x42, struct udmabuf_create), include/uapi/linux/udmabuf.h
            const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;
            const UDMABUF_FLAGS_CLOEXEC: u32 = 0x01;
            let device = libc::open(c"/dev/udmabuf".as_ptr(), libc::O_RDWR | libc::O_CLOEXEC);
            assert!(
                device >= 0,
                "open /dev/udmabuf: {}",
                std::io::Error::last_os_error()
            );
            let device = OwnedFd::from_raw_fd(device);
            let create = UdmabufCreate {
                memfd: memfd.as_raw_fd() as u32,
                flags: UDMABUF_FLAGS_CLOEXEC,
                offset: 0,
                size: size as u64,
            };
            let dmabuf = libc::ioctl(device.as_raw_fd(), UDMABUF_CREATE, &create);
            assert!(
                dmabuf >= 0,
                "UDMABUF_CREATE: {}",
                std::io::Error::last_os_error()
            );
            (memfd, OwnedFd::from_raw_fd(dmabuf))
        }
    }

    /// Overwrite the start of `memfd` with `bytes`, as a producer would.
    fn write_at(memfd: &OwnedFd, bytes: &[u8]) {
        let wrote =
            unsafe { libc::pwrite(memfd.as_raw_fd(), bytes.as_ptr().cast(), bytes.len(), 0) };
        assert_eq!(wrote, bytes.len() as isize);
    }

    /// How many open files share the DMA-buf `fd`, from its `fdinfo`.
    fn open_count(fd: &OwnedFd) -> u64 {
        let info =
            std::fs::read_to_string(format!("/proc/self/fdinfo/{}", fd.as_raw_fd())).unwrap();
        info.lines()
            .find_map(|line| line.strip_prefix("count:"))
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }

    /// Bytes that differ from row to row and texel to texel, with `seed`.
    fn pattern(len: usize, seed: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i.wrapping_mul(31) ^ (i / 1920) ^ seed) as u8)
            .collect()
    }

    /// The plane `desc` names in `bytes`, rows packed.
    fn plane(bytes: &[u8], desc: &DmabufTextureDescriptor, texel: u32) -> Vec<u8> {
        let row = (desc.width * texel) as usize;
        (0..desc.height as usize)
            .flat_map(|y| {
                let start = desc.offset as usize + y * desc.pitch as usize;
                bytes[start..start + row].iter().copied()
            })
            .collect()
    }

    /// Copy all of `texture` to a buffer in one submission and read it back,
    /// rows packed.
    fn read_back(ctx: &TestingContext, texture: &wgpu::Texture, texel: u32) -> Vec<u8> {
        let (width, height) = (texture.width(), texture.height());
        let row = width * texel;
        let padded =
            row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(padded * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        ctx.queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let mapped = buffer.slice(..).get_mapped_range().unwrap();
        (0..height as usize)
            .flat_map(|y| {
                let start = y * padded as usize;
                mapped[start..start + row as usize].to_vec()
            })
            .collect()
    }

    fn desc(
        format: TextureFormat,
        width: u32,
        height: u32,
        offset: u64,
        pitch: u64,
    ) -> DmabufTextureDescriptor<'static> {
        DmabufTextureDescriptor {
            label: None,
            width,
            height,
            format,
            offset,
            pitch,
        }
    }

    /// How many image barriers that transfer queue family ownership the
    /// device has recorded: an acquire and a release for each command buffer
    /// that uses a foreign texture.
    fn transfers(ctx: &TestingContext) -> isize {
        ctx.device
            .get_internal_counters()
            .hal
            .queue_family_transfers
            .read()
    }

    fn dup(fd: &OwnedFd) -> OwnedFd {
        fd.try_clone().unwrap()
    }

    #[apply(gpu_test!)]
    static DMABUF_NV12_PLANES_READ_BACK: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(params())
        .run_sync(|ctx| {
            if !ready(&ctx) {
                return;
            }
            // 1080p NV12 as uvcvideo writes it: Y, then interleaved UV, both
            // at a pitch of the width, which is not 256-aligned.
            let (width, height) = (1920, 1080);
            let bytes = pattern(width as usize * height as usize * 3 / 2, 0);
            let (_memfd, dmabuf) = udmabuf(&bytes);
            let y = desc(TextureFormat::R8Unorm, width, height, 0, u64::from(width));
            let uv = desc(
                TextureFormat::Rg8Unorm,
                width / 2,
                height / 2,
                u64::from(width * height),
                u64::from(width),
            );
            let y_texture = ctx.device.import_dmabuf_texture(dup(&dmabuf), &y).unwrap();
            let uv_texture = ctx.device.import_dmabuf_texture(dmabuf, &uv).unwrap();
            let before = transfers(&ctx);
            assert!(
                read_back(&ctx, &y_texture, 1) == plane(&bytes, &y, 1),
                "the Y plane"
            );
            assert!(
                read_back(&ctx, &uv_texture, 2) == plane(&bytes, &uv, 2),
                "the UV plane"
            );
            assert_eq!(
                transfers(&ctx) - before,
                4,
                "an acquire and a release per copy"
            );
        });

    #[apply(gpu_test!)]
    static DMABUF_NV12_WITH_A_PADDED_PITCH_READS_BACK: GpuTestConfiguration =
        GpuTestConfiguration::new()
            .parameters(params())
            .run_sync(|ctx| {
                if !ready(&ctx) {
                    return;
                }
                // Rows padded past the width: the UV plane starts at pitch * height,
                // not width * height.
                let (width, height, pitch) = (1918, 1080, 2048);
                let bytes = pattern(pitch as usize * height as usize * 3 / 2, 3);
                let (_memfd, dmabuf) = udmabuf(&bytes);
                let y = desc(TextureFormat::R8Unorm, width, height, 0, pitch);
                let uv = desc(
                    TextureFormat::Rg8Unorm,
                    width / 2,
                    height / 2,
                    pitch * u64::from(height),
                    pitch,
                );
                let y_texture = ctx.device.import_dmabuf_texture(dup(&dmabuf), &y).unwrap();
                let uv_texture = ctx.device.import_dmabuf_texture(dmabuf, &uv).unwrap();
                assert!(
                    read_back(&ctx, &y_texture, 1) == plane(&bytes, &y, 1),
                    "the Y plane"
                );
                assert!(
                    read_back(&ctx, &uv_texture, 2) == plane(&bytes, &uv, 2),
                    "the UV plane"
                );
            });

    #[apply(gpu_test!)]
    static DMABUF_YUYV_READS_BACK: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(params())
        .run_sync(|ctx| {
            if !ready(&ctx) {
                return;
            }
            // 1080p YUYV, sampled as Rgba8 at half width.
            let (width, height) = (960, 1080);
            let bytes = pattern(width as usize * 4 * height as usize, 0);
            let (_memfd, dmabuf) = udmabuf(&bytes);
            let yuyv = desc(
                TextureFormat::Rgba8Unorm,
                width,
                height,
                0,
                u64::from(width * 4),
            );
            let texture = ctx.device.import_dmabuf_texture(dmabuf, &yuyv).unwrap();
            assert!(read_back(&ctx, &texture, 4) == plane(&bytes, &yuyv, 4));
        });

    #[apply(gpu_test!)]
    static DMABUF_CONTENTS_FOLLOW_THE_PRODUCER: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(params())
        .run_sync(|ctx| {
            if !ready(&ctx) {
                return;
            }
            // The producer writes a new frame between two submissions; the
            // second sees it.
            let frame = desc(TextureFormat::Rgba8Unorm, 64, 64, 0, 256);
            let first = pattern(64 * 64 * 4, 0);
            let (memfd, dmabuf) = udmabuf(&first);
            let texture = ctx.device.import_dmabuf_texture(dmabuf, &frame).unwrap();
            let before = transfers(&ctx);
            assert!(
                read_back(&ctx, &texture, 4) == plane(&first, &frame, 4),
                "the first frame"
            );
            let second = pattern(64 * 64 * 4, 0x5a);
            write_at(&memfd, &second);
            assert!(
                read_back(&ctx, &texture, 4) == plane(&second, &frame, 4),
                "the second frame"
            );
            assert_eq!(
                transfers(&ctx) - before,
                4,
                "an acquire and a release per submission"
            );
        });

    #[apply(gpu_test!)]
    static DMABUF_SAMPLED_TWICE_THE_SAME_WAY: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(params())
        .run_sync(|ctx| {
            if !ready(&ctx) {
                return;
            }
            // Two command buffers in one submission that each copy from the
            // texture: the same use twice, so each needs its own acquire and
            // release, which the tracker alone would leave out.
            let frame = desc(TextureFormat::Rgba8Unorm, 64, 64, 0, 256);
            let bytes = pattern(64 * 64 * 4, 7);
            let (_memfd, dmabuf) = udmabuf(&bytes);
            let texture = ctx.device.import_dmabuf_texture(dmabuf, &frame).unwrap();
            let buffers = [(); 2].map(|()| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 256 * 64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            });
            let commands = buffers.each_ref().map(|buffer| {
                let mut encoder = ctx.device.create_command_encoder(&Default::default());
                encoder.copy_texture_to_buffer(
                    texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: None,
                        },
                    },
                    texture.size(),
                );
                encoder.finish()
            });
            let before = transfers(&ctx);
            ctx.queue.submit(commands);
            for buffer in &buffers {
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |result| result.unwrap());
            }
            ctx.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            for buffer in &buffers {
                assert!(
                    *buffer.slice(..).get_mapped_range().unwrap() == plane(&bytes, &frame, 4)[..]
                );
            }
            assert_eq!(
                transfers(&ctx) - before,
                4,
                "each command buffer acquires and releases, though both copy the same way"
            );
        });

    /// Import `fd` as `plane`, expecting a refusal, and check the import
    /// closed `fd` once: the other open file of the same buffer is then the
    /// only one.
    fn refused(
        ctx: &TestingContext,
        bytes: &[u8],
        plane: &DmabufTextureDescriptor,
    ) -> ImportDmabufError {
        let (_memfd, dmabuf) = udmabuf(bytes);
        let witness = dup(&dmabuf);
        assert_eq!(open_count(&witness), 2);
        let error = ctx.device.import_dmabuf_texture(dmabuf, plane).unwrap_err();
        assert_eq!(open_count(&witness), 1, "the refused fd is closed, once");
        error
    }

    #[apply(gpu_test!)]
    static DMABUF_REFUSES_A_FILE_THAT_IS_NOT_A_DMABUF: GpuTestConfiguration =
        GpuTestConfiguration::new()
            .parameters(params())
            .run_sync(|ctx| {
                if !ready(&ctx) {
                    return;
                }
                let (memfd, _dmabuf) = udmabuf(&pattern(4096, 0));
                let plane = desc(TextureFormat::Rgba8Unorm, 16, 16, 0, 64);
                let error = ctx.device.import_dmabuf_texture(memfd, &plane).unwrap_err();
                assert!(matches!(error, ImportDmabufError::NotDmabuf), "{error:?}");
            });

    #[apply(gpu_test!)]
    static DMABUF_REFUSES_A_PLANE_PAST_THE_END: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(params())
        .run_sync(|ctx| {
            if !ready(&ctx) {
                return;
            }
            // 64 rows of 256 bytes need 16384; the buffer holds one page.
            let plane = desc(TextureFormat::Rgba8Unorm, 64, 64, 0, 256);
            let error = refused(&ctx, &pattern(4096, 0), &plane);
            assert!(
                matches!(
                    error,
                    ImportDmabufError::TooSmall {
                        needed: 16384,
                        size: 4096
                    }
                ),
                "{error:?}"
            );
        });

    #[apply(gpu_test!)]
    static DMABUF_REFUSES_A_PITCH_SHORTER_THAN_A_ROW: GpuTestConfiguration =
        GpuTestConfiguration::new()
            .parameters(params())
            .run_sync(|ctx| {
                if !ready(&ctx) {
                    return;
                }
                let plane = desc(TextureFormat::Rgba8Unorm, 64, 4, 0, 128);
                let error = refused(&ctx, &pattern(4096, 0), &plane);
                assert!(
                    matches!(
                        error,
                        ImportDmabufError::PitchTooSmall {
                            pitch: 128,
                            row: 256
                        }
                    ),
                    "{error:?}"
                );
            });

    #[apply(gpu_test!)]
    static DMABUF_REFUSES_AN_UNLISTED_FORMAT: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(params())
        .run_sync(|ctx| {
            if !ready(&ctx) {
                return;
            }
            let plane = desc(TextureFormat::Rgba32Float, 16, 16, 0, 256);
            let error = refused(&ctx, &pattern(4096, 0), &plane);
            assert!(
                matches!(
                    error,
                    ImportDmabufError::UnsupportedFormat(TextureFormat::Rgba32Float)
                ),
                "{error:?}"
            );
        });
}
