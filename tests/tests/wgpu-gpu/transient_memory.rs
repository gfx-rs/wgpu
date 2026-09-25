//! Tests that `wgpu_hal::MemoryFlags::TRANSIENT` buffers do not pin the memory
//! blocks of long-lived resources.

use wgpu::{Backend, Backends};
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(TRANSIENT_BUFFERS_RELEASE_MEMORY_BLOCKS);
}

#[apply(gpu_test!)]
static TRANSIENT_BUFFERS_RELEASE_MEMORY_BLOCKS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // Only the Vulkan backend has a separate transient memory pool.
            .skip(FailureCase::backend(Backends::all() - Backends::VULKAN)),
    )
    .run_async(|ctx| async move {
        match ctx.adapter_info.backend {
            #[cfg(any(
                target_os = "windows",
                target_os = "linux",
                target_os = "android",
                target_os = "freebsd",
                target_os = "macos"
            ))]
            Backend::Vulkan => check_transient_buffers_release_memory_blocks(&ctx),
            other => unreachable!(
                "test is configured to skip all backends except Vulkan, but ran on {other:?}"
            ),
        }
    });

#[cfg(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "macos"
))]
fn check_transient_buffers_release_memory_blocks(ctx: &TestingContext) {
    use wgpu::hal::{self, Device as _};

    const MB: u64 = 1024 * 1024;
    // Host blocks start at MIN_HOST_BLOCK and double. Each transient buffer
    // leaves 1 MiB of slack, which a shared pool would fill with two
    // long-lived buffers.
    const MIN_HOST_BLOCK: u64 = 4 * MB;
    const TRANSIENT_SIZES: [u64; 3] = [3 * MB, 7 * MB, 15 * MB];
    const LONG_LIVED_SIZE: u64 = 384 * 1024;
    const LONG_LIVED_COUNT: usize = 2 * TRANSIENT_SIZES.len();

    // SAFETY: the hal device is only used to create and destroy buffers that
    // are never handed to wgpu-core.
    let hal_device =
        unsafe { ctx.device.as_hal::<hal::vulkan::Api>() }.expect("adapter backend mismatch");

    let create = |label: &'static str, size: u64, memory_flags: hal::MemoryFlags| {
        let (buffer, _) = unsafe {
            hal_device.create_buffer(&hal::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUses::MAP_WRITE | wgpu::BufferUses::COPY_SRC,
                memory_flags,
            })
        }
        .expect("failed to create buffer");
        buffer
    };
    let destroy = |buffer| unsafe { hal_device.destroy_buffer(buffer) };
    let report = || {
        ctx.device
            .generate_allocator_report()
            .expect("Vulkan backend should produce an allocator report")
    };
    let occupied_blocks = |report: &wgpu::AllocatorReport| {
        report
            .blocks
            .iter()
            .filter(|block| !block.allocations.is_empty())
            .count()
    };

    // Prime the transient pool so its retained empty block is in the baseline.
    destroy(create(
        "transient primer",
        TRANSIENT_SIZES[0],
        hal::MemoryFlags::TRANSIENT,
    ));
    let baseline = report();

    let transients: Vec<_> = TRANSIENT_SIZES
        .iter()
        .map(|&size| create("transient", size, hal::MemoryFlags::TRANSIENT))
        .collect();
    let with_transients = report();
    assert!(
        with_transients.blocks.len() > baseline.blocks.len(),
        "live transient buffers should occupy additional blocks\n\
         baseline: {baseline:?}\nwith transients: {with_transients:?}",
    );

    let long_lived: Vec<_> = (0..LONG_LIVED_COUNT)
        .map(|_| create("long-lived", LONG_LIVED_SIZE, hal::MemoryFlags::empty()))
        .collect();

    // Free newest-first so the retained block is the small one from the
    // baseline.
    for transient in transients.into_iter().rev() {
        destroy(transient);
    }

    let after_release = report();
    assert!(
        occupied_blocks(&after_release) <= occupied_blocks(&baseline) + 1,
        "long-lived buffers should share one new block; transient blocks must be released\n\
         baseline: {baseline:?}\nafter: {after_release:?}",
    );
    assert!(
        after_release.total_reserved_bytes - baseline.total_reserved_bytes <= 2 * MIN_HOST_BLOCK,
        "releasing transient buffers should return their blocks to the driver\n\
         baseline: {baseline:?}\nafter: {after_release:?}",
    );

    for buffer in long_lived {
        destroy(buffer);
    }
}
