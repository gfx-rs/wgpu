use wgpu::util::DeviceExt as _;
use wgpu::{Origin3d, TextureFormat, TextureUsages};
use wgpu_test::{
    apply, fail, gpu_test, valid, GpuTestConfiguration, GpuTestInitializer, TestParameters,
    TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(HOST_TEXTURE_MAP_CALLBACKS);
    vec.push(HOST_TEXTURE_MAP_STATE_VALIDATION);
    vec.push(HOST_TEXTURE_MAP_DEVICE_LOST);
    vec.push(HOST_TEXTURE_MAP_WITHOUT_QUEUE);
    vec.push(HOST_TEXTURE_OPTIMAL_FORMATS);
    vec.push(HOST_TEXTURE_ZERO_INIT);
    vec.push(HOST_TEXTURE_DEPTH);
    vec.push(HOST_TEXTURE_PLANAR_ROUNDTRIP);
    vec.push(HOST_TEXTURE_LAYOUTS);
    vec.push(HOST_TEXTURE_BC_LAYOUTS);
    vec.push(HOST_TEXTURE_BC_3D_LAYOUTS);
    vec.push(HOST_TEXTURE_MULTITHREADED);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_ORDERING);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_ORDERING_NO_IMPLICIT_HOST_COPY);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_COMPRESSED_ORDERING);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_COMPRESSED_ORDERING_NO_IMPLICIT_HOST_COPY);
    vec.push(HOST_TEXTURE_WRITE_TEXTURE_HOST_VISIBLE);
}

const PADDING_BYTE: u8 = 0xEE;
const HOST_RW: TextureUsages = TextureUsages::HOST_VISIBLE
    .union(TextureUsages::COPY_DST)
    .union(TextureUsages::COPY_SRC);

fn host_params() -> TestParameters {
    TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY)
}

fn extent(width: u32, height: u32, depth_or_array_layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }
}

fn desc(
    format: TextureFormat,
    size: wgpu::Extent3d,
    usage: TextureUsages,
) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
        texture_binding_view_dimension: None,
        mapped_at_creation: false,
    }
}

fn rgba_texture(
    device: &wgpu::Device,
    usage: TextureUsages,
    mapped_at_creation: bool,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        mapped_at_creation,
        ..desc(TextureFormat::Rgba8Unorm, extent(4, 4, 1), usage)
    })
}

fn host_copyable(ctx: &TestingContext, format: TextureFormat) -> bool {
    let copyable = ctx
        .adapter
        .get_texture_format_features(format)
        .allowed_usages
        .contains(TextureUsages::HOST_VISIBLE);
    if !copyable {
        log::info!("skipping: {format:?} is not host-copyable");
    }
    copyable
}

fn encoder(device: &wgpu::Device) -> wgpu::CommandEncoder {
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
}

async fn wait(ctx: &TestingContext) {
    ctx.async_poll(wgpu::PollType::wait_indefinitely())
        .await
        .unwrap();
}

type MapResult = Result<(), wgpu::TextureAsyncError>;
type MapResultReceiver = std::sync::mpsc::Receiver<MapResult>;

fn map_callback() -> (impl FnOnce(MapResult) + Send + 'static, MapResultReceiver) {
    let (tx, rx) = std::sync::mpsc::channel();
    (
        move |result| {
            let _ = tx.send(result);
        },
        rx,
    )
}

fn map_async(texture: &wgpu::Texture, mode: wgpu::MapMode) -> MapResultReceiver {
    let (callback, rx) = map_callback();
    texture.map_async(mode, callback);
    rx
}

fn map_on_submit(
    encoder: &wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mode: wgpu::MapMode,
) -> MapResultReceiver {
    let (callback, rx) = map_callback();
    encoder.map_texture_on_submit(texture, mode, callback);
    rx
}

fn expect_callback(rx: &MapResultReceiver, ok: bool, what: &str) {
    let result = rx
        .try_recv()
        .unwrap_or_else(|_| panic!("{what}: map callback did not fire"));
    assert_eq!(result.is_ok(), ok, "{what}: got {result:?}");
}

async fn map_and_wait(ctx: &TestingContext, texture: &wgpu::Texture, mode: wgpu::MapMode) {
    let rx = map_async(texture, mode);
    wait(ctx).await;
    expect_callback(&rx, true, "map");
}

fn host_range(mip_level: u32, origin: Origin3d) -> wgpu::TexelCopyTextureInfoBase<()> {
    wgpu::TexelCopyTextureInfoBase {
        texture: (),
        mip_level,
        origin,
        aspect: wgpu::TextureAspect::All,
    }
}

/// Non-zero bytes, so they can't be mistaken for zero-initialized memory.
fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| ((i + seed as usize) % 251) as u8 + 1)
        .collect()
}

/// Bytes per block row and number of block rows of a copy of `size`.
fn copy_geometry(format: TextureFormat, size: wgpu::Extent3d) -> (u32, u32) {
    let (block_width, block_height) = format.block_dimensions();
    let block_size = format.block_copy_size(None).unwrap();
    (
        size.width.div_ceil(block_width) * block_size,
        size.height.div_ceil(block_height),
    )
}

fn tight_layout(format: TextureFormat, size: wgpu::Extent3d) -> wgpu::TexelCopyBufferLayout {
    let (row_bytes, rows) = copy_geometry(format, size);
    wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(row_bytes),
        rows_per_image: Some(rows),
    }
}

/// Lays out tightly packed `dense` data according to `layout`, filling the
/// gaps with `PADDING_BYTE`.
fn pack(
    dense: &[u8],
    row_bytes: u32,
    rows: u32,
    depth: u32,
    layout: wgpu::TexelCopyBufferLayout,
) -> Vec<u8> {
    let row_bytes = row_bytes as usize;
    let bytes_per_row = layout.bytes_per_row.unwrap() as usize;
    let bytes_per_image = bytes_per_row * layout.rows_per_image.unwrap() as usize;
    let offset = layout.offset as usize;
    let len = offset
        + bytes_per_image * (depth as usize - 1)
        + bytes_per_row * (rows as usize - 1)
        + row_bytes;
    let mut out = vec![PADDING_BYTE; len];
    for z in 0..depth as usize {
        for y in 0..rows as usize {
            let src = (z * rows as usize + y) * row_bytes;
            let dst = offset + z * bytes_per_image + y * bytes_per_row;
            out[dst..dst + row_bytes].copy_from_slice(&dense[src..src + row_bytes]);
        }
    }
    out
}

/// Places tightly packed `dense` data for a copy at `origin` into an otherwise
/// zeroed, tightly packed image of `full_size`.
fn embed(
    format: TextureFormat,
    full_size: wgpu::Extent3d,
    origin: Origin3d,
    copy_size: wgpu::Extent3d,
    dense: &[u8],
) -> Vec<u8> {
    let (block_width, block_height) = format.block_dimensions();
    let block_size = format.block_copy_size(None).unwrap();
    let (full_row_bytes, full_rows) = copy_geometry(format, full_size);
    let (row_bytes, rows) = copy_geometry(format, copy_size);
    let (full_row_bytes, full_rows, row_bytes, rows) = (
        full_row_bytes as usize,
        full_rows as usize,
        row_bytes as usize,
        rows as usize,
    );
    let x = (origin.x / block_width * block_size) as usize;
    let y0 = (origin.y / block_height) as usize;
    let mut out = vec![0; full_row_bytes * full_rows * full_size.depth_or_array_layers as usize];
    for z in 0..copy_size.depth_or_array_layers as usize {
        for y in 0..rows {
            let src = (z * rows + y) * row_bytes;
            let dst = ((origin.z as usize + z) * full_rows + y0 + y) * full_row_bytes + x;
            out[dst..dst + row_bytes].copy_from_slice(&dense[src..src + row_bytes]);
        }
    }
    out
}

/// Reads a region of `texture` through a GPU copy, tightly packed.
async fn gpu_read(
    ctx: &TestingContext,
    texture: &wgpu::Texture,
    mip_level: u32,
    size: wgpu::Extent3d,
) -> Vec<u8> {
    let (row_bytes, rows) = copy_geometry(texture.format(), size);
    let bytes_per_row = row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(bytes_per_row * rows * size.depth_or_array_layers),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = encoder(&ctx.device);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(rows),
            },
        },
        size,
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    wait(ctx).await;
    let data = slice.get_mapped_range().unwrap();
    data.chunks(bytes_per_row as usize)
        .flat_map(|row| &row[..row_bytes as usize])
        .copied()
        .collect()
}

/// Records a GPU copy of tightly packed `dense` data into a region of `texture`.
fn gpu_write(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    mip_level: u32,
    origin: Origin3d,
    size: wgpu::Extent3d,
    dense: &[u8],
) {
    let (row_bytes, rows) = copy_geometry(texture.format(), size);
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(row_bytes.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
        rows_per_image: Some(rows),
    };
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &pack(dense, row_bytes, rows, size.depth_or_array_layers, layout),
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout,
        },
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
}

/// A command buffer that copies the whole 4x4 Rgba8 `texture` to a buffer.
fn copy_out(device: &wgpu::Device, texture: &wgpu::Texture) -> wgpu::CommandBuffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 256 * 4,
        usage: wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = encoder(device);
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        texture.size(),
    );
    encoder.finish()
}

fn host_read_all(texture: &wgpu::Texture) -> Vec<u8> {
    let size = texture.size();
    let layout = tight_layout(texture.format(), size);
    let mut out = vec![
        PADDING_BYTE;
        (layout.bytes_per_row.unwrap() * layout.rows_per_image.unwrap()) as usize
            * size.depth_or_array_layers as usize
    ];
    let mapped = texture.get_mapped_texture().unwrap();
    mapped.copy_to_memory(host_range(0, Origin3d::ZERO), &mut out, layout, size);
    out
}

fn host_write_all(texture: &wgpu::Texture, data: &[u8]) {
    let size = texture.size();
    let mapped = texture.get_mapped_texture_mut().unwrap();
    mapped.copy_from_memory(
        host_range(0, Origin3d::ZERO),
        data,
        tight_layout(texture.format(), size),
        size,
    );
}

fn write_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    origin: Origin3d,
    size: wgpu::Extent3d,
    data: &[u8],
) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        tight_layout(texture.format(), size),
        size,
    );
}

async fn host_texture_map_callbacks(ctx: TestingContext) {
    use wgpu::MapMode::{Read, Write};
    let device = &ctx.device;
    let new = || rgba_texture(device, TextureUsages::HOST_VISIBLE, false);

    let texture = new();
    assert!(texture.get_mapped_texture().is_err());
    assert!(texture.get_mapped_texture_mut().is_err());
    let rx = map_async(&texture, Read);
    wait(&ctx).await;
    expect_callback(&rx, true, "idle map");
    assert!(rx.try_recv().is_err(), "idle map: callback fired twice");
    assert!(
        texture.get_mapped_texture_mut().is_err(),
        "read map is writable"
    );
    assert_eq!(host_read_all(&texture), vec![0; 64]);
    let rx = valid(device, || map_async(&texture, Read));
    expect_callback(&rx, false, "map while mapped");
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mapped = texture.get_mapped_texture().unwrap();
        let unmap = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| texture.unmap()));
        assert!(unmap.is_err(), "unmap with a live MappedTexture must panic");
        drop(mapped);
    }
    texture.unmap();
    assert!(texture.get_mapped_texture().is_err());
    valid(device, || texture.unmap());

    let texture = rgba_texture(device, TextureUsages::COPY_DST, false);
    let rx = fail(device, || map_async(&texture, Read), Some("HOST_VISIBLE"));
    expect_callback(&rx, false, "not HOST_VISIBLE");

    let texture = new();
    let first = map_async(&texture, Write);
    let second = valid(device, || map_async(&texture, Write));
    expect_callback(&second, false, "second map");
    wait(&ctx).await;
    expect_callback(&first, true, "first map");
    assert!(texture.get_mapped_texture_mut().is_ok());
    assert!(
        texture.get_mapped_texture().is_err(),
        "write map is readable"
    );
    texture.unmap();

    let texture = new();
    let rx = map_async(&texture, Read);
    texture.destroy();
    expect_callback(&rx, false, "destroyed while pending");
    assert!(texture.get_mapped_texture().is_err());

    let texture = rgba_texture(device, TextureUsages::HOST_VISIBLE, true);
    texture.destroy();
    assert!(texture.get_mapped_texture_mut().is_err(), "destroy unmaps");
    #[cfg(not(target_arch = "wasm32"))]
    {
        let texture = rgba_texture(device, TextureUsages::HOST_VISIBLE, true);
        let mapped = texture.get_mapped_texture_mut().unwrap();
        let destroy = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| texture.destroy()));
        assert!(
            destroy.is_err(),
            "destroy with a live MappedTextureMut must panic"
        );
        drop(mapped);
        texture.destroy();
    }

    let texture = new();
    let cancelled = map_async(&texture, Read);
    texture.unmap();
    expect_callback(&cancelled, false, "unmap while pending");
    assert!(texture.get_mapped_texture().is_err());
    let rx = map_async(&texture, Write);
    wait(&ctx).await;
    expect_callback(&rx, true, "map after cancelled map");
    assert!(
        cancelled.try_recv().is_err(),
        "cancelled map called back twice"
    );
    assert!(texture.get_mapped_texture_mut().is_ok());
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_CALLBACKS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_map_callbacks);

async fn host_texture_map_state_validation(ctx: TestingContext) {
    use wgpu::MapMode::Read;
    let device = &ctx.device;
    let size = extent(4, 4, 1);

    let texture = rgba_texture(device, HOST_RW, true);
    let command_buffer = copy_out(device, &texture);
    fail(
        device,
        || ctx.queue.submit(Some(command_buffer)),
        Some("mapped or queued for mapping"),
    );
    fail(
        device,
        || write_texture(&ctx.queue, &texture, Origin3d::ZERO, size, &[0; 64]),
        Some("unmapped"),
    );
    texture.unmap();

    let data = pattern(64, 1);
    let mut write = encoder(device);
    gpu_write(device, &mut write, &texture, 0, Origin3d::ZERO, size, &data);
    ctx.queue.submit(Some(write.finish()));
    map_and_wait(&ctx, &texture, Read).await;
    assert_eq!(host_read_all(&texture), data, "map after GPU write");
    texture.unmap();

    let data = pattern(64, 2);
    let mut write = encoder(device);
    gpu_write(device, &mut write, &texture, 0, Origin3d::ZERO, size, &data);
    let rx = map_on_submit(&write, &texture, Read);
    valid(device, || ctx.queue.submit(Some(write.finish())));
    wait(&ctx).await;
    expect_callback(&rx, true, "map_texture_on_submit");
    assert_eq!(host_read_all(&texture), data, "map_texture_on_submit");
    texture.unmap();

    let before = pattern(64, 3);
    write_texture(&ctx.queue, &texture, Origin3d::ZERO, size, &before);
    let rx = map_async(&texture, Read);
    fail(
        device,
        || write_texture(&ctx.queue, &texture, Origin3d::ZERO, size, &data),
        Some("unmapped"),
    );
    let command_buffer = copy_out(device, &texture);
    fail(
        device,
        || ctx.queue.submit(Some(command_buffer)),
        Some("queued for mapping"),
    );
    wait(&ctx).await;
    expect_callback(&rx, true, "map after write_texture");
    assert_eq!(host_read_all(&texture), before, "map after write_texture");
    texture.unmap();

    let mut e = encoder(device);
    fail(
        device,
        || {
            e.transition_resources(
                core::iter::empty(),
                core::iter::once(wgpu::TextureTransition {
                    texture: &texture,
                    selector: None,
                    state: wgpu::TextureUses::HOST_COPY,
                }),
            );
            e.finish()
        },
        Some("HOST_COPY"),
    );

    let mut e = encoder(device);
    fail(
        device,
        || {
            e.transition_resources(
                core::iter::empty(),
                core::iter::once(wgpu::TextureTransition {
                    texture: &texture,
                    selector: None,
                    state: wgpu::TextureUses::UNINITIALIZED,
                }),
            );
            e.finish()
        },
        Some("UNINITIALIZED"),
    );

    let view = texture.create_view(&Default::default());
    let mut e = encoder(device);
    fail(
        device,
        || {
            let mut pass = e.begin_compute_pass(&Default::default());
            pass.transition_resources(
                core::iter::empty(),
                core::iter::once(wgpu::TextureTransition {
                    texture: &view,
                    selector: None,
                    state: wgpu::TextureUses::UNINITIALIZED,
                }),
            );
            drop(pass);
            e.finish()
        },
        Some("UNINITIALIZED"),
    );
}

async fn host_texture_map_device_lost(ctx: TestingContext) {
    let texture = rgba_texture(&ctx.device, TextureUsages::HOST_VISIBLE, false);
    let rx = map_async(&texture, wgpu::MapMode::Read);
    ctx.device.destroy();
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    expect_callback(&rx, false, "map with a lost device");
    assert!(texture.get_mapped_texture().is_err());
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_DEVICE_LOST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_map_device_lost);

async fn host_texture_map_without_queue(ctx: TestingContext) {
    use wgpu::MapMode::Read;
    let TestingContext { device, queue, .. } = ctx;
    let idle = rgba_texture(&device, TextureUsages::HOST_VISIBLE, false);
    let written = rgba_texture(&device, HOST_RW, false);
    let data = pattern(64, 4);
    write_texture(&queue, &written, Origin3d::ZERO, written.size(), &data);
    queue.submit(None);
    drop(queue);

    let rx = map_async(&idle, Read);
    expect_callback(&rx, true, "idle map without a queue");
    assert_eq!(host_read_all(&idle), vec![0; 64]);
    idle.unmap();

    // Backends that need a barrier after device writes can't map this texture.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let rx = map_async(&written, Read);
    let error = scope.pop().await;
    let result = rx.try_recv().expect("map without a queue: no callback");
    assert_eq!(result.is_ok(), error.is_none(), "{result:?}, {error:?}");
    if result.is_ok() {
        assert_eq!(host_read_all(&written), data, "map without a queue");
    }
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_WITHOUT_QUEUE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_map_without_queue);

#[apply(gpu_test!)]
static HOST_TEXTURE_OPTIMAL_FORMATS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_sync(host_texture_optimal_formats);

fn host_texture_optimal_formats(ctx: TestingContext) {
    let formats = [
        TextureFormat::Rgba8Unorm,
        TextureFormat::R8Unorm,
        TextureFormat::Rgba16Float,
        TextureFormat::Bc1RgbaUnorm,
        TextureFormat::Bc7RgbaUnorm,
        TextureFormat::Etc2Rgb8Unorm,
        TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: wgpu::AstcChannel::Unorm,
        },
    ];
    for format in formats {
        let features = ctx.adapter.get_texture_format_features(format);
        let host_visible = features
            .allowed_usages
            .contains(TextureUsages::HOST_VISIBLE);
        let optimal = features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::HOST_VISIBLE_OPTIMAL);
        assert!(
            !optimal || host_visible,
            "{format:?} is HOST_VISIBLE_OPTIMAL but doesn't allow HOST_VISIBLE"
        );
        if ctx.adapter_info.backend == wgpu::Backend::Metal {
            assert_eq!(
                optimal,
                host_visible && format.is_compressed(),
                "HOST_VISIBLE_OPTIMAL for {format:?}"
            );
        }
    }
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MAP_STATE_VALIDATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_map_state_validation);

async fn host_texture_zero_init(ctx: TestingContext) {
    let device = &ctx.device;
    let format = TextureFormat::Rgba8Unorm;
    let usage = TextureUsages::HOST_VISIBLE | TextureUsages::COPY_SRC;
    let size = extent(4, 4, 1);
    let zeros = vec![0; 64];

    let texture = rgba_texture(device, usage, true);
    let layout = tight_layout(format, size);
    let empty_copies = [
        (Origin3d::ZERO, extent(0, 4, 1)),
        (Origin3d::ZERO, extent(4, 0, 1)),
        (Origin3d::ZERO, extent(4, 4, 0)),
        (Origin3d { x: 4, y: 4, z: 0 }, extent(0, 0, 1)),
        (Origin3d { x: 0, y: 0, z: 1 }, extent(4, 4, 0)),
    ];
    {
        let mapped = texture.get_mapped_texture_mut().unwrap();
        let data = pattern(64, 4);
        valid(device, || {
            for (origin, size) in empty_copies {
                mapped.copy_from_memory(host_range(0, origin), &data, layout, size);
            }
        });
    }
    texture.unmap();
    map_and_wait(&ctx, &texture, wgpu::MapMode::Read).await;
    {
        let mapped = texture.get_mapped_texture().unwrap();
        let mut out = vec![PADDING_BYTE; 64];
        valid(device, || {
            for (origin, size) in empty_copies {
                mapped.copy_to_memory(host_range(0, origin), &mut out, layout, size);
            }
        });
        assert!(out.iter().all(|&b| b == PADDING_BYTE));
    }
    assert_eq!(host_read_all(&texture), zeros, "unwritten host read");
    texture.unmap();
    assert_eq!(
        gpu_read(&ctx, &texture, 0, size).await,
        zeros,
        "mapped, unwritten GPU read"
    );

    let texture = rgba_texture(device, usage, true);
    let origin = Origin3d { x: 1, y: 1, z: 0 };
    let part = extent(2, 2, 1);
    let data = pattern(16, 5);
    texture.get_mapped_texture_mut().unwrap().copy_from_memory(
        host_range(0, origin),
        &data,
        tight_layout(format, part),
        part,
    );
    let expected = embed(format, size, origin, part, &data);
    texture.unmap();
    map_and_wait(&ctx, &texture, wgpu::MapMode::Read).await;
    assert_eq!(host_read_all(&texture), expected, "partial write host read");
    texture.unmap();
    assert_eq!(
        gpu_read(&ctx, &texture, 0, size).await,
        expected,
        "partial write GPU read"
    );

    let texture = rgba_texture(device, usage, false);
    assert_eq!(
        gpu_read(&ctx, &texture, 0, size).await,
        zeros,
        "never mapped GPU read"
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_ZERO_INIT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_zero_init);

async fn host_texture_depth(ctx: TestingContext) {
    let format = TextureFormat::Depth32Float;
    if !host_copyable(&ctx, format) {
        return;
    }
    let usage = TextureUsages::HOST_VISIBLE | TextureUsages::COPY_SRC;
    let size = extent(4, 4, 1);
    let data: Vec<u8> = (0..16)
        .flat_map(|i| (i as f32 / 16.0).to_ne_bytes())
        .collect();

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        mapped_at_creation: true,
        ..desc(format, size, usage)
    });
    host_write_all(&texture, &data);
    texture.unmap();
    map_and_wait(&ctx, &texture, wgpu::MapMode::Read).await;
    assert_eq!(host_read_all(&texture), data, "depth host round-trip");
    texture.unmap();
    assert_eq!(gpu_read(&ctx, &texture, 0, size).await, data);

    // Depth zero-init uses a render pass clear instead of a copy.
    let texture = ctx.device.create_texture(&desc(format, size, usage));
    assert_eq!(gpu_read(&ctx, &texture, 0, size).await, vec![0; 64]);
}

#[apply(gpu_test!)]
static HOST_TEXTURE_DEPTH: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_depth);

async fn host_texture_planar_roundtrip(ctx: TestingContext) {
    if !host_copyable(&ctx, TextureFormat::NV12) {
        return;
    }
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        mapped_at_creation: true,
        ..desc(
            TextureFormat::NV12,
            extent(4, 4, 1),
            TextureUsages::HOST_VISIBLE | TextureUsages::COPY_SRC,
        )
    });
    let planes = [
        (wgpu::TextureAspect::Plane0, extent(4, 4, 1), 1, 0),
        (wgpu::TextureAspect::Plane1, extent(2, 2, 1), 2, 100),
    ]
    .map(|(aspect, size, bytes_per_texel, seed)| {
        let range = wgpu::TexelCopyTextureInfoBase {
            aspect,
            ..host_range(0, Origin3d::ZERO)
        };
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * bytes_per_texel),
            rows_per_image: Some(size.height),
        };
        let data = pattern((size.width * bytes_per_texel * size.height) as usize, seed);
        (range, layout, size, data)
    });
    let mapped = texture.get_mapped_texture_mut().unwrap();
    for (range, layout, size, data) in &planes {
        mapped.copy_from_memory(*range, data, *layout, *size);
    }
    drop(mapped);
    texture.unmap();
    map_and_wait(&ctx, &texture, wgpu::MapMode::Read).await;
    let mapped = texture.get_mapped_texture().unwrap();
    for (range, layout, size, data) in &planes {
        let mut readback = vec![PADDING_BYTE; data.len()];
        mapped.copy_to_memory(*range, &mut readback, *layout, *size);
        assert_eq!(&readback, data, "{:?} host round-trip", range.aspect);
    }
    drop(mapped);
    texture.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_PLANAR_ROUNDTRIP: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_TEXTURE_COPY | wgpu::Features::TEXTURE_FORMAT_NV12),
    )
    .run_async(host_texture_planar_roundtrip);

#[derive(Clone, Copy, Debug)]
struct LayoutCase {
    name: &'static str,
    format: TextureFormat,
    dimension: wgpu::TextureDimension,
    size: wgpu::Extent3d,
    mip_level_count: u32,
    mip_level: u32,
    origin: Origin3d,
    copy_size: wgpu::Extent3d,
    offset: u64,
    extra_bytes_per_row: u32,
    extra_rows_per_image: u32,
}

impl LayoutCase {
    const BASE: Self = Self {
        name: "",
        format: TextureFormat::Rgba8Unorm,
        dimension: wgpu::TextureDimension::D2,
        size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        mip_level: 0,
        origin: Origin3d::ZERO,
        copy_size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        offset: 0,
        extra_bytes_per_row: 0,
        extra_rows_per_image: 0,
    };

    fn create_texture(&self, device: &wgpu::Device, mapped_at_creation: bool) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(self.name),
            mip_level_count: self.mip_level_count,
            dimension: self.dimension,
            mapped_at_creation,
            ..desc(self.format, self.size, HOST_RW)
        })
    }
}

/// Copies `case` from the host and checks it through a GPU read, then copies it
/// on the GPU and checks it through host reads. Padding in the host layout must
/// stay untouched.
async fn run_layout_case(ctx: &TestingContext, case: &LayoutCase) {
    let name = case.name;
    let (row_bytes, rows) = copy_geometry(case.format, case.copy_size);
    let depth = case.copy_size.depth_or_array_layers;
    let layout = wgpu::TexelCopyBufferLayout {
        offset: case.offset,
        bytes_per_row: Some(row_bytes + case.extra_bytes_per_row),
        rows_per_image: Some(rows + case.extra_rows_per_image),
    };
    let dense = pattern((row_bytes * rows * depth) as usize, 0);
    let packed = pack(&dense, row_bytes, rows, depth, layout);
    let range = host_range(case.mip_level, case.origin);
    let mip_size = case
        .size
        .mip_level_size(case.mip_level, case.dimension)
        .physical_size(case.format);
    let expected = embed(case.format, mip_size, case.origin, case.copy_size, &dense);

    let texture = case.create_texture(&ctx.device, true);
    texture.get_mapped_texture_mut().unwrap().copy_from_memory(
        range,
        &packed,
        layout,
        case.copy_size,
    );
    texture.unmap();
    let gpu = gpu_read(ctx, &texture, case.mip_level, mip_size).await;
    assert_eq!(gpu, expected, "{name}: host write, GPU read");

    let texture = case.create_texture(&ctx.device, false);
    let mut encoder = encoder(&ctx.device);
    gpu_write(
        &ctx.device,
        &mut encoder,
        &texture,
        case.mip_level,
        case.origin,
        case.copy_size,
        &dense,
    );
    ctx.queue.submit(Some(encoder.finish()));
    map_and_wait(ctx, &texture, wgpu::MapMode::Read).await;
    let mapped = texture.get_mapped_texture().unwrap();
    let mut readback = vec![PADDING_BYTE; packed.len()];
    mapped.copy_to_memory(range, &mut readback, layout, case.copy_size);
    assert_eq!(readback, packed, "{name}: GPU write, host read");
    let mut full = vec![PADDING_BYTE; expected.len()];
    mapped.copy_to_memory(
        host_range(case.mip_level, Origin3d::ZERO),
        &mut full,
        tight_layout(case.format, mip_size),
        mip_size,
    );
    assert_eq!(full, expected, "{name}: GPU write, full mip host read");
    drop(mapped);
    texture.unmap();
}

async fn host_texture_layouts(ctx: TestingContext) {
    let base = LayoutCase::BASE;
    let cases = [
        LayoutCase {
            name: "mip level 1",
            size: extent(16, 16, 1),
            mip_level_count: 3,
            mip_level: 1,
            ..base
        },
        LayoutCase {
            name: "array layer 1",
            size: extent(8, 8, 3),
            origin: Origin3d { x: 0, y: 0, z: 1 },
            ..base
        },
        LayoutCase {
            name: "3d slices",
            dimension: wgpu::TextureDimension::D3,
            size: extent(8, 8, 4),
            origin: Origin3d { x: 0, y: 0, z: 1 },
            copy_size: extent(8, 8, 2),
            extra_rows_per_image: 3,
            ..base
        },
        LayoutCase {
            name: "non-zero origin",
            origin: Origin3d { x: 2, y: 1, z: 0 },
            copy_size: extent(3, 4, 1),
            ..base
        },
        LayoutCase {
            name: "padded layout",
            size: extent(8, 8, 2),
            copy_size: extent(5, 6, 2),
            offset: 8,
            extra_bytes_per_row: 12,
            extra_rows_per_image: 2,
            ..base
        },
        LayoutCase {
            name: "unaligned offset",
            size: extent(8, 8, 2),
            origin: Origin3d { x: 1, y: 1, z: 0 },
            copy_size: extent(5, 6, 2),
            offset: 7,
            extra_bytes_per_row: 12,
            extra_rows_per_image: 1,
            ..base
        },
        LayoutCase {
            name: "bytes_per_row not a multiple of the texel size",
            size: extent(8, 8, 2),
            origin: Origin3d { x: 1, y: 1, z: 0 },
            copy_size: extent(5, 6, 2),
            offset: 7,
            extra_bytes_per_row: 13,
            extra_rows_per_image: 1,
            ..base
        },
    ];
    for case in &cases {
        run_layout_case(&ctx, case).await;
    }
}

#[apply(gpu_test!)]
static HOST_TEXTURE_LAYOUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_layouts);

async fn host_texture_bc_layouts(ctx: TestingContext) {
    let format = TextureFormat::Bc1RgbaUnorm;
    if !host_copyable(&ctx, format) {
        return;
    }
    let base = LayoutCase {
        format,
        size: extent(16, 16, 1),
        copy_size: extent(16, 16, 1),
        ..LayoutCase::BASE
    };
    // Mips 3 and 4 are 2x2 and 1x1, smaller than a block.
    let small_mip = LayoutCase {
        mip_level_count: 5,
        copy_size: extent(4, 4, 1),
        ..base
    };
    let cases = [
        LayoutCase {
            name: "bc1 mip level 1",
            mip_level_count: 2,
            mip_level: 1,
            copy_size: extent(8, 8, 1),
            offset: 8,
            extra_bytes_per_row: 8,
            ..base
        },
        LayoutCase {
            name: "bc1 non-zero origin",
            origin: Origin3d { x: 4, y: 8, z: 0 },
            copy_size: extent(8, 4, 1),
            offset: 3,
            extra_bytes_per_row: 8,
            ..base
        },
        LayoutCase {
            name: "bc1 array layer 1",
            size: extent(16, 16, 3),
            origin: Origin3d { x: 0, y: 0, z: 1 },
            copy_size: extent(16, 16, 2),
            extra_rows_per_image: 1,
            ..base
        },
        LayoutCase {
            name: "bc1 mip level 3",
            mip_level: 3,
            ..small_mip
        },
        LayoutCase {
            name: "bc1 mip level 4",
            mip_level: 4,
            ..small_mip
        },
    ];
    for case in &cases {
        run_layout_case(&ctx, case).await;
    }

    let block = small_mip.copy_size;
    let texture = small_mip.create_texture(&ctx.device, false);
    map_and_wait(&ctx, &texture, wgpu::MapMode::Read).await;
    {
        let mapped = texture.get_mapped_texture().unwrap();
        for mip in 2..5 {
            let mut readback = vec![PADDING_BYTE; 8];
            mapped.copy_to_memory(
                host_range(mip, Origin3d::ZERO),
                &mut readback,
                tight_layout(format, block),
                block,
            );
            assert_eq!(readback, [0; 8], "mip {mip}: unwritten host read");
        }
    }
    texture.unmap();
    for mip in 2..5 {
        let gpu = gpu_read(&ctx, &texture, mip, block).await;
        assert_eq!(gpu, [0; 8], "mip {mip}: unwritten GPU read");
    }
}

#[apply(gpu_test!)]
static HOST_TEXTURE_BC_LAYOUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::HOST_TEXTURE_COPY | wgpu::Features::TEXTURE_COMPRESSION_BC),
    )
    .run_async(host_texture_bc_layouts);

async fn host_texture_bc_3d_layouts(ctx: TestingContext) {
    let format = TextureFormat::Bc1RgbaUnorm;
    if !host_copyable(&ctx, format) {
        return;
    }
    let base = LayoutCase {
        format,
        dimension: wgpu::TextureDimension::D3,
        offset: 1,
        ..LayoutCase::BASE
    };
    let cases = [
        LayoutCase {
            name: "bc1 3d whole texture",
            size: extent(4, 4, 2),
            copy_size: extent(4, 4, 2),
            ..base
        },
        LayoutCase {
            name: "bc1 3d sub-region",
            size: extent(8, 8, 3),
            origin: Origin3d { x: 4, y: 4, z: 1 },
            copy_size: extent(4, 4, 2),
            extra_bytes_per_row: 8,
            ..base
        },
    ];
    for case in &cases {
        run_layout_case(&ctx, case).await;
    }

    let size = extent(8, 8, 2);
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        dimension: wgpu::TextureDimension::D3,
        ..desc(
            format,
            size,
            TextureUsages::COPY_DST | TextureUsages::COPY_SRC,
        )
    });
    let origin = Origin3d { x: 0, y: 4, z: 0 };
    let copy_size = extent(8, 4, 2);
    let dense = pattern(32, 9);
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 1,
        bytes_per_row: Some(16),
        rows_per_image: Some(1),
    };
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            origin,
            ..texture.as_image_copy()
        },
        &pack(&dense, 16, 1, 2, layout),
        layout,
        copy_size,
    );
    assert_eq!(
        gpu_read(&ctx, &texture, 0, size).await,
        embed(format, size, origin, copy_size, &dense),
        "bc1 3d write_texture"
    );
}

#[apply(gpu_test!)]
static HOST_TEXTURE_BC_3D_LAYOUTS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(
        wgpu::Features::HOST_TEXTURE_COPY
            | wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_BC_SLICED_3D,
    ))
    .run_async(host_texture_bc_3d_layouts);

#[cfg(not(target_arch = "wasm32"))]
async fn host_texture_multithreaded(ctx: TestingContext) {
    const THREADS: u32 = 4;
    const SIZE: u32 = 64;
    const ROW_BYTES: usize = SIZE as usize * 4;

    let create = || {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            mapped_at_creation: true,
            ..desc(
                TextureFormat::Rgba8Unorm,
                extent(SIZE, SIZE, 1),
                TextureUsages::HOST_VISIBLE | TextureUsages::COPY_SRC,
            )
        })
    };
    let write_rows = |mapped: &wgpu::MappedTextureMut, data: &[u8], rows: core::ops::Range<u32>| {
        for y in rows {
            let start = y as usize * ROW_BYTES;
            mapped.copy_from_memory(
                host_range(0, Origin3d { x: 0, y, z: 0 }),
                &data[start..start + ROW_BYTES],
                wgpu::TexelCopyBufferLayout::default(),
                extent(SIZE, 1, 1),
            );
        }
    };

    let own_textures: Vec<_> = (0..THREADS).map(|_| create()).collect();
    let own_data: Vec<_> = (0..THREADS)
        .map(|i| pattern(ROW_BYTES * SIZE as usize, i as u8 * 31))
        .collect();
    let shared = create();
    let shared_data = pattern(ROW_BYTES * SIZE as usize, 200);
    let unwritten = create();

    let shared_mapped = shared.get_mapped_texture_mut().unwrap();
    std::thread::scope(|s| {
        for (i, (texture, data)) in own_textures.iter().zip(&own_data).enumerate() {
            let shared_mapped = &shared_mapped;
            let shared_data = &shared_data;
            s.spawn(move || {
                let mapped = texture.get_mapped_texture_mut().unwrap();
                write_rows(&mapped, data, 0..SIZE);
                let band = SIZE / THREADS;
                let i = i as u32;
                write_rows(shared_mapped, shared_data, i * band..(i + 1) * band);
            });
        }
    });
    drop(shared_mapped);
    for texture in [&shared, &unwritten] {
        texture.unmap();
        map_and_wait(&ctx, texture, wgpu::MapMode::Read).await;
    }

    std::thread::scope(|s| {
        for _ in 0..THREADS {
            s.spawn(|| {
                for _ in 0..8 {
                    assert_eq!(host_read_all(&shared), shared_data);
                    assert!(host_read_all(&unwritten).iter().all(|&b| b == 0));
                }
            });
        }
    });

    for (texture, data) in own_textures
        .iter()
        .zip(&own_data)
        .chain([(&shared, &shared_data)])
    {
        texture.unmap();
        assert_eq!(&gpu_read(&ctx, texture, 0, texture.size()).await, data);
    }
    unwritten.unmap();
}

#[apply(gpu_test!)]
static HOST_TEXTURE_MULTITHREADED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::HOST_TEXTURE_COPY))
    .run_async(|ctx| async move {
        #[cfg(not(target_arch = "wasm32"))]
        host_texture_multithreaded(ctx).await;
        #[cfg(target_arch = "wasm32")]
        let _ = ctx;
    });

fn clear_pass(encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture, color: wgpu::Color) {
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
}

/// `Queue::write_texture` may use host copies internally, but must keep its
/// ordering relative to submissions.
async fn write_texture_ordering(ctx: TestingContext) {
    let ctx = &ctx;
    let size = extent(4, 4, 1);
    let create = || {
        rgba_texture(
            &ctx.device,
            TextureUsages::COPY_DST | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT,
            false,
        )
    };
    let red_bytes = [255u8, 0, 0, 255].repeat(16);
    let b = pattern(64, 7);
    let c = pattern(16, 8);
    let clear = |texture: &wgpu::Texture| {
        let mut encoder = encoder(&ctx.device);
        clear_pass(&mut encoder, texture, wgpu::Color::RED);
        encoder.finish()
    };
    let write = |texture: &wgpu::Texture| {
        write_texture(&ctx.queue, texture, Origin3d::ZERO, size, &b);
    };
    let read = |texture: wgpu::Texture| async move { gpu_read(ctx, &texture, 0, size).await };

    let t = create();
    write(&t);
    assert_eq!(read(t).await, b, "write only");

    let t = create();
    ctx.queue.submit(Some(clear(&t)));
    write(&t);
    assert_eq!(read(t).await, b, "clear then write");

    let t = create();
    write(&t);
    ctx.queue.submit(Some(clear(&t)));
    assert_eq!(read(t).await, red_bytes, "write then clear");

    let t = create();
    let clear_command_buffer = clear(&t);
    write(&t);
    ctx.queue.submit(Some(clear_command_buffer));
    assert_eq!(
        read(t).await,
        red_bytes,
        "clear recorded, then write, then clear submitted"
    );

    let origin = Origin3d { x: 1, y: 2, z: 0 };
    let part = extent(2, 2, 1);
    let t = create();
    write(&t);
    write_texture(&ctx.queue, &t, origin, part, &c);
    let mut expected = b.clone();
    for (y, row) in c.chunks(8).enumerate() {
        let start = (y + 2) * 16 + 4;
        expected[start..start + 8].copy_from_slice(row);
    }
    assert_eq!(read(t).await, expected, "write then partial write");

    let t = create();
    write_texture(&ctx.queue, &t, origin, part, &c);
    assert_eq!(
        read(t).await,
        embed(TextureFormat::Rgba8Unorm, size, origin, part, &c),
        "partial write"
    );

    let t = create();
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 3,
        bytes_per_row: Some(17),
        rows_per_image: Some(4),
    };
    ctx.queue
        .write_texture(t.as_image_copy(), &pack(&b, 16, 4, 1, layout), layout, size);
    assert_eq!(read(t).await, b, "unaligned layout");
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_ORDERING: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(write_texture_ordering);

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_ORDERING_NO_IMPLICIT_HOST_COPY: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .instance_flags(wgpu::InstanceFlags::DISABLE_IMPLICIT_HOST_TEXTURE_COPY),
        )
        .run_async(write_texture_ordering);

/// Block-compressed formats are the ones Metal lets `write_texture` copy into
/// from the host.
async fn write_texture_compressed_ordering(ctx: TestingContext) {
    let ctx = &ctx;
    let format = TextureFormat::Bc1RgbaUnorm;
    let size = extent(16, 16, 1);
    let create = || {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: TextureUsages::COPY_DST
                | TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
            texture_binding_view_dimension: None,
            mapped_at_creation: false,
        })
    };
    let a = pattern(128, 3);
    let b = pattern(128, 9);
    let c = pattern(32, 5);
    let write = |texture: &wgpu::Texture, data: &[u8]| {
        write_texture(&ctx.queue, texture, Origin3d::ZERO, size, data);
    };
    let copy = |src: &wgpu::Texture, dst: &wgpu::Texture| {
        let mut encoder = encoder(&ctx.device);
        encoder.copy_texture_to_texture(src.as_image_copy(), dst.as_image_copy(), size);
        encoder.finish()
    };
    let read = |texture: wgpu::Texture| async move { gpu_read(ctx, &texture, 0, size).await };

    let src = create();
    write(&src, &b);

    let t = create();
    write(&t, &a);
    assert_eq!(read(t).await, a, "write only");

    let origin = Origin3d { x: 4, y: 8, z: 0 };
    let part = extent(8, 8, 1);
    let t = create();
    write_texture(&ctx.queue, &t, origin, part, &c);
    assert_eq!(
        read(t).await,
        embed(format, size, origin, part, &c),
        "partial write"
    );

    let t = create();
    write(&t, &a);
    ctx.queue.submit(Some(copy(&src, &t)));
    assert_eq!(read(t).await, b, "write then copy");

    let t = create();
    ctx.queue.submit(Some(copy(&src, &t)));
    write(&t, &a);
    assert_eq!(read(t).await, a, "copy then write");

    let t = create();
    let copy_command_buffer = copy(&src, &t);
    write(&t, &a);
    ctx.queue.submit(Some(copy_command_buffer));
    assert_eq!(
        read(t).await,
        b,
        "copy recorded, then write, then copy submitted"
    );
}

fn compressed_params() -> TestParameters {
    TestParameters::default().features(wgpu::Features::TEXTURE_COMPRESSION_BC)
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_COMPRESSED_ORDERING: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(compressed_params())
        .run_async(write_texture_compressed_ordering);

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_COMPRESSED_ORDERING_NO_IMPLICIT_HOST_COPY: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            compressed_params()
                .instance_flags(wgpu::InstanceFlags::DISABLE_IMPLICIT_HOST_TEXTURE_COPY),
        )
        .run_async(write_texture_compressed_ordering);

async fn host_texture_write_texture_host_visible(ctx: TestingContext) {
    let texture = rgba_texture(&ctx.device, HOST_RW, false);
    let size = texture.size();
    let b = pattern(64, 10);
    let c = pattern(64, 11);
    let d = pattern(64, 12);

    write_texture(&ctx.queue, &texture, Origin3d::ZERO, size, &b);
    map_and_wait(&ctx, &texture, wgpu::MapMode::Read).await;
    assert_eq!(host_read_all(&texture), b);
    texture.unmap();

    let layout = wgpu::TexelCopyBufferLayout {
        offset: 3,
        bytes_per_row: Some(17),
        rows_per_image: Some(4),
    };
    ctx.queue.write_texture(
        texture.as_image_copy(),
        &pack(&c, 16, 4, 1, layout),
        layout,
        size,
    );
    assert_eq!(gpu_read(&ctx, &texture, 0, size).await, c);

    write_texture(&ctx.queue, &texture, Origin3d::ZERO, size, &d);
    map_and_wait(&ctx, &texture, wgpu::MapMode::Read).await;
    assert_eq!(host_read_all(&texture), d);
    texture.unmap();
    map_and_wait(&ctx, &texture, wgpu::MapMode::Write).await;
    host_write_all(&texture, &b);
    texture.unmap();
    assert_eq!(gpu_read(&ctx, &texture, 0, size).await, b);
}

#[apply(gpu_test!)]
static HOST_TEXTURE_WRITE_TEXTURE_HOST_VISIBLE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(host_params())
    .run_async(host_texture_write_texture_host_visible);
