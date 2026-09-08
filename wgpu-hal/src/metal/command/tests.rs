//! All tests here access a real Metal GPU, including fence setup, non-ray encoder,
//! and descriptor cases; this is not a CPU-only suite. The ignored capacity test
//! retains 4095 native command buffers and requires explicit hardware authorization.

use super::*;
use crate::metal as m;
use crate::{Adapter as _, Device as _, Queue as _};
use alloc::vec;
use core::{cell::RefCell, sync::atomic};
use objc2_metal::{MTLCommandBufferStatus, MTLCreateSystemDefaultDevice, MTLResourceOptions};

#[derive(Clone, Debug)]
struct Event {
    buffer: usize,
    encoder: Option<usize>,
    operation: &'static str,
}

std::thread_local! {
    static EVENTS: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
}

pub(in crate::metal) fn identity<T: ?Sized>(object: &T) -> usize {
    core::ptr::from_ref(object).cast::<()>() as usize
}

pub(in crate::metal) fn record(
    buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    encoder: Option<usize>,
    operation: &'static str,
) {
    std::eprintln!(
        "Metal emission buffer={:#x} encoder={encoder:?} {operation}",
        identity(buffer)
    );
    EVENTS.with_borrow_mut(|events| {
        events.push(Event {
            buffer: identity(buffer),
            encoder,
            operation,
        })
    });
}

impl m::CommandEncoder {
    pub(super) fn trace<T: ?Sized>(&self, encoder: &ProtocolObject<T>, operation: &'static str) {
        record(
            self.raw_cmd_buf.as_ref().unwrap(),
            Some(identity(encoder)),
            operation,
        );
    }
}

fn events() -> Vec<Event> {
    EVENTS.with_borrow(|events| events.clone())
}

fn open() -> Option<crate::OpenDevice<m::Api>> {
    let raw = MTLCreateSystemDefaultDevice()?;
    let exposed = m::AdapterShared::expose(raw);
    if !exposed
        .features
        .contains(wgt::Features::EXPERIMENTAL_RAY_QUERY)
    {
        std::eprintln!("SKIP: Metal ray queries unavailable");
        return None;
    }
    Some(unsafe {
        exposed
            .adapter
            .open(
                wgt::Features::EXPERIMENTAL_RAY_QUERY,
                &exposed.capabilities.limits,
                &wgt::MemoryHints::default(),
            )
            .unwrap()
    })
}

fn encoder(device: &m::Device, queue: &m::Queue) -> m::CommandEncoder {
    unsafe {
        device
            .create_command_encoder(&crate::CommandEncoderDescriptor { label: None, queue })
            .unwrap()
    }
}

fn buffer(device: &m::Device, size: usize) -> m::Buffer {
    m::Buffer {
        raw: device
            .shared
            .device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .unwrap(),
        size: size as u64,
    }
}

fn triangles(vertex: &m::Buffer) -> crate::AccelerationStructureEntries<'_, m::Buffer> {
    crate::AccelerationStructureEntries::Triangles(vec![crate::AccelerationStructureTriangles {
        vertex_buffer: Some(vertex),
        vertex_format: wgt::VertexFormat::Float32x3,
        first_vertex: 0,
        vertex_count: 3,
        vertex_stride: 12,
        indices: None,
        transform: None,
        flags: wgt::AccelerationStructureGeometryFlags::OPAQUE,
    }])
}

fn storage(
    device: &m::Device,
    entries: &crate::AccelerationStructureEntries<'_, m::Buffer>,
) -> (m::AccelerationStructure, m::Buffer) {
    let sizes = unsafe {
        device.get_acceleration_structure_build_sizes(
            &crate::GetAccelerationStructureBuildSizesDescriptor {
                entries,
                flags: wgt::AccelerationStructureFlags::empty(),
            },
        )
    };
    let structure = unsafe {
        device
            .create_acceleration_structure(&crate::AccelerationStructureDescriptor {
                label: None,
                size: sizes.acceleration_structure_size,
                format: match entries {
                    crate::AccelerationStructureEntries::Instances(_) => {
                        crate::AccelerationStructureFormat::TopLevel
                    }
                    _ => crate::AccelerationStructureFormat::BottomLevel,
                },
                allow_compaction: true,
            })
            .unwrap()
    };
    (structure, buffer(device, sizes.build_scratch_size as usize))
}

unsafe fn build(
    encoder: &mut m::CommandEncoder,
    entries: &crate::AccelerationStructureEntries<'_, m::Buffer>,
    structure: &m::AccelerationStructure,
    scratch: &m::Buffer,
) {
    unsafe {
        encoder.build_acceleration_structures(
            1,
            [crate::BuildAccelerationStructureDescriptor {
                entries,
                mode: crate::AccelerationStructureBuildMode::Build,
                flags: wgt::AccelerationStructureFlags::empty(),
                source_acceleration_structure: None,
                destination_acceleration_structure: structure,
                scratch_buffer: scratch,
                scratch_buffer_offset: 0,
            }],
        );
    }
}

fn assert_recording(id: usize, expected: &[&str]) {
    let recording: Vec<_> = events()
        .into_iter()
        .filter(|event| event.buffer == id)
        .collect();
    assert_eq!(
        recording
            .iter()
            .map(|event| event.operation)
            .collect::<Vec<_>>(),
        expected
    );
    for group in recording.chunks(4) {
        if group.len() == 4 && group[0].operation == "as-wait" {
            assert!(group.iter().all(|event| event.encoder == group[0].encoder));
        }
    }
}

#[test]
fn fence_setup_failure_is_fallible() {
    let mut sync = m::AccelerationStructureSync::default();
    assert!(matches!(
        sync.initialize_fence(|| None),
        Err(crate::DeviceError::OutOfMemory)
    ));
    assert!(sync.fence.is_none());
    assert!(!sync.seeded);
    let Some(opened) = open() else { return };
    let fence = opened.device.shared.device.newFence().unwrap();
    sync.initialize_fence(|| Some(fence.clone())).unwrap();
    sync.initialize_fence(|| panic!("initialized fence must be reused"))
        .unwrap();
    assert_eq!(identity(&**sync.fence.as_ref().unwrap()), identity(&*fence));
}

#[test]
fn non_ray_encoder_does_not_allocate_as_fence() {
    let Some(raw) = MTLCreateSystemDefaultDevice() else {
        return;
    };
    let exposed = m::AdapterShared::expose(raw);
    let opened = unsafe {
        exposed
            .adapter
            .open(
                wgt::Features::empty(),
                &exposed.capabilities.limits,
                &wgt::MemoryHints::default(),
            )
            .unwrap()
    };
    let enc = encoder(&opened.device, &opened.queue);
    assert!(opened
        .queue
        .shared
        .acceleration_structure_sync
        .lock()
        .fence
        .is_none());
    drop(enc);
}

#[test]
#[ignore = "opt-in native queue-capacity stress; may make the machine/display unresponsive"]
fn first_as_at_native_recording_capacity() {
    const CHILD: &str = "WGPU_METAL_CAPACITY_TEST_CHILD";
    const CHILD_COMPLETED: i32 = 42;
    if std::env::var_os(CHILD).is_none() {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "metal::command::tests::first_as_at_native_recording_capacity",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .spawn()
            .unwrap();
        // This userspace deadline cannot protect against GPU/driver hangs or display freezes.
        let deadline = std::time::Instant::now() + core::time::Duration::from_secs(60);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                // A zero-test libtest invocation exits 0, not CHILD_COMPLETED.
                assert_eq!(
                    status.code(),
                    Some(CHILD_COMPLETED),
                    "capacity subprocess did not complete the native test: {status}"
                );
                return;
            }
            if std::time::Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("native capacity subprocess failed to make bounded progress");
            }
            std::thread::sleep(core::time::Duration::from_millis(20));
        }
    }
    assert_eq!(std::env::var(CHILD).as_deref(), Ok("1"));
    run_native_recording_capacity();
    // Signal completion only after the native checks and resource drops have finished.
    std::process::exit(CHILD_COMPLETED);
}

fn run_native_recording_capacity() {
    let crate::OpenDevice { device, queue } =
        open().expect("opt-in native capacity test requires a Metal ray-query device");
    let vertex = buffer(&device, 36);
    let vertices = [-1.0f32, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
    unsafe {
        core::ptr::copy_nonoverlapping(
            vertices.as_ptr().cast::<u8>(),
            vertex.raw.contents().as_ptr().cast(),
            36,
        );
    }
    let entries = triangles(&vertex);
    let (blas, scratch) = storage(&device, &entries);
    let mut enc = encoder(&device, &queue);
    unsafe {
        enc.begin_encoding(None).unwrap();
        build(&mut enc, &entries, &blas, &scratch);
        let participant = enc.end_encoding().unwrap();
        let mut held = Vec::new();
        for _ in 1..adapter::MAX_UNSUBMITTED_COMMAND_BUFFERS {
            enc.begin_encoding(None).unwrap();
            held.push(enc.end_encoding().unwrap());
        }
        assert!(matches!(
            enc.begin_encoding(None),
            Err(crate::DeviceError::Lost)
        ));
        assert_eq!(
            queue
                .shared
                .command_buffer_created_not_submitted
                .load(atomic::Ordering::Acquire),
            adapter::MAX_UNSUBMITTED_COMMAND_BUFFERS
        );
        let fence = device.create_fence().unwrap();
        // Internal empty-submit and idle joins must also progress with held recordings.
        queue.submit(&[], &[], (&fence, 1)).unwrap();
        queue.wait_for_idle().unwrap();
        assert!(!queue.shared.acceleration_structure_sync.lock().seeded);
        queue.submit(&[&participant], &[], (&fence, 2)).unwrap();
        assert!(device
            .wait(&fence, 2, Some(core::time::Duration::from_secs(30)))
            .unwrap());
        let emitted = events();
        let seed = emitted
            .iter()
            .position(|e| e.operation == "seed-commit")
            .unwrap();
        let commit = emitted
            .iter()
            .position(|e| e.operation == "commit" && e.buffer == identity(&*participant.raw))
            .unwrap();
        assert!(seed < commit);
        assert!(emitted.iter().any(|e| e.operation == "retire-wait"));
        enc.reset_all(held.into_iter().chain([participant]));
        assert_eq!(
            queue
                .shared
                .command_buffer_created_not_submitted
                .load(atomic::Ordering::Acquire),
            0
        );
        drop(enc);
        device.destroy_fence(fence);
        device.destroy_acceleration_structure(blas);
    }
}

#[test]
fn encoder_finish_discard_reuse_and_submit_order() {
    let Some(crate::OpenDevice { device, queue }) = open() else {
        return;
    };
    let vertex = buffer(&device, 36);
    let vertices = [-1.0f32, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
    unsafe {
        core::ptr::copy_nonoverlapping(
            vertices.as_ptr().cast::<u8>(),
            vertex.raw.contents().as_ptr().cast(),
            36,
        );
    }
    let entries = triangles(&vertex);
    let (blas, scratch) = storage(&device, &entries);
    let mut enc = encoder(&device, &queue);
    unsafe {
        enc.begin_encoding(None).unwrap();
        build(&mut enc, &entries, &blas, &scratch);
        enc.enter_blit();
        let abandoned = enc.end_encoding().unwrap();
        let abandoned_id = identity(&*abandoned.raw);
        assert_recording(abandoned_id, &["as-wait", "build", "as-update", "as-end"]);
        assert!(abandoned.contains_acceleration_structure_commands);
        assert!(!enc.state.contains_acceleration_structure_commands);

        enc.begin_encoding(None).unwrap();
        build(&mut enc, &entries, &blas, &scratch);
        let discarded_raw = enc.raw_cmd_buf.as_ref().unwrap().clone();
        let discarded_id = identity(&*discarded_raw);
        enc.discard_encoding();
        assert_recording(discarded_id, &["as-wait", "build", "as-end"]);

        enc.begin_encoding(None).unwrap();
        let ordinary = enc.end_encoding().unwrap();
        assert!(!ordinary.contains_acceleration_structure_commands);
        assert!(!queue.shared.acceleration_structure_sync.lock().seeded);

        enc.begin_encoding(None).unwrap();
        build(&mut enc, &entries, &blas, &scratch);
        let submitted = enc.end_encoding().unwrap();
        let submitted_id = identity(&*submitted.raw);
        assert_recording(submitted_id, &["as-wait", "build", "as-update", "as-end"]);
        let fence = device.create_fence().unwrap();
        queue
            .submit(&[&submitted, &ordinary], &[], (&fence, 1))
            .unwrap();
        assert!(device
            .wait(&fence, 1, Some(core::time::Duration::from_secs(30)))
            .unwrap());
        assert_eq!(submitted.raw.status(), MTLCommandBufferStatus::Completed);
        assert!(queue.shared.acceleration_structure_sync.lock().seeded);
        let commits: Vec<_> = events()
            .into_iter()
            .filter(|event| event.operation.ends_with("commit"))
            .collect();
        assert_eq!(commits.len(), 3);
        assert_eq!(commits[0].operation, "seed-commit");
        assert_eq!(commits[1].buffer, submitted_id);
        assert_eq!(commits[2].buffer, identity(&*ordinary.raw));
        assert!(!commits.iter().any(|event| event.buffer == abandoned_id));
        let retirement: Vec<_> = events()
            .into_iter()
            .filter(|event| event.operation == "retire-wait")
            .collect();
        assert_eq!(retirement.len(), 1);
        assert_eq!(retirement[0].buffer, identity(&*ordinary.raw));

        EVENTS.with_borrow_mut(Vec::clear);
        enc.begin_encoding(None).unwrap();
        enc.enter_blit();
        let post = enc.end_encoding().unwrap();
        queue.submit(&[&post], &[], (&fence, 2)).unwrap();
        queue.submit(&[], &[], (&fence, 3)).unwrap();
        assert!(device
            .wait(&fence, 3, Some(core::time::Duration::from_secs(30)))
            .unwrap());
        assert_eq!(
            events()
                .iter()
                .map(|event| event.operation)
                .collect::<Vec<_>>(),
            ["commit"]
        );
        drop(abandoned);
        assert_eq!(
            queue
                .shared
                .command_buffer_created_not_submitted
                .load(atomic::Ordering::Acquire),
            0
        );
        device.destroy_fence(fence);
    }
}

#[test]
fn query_only_compute_and_render_passes_join_once() {
    let Some(crate::OpenDevice { device, queue }) = open() else {
        return;
    };
    let vertex = buffer(&device, 36);
    let entries = triangles(&vertex);
    let (blas, _) = storage(&device, &entries);
    let mut group = m::BindGroup::default();
    group
        .buffers
        .push(m::BufferLikeResource::AccelerationStructure(NonNull::from(
            &*blas.raw,
        )));
    group.counters.cs.buffers = 1;
    group.counters.vs.buffers = 1;
    group.counters.fs.buffers = 1;
    let info = m::BindGroupLayoutInfo {
        base_resource_indices: Default::default(),
    };
    let mut enc = encoder(&device, &queue);
    let texture_desc = objc2_metal::MTLTextureDescriptor::new();
    texture_desc.setPixelFormat(objc2_metal::MTLPixelFormat::RGBA8Unorm);
    unsafe {
        texture_desc.setWidth(1);
        texture_desc.setHeight(1);
    }
    texture_desc.setUsage(objc2_metal::MTLTextureUsage::RenderTarget);
    let target = m::TextureView {
        raw: device
            .shared
            .device
            .newTextureWithDescriptor(&texture_desc)
            .unwrap(),
        aspects: crate::FormatAspects::COLOR,
    };
    unsafe {
        enc.begin_encoding(None).unwrap();
        enc.begin_compute_pass(&crate::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        let raw = enc.state.compute.as_ref().unwrap().clone();
        for _ in 0..2 {
            enc.update_bind_group_state(
                Encoder::Compute(&raw),
                Default::default(),
                &info,
                &[],
                0,
                &group,
            );
        }
        enc.end_compute_pass();
        let compute = enc.end_encoding().unwrap();
        assert!(compute.contains_acceleration_structure_commands);
        assert_recording(identity(&*compute.raw), &["compute-wait", "compute-update"]);

        enc.begin_encoding(None).unwrap();
        enc.begin_render_pass(&crate::RenderPassDescriptor {
            label: None,
            extent: wgt::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            sample_count: 1,
            color_attachments: &[Some(crate::ColorAttachment {
                target: crate::Attachment {
                    view: &target,
                    usage: wgt::TextureUses::COLOR_TARGET,
                },
                depth_slice: None,
                resolve_target: None,
                ops: crate::AttachmentOps::STORE | crate::AttachmentOps::LOAD_CLEAR,
                clear_value: wgt::Color::BLACK,
            })],
            depth_stencil_attachment: None,
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        })
        .unwrap();
        let raw = enc.state.render.as_ref().unwrap().clone();
        enc.update_bind_group_state(
            Encoder::Vertex(&raw),
            Default::default(),
            &info,
            &[],
            0,
            &group,
        );
        enc.update_bind_group_state(
            Encoder::Fragment(&raw),
            Default::default(),
            &info,
            &[],
            0,
            &group,
        );
        enc.end_render_pass();
        let render = enc.end_encoding().unwrap();
        assert!(render.contains_acceleration_structure_commands);
        assert_recording(identity(&*render.raw), &["render-wait", "render-update"]);
        assert!(!queue.shared.acceleration_structure_sync.lock().seeded);
        // These bindings deliberately never dispatch or draw against the unbuilt AS.
        drop((compute, render));
        enc.begin_encoding(None).unwrap();
        enc.begin_compute_pass(&crate::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        enc.end_compute_pass();
        assert!(
            !enc.end_encoding()
                .unwrap()
                .contains_acceleration_structure_commands
        );
    }
}

#[test]
fn tlas_recorded_before_blas_and_compact_size_retirement() {
    let Some(crate::OpenDevice { mut device, queue }) = open() else {
        return;
    };
    Arc::get_mut(&mut device.shared)
        .unwrap()
        .settings
        .retain_command_buffer_references = false;
    let vertex = buffer(&device, 36);
    let vertices = [-1.0f32, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
    unsafe {
        core::ptr::copy_nonoverlapping(
            vertices.as_ptr().cast::<u8>(),
            vertex.raw.contents().as_ptr().cast(),
            36,
        );
    }
    let entries = triangles(&vertex);
    let (blas, scratch) = storage(&device, &entries);
    let instances = device.tlas_instance_to_bytes(crate::TlasInstance {
        transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        custom_data: 7,
        mask: 255,
        blas_address: unsafe { device.get_acceleration_structure_device_address(&blas) },
    });
    let instance_buffer = buffer(&device, instances.len());
    unsafe {
        core::ptr::copy_nonoverlapping(
            instances.as_ptr(),
            instance_buffer.raw.contents().as_ptr().cast(),
            instances.len(),
        );
    }
    let tlas_entries =
        crate::AccelerationStructureEntries::Instances(crate::AccelerationStructureInstances {
            buffer: Some(&instance_buffer),
            offset: 0,
            count: 1,
        });
    let (tlas, tlas_scratch) = storage(&device, &tlas_entries);
    let compact_size = buffer(&device, 8);
    let fence = unsafe { device.create_fence().unwrap() };
    let mut producer = encoder(&device, &queue);
    let mut consumer = encoder(&device, &queue);
    for split in [false, true] {
        EVENTS.with_borrow_mut(Vec::clear);
        unsafe {
            consumer.begin_encoding(None).unwrap();
            build(&mut consumer, &tlas_entries, &tlas, &tlas_scratch);
            consumer.read_acceleration_structure_compact_size(&tlas, &compact_size);
            let consumer_buffer = consumer.end_encoding().unwrap();
            producer.begin_encoding(None).unwrap();
            build(&mut producer, &entries, &blas, &scratch);
            let producer_buffer = producer.end_encoding().unwrap();
            m::CommandEncoder::set_acceleration_structure_dependencies(
                &[&consumer_buffer],
                &[&blas],
            );
            let consumer_id = identity(&*consumer_buffer.raw);
            assert_recording(
                consumer_id,
                &[
                    "as-wait",
                    "build",
                    "as-update",
                    "as-end",
                    "as-wait",
                    "compact-size",
                    "as-update",
                    "as-end",
                ],
            );
            if split {
                queue.submit(&[&producer_buffer], &[], (&fence, 2)).unwrap();
                queue.submit(&[&consumer_buffer], &[], (&fence, 3)).unwrap();
            } else {
                queue
                    .submit(&[&producer_buffer, &consumer_buffer], &[], (&fence, 1))
                    .unwrap();
            }
            assert!(device
                .wait(
                    &fence,
                    if split { 3 } else { 1 },
                    Some(core::time::Duration::from_secs(30))
                )
                .unwrap());
            assert_eq!(
                consumer_buffer.raw.status(),
                MTLCommandBufferStatus::Completed
            );
            assert_ne!(compact_size.raw.contents().cast::<u32>().as_ptr().read(), 0);
            let commits: Vec<_> = events()
                .into_iter()
                .filter(|event| event.operation == "commit")
                .map(|event| event.buffer)
                .collect();
            assert_eq!(commits, [identity(&*producer_buffer.raw), consumer_id]);
            let retirement: Vec<_> = events()
                .into_iter()
                .filter(|event| event.operation == "retire-wait")
                .map(|event| event.buffer)
                .collect();
            if split {
                assert_eq!(retirement, [identity(&*producer_buffer.raw), consumer_id]);
            } else {
                assert_eq!(retirement, [consumer_id]);
            }
        }
    }
    unsafe {
        device.destroy_fence(fence);
    }
}

#[test]
fn independent_build_batch_shares_encoder_and_queue_fence() {
    let Some(crate::OpenDevice { device, queue }) = open() else {
        return;
    };
    let vertex = buffer(&device, 36);
    let entries = triangles(&vertex);
    let (first, first_scratch) = storage(&device, &entries);
    let (second, second_scratch) = storage(&device, &entries);
    let mut first_encoder = encoder(&device, &queue);
    let mut second_encoder = encoder(&device, &queue);
    let fence = first_encoder.acceleration_structure_fence();
    assert_eq!(
        identity(&*fence),
        identity(&*second_encoder.acceleration_structure_fence())
    );
    unsafe {
        first_encoder.begin_encoding(None).unwrap();
        second_encoder.begin_encoding(None).unwrap();
        first_encoder.build_acceleration_structures(
            2,
            [(&first, &first_scratch), (&second, &second_scratch)].map(|(destination, scratch)| {
                crate::BuildAccelerationStructureDescriptor {
                    entries: &entries,
                    mode: crate::AccelerationStructureBuildMode::Build,
                    flags: wgt::AccelerationStructureFlags::empty(),
                    source_acceleration_structure: None,
                    destination_acceleration_structure: destination,
                    scratch_buffer: scratch,
                    scratch_buffer_offset: 0,
                }
            }),
        );
        let batch = first_encoder.end_encoding().unwrap();
        assert_recording(
            identity(&*batch.raw),
            &["as-wait", "build", "build", "as-update", "as-end"],
        );
        let ids: Vec<_> = events()
            .into_iter()
            .filter(|event| event.buffer == identity(&*batch.raw))
            .map(|event| event.encoder)
            .collect();
        assert!(ids.iter().all(|id| *id == ids[0]));
        second_encoder.discard_encoding();
        assert!(!queue.shared.acceleration_structure_sync.lock().seeded);
    }
}

#[test]
fn concurrent_recordings_use_commit_order() {
    let Some(crate::OpenDevice { device, queue }) = open() else {
        return;
    };
    let vertex = buffer(&device, 36);
    let vertices = [-1.0f32, -1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0];
    unsafe {
        core::ptr::copy_nonoverlapping(
            vertices.as_ptr().cast::<u8>(),
            vertex.raw.contents().as_ptr().cast(),
            36,
        );
    }
    let entries = triangles(&vertex);
    let (first, first_scratch) = storage(&device, &entries);
    let (second, second_scratch) = storage(&device, &entries);
    let rendezvous = std::sync::Barrier::new(2);
    let record_build = |structure: &m::AccelerationStructure, scratch: &m::Buffer| {
        let mut enc = encoder(&device, &queue);
        unsafe {
            enc.begin_encoding(None).unwrap();
            build(&mut enc, &entries, structure, scratch);
            rendezvous.wait();
            let buffer = enc.end_encoding().unwrap();
            assert_recording(
                identity(&*buffer.raw),
                &["as-wait", "build", "as-update", "as-end"],
            );
            buffer
        }
    };
    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| record_build(&first, &first_scratch));
        let second = scope.spawn(|| record_build(&second, &second_scratch));
        (first.join().unwrap(), second.join().unwrap())
    });
    unsafe {
        let fence = device.create_fence().unwrap();
        queue.submit(&[&second, &first], &[], (&fence, 1)).unwrap();
        assert!(device
            .wait(&fence, 1, Some(core::time::Duration::from_secs(30)))
            .unwrap());
        let commits: Vec<_> = events()
            .into_iter()
            .filter(|event| event.operation == "commit")
            .map(|event| event.buffer)
            .collect();
        assert_eq!(commits, [identity(&*second.raw), identity(&*first.raw)]);
        device.destroy_fence(fence);
    }
}
