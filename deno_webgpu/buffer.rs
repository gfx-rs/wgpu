// Copyright 2018-2024 the Deno authors. All rights reserved. MIT license.

use super::error::WebGpuResult;
use super::wgpu_types;
use deno_core::futures::channel::oneshot;
use deno_core::op2;
use deno_core::OpState;
use deno_core::Resource;
use deno_core::ResourceId;
use std::borrow::Cow;
use std::cell::RefCell;
use std::ptr::NonNull;
use std::rc::Rc;
use std::time::Duration;
use wgpu_core::resource::BufferAccessResult;

#[derive(Debug, thiserror::Error)]
pub enum BufferError {
    #[error(transparent)]
    Resource(deno_core::error::AnyError),
    #[error("usage is not valid")]
    InvalidUsage,
    #[error(transparent)]
    Access(#[from] wgpu_core::resource::BufferAccessError),
    #[error(transparent)]
    Canceled(#[from] oneshot::Canceled),
}

pub(crate) struct WebGpuBuffer(
    pub(crate) super::Instance,
    pub(crate) wgpu_core::id::BufferId,
);
impl Resource for WebGpuBuffer {
    fn name(&self) -> Cow<str> {
        "webGPUBuffer".into()
    }

    fn close(self: Rc<Self>) {
        self.0.buffer_drop(self.1);
    }
}

struct WebGpuBufferMapped(NonNull<u8>, usize);
impl Resource for WebGpuBufferMapped {
    fn name(&self) -> Cow<str> {
        "webGPUBufferMapped".into()
    }
}

#[op2]
#[serde]
pub fn op_webgpu_create_buffer(
    state: &mut OpState,
    #[smi] device_rid: ResourceId,
    #[string] label: Cow<str>,
    #[number] size: u64,
    usage: u32,
    mapped_at_creation: bool,
) -> Result<WebGpuResult, BufferError> {
    let instance = state.borrow::<super::Instance>();
    let device_resource = state
        .resource_table
        .get::<super::WebGpuDevice>(device_rid)
        .map_err(BufferError::Resource)?;
    let device = device_resource.1;

    let descriptor = wgpu_core::resource::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu_types::BufferUsages::from_bits(usage).ok_or(BufferError::InvalidUsage)?,
        mapped_at_creation,
    };

    gfx_put!(instance.device_create_buffer(
    device,
    &descriptor,
    None
  ) => state, WebGpuBuffer)
}

#[op2(async)]
#[serde]
pub async fn op_webgpu_buffer_get_map_async(
    state: Rc<RefCell<OpState>>,
    #[smi] buffer_rid: ResourceId,
    #[smi] device_rid: ResourceId,
    mode: u32,
    #[number] offset: u64,
    #[number] size: u64,
) -> Result<WebGpuResult, BufferError> {
    let (sender, receiver) = oneshot::channel::<BufferAccessResult>();

    let device;
    {
        let state_ = state.borrow();
        let instance = state_.borrow::<super::Instance>();
        let buffer_resource = state_
            .resource_table
            .get::<WebGpuBuffer>(buffer_rid)
            .map_err(BufferError::Resource)?;
        let buffer = buffer_resource.1;
        let device_resource = state_
            .resource_table
            .get::<super::WebGpuDevice>(device_rid)
            .map_err(BufferError::Resource)?;
        device = device_resource.1;

        let callback = Box::new(move |status| {
            sender.send(status).unwrap();
        });

        // TODO(lucacasonato): error handling
        let maybe_err = instance
            .buffer_map_async(
                buffer,
                offset,
                Some(size),
                wgpu_core::resource::BufferMapOperation {
                    host: match mode {
                        1 => wgpu_core::device::HostMap::Read,
                        2 => wgpu_core::device::HostMap::Write,
                        _ => unreachable!(),
                    },
                    callback: Some(callback),
                },
            )
            .err();

        if maybe_err.is_some() {
            return Ok(WebGpuResult::maybe_err(maybe_err));
        }
    }

    let done = Rc::new(RefCell::new(false));
    let done_ = done.clone();
    let device_poll_fut = async move {
        while !*done.borrow() {
            {
                let state = state.borrow();
                let instance = state.borrow::<super::Instance>();
                instance
                    .device_poll(device, wgpu_types::Maintain::wait())
                    .unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        Ok::<(), BufferError>(())
    };

    let receiver_fut = async move {
        receiver.await??;
        let mut done = done_.borrow_mut();
        *done = true;
        Ok::<(), BufferError>(())
    };

    tokio::try_join!(device_poll_fut, receiver_fut)?;

    Ok(WebGpuResult::empty())
}

#[op2]
#[serde]
pub fn op_webgpu_buffer_get_mapped_range(
    state: &mut OpState,
    #[smi] buffer_rid: ResourceId,
    #[number] offset: u64,
    #[number] size: Option<u64>,
    #[buffer] buf: &mut [u8],
) -> Result<WebGpuResult, BufferError> {
    let instance = state.borrow::<super::Instance>();
    let buffer_resource = state
        .resource_table
        .get::<WebGpuBuffer>(buffer_rid)
        .map_err(BufferError::Resource)?;
    let buffer = buffer_resource.1;

    let (slice_pointer, range_size) = instance
        .buffer_get_mapped_range(buffer, offset, size)
        .map_err(BufferError::Access)?;

    // SAFETY: guarantee to be safe from wgpu
    let slice =
        unsafe { std::slice::from_raw_parts_mut(slice_pointer.as_ptr(), range_size as usize) };
    buf.copy_from_slice(slice);

    let rid = state
        .resource_table
        .add(WebGpuBufferMapped(slice_pointer, range_size as usize));

    Ok(WebGpuResult::rid(rid))
}

#[op2]
#[serde]
pub fn op_webgpu_buffer_unmap(
    state: &mut OpState,
    #[smi] buffer_rid: ResourceId,
    #[smi] mapped_rid: ResourceId,
    #[buffer] buf: Option<&[u8]>,
) -> Result<WebGpuResult, BufferError> {
    let mapped_resource = state
        .resource_table
        .take::<WebGpuBufferMapped>(mapped_rid)
        .map_err(BufferError::Resource)?;
    let instance = state.borrow::<super::Instance>();
    let buffer_resource = state
        .resource_table
        .get::<WebGpuBuffer>(buffer_rid)
        .map_err(BufferError::Resource)?;
    let buffer = buffer_resource.1;

    if let Some(buf) = buf {
        // SAFETY: guarantee to be safe from wgpu
        let slice = unsafe {
            std::slice::from_raw_parts_mut(mapped_resource.0.as_ptr(), mapped_resource.1)
        };
        slice.copy_from_slice(buf);
    }

    gfx_ok!(instance.buffer_unmap(buffer))
}
