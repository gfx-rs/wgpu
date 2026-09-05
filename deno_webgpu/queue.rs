// Copyright 2018-2025 the Deno authors. MIT license.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use deno_core::cppgc::Ptr;
use deno_core::futures::channel::oneshot;
use deno_core::op2;
use deno_core::v8;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use deno_error::JsErrorBox;

use crate::buffer::GPUBuffer;
use crate::command_buffer::GPUCommandBuffer;
use crate::error::GPUGenericError;
use crate::get_data_slice;
use crate::texture::GPUTexture;
use crate::texture::GPUTextureAspect;
use crate::webidl::GPUExtent3D;
use crate::webidl::GPUOrigin3D;

pub struct GPUQueue {
  pub label: String,

  pub wgpu_queue: Arc<wgpu_core::device::queue::Queue>,
  pub wgpu_device: Arc<wgpu_core::device::Device>,
}

impl GarbageCollected for GPUQueue {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUQueue"
  }
}

#[op2]
impl GPUQueue {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUQueue, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn label(&self) -> String {
    self.label.clone()
  }
  #[setter]
  #[string]
  fn label(&self, #[webidl] _label: String) {
    // TODO(@crowlKats): no-op, needs wpgu to implement changing the label
  }

  #[reentrant]
  #[required(1)]
  #[undefined]
  fn submit(
    &self,
    #[webidl] command_buffers: Vec<Ptr<GPUCommandBuffer>>,
  ) -> Result<(), JsErrorBox> {
    let ids = command_buffers
      .into_iter()
      .map(|cb| cb.wgpu_command_buffer.clone())
      .collect::<Vec<_>>();

    self.wgpu_queue.submit(&ids);

    Ok(())
  }

  // In the successful case, the promise should resolve to undefined, but
  // `#[undefined]` does not seem to work here.
  // https://github.com/denoland/deno/issues/29603
  #[async_method]
  async fn on_submitted_work_done(&self) -> Result<(), JsErrorBox> {
    let (sender, receiver) = oneshot::channel::<()>();

    let callback = Box::new(move || {
      sender.send(()).unwrap();
    });

    self.wgpu_queue.on_submitted_work_done(callback);

    let done = Rc::new(RefCell::new(false));
    let done_ = done.clone();
    let device_poll_fut = async move {
      while !*done.borrow() {
        {
          self
            .wgpu_device
            .poll(wgpu_types::PollType::wait_indefinitely())
            .unwrap();
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
      }
      Ok::<(), JsErrorBox>(())
    };

    let receiver_fut = async move {
      receiver
        .await
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;
      let mut done = done_.borrow_mut();
      *done = true;
      Ok::<(), JsErrorBox>(())
    };

    tokio::try_join!(device_poll_fut, receiver_fut)?;

    Ok(())
  }

  #[required(3)]
  #[undefined]
  fn write_buffer<'a>(
    &self,
    scope: &mut v8::HandleScope<'a>,
    #[webidl] buffer: Ptr<GPUBuffer>,
    #[webidl(options(enforce_range = true))] buffer_offset: u64,
    data_arg: v8::Local<'a, v8::Value>,
    #[webidl(default = 0, options(enforce_range = true))] data_offset: u64,
    #[webidl(options(enforce_range = true))] size: Option<u64>,
  ) -> Result<(), JsErrorBox> {
    let data = get_data_slice(scope, data_arg, data_offset, size)?;

    self.wgpu_queue.write_buffer(
      buffer.wgpu_buffer.clone(),
      buffer_offset,
      data,
    );

    Ok(())
  }

  #[reentrant]
  #[required(4)]
  #[undefined]
  fn write_texture(
    &self,
    #[webidl] destination: GPUTexelCopyTextureInfo,
    #[anybuffer] buf: &[u8],
    #[webidl] data_layout: GPUTexelCopyBufferLayout,
    #[webidl] size: GPUExtent3D,
  ) {
    let destination = wgpu_types::TexelCopyTextureInfo {
      texture: destination.texture.wgpu_texture.clone(),
      mip_level: destination.mip_level,
      origin: destination.origin.into(),
      aspect: destination.aspect.into(),
    };

    let data_layout = wgpu_types::TexelCopyBufferLayout {
      offset: data_layout.offset,
      bytes_per_row: data_layout.bytes_per_row,
      rows_per_image: data_layout.rows_per_image,
    };

    self
      .wgpu_queue
      .write_texture(destination, buf, &data_layout, &size.into());
  }
}

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUTexelCopyTextureInfo {
  pub texture: Ptr<GPUTexture>,
  #[webidl(default = 0)]
  #[options(enforce_range = true)]
  pub mip_level: u32,
  #[webidl(default = Default::default())]
  pub origin: GPUOrigin3D,
  #[webidl(default = GPUTextureAspect::All)]
  pub aspect: GPUTextureAspect,
}

#[derive(WebIDL)]
#[webidl(dictionary)]
struct GPUTexelCopyBufferLayout {
  #[webidl(default = 0)]
  #[options(enforce_range = true)]
  offset: u64,
  #[options(enforce_range = true)]
  bytes_per_row: Option<u32>,
  #[options(enforce_range = true)]
  rows_per_image: Option<u32>,
}
