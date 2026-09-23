// Copyright 2018-2025 the Deno authors. MIT license.

#![allow(clippy::nonminimal_bool)]

use std::cell::RefCell;
use std::fmt::Display;
use std::ops::Range;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;

use deno_core::futures::channel::oneshot;
use deno_core::op2;
use deno_core::v8;
use deno_core::webidl::WebIdlInterfaceConverter;
use deno_core::GarbageCollected;
use deno_core::WebIDL;
use deno_error::JsErrorBox;
use wgpu_core::device::HostMap as MapMode;
use wgpu_core::resource::Labeled as _;

use crate::error::GPUGenericError;

#[derive(WebIDL)]
#[webidl(dictionary)]
pub(crate) struct GPUBufferDescriptor {
  #[webidl(default = String::new())]
  pub label: String,

  pub size: u64,
  #[options(enforce_range = true)]
  pub usage: u32,
  #[webidl(default = false)]
  pub mapped_at_creation: bool,
}

#[derive(Debug, thiserror::Error, deno_error::JsError)]
pub enum BufferError {
  #[class(generic)]
  #[error(transparent)]
  Canceled(#[from] oneshot::Canceled),
  #[class("DOMExceptionOperationError")]
  #[error(transparent)]
  Access(wgpu_core::resource::BufferAccessError),
  #[class("DOMExceptionAbortError")]
  #[error("{0}")]
  Aborted(&'static str),
  #[class("DOMExceptionOperationError")]
  #[error("{0}")]
  Operation(&'static str),
  #[class(inherit)]
  #[error(transparent)]
  Other(#[from] JsErrorBox),
}

impl From<wgpu_core::resource::BufferAccessError> for BufferError {
  fn from(err: wgpu_core::resource::BufferAccessError) -> Self {
    match err {
      wgpu_core::resource::BufferAccessError::Device(
        wgpu_core::device::DeviceError::Lost,
      ) => BufferError::Aborted("Device lost"),
      err => BufferError::Access(err),
    }
  }
}

pub(crate) struct View {
  range: Range<u64>,
  array_buffer: v8::Global<v8::ArrayBuffer>,
}

/// Returns true if two non-inclusive ranges overlap
// https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-if-two-ranges-overlap
fn range_overlap<T: std::cmp::PartialOrd>(
  range1: &Range<T>,
  range2: &Range<T>,
) -> bool {
  range1.start < range2.end && range2.start < range1.end
}

pub(crate) enum BufferMapState {
  Unmapped,
  Pending,
  Mapped {
    mode: MapMode,
    range: Range<u64>,
    views: Vec<View>,
  },
}

impl Display for BufferMapState {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      BufferMapState::Unmapped => write!(f, "unmapped"),
      BufferMapState::Pending => write!(f, "pending"),
      BufferMapState::Mapped { .. } => write!(f, "mapped"),
    }
  }
}

pub struct GPUBuffer {
  pub wgpu_buffer: Arc<wgpu_core::resource::Buffer>,
  pub wgpu_device: Arc<wgpu_core::device::Device>,

  pub usage: u32,

  pub map_state: RefCell<BufferMapState>,
  /// None if buffer is not mappable or OOM while creating the buffer
  pub data: RefCell<Option<Vec<u8>>>,
}

impl WebIdlInterfaceConverter for GPUBuffer {
  const NAME: &'static str = "GPUBuffer";
}

impl GarbageCollected for GPUBuffer {
  fn get_name(&self) -> &'static std::ffi::CStr {
    c"GPUBuffer"
  }
}

#[op2]
impl GPUBuffer {
  #[constructor]
  #[cppgc]
  fn constructor(_: bool) -> Result<GPUBuffer, GPUGenericError> {
    Err(GPUGenericError::InvalidConstructor)
  }

  #[getter]
  #[string]
  fn label(&self) -> String {
    self.wgpu_buffer.label().to_string()
  }
  #[setter]
  #[string]
  fn label(&self, #[webidl] _label: String) {
    // TODO(@crowlKats): no-op, needs wpgu to implement changing the label
  }

  #[getter]
  #[number]
  fn size(&self) -> u64 {
    self.wgpu_buffer.size()
  }
  #[getter]
  fn usage(&self) -> u32 {
    self.usage
  }

  #[getter]
  #[string]
  fn map_state(&self) -> String {
    self.map_state.borrow().to_string()
  }

  // In the successful case, the promise should resolve to undefined, but
  // `#[undefined]` does not seem to work here.
  // https://github.com/denoland/deno/issues/29603
  #[async_method]
  async fn map_async(
    &self,
    #[webidl(options(enforce_range = true))] mode: u32,
    #[webidl(default = 0)] offset: u64,
    #[webidl] size: Option<u64>,
  ) -> Result<(), BufferError> {
    let read_mode = (mode & 0x0001) == 0x0001;
    let write_mode = (mode & 0x0002) == 0x0002;
    if (read_mode && write_mode) || (!read_mode && !write_mode) {
      return Err(BufferError::Operation(
        "exactly one of READ or WRITE map mode must be set",
      ));
    }

    let mode = if read_mode {
      MapMode::Read
    } else {
      assert!(write_mode);
      MapMode::Write
    };

    {
      *self.map_state.borrow_mut() = BufferMapState::Pending;
    }

    let (sender, receiver) =
      oneshot::channel::<wgpu_core::resource::BufferAccessResult>();

    {
      let callback = Box::new(move |status| {
        sender.send(status).unwrap();
      });

      self.wgpu_buffer.map_async(
        offset,
        size,
        wgpu_core::resource::BufferMapOperation {
          host: mode,
          callback: Some(callback),
        },
      );
    }

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
      Ok::<(), BufferError>(())
    };

    let receiver_fut = async move {
      receiver.await??;
      let mut done = done_.borrow_mut();
      *done = true;
      Ok::<(), BufferError>(())
    };

    tokio::try_join!(device_poll_fut, receiver_fut)?;

    let mapping = self.wgpu_buffer.get_mapped_range(offset, size)?;

    if mode == MapMode::Read {
      let slice = mapping.read_slice();
      let mut data = self.data.borrow_mut();
      let data = data
        .as_mut()
        .ok_or(JsErrorBox::range_error("Buffer failed allocating"))?;
      data[offset as usize..(offset + slice.len() as u64) as usize]
        .clone_from_slice(slice);
    }

    self.map_state.replace(BufferMapState::Mapped {
      mode,
      range: offset..(offset + mapping.len()),
      views: vec![],
    });

    Ok(())
  }

  fn get_mapped_range<'s>(
    &self,
    scope: &mut v8::HandleScope<'s>,
    #[webidl(default = 0)] offset: u64,
    #[webidl] size: Option<u64>,
  ) -> Result<v8::Local<'s, v8::ArrayBuffer>, BufferError> {
    let size = size.unwrap_or_else(|| self.wgpu_buffer.size() - offset);
    let BufferMapState::Mapped {
      mode: _,
      range,
      views,
    } = &mut *self.map_state.borrow_mut()
    else {
      return Err(BufferError::Operation("Buffer is not mapped"));
    };

    if !(offset.is_multiple_of(8)) {
      return Err(BufferError::Operation("Offset must be a multiple of 8"));
    }

    if !(size.is_multiple_of(4)) {
      return Err(BufferError::Operation("Size must be a multiple of 4"));
    }

    if !(offset >= range.start) {
      return Err(BufferError::Operation("Offset is out of range"));
    }

    if !(offset + size <= range.end) {
      return Err(BufferError::Operation("Size is out of range"));
    }

    let range = offset..(offset + size);

    if views.iter().any(|view| range_overlap(&view.range, &range)) {
      return Err(BufferError::Operation(
        "Overlapping mapped ranges are not allowed",
      ));
    }

    let data = self.data.borrow();
    let data = data.as_ref().expect("mapAsync succeeded");

    unsafe extern "C" fn noop_deleter_callback(
      _data: *mut std::ffi::c_void,
      _byte_length: usize,
      _deleter_data: *mut std::ffi::c_void,
    ) {
    }

    // SAFETY: creating a backing store from the pointer and length provided by wgpu
    let bs = unsafe {
      v8::ArrayBuffer::new_backing_store_from_ptr(
        data.as_ptr().add(offset as usize) as _,
        size as usize,
        noop_deleter_callback,
        std::ptr::null_mut(),
      )
    };

    let shared_bs = bs.make_shared();
    let ab = v8::ArrayBuffer::with_backing_store(scope, &shared_bs);

    views.push(View {
      range,
      array_buffer: v8::Global::new(scope, ab),
    });

    Ok(ab)
  }

  #[nofast]
  #[undefined]
  fn unmap(&self, scope: &mut v8::HandleScope) -> Result<(), BufferError> {
    if let BufferMapState::Mapped { mode, range, views } =
      self.map_state.replace(BufferMapState::Unmapped)
    {
      for ab in views {
        let ab = ab.array_buffer.open(scope);
        ab.detach(None);
      }

      if mode == MapMode::Write {
        if let Ok(mapping) = self
          .wgpu_buffer
          .get_mapped_range(range.start, Some(range.end - range.start))
        {
          let data = self.data.borrow();
          let data = data.as_ref().expect("mapAsync succeeded");
          mapping
            .write_slice()
            .copy_from_slice(&data[range.start as usize..range.end as usize]);
        }
      }
    }

    self.wgpu_buffer.unmap();

    Ok(())
  }

  #[fast]
  #[undefined]
  fn destroy(&self) {
    self.wgpu_buffer.destroy();
  }
}
