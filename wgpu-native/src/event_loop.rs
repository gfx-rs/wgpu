use crate::{
    device::BufferMapPendingCallback,
    BufferMapAsyncStatus,
    BufferMapOperation,
    instance::RequestAdapterCallback,
    id::AdapterId,
    hub::{GLOBAL, Global}
};

#[derive(Debug)]
pub struct EventLoop {
    scheduled: Vec<Event>,
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct EventLoopId(*mut EventLoop);

impl EventLoopId {
    pub(crate) fn schedule(&self, event: Event) {
        let event_loop = unsafe { self.0.as_mut().unwrap() };
        event_loop.scheduled.push(event);
    }

    fn process(&self, global: &Global) {
        let _ = global.hubs.maintain_all_devices(global, *self);
        let event_loop = unsafe { self.0.as_mut().unwrap() };
        for event in event_loop.scheduled.drain(..) {
            match event {
                Event::RequestAdapterCallback(callback, ref adapter_id, userdata) => {
                    callback(adapter_id as *const _, userdata);
                }
                Event::BufferMapPendingCallback(callback) => {
                    let (operation, result) = callback;
                    // Note: this logic is specifically moved out of `handle_mapping()` in order to
                    // have nothing locked by the time we execute users callback code.
                    let (status, ptr) = match result {
                        Ok(ptr) => (BufferMapAsyncStatus::Success, ptr),
                        Err(e) => {
                            log::error!("failed to map buffer: {:?}", e);
                            (BufferMapAsyncStatus::Error, std::ptr::null_mut())
                        }
                    };
                    match operation {
                        BufferMapOperation::Read(_, on_read, userdata) => on_read(status, ptr, userdata),
                        BufferMapOperation::Write(_, on_write, userdata) => on_write(status, ptr, userdata),
                    }
                }
            }
        }
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_create_event_loop() -> EventLoopId {
    let event_loop = EventLoop {
        scheduled: Vec::new()
    };
    EventLoopId(Box::into_raw(Box::new(event_loop)))
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_destroy_event_loop(event_loop_id: EventLoopId) {
    unsafe {
        let _ = Box::from_raw(event_loop_id.0);
    }
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_process_events(event_loop_id: EventLoopId) {
    event_loop_id.process(&*GLOBAL);
}

#[derive(Debug)]
pub(crate) enum Event {
    RequestAdapterCallback(RequestAdapterCallback, AdapterId, *mut std::ffi::c_void),
    BufferMapPendingCallback(BufferMapPendingCallback),
}
