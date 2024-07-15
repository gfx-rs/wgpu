use wgt::WasmNotSendSync;

use crate::{BufferBinding, CommandEncoder, Device};

// TODO: docs
pub trait DynResource: WasmNotSendSync + std::fmt::Debug + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

trait DynResourceExt {
    fn expect_downcast_ref<T: DynResource>(&self) -> &T;
    fn expect_downcast_mut<T: DynResource>(&mut self) -> &mut T;
}

impl<R: DynResource + ?Sized> DynResourceExt for R {
    fn expect_downcast_ref<'a, T: DynResource>(&'a self) -> &'a T {
        self.as_any()
            .downcast_ref()
            .expect("Resource doesn't have the expected backend type.")
    }

    fn expect_downcast_mut<'a, T: DynResource>(&'a mut self) -> &'a mut T {
        self.as_any_mut()
            .downcast_mut()
            .expect("Resource doesn't have the expected backend type.")
    }
}

// TODO: actually use this one
pub trait DynBuffer: DynResource {}

pub trait DynDevice {
    unsafe fn destroy_buffer(&self, buffer: Box<dyn DynBuffer>);
}

impl<D: Device> DynDevice for D {
    unsafe fn destroy_buffer(&self, mut buffer: Box<dyn DynBuffer>) {
        // Ideally, we'd cast the box and then unbox it with `Box::into_inner`.
        // Unfortunately, the latter is only available on nightly Rust.
        //
        // Another better alternative would be for `D::destroy_buffer` to take a `Box<D::A::Buffer>`.
        // However, that would require casting the box first to `Box<dyn Any>` for which we need
        // super trait casting (https://rust-lang.github.io/rfcs/3324-dyn-upcasting.html)
        // which as of writing is still being stabilized.
        let buffer = buffer.expect_downcast_mut();
        unsafe { self.destroy_buffer(buffer) };
    }
}

pub trait DynCommandEncoder {
    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: BufferBinding<'a, dyn DynBuffer>,
        format: wgt::IndexFormat,
    );

    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: BufferBinding<'a, dyn DynBuffer>,
    );
}

impl<C: CommandEncoder> DynCommandEncoder for C {
    unsafe fn set_index_buffer<'a>(
        &mut self,
        binding: BufferBinding<'a, dyn DynBuffer>,
        format: wgt::IndexFormat,
    ) {
        let binding = binding.expect_downcast();
        unsafe { self.set_index_buffer(binding, format) };
    }

    unsafe fn set_vertex_buffer<'a>(
        &mut self,
        index: u32,
        binding: BufferBinding<'a, dyn DynBuffer>,
    ) {
        let binding = binding.expect_downcast();
        unsafe { self.set_vertex_buffer(index, binding) };
    }
}

impl<'a> BufferBinding<'a, dyn DynBuffer> {
    pub fn expect_downcast<B: DynBuffer>(self) -> BufferBinding<'a, B> {
        BufferBinding {
            buffer: self.buffer.expect_downcast_ref(),
            offset: self.offset,
            size: self.size,
        }
    }
}
