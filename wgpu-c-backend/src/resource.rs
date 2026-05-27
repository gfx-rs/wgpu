use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use wgpu::custom::*;
use wgpu_native::{native, *};

use crate::conv;

// ── Macro for simple opaque-pointer resources ────────────────────────────────

macro_rules! c_resource {
    ($name:ident, $ptr_ty:ty, $release:ident) => {
        pub struct $name {
            pub(crate) ptr: $ptr_ty,
        }
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($name))
                    .field("ptr", &self.ptr)
                    .finish()
            }
        }
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
        impl Drop for $name {
            fn drop(&mut self) {
                unsafe { $release(self.ptr) };
            }
        }
    };
}

// ── CBuffer ───────────────────────────────────────────────────────────────────

pub struct CBuffer {
    pub(crate) ptr: native::WGPUBuffer,
    // Tracks whether the buffer is currently mapped. Set to true by a
    // successful map_async callback; reset to false by unmap(). Prevents
    // calling wgpuBufferGetMappedRange on an unmapped buffer, which would
    // cause handle_error_fatal to panic inside extern "C" → SIGSEGV.
    is_mapped: Arc<AtomicBool>,
}
impl std::fmt::Debug for CBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CBuffer")
            .field("ptr", &self.ptr)
            .field("is_mapped", &self.is_mapped.load(Ordering::Relaxed))
            .finish()
    }
}
unsafe impl Send for CBuffer {}
unsafe impl Sync for CBuffer {}
impl Drop for CBuffer {
    fn drop(&mut self) {
        unsafe { wgpuBufferRelease(self.ptr) };
    }
}

impl CBuffer {
    pub(crate) fn new(ptr: native::WGPUBuffer) -> Self {
        CBuffer {
            ptr,
            is_mapped: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn new_mapped_at_creation(ptr: native::WGPUBuffer) -> Self {
        CBuffer {
            ptr,
            is_mapped: Arc::new(AtomicBool::new(true)),
        }
    }
}

impl BufferInterface for CBuffer {
    fn map_async(
        &self,
        mode: wgpu::MapMode,
        range: std::ops::Range<wgpu::BufferAddress>,
        callback: BufferMapCallback,
    ) {
        struct Out {
            callback: Option<BufferMapCallback>,
            is_mapped: Arc<AtomicBool>,
        }

        unsafe extern "C" fn cb(
            status: native::WGPUMapAsyncStatus,
            _message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = unsafe { Box::from_raw(userdata1 as *mut Out) };
            let result = match status {
                native::WGPUMapAsyncStatus_Success => {
                    out.is_mapped.store(true, Ordering::Release);
                    Ok(())
                }
                _ => Err(wgpu::BufferAsyncError),
            };
            if let Some(callback) = out.callback {
                crate::catch_callback_panic(|| callback(result));
            }
        }

        let c_mode = match mode {
            wgpu::MapMode::Read => native::WGPUMapMode_Read,
            wgpu::MapMode::Write => native::WGPUMapMode_Write,
        };

        let out = Box::new(Out {
            callback: Some(callback),
            is_mapped: Arc::clone(&self.is_mapped),
        });
        let callback_info = native::WGPUBufferMapCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(cb),
            userdata1: Box::into_raw(out).cast(),
            userdata2: std::ptr::null_mut(),
        };

        unsafe {
            wgpuBufferMapAsync(
                self.ptr,
                c_mode,
                range.start as usize,
                (range.end - range.start) as usize,
                callback_info,
            )
        };
        // Re-raise any panic that occurred if the callback fired synchronously.
        crate::resume_callback_panic();
    }

    fn get_mapped_range(
        &self,
        sub_range: std::ops::Range<wgpu::BufferAddress>,
    ) -> Result<DispatchBufferMappedRange, wgpu::MapRangeError> {
        let offset = sub_range.start as usize;
        let size = (sub_range.end - sub_range.start) as usize;

        // Guard against calling wgpuBufferGetMappedRange on an unmapped buffer:
        // that function calls handle_error_fatal which panics inside extern "C",
        // causing UB / SIGSEGV. Panic in Rust instead.
        if !self.is_mapped.load(Ordering::Acquire) {
            panic!("get_mapped_range called on unmapped buffer");
        }

        let ptr = unsafe { wgpuBufferGetMappedRange(self.ptr, offset, size) };
        let (ptr, read_only) = if ptr.is_null() {
            // wgpuBufferGetMappedRange returns null for MapMode::Read buffers.
            // Fall back to the const variant; write_slice will panic if called.
            let cp = unsafe { wgpuBufferGetConstMappedRange(self.ptr, offset, size) };
            if cp.is_null() {
                panic!("wgpu-native: buffer mapped range pointer is null");
            }
            (cp as *mut u8, true)
        } else {
            (ptr, false)
        };

        Ok(DispatchBufferMappedRange::custom(CBufferMappedRange {
            ptr: ptr.cast::<u8>(),
            len: size,
            read_only,
        }))
    }

    fn unmap(&self) {
        self.is_mapped.store(false, Ordering::Release);
        unsafe { wgpuBufferUnmap(self.ptr) };
    }

    fn destroy(&self) {
        unsafe { wgpuBufferDestroy(self.ptr) };
    }
}

// ── CBufferMappedRange ────────────────────────────────────────────────────────

pub struct CBufferMappedRange {
    pub(crate) ptr: *mut u8,
    pub(crate) len: usize,
    // True when the buffer was mapped MapMode::Read: wgpuBufferGetMappedRange returns
    // null for read-only mappings, so ptr comes from wgpuBufferGetConstMappedRange.
    // write_slice on a read-only pointer is UB; we panic here as a dispatch-layer guard
    // (the wgpu public API should have prevented this via MapMode checks already).
    read_only: bool,
}
impl std::fmt::Debug for CBufferMappedRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CBufferMappedRange")
            .field("len", &self.len)
            .finish()
    }
}
unsafe impl Send for CBufferMappedRange {}
unsafe impl Sync for CBufferMappedRange {}

impl BufferMappedRangeInterface for CBufferMappedRange {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn read_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        if self.read_only {
            panic!("write_slice called on a read-only (MapMode::Read) buffer mapped range");
        }
        let nn =
            unsafe { NonNull::slice_from_raw_parts(NonNull::new_unchecked(self.ptr), self.len) };
        unsafe { wgpu::WriteOnly::new(nn) }
    }
}

// ── CTexture ──────────────────────────────────────────────────────────────────

c_resource!(CTexture, native::WGPUTexture, wgpuTextureRelease);

impl TextureInterface for CTexture {
    fn create_view(&self, desc: &wgpu::TextureViewDescriptor<'_>) -> DispatchTextureView {
        let label = desc.label.map(|s| s.to_owned());
        let label_sv = label
            .as_deref()
            .map(conv::str_to_string_view)
            .unwrap_or(conv::null_string_view());
        let c_desc = native::WGPUTextureViewDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: label_sv,
            format: desc
                .format
                .map(conv::texture_format_to_native)
                .unwrap_or(native::WGPUTextureFormat_Undefined),
            dimension: desc
                .dimension
                .map(conv::texture_view_dimension_to_native)
                .unwrap_or(native::WGPUTextureViewDimension_Undefined),
            baseMipLevel: desc.base_mip_level,
            mipLevelCount: desc
                .mip_level_count
                .unwrap_or(native::WGPU_MIP_LEVEL_COUNT_UNDEFINED),
            baseArrayLayer: desc.base_array_layer,
            arrayLayerCount: desc
                .array_layer_count
                .unwrap_or(native::WGPU_ARRAY_LAYER_COUNT_UNDEFINED),
            aspect: conv::texture_aspect_to_native(desc.aspect),
            usage: desc
                .usage
                .map(conv::texture_usage_to_native)
                .unwrap_or_else(|| unsafe { wgpuTextureGetUsage(self.ptr) }),
        };
        let ptr = unsafe { wgpuTextureCreateView(self.ptr, Some(&c_desc)) };
        DispatchTextureView::custom(CTextureView { ptr })
    }

    fn destroy(&self) {
        unsafe { wgpuTextureDestroy(self.ptr) };
    }
}

// ── CTextureView ──────────────────────────────────────────────────────────────

c_resource!(
    CTextureView,
    native::WGPUTextureView,
    wgpuTextureViewRelease
);

impl TextureViewInterface for CTextureView {}

// ── CSampler ──────────────────────────────────────────────────────────────────

c_resource!(CSampler, native::WGPUSampler, wgpuSamplerRelease);

impl SamplerInterface for CSampler {}

// ── CShaderModule ─────────────────────────────────────────────────────────────

c_resource!(
    CShaderModule,
    native::WGPUShaderModule,
    wgpuShaderModuleRelease
);

impl ShaderModuleInterface for CShaderModule {
    fn get_compilation_info(&self) -> std::pin::Pin<Box<dyn ShaderCompilationInfoFuture>> {
        struct Out {
            messages: Vec<wgpu::CompilationMessage>,
        }

        unsafe extern "C" fn callback(
            _status: native::WGPUCompilationInfoRequestStatus,
            info: *const native::WGPUCompilationInfo,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = &mut *(userdata1 as *mut Out);
            if info.is_null() {
                return;
            }
            let info = &*info;
            let messages_slice = if info.messageCount == 0 || info.messages.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(info.messages, info.messageCount)
            };
            out.messages = messages_slice
                .iter()
                .map(|m| {
                    let message_type = match m.type_ {
                        native::WGPUCompilationMessageType_Warning => {
                            wgpu::CompilationMessageType::Warning
                        }
                        native::WGPUCompilationMessageType_Info => {
                            wgpu::CompilationMessageType::Info
                        }
                        _ => wgpu::CompilationMessageType::Error,
                    };
                    let location = if m.lineNum > 0 {
                        Some(wgpu::SourceLocation {
                            line_number: m.lineNum as u32,
                            line_position: m.linePos as u32,
                            offset: m.offset as u32,
                            length: m.length as u32,
                        })
                    } else {
                        None
                    };
                    wgpu::CompilationMessage {
                        message: crate::conv::string_view_to_string(m.message),
                        message_type,
                        location,
                    }
                })
                .collect();
        }

        let mut out = Out { messages: vec![] };
        let callback_info = native::WGPUCompilationInfoCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(callback),
            userdata1: std::ptr::addr_of_mut!(out).cast(),
            userdata2: std::ptr::null_mut(),
        };
        unsafe { wgpuShaderModuleGetCompilationInfo(self.ptr, callback_info) };
        Box::pin(std::future::ready(wgpu::CompilationInfo {
            messages: out.messages,
        }))
    }
}

// ── CBindGroupLayout ──────────────────────────────────────────────────────────

pub struct CBindGroupLayout {
    pub(crate) ptr: native::WGPUBindGroupLayout,
    pub(crate) device_ptr: native::WGPUDevice,
}
impl std::fmt::Debug for CBindGroupLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CBindGroupLayout")
            .field("ptr", &self.ptr)
            .finish()
    }
}
unsafe impl Send for CBindGroupLayout {}
unsafe impl Sync for CBindGroupLayout {}
impl Drop for CBindGroupLayout {
    fn drop(&mut self) {
        unsafe { wgpuBindGroupLayoutRelease(self.ptr) };
    }
}

impl BindGroupLayoutInterface for CBindGroupLayout {}

// ── CBindGroup ────────────────────────────────────────────────────────────────

c_resource!(CBindGroup, native::WGPUBindGroup, wgpuBindGroupRelease);

impl BindGroupInterface for CBindGroup {}

// ── CPipelineLayout ───────────────────────────────────────────────────────────

c_resource!(
    CPipelineLayout,
    native::WGPUPipelineLayout,
    wgpuPipelineLayoutRelease
);

impl PipelineLayoutInterface for CPipelineLayout {}

// ── CRenderPipeline ───────────────────────────────────────────────────────────

c_resource!(
    CRenderPipeline,
    native::WGPURenderPipeline,
    wgpuRenderPipelineRelease
);

impl RenderPipelineInterface for CRenderPipeline {
    fn get_bind_group_layout(&self, index: u32) -> DispatchBindGroupLayout {
        let ptr = unsafe { wgpuRenderPipelineGetBindGroupLayout(self.ptr, index) };
        DispatchBindGroupLayout::custom(CBindGroupLayout {
            ptr,
            device_ptr: std::ptr::null_mut(),
        })
    }
}

// ── CComputePipeline ──────────────────────────────────────────────────────────

c_resource!(
    CComputePipeline,
    native::WGPUComputePipeline,
    wgpuComputePipelineRelease
);

impl ComputePipelineInterface for CComputePipeline {
    fn get_bind_group_layout(&self, index: u32) -> DispatchBindGroupLayout {
        let ptr = unsafe { wgpuComputePipelineGetBindGroupLayout(self.ptr, index) };
        DispatchBindGroupLayout::custom(CBindGroupLayout {
            ptr,
            device_ptr: std::ptr::null_mut(),
        })
    }
}

// ── CPipelineCache ────────────────────────────────────────────────────────────

c_resource!(
    CPipelineCache,
    native::WGPUPipelineCache,
    wgpuPipelineCacheRelease
);

impl PipelineCacheInterface for CPipelineCache {
    fn get_data(&self) -> Option<Vec<u8>> {
        let size = unsafe { wgpuPipelineCacheGetData(self.ptr, std::ptr::null_mut()) };
        if size == 0 {
            return None;
        }
        let mut buf = vec![0u8; size];
        unsafe { wgpuPipelineCacheGetData(self.ptr, buf.as_mut_ptr()) };
        Some(buf)
    }
}

// ── CQuerySet ─────────────────────────────────────────────────────────────────

c_resource!(CQuerySet, native::WGPUQuerySet, wgpuQuerySetRelease);

impl QuerySetInterface for CQuerySet {}

// ── CCommandBuffer ────────────────────────────────────────────────────────────

pub struct CCommandBuffer {
    pub(crate) ptr: native::WGPUCommandBuffer,
    /// Device that created this command buffer. Used to detect cross-device submission.
    pub(crate) device_ptr: native::WGPUDevice,
}
impl std::fmt::Debug for CCommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CCommandBuffer")
            .field("ptr", &self.ptr)
            .finish()
    }
}
unsafe impl Send for CCommandBuffer {}
unsafe impl Sync for CCommandBuffer {}
impl Drop for CCommandBuffer {
    fn drop(&mut self) {
        unsafe { wgpuCommandBufferRelease(self.ptr) };
    }
}

impl CommandBufferInterface for CCommandBuffer {}

// ── CRenderBundle ─────────────────────────────────────────────────────────────

c_resource!(
    CRenderBundle,
    native::WGPURenderBundle,
    wgpuRenderBundleRelease
);

impl RenderBundleInterface for CRenderBundle {}

// ── CBlas ─────────────────────────────────────────────────────────────────────

pub struct CBlas {
    pub(crate) ptr: native::WGPUBlas,
}
impl std::fmt::Debug for CBlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CBlas").field("ptr", &self.ptr).finish()
    }
}
unsafe impl Send for CBlas {}
unsafe impl Sync for CBlas {}
impl Drop for CBlas {
    fn drop(&mut self) {
        unsafe { wgpuBlasRelease(self.ptr) };
    }
}

impl BlasInterface for CBlas {
    fn prepare_compact_async(&self, callback: BlasCompactCallback) {
        unsafe extern "C" fn compact_cb(
            success: native::WGPUBool,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let cb = unsafe { *Box::from_raw(userdata1 as *mut BlasCompactCallback) };
            if success != 0 {
                cb(Ok(()));
            } else {
                cb(Err(wgpu::BlasAsyncError));
            }
        }
        let boxed: Box<BlasCompactCallback> = Box::new(callback);
        let callback_info = native::WGPUBlasCompactCallbackInfo {
            nextInChain: std::ptr::null(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(compact_cb),
            userdata1: Box::into_raw(boxed) as *mut _,
            userdata2: std::ptr::null_mut(),
        };
        unsafe { wgpuBlasPrepareCompactAsync(self.ptr, callback_info) };
    }

    fn ready_for_compaction(&self) -> bool {
        unsafe { wgpuBlasReadyForCompaction(self.ptr) != 0 }
    }
}

// ── CTlas ─────────────────────────────────────────────────────────────────────

c_resource!(CTlas, native::WGPUTlas, wgpuTlasRelease);

impl TlasInterface for CTlas {}

// ── CExternalTexture ──────────────────────────────────────────────────────────

c_resource!(
    CExternalTexture,
    native::WGPUExternalTexture,
    wgpuExternalTextureRelease
);

impl ExternalTextureInterface for CExternalTexture {
    fn destroy(&self) {
        // wgpu-native does not expose wgpuExternalTextureDestroy; the underlying
        // resource is released when CExternalTexture is dropped via wgpuExternalTextureRelease.
    }
}

// ── CQueueWriteBuffer ─────────────────────────────────────────────────────────
//
// wgpu-native has no GPU staging buffer API, so we use a CPU Vec that is
// flushed to the GPU via wgpuQueueWriteBuffer in write_staging_buffer.
// This lets Queue::write_buffer_with work correctly.

pub struct CQueueWriteBuffer {
    pub(crate) data: Vec<u8>,
}

impl std::fmt::Debug for CQueueWriteBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CQueueWriteBuffer")
            .field("len", &self.data.len())
            .finish()
    }
}

unsafe impl Send for CQueueWriteBuffer {}
unsafe impl Sync for CQueueWriteBuffer {}

impl QueueWriteBufferInterface for CQueueWriteBuffer {
    fn len(&self) -> usize {
        self.data.len()
    }

    unsafe fn write_slice(&mut self) -> wgpu::WriteOnly<'_, [u8]> {
        let nn = unsafe {
            NonNull::slice_from_raw_parts(
                NonNull::new_unchecked(self.data.as_mut_ptr()),
                self.data.len(),
            )
        };
        unsafe { wgpu::WriteOnly::new(nn) }
    }
}
