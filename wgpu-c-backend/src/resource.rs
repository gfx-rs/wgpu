use std::ptr::NonNull;

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

c_resource!(CBuffer, native::WGPUBuffer, wgpuBufferRelease);

impl BufferInterface for CBuffer {
    fn map_async(
        &self,
        mode: wgpu::MapMode,
        range: std::ops::Range<wgpu::BufferAddress>,
        callback: BufferMapCallback,
    ) {
        struct Out {
            callback: Option<BufferMapCallback>,
        }

        unsafe extern "C" fn cb(
            status: native::WGPUMapAsyncStatus,
            _message: native::WGPUStringView,
            userdata1: *mut std::ffi::c_void,
            _userdata2: *mut std::ffi::c_void,
        ) {
            let out = &mut *(userdata1 as *mut Out);
            let result = match status {
                native::WGPUMapAsyncStatus_Success => Ok(()),
                _ => Err(wgpu::BufferAsyncError),
            };
            if let Some(cb) = out.callback.take() {
                cb(result);
            }
        }

        let c_mode = match mode {
            wgpu::MapMode::Read => native::WGPUMapMode_Read,
            wgpu::MapMode::Write => native::WGPUMapMode_Write,
        };

        let mut out = Out {
            callback: Some(callback),
        };
        let callback_info = native::WGPUBufferMapCallbackInfo {
            nextInChain: std::ptr::null_mut(),
            mode: native::WGPUCallbackMode_AllowSpontaneous,
            callback: Some(cb),
            userdata1: std::ptr::addr_of_mut!(out).cast(),
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
        // wgpu-native fires the callback synchronously; `out.callback` is consumed by now.
    }

    fn get_mapped_range(
        &self,
        sub_range: std::ops::Range<wgpu::BufferAddress>,
    ) -> Result<DispatchBufferMappedRange, wgpu::MapRangeError> {
        let offset = sub_range.start as usize;
        let size = (sub_range.end - sub_range.start) as usize;

        let ptr = unsafe { wgpuBufferGetMappedRange(self.ptr, offset, size) };
        let (ptr, _is_const): (*mut u8, bool) = if ptr.is_null() {
            let cp = unsafe { wgpuBufferGetConstMappedRange(self.ptr, offset, size) };
            (cp as *mut u8, true)
        } else {
            (ptr, false)
        };

        if ptr.is_null() {
            panic!("wgpu-native: buffer mapped range pointer is null");
        }

        Ok(DispatchBufferMappedRange::custom(CBufferMappedRange {
            ptr: ptr.cast::<u8>(),
            len: size,
        }))
    }

    fn unmap(&self) {
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
                .unwrap_or(native::WGPUTextureUsage_None),
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
        // wgpu-native does not implement wgpuShaderModuleGetCompilationInfo.
        unimplemented!("wgpu-native does not implement wgpuShaderModuleGetCompilationInfo")
    }
}

// ── CBindGroupLayout ──────────────────────────────────────────────────────────

c_resource!(
    CBindGroupLayout,
    native::WGPUBindGroupLayout,
    wgpuBindGroupLayoutRelease
);

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
        DispatchBindGroupLayout::custom(CBindGroupLayout { ptr })
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
        DispatchBindGroupLayout::custom(CBindGroupLayout { ptr })
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

c_resource!(
    CCommandBuffer,
    native::WGPUCommandBuffer,
    wgpuCommandBufferRelease
);

impl CommandBufferInterface for CCommandBuffer {}

// ── CRenderBundle ─────────────────────────────────────────────────────────────

c_resource!(
    CRenderBundle,
    native::WGPURenderBundle,
    wgpuRenderBundleRelease
);

impl RenderBundleInterface for CRenderBundle {}
