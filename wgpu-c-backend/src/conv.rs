#![allow(dead_code)]
use wgpu_native::native;

// ── Instance ──────────────────────────────────────────────────────────────────

pub fn backends_to_native(backends: wgpu::Backends) -> native::WGPUInstanceBackend {
    let mut result: native::WGPUInstanceBackend = 0;
    if backends.contains(wgpu::Backends::BROWSER_WEBGPU) {
        result |= native::WGPUInstanceBackend_BrowserWebGPU;
    }
    if backends.contains(wgpu::Backends::VULKAN) {
        result |= native::WGPUInstanceBackend_Vulkan;
    }
    if backends.contains(wgpu::Backends::GL) {
        result |= native::WGPUInstanceBackend_GL;
    }
    if backends.contains(wgpu::Backends::METAL) {
        result |= native::WGPUInstanceBackend_Metal;
    }
    if backends.contains(wgpu::Backends::DX12) {
        result |= native::WGPUInstanceBackend_DX12;
    }
    result
}

pub fn instance_flags_to_native(flags: wgpu::InstanceFlags) -> native::WGPUInstanceFlag {
    let mut result: native::WGPUInstanceFlag = 0;
    if flags.contains(wgpu::InstanceFlags::DEBUG) {
        result |= native::WGPUInstanceFlag_Debug;
    }
    if flags.contains(wgpu::InstanceFlags::VALIDATION) {
        result |= native::WGPUInstanceFlag_Validation;
    }
    if flags.contains(wgpu::InstanceFlags::DISCARD_HAL_LABELS) {
        result |= native::WGPUInstanceFlag_DiscardHalLabels;
    }
    if flags.contains(wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER) {
        result |= native::WGPUInstanceFlag_AllowUnderlyingNoncompliantAdapter;
    }
    if flags.contains(wgpu::InstanceFlags::GPU_BASED_VALIDATION) {
        result |= native::WGPUInstanceFlag_GPUBasedValidation;
    }
    if flags.contains(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL) {
        result |= native::WGPUInstanceFlag_ValidationIndirectCall;
    }
    if flags.contains(wgpu::InstanceFlags::AUTOMATIC_TIMESTAMP_NORMALIZATION) {
        result |= native::WGPUInstanceFlag_AutomaticTimestampNormalization;
    }
    result
}

pub fn dx12_compiler_to_native(compiler: &wgpu::Dx12Compiler) -> native::WGPUDx12Compiler {
    match compiler {
        wgpu::Dx12Compiler::Fxc => native::WGPUDx12Compiler_Fxc,
        wgpu::Dx12Compiler::DynamicDxc { .. } => native::WGPUDx12Compiler_Dxc,
        wgpu::Dx12Compiler::StaticDxc => native::WGPUDx12Compiler_Dxc,
        wgpu::Dx12Compiler::Auto => native::WGPUDx12Compiler_Undefined,
    }
}

pub fn dx12_swapchain_kind_to_native(
    kind: wgpu::Dx12SwapchainKind,
) -> native::WGPUDx12SwapchainKind {
    match kind {
        wgpu::Dx12SwapchainKind::DxgiFromHwnd => native::WGPUDx12SwapchainKind_DxgiFromHwnd,
        wgpu::Dx12SwapchainKind::DxgiFromVisual => native::WGPUDx12SwapchainKind_DxgiFromVisual,
    }
}

pub fn gles3_minor_version_to_native(
    version: wgpu::Gles3MinorVersion,
) -> native::WGPUGles3MinorVersion {
    match version {
        wgpu::Gles3MinorVersion::Automatic => native::WGPUGles3MinorVersion_Automatic,
        wgpu::Gles3MinorVersion::Version0 => native::WGPUGles3MinorVersion_Version0,
        wgpu::Gles3MinorVersion::Version1 => native::WGPUGles3MinorVersion_Version1,
        wgpu::Gles3MinorVersion::Version2 => native::WGPUGles3MinorVersion_Version2,
    }
}

pub fn gl_fence_behavior_to_native(
    behavior: wgpu::GlFenceBehavior,
) -> native::WGPUGLFenceBehaviour {
    match behavior {
        wgpu::GlFenceBehavior::Normal => native::WGPUGLFenceBehaviour_Normal,
        wgpu::GlFenceBehavior::AutoFinish => native::WGPUGLFenceBehaviour_AutoFinish,
    }
}

// ── Strings ───────────────────────────────────────────────────────────────────

pub fn null_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: usize::MAX, // WGPU_STRLEN = undefined/null optional string
    }
}

pub fn str_to_string_view(s: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: s.as_ptr() as *const std::os::raw::c_char,
        length: s.len(),
    }
}

pub fn opt_str_to_string_view(s: Option<&str>) -> native::WGPUStringView {
    s.map(str_to_string_view).unwrap_or(null_string_view())
}

/// SAFETY: caller must ensure the WGPUStringView data pointer is valid.
pub unsafe fn string_view_to_string(sv: native::WGPUStringView) -> String {
    if sv.data.is_null() {
        return String::new();
    }
    let len = if sv.length == usize::MAX {
        // WGPU_STRLEN: null-terminated
        unsafe { std::ffi::CStr::from_ptr(sv.data as *const std::ffi::c_char) }
            .to_bytes()
            .len()
    } else {
        sv.length
    };
    if len == 0 {
        return String::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(sv.data as *const u8, len) };
    String::from_utf8_lossy(slice).into_owned()
}

// ── Features ──────────────────────────────────────────────────────────────────

pub fn map_feature(f: native::WGPUFeatureName) -> Option<wgpu::Features> {
    use wgpu::Features;
    match f {
        native::WGPUFeatureName_DepthClipControl => Some(Features::DEPTH_CLIP_CONTROL),
        native::WGPUFeatureName_Depth32FloatStencil8 => Some(Features::DEPTH32FLOAT_STENCIL8),
        native::WGPUFeatureName_TextureCompressionBC => Some(Features::TEXTURE_COMPRESSION_BC),
        native::WGPUFeatureName_TextureCompressionBCSliced3D => {
            Some(Features::TEXTURE_COMPRESSION_BC_SLICED_3D)
        }
        native::WGPUFeatureName_TextureCompressionETC2 => Some(Features::TEXTURE_COMPRESSION_ETC2),
        native::WGPUFeatureName_TextureCompressionASTC => Some(Features::TEXTURE_COMPRESSION_ASTC),
        native::WGPUFeatureName_TextureCompressionASTCSliced3D => {
            Some(Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D)
        }
        native::WGPUFeatureName_TimestampQuery => Some(Features::TIMESTAMP_QUERY),
        native::WGPUFeatureName_IndirectFirstInstance => Some(Features::INDIRECT_FIRST_INSTANCE),
        native::WGPUFeatureName_ShaderF16 => Some(Features::SHADER_F16),
        native::WGPUFeatureName_RG11B10UfloatRenderable => Some(Features::RG11B10UFLOAT_RENDERABLE),
        native::WGPUFeatureName_BGRA8UnormStorage => Some(Features::BGRA8UNORM_STORAGE),
        native::WGPUFeatureName_Float32Filterable => Some(Features::FLOAT32_FILTERABLE),
        native::WGPUFeatureName_Float32Blendable => Some(Features::FLOAT32_BLENDABLE),
        native::WGPUFeatureName_ClipDistances => Some(Features::CLIP_DISTANCES),
        native::WGPUFeatureName_DualSourceBlending => Some(Features::DUAL_SOURCE_BLENDING),
        native::WGPUFeatureName_PrimitiveIndex => Some(Features::PRIMITIVE_INDEX),
        native::WGPUNativeFeature_AddressModeClampToZero => {
            Some(Features::ADDRESS_MODE_CLAMP_TO_ZERO)
        }
        native::WGPUNativeFeature_AddressModeClampToBorder => {
            Some(Features::ADDRESS_MODE_CLAMP_TO_BORDER)
        }
        native::WGPUNativeFeature_PassthroughShaders => Some(Features::PASSTHROUGH_SHADERS),
        native::WGPUNativeFeature_Immediates => Some(Features::IMMEDIATES),
        native::WGPUNativeFeature_TextureAdapterSpecificFormatFeatures => {
            Some(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        }
        native::WGPUNativeFeature_MultiDrawIndirectCount => {
            Some(Features::MULTI_DRAW_INDIRECT_COUNT)
        }
        native::WGPUNativeFeature_VertexWritableStorage => Some(Features::VERTEX_WRITABLE_STORAGE),
        native::WGPUNativeFeature_TextureBindingArray => Some(Features::TEXTURE_BINDING_ARRAY),
        native::WGPUNativeFeature_SampledTextureAndStorageBufferArrayNonUniformIndexing => {
            Some(Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING)
        }
        native::WGPUNativeFeature_PipelineStatisticsQuery => {
            Some(Features::PIPELINE_STATISTICS_QUERY)
        }
        native::WGPUNativeFeature_StorageResourceBindingArray => {
            Some(Features::STORAGE_RESOURCE_BINDING_ARRAY)
        }
        native::WGPUNativeFeature_PartiallyBoundBindingArray => {
            Some(Features::PARTIALLY_BOUND_BINDING_ARRAY)
        }
        native::WGPUNativeFeature_TextureFormat16bitNorm => {
            Some(Features::TEXTURE_FORMAT_16BIT_NORM)
        }
        native::WGPUNativeFeature_TextureCompressionAstcHdr => {
            Some(Features::TEXTURE_COMPRESSION_ASTC_HDR)
        }
        native::WGPUNativeFeature_MappablePrimaryBuffers => {
            Some(Features::MAPPABLE_PRIMARY_BUFFERS)
        }
        native::WGPUNativeFeature_BufferBindingArray => Some(Features::BUFFER_BINDING_ARRAY),
        native::WGPUNativeFeature_StorageTextureArrayNonUniformIndexing => {
            Some(Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING)
        }
        native::WGPUNativeFeature_PolygonModeLine => Some(Features::POLYGON_MODE_LINE),
        native::WGPUNativeFeature_PolygonModePoint => Some(Features::POLYGON_MODE_POINT),
        native::WGPUNativeFeature_ConservativeRasterization => {
            Some(Features::CONSERVATIVE_RASTERIZATION)
        }
        native::WGPUNativeFeature_ClearTexture => Some(Features::CLEAR_TEXTURE),
        native::WGPUNativeFeature_Multiview => Some(Features::MULTIVIEW),
        native::WGPUNativeFeature_VertexAttribute64bit => Some(Features::VERTEX_ATTRIBUTE_64BIT),
        native::WGPUNativeFeature_TextureFormatNv12 => Some(Features::TEXTURE_FORMAT_NV12),
        native::WGPUNativeFeature_RayQuery => Some(Features::EXPERIMENTAL_RAY_QUERY),
        native::WGPUNativeFeature_ShaderF64 => Some(Features::SHADER_F64),
        native::WGPUNativeFeature_ShaderI16 => Some(Features::SHADER_I16),
        native::WGPUNativeFeature_ShaderEarlyDepthTest => Some(Features::SHADER_EARLY_DEPTH_TEST),
        native::WGPUFeatureName_Subgroups => Some(Features::SUBGROUP),
        native::WGPUNativeFeature_Subgroup => Some(Features::SUBGROUP),
        native::WGPUNativeFeature_SubgroupVertex => Some(Features::SUBGROUP_VERTEX),
        native::WGPUNativeFeature_SubgroupBarrier => Some(Features::SUBGROUP_BARRIER),
        native::WGPUNativeFeature_TimestampQueryInsideEncoders => {
            Some(Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
        }
        native::WGPUNativeFeature_TimestampQueryInsidePasses => {
            Some(Features::TIMESTAMP_QUERY_INSIDE_PASSES)
        }
        native::WGPUNativeFeature_ShaderInt64 => Some(Features::SHADER_INT64),
        native::WGPUNativeFeature_ShaderFloat32Atomic => Some(Features::SHADER_FLOAT32_ATOMIC),
        native::WGPUNativeFeature_TextureAtomic => Some(Features::TEXTURE_ATOMIC),
        native::WGPUNativeFeature_TextureFormatP010 => Some(Features::TEXTURE_FORMAT_P010),
        native::WGPUNativeFeature_PipelineCache => Some(Features::PIPELINE_CACHE),
        native::WGPUNativeFeature_ShaderInt64AtomicMinMax => {
            Some(Features::SHADER_INT64_ATOMIC_MIN_MAX)
        }
        native::WGPUNativeFeature_ShaderInt64AtomicAllOps => {
            Some(Features::SHADER_INT64_ATOMIC_ALL_OPS)
        }
        native::WGPUNativeFeature_TextureInt64Atomic => Some(Features::TEXTURE_INT64_ATOMIC),
        native::WGPUNativeFeature_ShaderBarycentrics => Some(Features::SHADER_BARYCENTRICS),
        native::WGPUNativeFeature_SelectiveMultiview => Some(Features::SELECTIVE_MULTIVIEW),
        native::WGPUNativeFeature_MultisampleArray => Some(Features::MULTISAMPLE_ARRAY),
        native::WGPUNativeFeature_CooperativeMatrix => {
            Some(Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
        }
        native::WGPUNativeFeature_MeshShader => Some(Features::EXPERIMENTAL_MESH_SHADER),
        native::WGPUNativeFeature_RayHitVertexReturn => {
            Some(Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN)
        }
        native::WGPUNativeFeature_MeshShaderMultiview => {
            Some(Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW)
        }
        native::WGPUNativeFeature_MeshShaderPoints => {
            Some(Features::EXPERIMENTAL_MESH_SHADER_POINTS)
        }
        native::WGPUNativeFeature_ShaderPerVertex => Some(Features::SHADER_PER_VERTEX),
        native::WGPUNativeFeature_ShaderDrawIndex => Some(Features::SHADER_DRAW_INDEX),
        native::WGPUNativeFeature_AccelerationStructureBindingArray => {
            Some(Features::ACCELERATION_STRUCTURE_BINDING_ARRAY)
        }
        native::WGPUNativeFeature_MemoryDecorationCoherent => {
            Some(Features::MEMORY_DECORATION_COHERENT)
        }
        native::WGPUNativeFeature_MemoryDecorationVolatile => {
            Some(Features::MEMORY_DECORATION_VOLATILE)
        }
        native::WGPUNativeFeature_ExternalTexture => Some(Features::EXTERNAL_TEXTURE),
        native::WGPUNativeFeature_ExtendedAccelerationStructureVertexFormats => {
            Some(Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS)
        }
        native::WGPUNativeFeature_VulkanExternalMemoryFd => {
            Some(Features::VULKAN_EXTERNAL_MEMORY_FD)
        }
        native::WGPUNativeFeature_VulkanExternalMemoryDmaBuf => {
            Some(Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF)
        }
        native::WGPUNativeFeature_VulkanGoogleDisplayTiming => {
            Some(Features::VULKAN_GOOGLE_DISPLAY_TIMING)
        }
        _ => None,
    }
}

pub fn map_supported_features(sf: &native::WGPUSupportedFeatures) -> wgpu::Features {
    let slice = unsafe { std::slice::from_raw_parts(sf.features, sf.featureCount) };
    let mut result = wgpu::Features::empty();
    for &f in slice {
        if let Some(feat) = map_feature(f) {
            result.insert(feat);
        }
    }
    result
}

pub fn features_to_native(features: wgpu::Features) -> Vec<native::WGPUFeatureName> {
    use wgpu::Features;
    let mut out = Vec::new();
    macro_rules! push {
        ($feat:expr, $name:expr) => {
            if features.contains($feat) {
                out.push($name);
            }
        };
    }
    push!(
        Features::DEPTH_CLIP_CONTROL,
        native::WGPUFeatureName_DepthClipControl
    );
    push!(
        Features::DEPTH32FLOAT_STENCIL8,
        native::WGPUFeatureName_Depth32FloatStencil8
    );
    push!(
        Features::TEXTURE_COMPRESSION_BC,
        native::WGPUFeatureName_TextureCompressionBC
    );
    push!(
        Features::TEXTURE_COMPRESSION_BC_SLICED_3D,
        native::WGPUFeatureName_TextureCompressionBCSliced3D
    );
    push!(
        Features::TEXTURE_COMPRESSION_ETC2,
        native::WGPUFeatureName_TextureCompressionETC2
    );
    push!(
        Features::TEXTURE_COMPRESSION_ASTC,
        native::WGPUFeatureName_TextureCompressionASTC
    );
    push!(
        Features::TEXTURE_COMPRESSION_ASTC_SLICED_3D,
        native::WGPUFeatureName_TextureCompressionASTCSliced3D
    );
    push!(
        Features::TIMESTAMP_QUERY,
        native::WGPUFeatureName_TimestampQuery
    );
    push!(
        Features::INDIRECT_FIRST_INSTANCE,
        native::WGPUFeatureName_IndirectFirstInstance
    );
    push!(Features::SHADER_F16, native::WGPUFeatureName_ShaderF16);
    push!(
        Features::RG11B10UFLOAT_RENDERABLE,
        native::WGPUFeatureName_RG11B10UfloatRenderable
    );
    push!(
        Features::BGRA8UNORM_STORAGE,
        native::WGPUFeatureName_BGRA8UnormStorage
    );
    push!(
        Features::FLOAT32_FILTERABLE,
        native::WGPUFeatureName_Float32Filterable
    );
    push!(
        Features::FLOAT32_BLENDABLE,
        native::WGPUFeatureName_Float32Blendable
    );
    push!(
        Features::CLIP_DISTANCES,
        native::WGPUFeatureName_ClipDistances
    );
    push!(
        Features::DUAL_SOURCE_BLENDING,
        native::WGPUFeatureName_DualSourceBlending
    );
    push!(
        Features::PRIMITIVE_INDEX,
        native::WGPUFeatureName_PrimitiveIndex
    );
    push!(
        Features::ADDRESS_MODE_CLAMP_TO_ZERO,
        native::WGPUNativeFeature_AddressModeClampToZero
    );
    push!(
        Features::ADDRESS_MODE_CLAMP_TO_BORDER,
        native::WGPUNativeFeature_AddressModeClampToBorder
    );
    push!(
        Features::PASSTHROUGH_SHADERS,
        native::WGPUNativeFeature_PassthroughShaders
    );
    push!(Features::IMMEDIATES, native::WGPUNativeFeature_Immediates);
    push!(
        Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        native::WGPUNativeFeature_TextureAdapterSpecificFormatFeatures
    );
    push!(
        Features::MULTI_DRAW_INDIRECT_COUNT,
        native::WGPUNativeFeature_MultiDrawIndirectCount
    );
    push!(
        Features::VERTEX_WRITABLE_STORAGE,
        native::WGPUNativeFeature_VertexWritableStorage
    );
    push!(
        Features::TEXTURE_BINDING_ARRAY,
        native::WGPUNativeFeature_TextureBindingArray
    );
    push!(
        Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        native::WGPUNativeFeature_SampledTextureAndStorageBufferArrayNonUniformIndexing
    );
    push!(
        Features::PIPELINE_STATISTICS_QUERY,
        native::WGPUNativeFeature_PipelineStatisticsQuery
    );
    push!(
        Features::STORAGE_RESOURCE_BINDING_ARRAY,
        native::WGPUNativeFeature_StorageResourceBindingArray
    );
    push!(
        Features::PARTIALLY_BOUND_BINDING_ARRAY,
        native::WGPUNativeFeature_PartiallyBoundBindingArray
    );
    push!(
        Features::TEXTURE_FORMAT_16BIT_NORM,
        native::WGPUNativeFeature_TextureFormat16bitNorm
    );
    push!(
        Features::TEXTURE_COMPRESSION_ASTC_HDR,
        native::WGPUNativeFeature_TextureCompressionAstcHdr
    );
    push!(
        Features::MAPPABLE_PRIMARY_BUFFERS,
        native::WGPUNativeFeature_MappablePrimaryBuffers
    );
    push!(
        Features::BUFFER_BINDING_ARRAY,
        native::WGPUNativeFeature_BufferBindingArray
    );
    push!(
        Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING,
        native::WGPUNativeFeature_StorageTextureArrayNonUniformIndexing
    );
    push!(
        Features::POLYGON_MODE_LINE,
        native::WGPUNativeFeature_PolygonModeLine
    );
    push!(
        Features::POLYGON_MODE_POINT,
        native::WGPUNativeFeature_PolygonModePoint
    );
    push!(
        Features::CONSERVATIVE_RASTERIZATION,
        native::WGPUNativeFeature_ConservativeRasterization
    );
    push!(
        Features::CLEAR_TEXTURE,
        native::WGPUNativeFeature_ClearTexture
    );
    push!(Features::MULTIVIEW, native::WGPUNativeFeature_Multiview);
    push!(
        Features::VERTEX_ATTRIBUTE_64BIT,
        native::WGPUNativeFeature_VertexAttribute64bit
    );
    push!(
        Features::TEXTURE_FORMAT_NV12,
        native::WGPUNativeFeature_TextureFormatNv12
    );
    push!(
        Features::EXPERIMENTAL_RAY_QUERY,
        native::WGPUNativeFeature_RayQuery
    );
    push!(Features::SHADER_F64, native::WGPUNativeFeature_ShaderF64);
    push!(Features::SHADER_I16, native::WGPUNativeFeature_ShaderI16);
    push!(
        Features::SHADER_EARLY_DEPTH_TEST,
        native::WGPUNativeFeature_ShaderEarlyDepthTest
    );
    push!(Features::SUBGROUP, native::WGPUNativeFeature_Subgroup);
    push!(
        Features::SUBGROUP_VERTEX,
        native::WGPUNativeFeature_SubgroupVertex
    );
    push!(
        Features::SUBGROUP_BARRIER,
        native::WGPUNativeFeature_SubgroupBarrier
    );
    push!(
        Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
        native::WGPUNativeFeature_TimestampQueryInsideEncoders
    );
    push!(
        Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        native::WGPUNativeFeature_TimestampQueryInsidePasses
    );
    push!(
        Features::SHADER_INT64,
        native::WGPUNativeFeature_ShaderInt64
    );
    push!(
        Features::SHADER_FLOAT32_ATOMIC,
        native::WGPUNativeFeature_ShaderFloat32Atomic
    );
    push!(
        Features::TEXTURE_ATOMIC,
        native::WGPUNativeFeature_TextureAtomic
    );
    push!(
        Features::TEXTURE_FORMAT_P010,
        native::WGPUNativeFeature_TextureFormatP010
    );
    push!(
        Features::PIPELINE_CACHE,
        native::WGPUNativeFeature_PipelineCache
    );
    push!(
        Features::SHADER_INT64_ATOMIC_MIN_MAX,
        native::WGPUNativeFeature_ShaderInt64AtomicMinMax
    );
    push!(
        Features::SHADER_INT64_ATOMIC_ALL_OPS,
        native::WGPUNativeFeature_ShaderInt64AtomicAllOps
    );
    push!(
        Features::TEXTURE_INT64_ATOMIC,
        native::WGPUNativeFeature_TextureInt64Atomic
    );
    push!(
        Features::SHADER_BARYCENTRICS,
        native::WGPUNativeFeature_ShaderBarycentrics
    );
    push!(
        Features::SELECTIVE_MULTIVIEW,
        native::WGPUNativeFeature_SelectiveMultiview
    );
    push!(
        Features::MULTISAMPLE_ARRAY,
        native::WGPUNativeFeature_MultisampleArray
    );
    push!(
        Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        native::WGPUNativeFeature_CooperativeMatrix
    );
    push!(
        Features::EXPERIMENTAL_MESH_SHADER,
        native::WGPUNativeFeature_MeshShader
    );
    push!(
        Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN,
        native::WGPUNativeFeature_RayHitVertexReturn
    );
    push!(
        Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW,
        native::WGPUNativeFeature_MeshShaderMultiview
    );
    push!(
        Features::EXPERIMENTAL_MESH_SHADER_POINTS,
        native::WGPUNativeFeature_MeshShaderPoints
    );
    push!(
        Features::SHADER_PER_VERTEX,
        native::WGPUNativeFeature_ShaderPerVertex
    );
    push!(
        Features::SHADER_DRAW_INDEX,
        native::WGPUNativeFeature_ShaderDrawIndex
    );
    push!(
        Features::ACCELERATION_STRUCTURE_BINDING_ARRAY,
        native::WGPUNativeFeature_AccelerationStructureBindingArray
    );
    push!(
        Features::MEMORY_DECORATION_COHERENT,
        native::WGPUNativeFeature_MemoryDecorationCoherent
    );
    push!(
        Features::MEMORY_DECORATION_VOLATILE,
        native::WGPUNativeFeature_MemoryDecorationVolatile
    );
    push!(Features::EXTERNAL_TEXTURE, native::WGPUNativeFeature_ExternalTexture);
    push!(
        Features::EXTENDED_ACCELERATION_STRUCTURE_VERTEX_FORMATS,
        native::WGPUNativeFeature_ExtendedAccelerationStructureVertexFormats
    );
    push!(
        Features::VULKAN_EXTERNAL_MEMORY_FD,
        native::WGPUNativeFeature_VulkanExternalMemoryFd
    );
    push!(
        Features::VULKAN_EXTERNAL_MEMORY_DMA_BUF,
        native::WGPUNativeFeature_VulkanExternalMemoryDmaBuf
    );
    push!(
        Features::VULKAN_GOOGLE_DISPLAY_TIMING,
        native::WGPUNativeFeature_VulkanGoogleDisplayTiming
    );
    out
}

// ── Limits ────────────────────────────────────────────────────────────────────

pub fn limits_to_native(l: &wgpu::Limits) -> native::WGPULimits {
    // SAFETY: undefined sentinel fields stay as WGPU_LIMIT_U32_UNDEFINED = u32::MAX
    let mut out: native::WGPULimits = unsafe { std::mem::zeroed() };
    out.maxTextureDimension1D = l.max_texture_dimension_1d;
    out.maxTextureDimension2D = l.max_texture_dimension_2d;
    out.maxTextureDimension3D = l.max_texture_dimension_3d;
    out.maxTextureArrayLayers = l.max_texture_array_layers;
    out.maxBindGroups = l.max_bind_groups;
    out.maxBindingsPerBindGroup = l.max_bindings_per_bind_group;
    out.maxDynamicUniformBuffersPerPipelineLayout =
        l.max_dynamic_uniform_buffers_per_pipeline_layout;
    out.maxDynamicStorageBuffersPerPipelineLayout =
        l.max_dynamic_storage_buffers_per_pipeline_layout;
    out.maxSampledTexturesPerShaderStage = l.max_sampled_textures_per_shader_stage;
    out.maxSamplersPerShaderStage = l.max_samplers_per_shader_stage;
    out.maxStorageBuffersPerShaderStage = l.max_storage_buffers_per_shader_stage;
    out.maxStorageTexturesPerShaderStage = l.max_storage_textures_per_shader_stage;
    out.maxUniformBuffersPerShaderStage = l.max_uniform_buffers_per_shader_stage;
    out.maxUniformBufferBindingSize = l.max_uniform_buffer_binding_size;
    out.maxStorageBufferBindingSize = l.max_storage_buffer_binding_size;
    out.minUniformBufferOffsetAlignment = l.min_uniform_buffer_offset_alignment;
    out.minStorageBufferOffsetAlignment = l.min_storage_buffer_offset_alignment;
    out.maxVertexBuffers = l.max_vertex_buffers;
    out.maxBufferSize = l.max_buffer_size;
    out.maxVertexAttributes = l.max_vertex_attributes;
    out.maxVertexBufferArrayStride = l.max_vertex_buffer_array_stride;
    out.maxInterStageShaderVariables = l.max_inter_stage_shader_variables;
    out.maxColorAttachments = l.max_color_attachments;
    out.maxColorAttachmentBytesPerSample = l.max_color_attachment_bytes_per_sample;
    out.maxComputeWorkgroupStorageSize = l.max_compute_workgroup_storage_size;
    out.maxComputeInvocationsPerWorkgroup = l.max_compute_invocations_per_workgroup;
    out.maxComputeWorkgroupSizeX = l.max_compute_workgroup_size_x;
    out.maxComputeWorkgroupSizeY = l.max_compute_workgroup_size_y;
    out.maxComputeWorkgroupSizeZ = l.max_compute_workgroup_size_z;
    out.maxComputeWorkgroupsPerDimension = l.max_compute_workgroups_per_dimension;
    out.maxImmediateSize = l.max_immediate_size;
    out
}

pub fn map_limits(c: &native::WGPULimits, extras: Option<&native::WGPUNativeLimits>) -> wgpu::Limits {
    let mut l = wgpu::Limits::default();
    // wgpuAdapterGetLimits / wgpuDeviceGetLimits always fill all standard fields with the real
    // hardware values.  Do NOT use a sentinel check here: u32::MAX is a valid hardware limit (e.g.
    // Metal reports u32::MAX for maxBindingsPerBindGroup) and skipping it would leave the field at
    // the WebGPU default (1000), which differs from what wgpu-core reports directly.
    macro_rules! set {
        ($field:ident, $src:expr) => {
            l.$field = $src as _;
        };
    }
    set!(max_texture_dimension_1d, c.maxTextureDimension1D);
    set!(max_texture_dimension_2d, c.maxTextureDimension2D);
    set!(max_texture_dimension_3d, c.maxTextureDimension3D);
    set!(max_texture_array_layers, c.maxTextureArrayLayers);
    set!(max_bind_groups, c.maxBindGroups);
    set!(max_bindings_per_bind_group, c.maxBindingsPerBindGroup);
    set!(
        max_dynamic_uniform_buffers_per_pipeline_layout,
        c.maxDynamicUniformBuffersPerPipelineLayout
    );
    set!(
        max_dynamic_storage_buffers_per_pipeline_layout,
        c.maxDynamicStorageBuffersPerPipelineLayout
    );
    set!(
        max_sampled_textures_per_shader_stage,
        c.maxSampledTexturesPerShaderStage
    );
    set!(max_samplers_per_shader_stage, c.maxSamplersPerShaderStage);
    set!(
        max_storage_buffers_per_shader_stage,
        c.maxStorageBuffersPerShaderStage
    );
    set!(
        max_storage_textures_per_shader_stage,
        c.maxStorageTexturesPerShaderStage
    );
    set!(
        max_uniform_buffers_per_shader_stage,
        c.maxUniformBuffersPerShaderStage
    );
    set!(max_uniform_buffer_binding_size, c.maxUniformBufferBindingSize);
    set!(max_storage_buffer_binding_size, c.maxStorageBufferBindingSize);
    set!(
        min_uniform_buffer_offset_alignment,
        c.minUniformBufferOffsetAlignment
    );
    set!(
        min_storage_buffer_offset_alignment,
        c.minStorageBufferOffsetAlignment
    );
    set!(max_vertex_buffers, c.maxVertexBuffers);
    set!(max_buffer_size, c.maxBufferSize);
    set!(max_vertex_attributes, c.maxVertexAttributes);
    set!(max_vertex_buffer_array_stride, c.maxVertexBufferArrayStride);
    set!(
        max_inter_stage_shader_variables,
        c.maxInterStageShaderVariables
    );
    set!(max_color_attachments, c.maxColorAttachments);
    set!(
        max_color_attachment_bytes_per_sample,
        c.maxColorAttachmentBytesPerSample
    );
    set!(
        max_compute_workgroup_storage_size,
        c.maxComputeWorkgroupStorageSize
    );
    set!(
        max_compute_invocations_per_workgroup,
        c.maxComputeInvocationsPerWorkgroup
    );
    set!(max_compute_workgroup_size_x, c.maxComputeWorkgroupSizeX);
    set!(max_compute_workgroup_size_y, c.maxComputeWorkgroupSizeY);
    set!(max_compute_workgroup_size_z, c.maxComputeWorkgroupSizeZ);
    set!(
        max_compute_workgroups_per_dimension,
        c.maxComputeWorkgroupsPerDimension
    );
    set!(max_immediate_size, c.maxImmediateSize);
    if let Some(n) = extras {
        set!(max_non_sampler_bindings, n.maxNonSamplerBindings);
        set!(
            max_binding_array_elements_per_shader_stage,
            n.maxBindingArrayElementsPerShaderStage
        );
        set!(
            max_binding_array_sampler_elements_per_shader_stage,
            n.maxBindingArraySamplerElementsPerShaderStage
        );
        set!(max_multiview_view_count, n.maxMultiviewViewCount);
        set!(
            max_binding_array_acceleration_structure_elements_per_shader_stage,
            n.maxBindingArrayAccelerationStructureElementsPerShaderStage
        );
        set!(max_task_workgroup_total_count, n.maxTaskWorkgroupTotalCount);
        set!(max_task_workgroups_per_dimension, n.maxTaskWorkgroupsPerDimension);
        set!(max_mesh_workgroup_total_count, n.maxMeshWorkgroupTotalCount);
        set!(max_mesh_workgroups_per_dimension, n.maxMeshWorkgroupsPerDimension);
        set!(max_task_invocations_per_workgroup, n.maxTaskInvocationsPerWorkgroup);
        set!(max_task_invocations_per_dimension, n.maxTaskInvocationsPerDimension);
        set!(max_mesh_invocations_per_workgroup, n.maxMeshInvocationsPerWorkgroup);
        set!(max_mesh_invocations_per_dimension, n.maxMeshInvocationsPerDimension);
        set!(max_task_payload_size, n.maxTaskPayloadSize);
        set!(max_mesh_output_vertices, n.maxMeshOutputVertices);
        set!(max_mesh_output_primitives, n.maxMeshOutputPrimitives);
        set!(max_mesh_output_layers, n.maxMeshOutputLayers);
        set!(max_mesh_multiview_view_count, n.maxMeshMultiviewViewCount);
        set!(max_blas_primitive_count, n.maxBlasPrimitiveCount as u32);
        set!(max_blas_geometry_count, n.maxBlasGeometryCount as u32);
        set!(max_tlas_instance_count, n.maxTlasInstanceCount as u32);
        set!(
            max_acceleration_structures_per_shader_stage,
            n.maxAccelerationStructuresPerShaderStage
        );
    }
    l
}

pub fn map_texture_format_capabilities(
    caps: &native::WGPUNativeTextureFormatCapabilities,
) -> wgpu::TextureFormatFeatures {
    wgpu::TextureFormatFeatures {
        allowed_usages: wgpu::TextureUsages::from_bits_truncate(caps.allowedUsages as u32),
        flags: wgpu::TextureFormatFeatureFlags::from_bits_truncate(caps.flags),
    }
}

// ── Adapter info ──────────────────────────────────────────────────────────────

pub fn map_backend_from_native(b: native::WGPUBackendType) -> wgpu::Backend {
    match b {
        native::WGPUBackendType_Vulkan => wgpu::Backend::Vulkan,
        native::WGPUBackendType_Metal => wgpu::Backend::Metal,
        native::WGPUBackendType_D3D12 => wgpu::Backend::Dx12,
        native::WGPUBackendType_OpenGL | native::WGPUBackendType_OpenGLES => wgpu::Backend::Gl,
        native::WGPUBackendType_WebGPU => wgpu::Backend::BrowserWebGpu,
        _ => wgpu::Backend::Noop,
    }
}

pub fn map_device_type_from_native(t: native::WGPUAdapterType) -> wgpu::DeviceType {
    match t {
        native::WGPUAdapterType_DiscreteGPU => wgpu::DeviceType::DiscreteGpu,
        native::WGPUAdapterType_IntegratedGPU => wgpu::DeviceType::IntegratedGpu,
        native::WGPUAdapterType_CPU => wgpu::DeviceType::Cpu,
        _ => wgpu::DeviceType::Other,
    }
}

// ── Texture format ────────────────────────────────────────────────────────────

pub fn map_texture_format(v: native::WGPUTextureFormat) -> Option<wgpu::TextureFormat> {
    use wgpu::{AstcBlock, AstcChannel, TextureFormat as TF};
    match v {
        native::WGPUTextureFormat_Undefined => None,
        native::WGPUTextureFormat_R8Unorm => Some(TF::R8Unorm),
        native::WGPUTextureFormat_R8Snorm => Some(TF::R8Snorm),
        native::WGPUTextureFormat_R8Uint => Some(TF::R8Uint),
        native::WGPUTextureFormat_R8Sint => Some(TF::R8Sint),
        native::WGPUTextureFormat_R16Uint => Some(TF::R16Uint),
        native::WGPUTextureFormat_R16Sint => Some(TF::R16Sint),
        native::WGPUTextureFormat_R16Float => Some(TF::R16Float),
        native::WGPUTextureFormat_RG8Unorm => Some(TF::Rg8Unorm),
        native::WGPUTextureFormat_RG8Snorm => Some(TF::Rg8Snorm),
        native::WGPUTextureFormat_RG8Uint => Some(TF::Rg8Uint),
        native::WGPUTextureFormat_RG8Sint => Some(TF::Rg8Sint),
        native::WGPUTextureFormat_R32Float => Some(TF::R32Float),
        native::WGPUTextureFormat_R32Uint => Some(TF::R32Uint),
        native::WGPUTextureFormat_R32Sint => Some(TF::R32Sint),
        native::WGPUTextureFormat_RG16Unorm => Some(TF::Rg16Unorm),
        native::WGPUTextureFormat_RG16Snorm => Some(TF::Rg16Snorm),
        native::WGPUTextureFormat_RG16Uint => Some(TF::Rg16Uint),
        native::WGPUTextureFormat_RG16Sint => Some(TF::Rg16Sint),
        native::WGPUTextureFormat_RG16Float => Some(TF::Rg16Float),
        native::WGPUTextureFormat_RGBA8Unorm => Some(TF::Rgba8Unorm),
        native::WGPUTextureFormat_RGBA8UnormSrgb => Some(TF::Rgba8UnormSrgb),
        native::WGPUTextureFormat_RGBA8Snorm => Some(TF::Rgba8Snorm),
        native::WGPUTextureFormat_RGBA8Uint => Some(TF::Rgba8Uint),
        native::WGPUTextureFormat_RGBA8Sint => Some(TF::Rgba8Sint),
        native::WGPUTextureFormat_BGRA8Unorm => Some(TF::Bgra8Unorm),
        native::WGPUTextureFormat_BGRA8UnormSrgb => Some(TF::Bgra8UnormSrgb),
        native::WGPUTextureFormat_RGB10A2Uint => Some(TF::Rgb10a2Uint),
        native::WGPUTextureFormat_RGB10A2Unorm => Some(TF::Rgb10a2Unorm),
        native::WGPUTextureFormat_RG11B10Ufloat => Some(TF::Rg11b10Ufloat),
        native::WGPUTextureFormat_RGB9E5Ufloat => Some(TF::Rgb9e5Ufloat),
        native::WGPUTextureFormat_RG32Float => Some(TF::Rg32Float),
        native::WGPUTextureFormat_RG32Uint => Some(TF::Rg32Uint),
        native::WGPUTextureFormat_RG32Sint => Some(TF::Rg32Sint),
        native::WGPUTextureFormat_RGBA16Unorm => Some(TF::Rgba16Unorm),
        native::WGPUTextureFormat_RGBA16Snorm => Some(TF::Rgba16Snorm),
        native::WGPUTextureFormat_RGBA16Uint => Some(TF::Rgba16Uint),
        native::WGPUTextureFormat_RGBA16Sint => Some(TF::Rgba16Sint),
        native::WGPUTextureFormat_RGBA16Float => Some(TF::Rgba16Float),
        native::WGPUTextureFormat_RGBA32Float => Some(TF::Rgba32Float),
        native::WGPUTextureFormat_RGBA32Uint => Some(TF::Rgba32Uint),
        native::WGPUTextureFormat_RGBA32Sint => Some(TF::Rgba32Sint),
        native::WGPUTextureFormat_Stencil8 => Some(TF::Stencil8),
        native::WGPUTextureFormat_Depth16Unorm => Some(TF::Depth16Unorm),
        native::WGPUTextureFormat_Depth24Plus => Some(TF::Depth24Plus),
        native::WGPUTextureFormat_Depth24PlusStencil8 => Some(TF::Depth24PlusStencil8),
        native::WGPUTextureFormat_Depth32Float => Some(TF::Depth32Float),
        native::WGPUTextureFormat_Depth32FloatStencil8 => Some(TF::Depth32FloatStencil8),
        native::WGPUTextureFormat_BC1RGBAUnorm => Some(TF::Bc1RgbaUnorm),
        native::WGPUTextureFormat_BC1RGBAUnormSrgb => Some(TF::Bc1RgbaUnormSrgb),
        native::WGPUTextureFormat_BC2RGBAUnorm => Some(TF::Bc2RgbaUnorm),
        native::WGPUTextureFormat_BC2RGBAUnormSrgb => Some(TF::Bc2RgbaUnormSrgb),
        native::WGPUTextureFormat_BC3RGBAUnorm => Some(TF::Bc3RgbaUnorm),
        native::WGPUTextureFormat_BC3RGBAUnormSrgb => Some(TF::Bc3RgbaUnormSrgb),
        native::WGPUTextureFormat_BC4RUnorm => Some(TF::Bc4RUnorm),
        native::WGPUTextureFormat_BC4RSnorm => Some(TF::Bc4RSnorm),
        native::WGPUTextureFormat_BC5RGUnorm => Some(TF::Bc5RgUnorm),
        native::WGPUTextureFormat_BC5RGSnorm => Some(TF::Bc5RgSnorm),
        native::WGPUTextureFormat_BC6HRGBUfloat => Some(TF::Bc6hRgbUfloat),
        native::WGPUTextureFormat_BC6HRGBFloat => Some(TF::Bc6hRgbFloat),
        native::WGPUTextureFormat_BC7RGBAUnorm => Some(TF::Bc7RgbaUnorm),
        native::WGPUTextureFormat_BC7RGBAUnormSrgb => Some(TF::Bc7RgbaUnormSrgb),
        native::WGPUTextureFormat_ETC2RGB8Unorm => Some(TF::Etc2Rgb8Unorm),
        native::WGPUTextureFormat_ETC2RGB8UnormSrgb => Some(TF::Etc2Rgb8UnormSrgb),
        native::WGPUTextureFormat_ETC2RGB8A1Unorm => Some(TF::Etc2Rgb8A1Unorm),
        native::WGPUTextureFormat_ETC2RGB8A1UnormSrgb => Some(TF::Etc2Rgb8A1UnormSrgb),
        native::WGPUTextureFormat_ETC2RGBA8Unorm => Some(TF::Etc2Rgba8Unorm),
        native::WGPUTextureFormat_ETC2RGBA8UnormSrgb => Some(TF::Etc2Rgba8UnormSrgb),
        native::WGPUTextureFormat_EACR11Unorm => Some(TF::EacR11Unorm),
        native::WGPUTextureFormat_EACR11Snorm => Some(TF::EacR11Snorm),
        native::WGPUTextureFormat_EACRG11Unorm => Some(TF::EacRg11Unorm),
        native::WGPUTextureFormat_EACRG11Snorm => Some(TF::EacRg11Snorm),
        native::WGPUTextureFormat_ASTC4x4Unorm => Some(TF::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC4x4UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B4x4,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC5x4Unorm => Some(TF::Astc {
            block: AstcBlock::B5x4,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC5x4UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B5x4,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC5x5Unorm => Some(TF::Astc {
            block: AstcBlock::B5x5,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC5x5UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B5x5,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC6x5Unorm => Some(TF::Astc {
            block: AstcBlock::B6x5,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC6x5UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B6x5,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC6x6Unorm => Some(TF::Astc {
            block: AstcBlock::B6x6,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC6x6UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B6x6,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC8x5Unorm => Some(TF::Astc {
            block: AstcBlock::B8x5,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC8x5UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B8x5,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC8x6Unorm => Some(TF::Astc {
            block: AstcBlock::B8x6,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC8x6UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B8x6,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC8x8Unorm => Some(TF::Astc {
            block: AstcBlock::B8x8,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC8x8UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B8x8,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC10x5Unorm => Some(TF::Astc {
            block: AstcBlock::B10x5,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC10x5UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B10x5,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC10x6Unorm => Some(TF::Astc {
            block: AstcBlock::B10x6,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC10x6UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B10x6,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC10x8Unorm => Some(TF::Astc {
            block: AstcBlock::B10x8,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC10x8UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B10x8,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC10x10Unorm => Some(TF::Astc {
            block: AstcBlock::B10x10,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC10x10UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B10x10,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC12x10Unorm => Some(TF::Astc {
            block: AstcBlock::B12x10,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC12x10UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B12x10,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUTextureFormat_ASTC12x12Unorm => Some(TF::Astc {
            block: AstcBlock::B12x12,
            channel: AstcChannel::Unorm,
        }),
        native::WGPUTextureFormat_ASTC12x12UnormSrgb => Some(TF::Astc {
            block: AstcBlock::B12x12,
            channel: AstcChannel::UnormSrgb,
        }),
        native::WGPUNativeTextureFormat_R16Unorm => Some(TF::R16Unorm),
        native::WGPUNativeTextureFormat_R16Snorm => Some(TF::R16Snorm),
        native::WGPUNativeTextureFormat_Rg16Unorm => Some(TF::Rg16Unorm),
        native::WGPUNativeTextureFormat_Rg16Snorm => Some(TF::Rg16Snorm),
        native::WGPUNativeTextureFormat_Rgba16Unorm => Some(TF::Rgba16Unorm),
        native::WGPUNativeTextureFormat_Rgba16Snorm => Some(TF::Rgba16Snorm),
        native::WGPUNativeTextureFormat_NV12 => Some(TF::NV12),
        native::WGPUNativeTextureFormat_P010 => Some(TF::P010),
        native::WGPUNativeTextureFormat_Astc4x4Sfloat => Some(TF::Astc { block: AstcBlock::B4x4, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc5x4Sfloat => Some(TF::Astc { block: AstcBlock::B5x4, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc5x5Sfloat => Some(TF::Astc { block: AstcBlock::B5x5, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc6x5Sfloat => Some(TF::Astc { block: AstcBlock::B6x5, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc6x6Sfloat => Some(TF::Astc { block: AstcBlock::B6x6, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc8x5Sfloat => Some(TF::Astc { block: AstcBlock::B8x5, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc8x6Sfloat => Some(TF::Astc { block: AstcBlock::B8x6, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc8x8Sfloat => Some(TF::Astc { block: AstcBlock::B8x8, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc10x5Sfloat => Some(TF::Astc { block: AstcBlock::B10x5, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc10x6Sfloat => Some(TF::Astc { block: AstcBlock::B10x6, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc10x8Sfloat => Some(TF::Astc { block: AstcBlock::B10x8, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc10x10Sfloat => Some(TF::Astc { block: AstcBlock::B10x10, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc12x10Sfloat => Some(TF::Astc { block: AstcBlock::B12x10, channel: AstcChannel::Hdr }),
        native::WGPUNativeTextureFormat_Astc12x12Sfloat => Some(TF::Astc { block: AstcBlock::B12x12, channel: AstcChannel::Hdr }),
        _ => None,
    }
}

pub fn texture_format_to_native(f: wgpu::TextureFormat) -> native::WGPUTextureFormat {
    use wgpu::TextureFormat as TF;
    match f {
        TF::R8Unorm => native::WGPUTextureFormat_R8Unorm,
        TF::R8Snorm => native::WGPUTextureFormat_R8Snorm,
        TF::R8Uint => native::WGPUTextureFormat_R8Uint,
        TF::R8Sint => native::WGPUTextureFormat_R8Sint,
        TF::R16Uint => native::WGPUTextureFormat_R16Uint,
        TF::R16Sint => native::WGPUTextureFormat_R16Sint,
        TF::R16Float => native::WGPUTextureFormat_R16Float,
        TF::Rg8Unorm => native::WGPUTextureFormat_RG8Unorm,
        TF::Rg8Snorm => native::WGPUTextureFormat_RG8Snorm,
        TF::Rg8Uint => native::WGPUTextureFormat_RG8Uint,
        TF::Rg8Sint => native::WGPUTextureFormat_RG8Sint,
        TF::R32Float => native::WGPUTextureFormat_R32Float,
        TF::R32Uint => native::WGPUTextureFormat_R32Uint,
        TF::R32Sint => native::WGPUTextureFormat_R32Sint,
        TF::Rg16Unorm => native::WGPUTextureFormat_RG16Unorm,
        TF::Rg16Snorm => native::WGPUTextureFormat_RG16Snorm,
        TF::Rg16Uint => native::WGPUTextureFormat_RG16Uint,
        TF::Rg16Sint => native::WGPUTextureFormat_RG16Sint,
        TF::Rg16Float => native::WGPUTextureFormat_RG16Float,
        TF::Rgba8Unorm => native::WGPUTextureFormat_RGBA8Unorm,
        TF::Rgba8UnormSrgb => native::WGPUTextureFormat_RGBA8UnormSrgb,
        TF::Rgba8Snorm => native::WGPUTextureFormat_RGBA8Snorm,
        TF::Rgba8Uint => native::WGPUTextureFormat_RGBA8Uint,
        TF::Rgba8Sint => native::WGPUTextureFormat_RGBA8Sint,
        TF::Bgra8Unorm => native::WGPUTextureFormat_BGRA8Unorm,
        TF::Bgra8UnormSrgb => native::WGPUTextureFormat_BGRA8UnormSrgb,
        TF::Rgb10a2Uint => native::WGPUTextureFormat_RGB10A2Uint,
        TF::Rgb10a2Unorm => native::WGPUTextureFormat_RGB10A2Unorm,
        TF::Rg11b10Ufloat => native::WGPUTextureFormat_RG11B10Ufloat,
        TF::Rgb9e5Ufloat => native::WGPUTextureFormat_RGB9E5Ufloat,
        TF::Rg32Float => native::WGPUTextureFormat_RG32Float,
        TF::Rg32Uint => native::WGPUTextureFormat_RG32Uint,
        TF::Rg32Sint => native::WGPUTextureFormat_RG32Sint,
        TF::Rgba16Unorm => native::WGPUTextureFormat_RGBA16Unorm,
        TF::Rgba16Snorm => native::WGPUTextureFormat_RGBA16Snorm,
        TF::Rgba16Uint => native::WGPUTextureFormat_RGBA16Uint,
        TF::Rgba16Sint => native::WGPUTextureFormat_RGBA16Sint,
        TF::Rgba16Float => native::WGPUTextureFormat_RGBA16Float,
        TF::Rgba32Float => native::WGPUTextureFormat_RGBA32Float,
        TF::Rgba32Uint => native::WGPUTextureFormat_RGBA32Uint,
        TF::Rgba32Sint => native::WGPUTextureFormat_RGBA32Sint,
        TF::Stencil8 => native::WGPUTextureFormat_Stencil8,
        TF::Depth16Unorm => native::WGPUTextureFormat_Depth16Unorm,
        TF::Depth24Plus => native::WGPUTextureFormat_Depth24Plus,
        TF::Depth24PlusStencil8 => native::WGPUTextureFormat_Depth24PlusStencil8,
        TF::Depth32Float => native::WGPUTextureFormat_Depth32Float,
        TF::Depth32FloatStencil8 => native::WGPUTextureFormat_Depth32FloatStencil8,
        TF::Bc1RgbaUnorm => native::WGPUTextureFormat_BC1RGBAUnorm,
        TF::Bc1RgbaUnormSrgb => native::WGPUTextureFormat_BC1RGBAUnormSrgb,
        TF::Bc2RgbaUnorm => native::WGPUTextureFormat_BC2RGBAUnorm,
        TF::Bc2RgbaUnormSrgb => native::WGPUTextureFormat_BC2RGBAUnormSrgb,
        TF::Bc3RgbaUnorm => native::WGPUTextureFormat_BC3RGBAUnorm,
        TF::Bc3RgbaUnormSrgb => native::WGPUTextureFormat_BC3RGBAUnormSrgb,
        TF::Bc4RUnorm => native::WGPUTextureFormat_BC4RUnorm,
        TF::Bc4RSnorm => native::WGPUTextureFormat_BC4RSnorm,
        TF::Bc5RgUnorm => native::WGPUTextureFormat_BC5RGUnorm,
        TF::Bc5RgSnorm => native::WGPUTextureFormat_BC5RGSnorm,
        TF::Bc6hRgbUfloat => native::WGPUTextureFormat_BC6HRGBUfloat,
        TF::Bc6hRgbFloat => native::WGPUTextureFormat_BC6HRGBFloat,
        TF::Bc7RgbaUnorm => native::WGPUTextureFormat_BC7RGBAUnorm,
        TF::Bc7RgbaUnormSrgb => native::WGPUTextureFormat_BC7RGBAUnormSrgb,
        TF::Etc2Rgb8Unorm => native::WGPUTextureFormat_ETC2RGB8Unorm,
        TF::Etc2Rgb8UnormSrgb => native::WGPUTextureFormat_ETC2RGB8UnormSrgb,
        TF::Etc2Rgb8A1Unorm => native::WGPUTextureFormat_ETC2RGB8A1Unorm,
        TF::Etc2Rgb8A1UnormSrgb => native::WGPUTextureFormat_ETC2RGB8A1UnormSrgb,
        TF::Etc2Rgba8Unorm => native::WGPUTextureFormat_ETC2RGBA8Unorm,
        TF::Etc2Rgba8UnormSrgb => native::WGPUTextureFormat_ETC2RGBA8UnormSrgb,
        TF::EacR11Unorm => native::WGPUTextureFormat_EACR11Unorm,
        TF::EacR11Snorm => native::WGPUTextureFormat_EACR11Snorm,
        TF::EacRg11Unorm => native::WGPUTextureFormat_EACRG11Unorm,
        TF::EacRg11Snorm => native::WGPUTextureFormat_EACRG11Snorm,
        TF::Astc { block, channel } => astc_to_native(block, channel),
        TF::R16Unorm => native::WGPUNativeTextureFormat_R16Unorm,
        TF::R16Snorm => native::WGPUNativeTextureFormat_R16Snorm,
        TF::NV12 => native::WGPUNativeTextureFormat_NV12,
        TF::P010 => native::WGPUNativeTextureFormat_P010,
        TF::R64Uint => wgpu_native::conv::WGPU_NATIVE_TEXTURE_FORMAT_R64_UINT,
    }
}

fn astc_to_native(block: wgpu::AstcBlock, channel: wgpu::AstcChannel) -> native::WGPUTextureFormat {
    use wgpu::{AstcBlock as B, AstcChannel as C};
    match (block, channel) {
        (B::B4x4, C::Unorm) => native::WGPUTextureFormat_ASTC4x4Unorm,
        (B::B4x4, C::UnormSrgb) => native::WGPUTextureFormat_ASTC4x4UnormSrgb,
        (B::B5x4, C::Unorm) => native::WGPUTextureFormat_ASTC5x4Unorm,
        (B::B5x4, C::UnormSrgb) => native::WGPUTextureFormat_ASTC5x4UnormSrgb,
        (B::B5x5, C::Unorm) => native::WGPUTextureFormat_ASTC5x5Unorm,
        (B::B5x5, C::UnormSrgb) => native::WGPUTextureFormat_ASTC5x5UnormSrgb,
        (B::B6x5, C::Unorm) => native::WGPUTextureFormat_ASTC6x5Unorm,
        (B::B6x5, C::UnormSrgb) => native::WGPUTextureFormat_ASTC6x5UnormSrgb,
        (B::B6x6, C::Unorm) => native::WGPUTextureFormat_ASTC6x6Unorm,
        (B::B6x6, C::UnormSrgb) => native::WGPUTextureFormat_ASTC6x6UnormSrgb,
        (B::B8x5, C::Unorm) => native::WGPUTextureFormat_ASTC8x5Unorm,
        (B::B8x5, C::UnormSrgb) => native::WGPUTextureFormat_ASTC8x5UnormSrgb,
        (B::B8x6, C::Unorm) => native::WGPUTextureFormat_ASTC8x6Unorm,
        (B::B8x6, C::UnormSrgb) => native::WGPUTextureFormat_ASTC8x6UnormSrgb,
        (B::B8x8, C::Unorm) => native::WGPUTextureFormat_ASTC8x8Unorm,
        (B::B8x8, C::UnormSrgb) => native::WGPUTextureFormat_ASTC8x8UnormSrgb,
        (B::B10x5, C::Unorm) => native::WGPUTextureFormat_ASTC10x5Unorm,
        (B::B10x5, C::UnormSrgb) => native::WGPUTextureFormat_ASTC10x5UnormSrgb,
        (B::B10x6, C::Unorm) => native::WGPUTextureFormat_ASTC10x6Unorm,
        (B::B10x6, C::UnormSrgb) => native::WGPUTextureFormat_ASTC10x6UnormSrgb,
        (B::B10x8, C::Unorm) => native::WGPUTextureFormat_ASTC10x8Unorm,
        (B::B10x8, C::UnormSrgb) => native::WGPUTextureFormat_ASTC10x8UnormSrgb,
        (B::B10x10, C::Unorm) => native::WGPUTextureFormat_ASTC10x10Unorm,
        (B::B10x10, C::UnormSrgb) => native::WGPUTextureFormat_ASTC10x10UnormSrgb,
        (B::B12x10, C::Unorm) => native::WGPUTextureFormat_ASTC12x10Unorm,
        (B::B12x10, C::UnormSrgb) => native::WGPUTextureFormat_ASTC12x10UnormSrgb,
        (B::B12x12, C::Unorm) => native::WGPUTextureFormat_ASTC12x12Unorm,
        (B::B12x12, C::UnormSrgb) => native::WGPUTextureFormat_ASTC12x12UnormSrgb,
        (B::B4x4,  C::Hdr) => native::WGPUNativeTextureFormat_Astc4x4Sfloat,
        (B::B5x4,  C::Hdr) => native::WGPUNativeTextureFormat_Astc5x4Sfloat,
        (B::B5x5,  C::Hdr) => native::WGPUNativeTextureFormat_Astc5x5Sfloat,
        (B::B6x5,  C::Hdr) => native::WGPUNativeTextureFormat_Astc6x5Sfloat,
        (B::B6x6,  C::Hdr) => native::WGPUNativeTextureFormat_Astc6x6Sfloat,
        (B::B8x5,  C::Hdr) => native::WGPUNativeTextureFormat_Astc8x5Sfloat,
        (B::B8x6,  C::Hdr) => native::WGPUNativeTextureFormat_Astc8x6Sfloat,
        (B::B8x8,  C::Hdr) => native::WGPUNativeTextureFormat_Astc8x8Sfloat,
        (B::B10x5, C::Hdr) => native::WGPUNativeTextureFormat_Astc10x5Sfloat,
        (B::B10x6, C::Hdr) => native::WGPUNativeTextureFormat_Astc10x6Sfloat,
        (B::B10x8, C::Hdr) => native::WGPUNativeTextureFormat_Astc10x8Sfloat,
        (B::B10x10, C::Hdr) => native::WGPUNativeTextureFormat_Astc10x10Sfloat,
        (B::B12x10, C::Hdr) => native::WGPUNativeTextureFormat_Astc12x10Sfloat,
        (B::B12x12, C::Hdr) => native::WGPUNativeTextureFormat_Astc12x12Sfloat,
    }
}

// ── Usages ────────────────────────────────────────────────────────────────────

/// All wgpu::BufferUsages bits that buffer_usage_to_native knows how to map.
/// Any bits outside this set are unknown to the C API and will cause validation failure.
pub const KNOWN_BUFFER_USAGE_BITS: wgpu::BufferUsages = wgpu::BufferUsages::MAP_READ
    .union(wgpu::BufferUsages::MAP_WRITE)
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::INDEX)
    .union(wgpu::BufferUsages::VERTEX)
    .union(wgpu::BufferUsages::UNIFORM)
    .union(wgpu::BufferUsages::STORAGE)
    .union(wgpu::BufferUsages::INDIRECT)
    .union(wgpu::BufferUsages::QUERY_RESOLVE)
    .union(wgpu::BufferUsages::BLAS_INPUT)
    .union(wgpu::BufferUsages::TLAS_INPUT);

pub fn buffer_usage_to_native(u: wgpu::BufferUsages) -> native::WGPUBufferUsage {
    let mut out: native::WGPUBufferUsage = 0;
    if u.contains(wgpu::BufferUsages::MAP_READ) {
        out |= native::WGPUBufferUsage_MapRead;
    }
    if u.contains(wgpu::BufferUsages::MAP_WRITE) {
        out |= native::WGPUBufferUsage_MapWrite;
    }
    if u.contains(wgpu::BufferUsages::COPY_SRC) {
        out |= native::WGPUBufferUsage_CopySrc;
    }
    if u.contains(wgpu::BufferUsages::COPY_DST) {
        out |= native::WGPUBufferUsage_CopyDst;
    }
    if u.contains(wgpu::BufferUsages::INDEX) {
        out |= native::WGPUBufferUsage_Index;
    }
    if u.contains(wgpu::BufferUsages::VERTEX) {
        out |= native::WGPUBufferUsage_Vertex;
    }
    if u.contains(wgpu::BufferUsages::UNIFORM) {
        out |= native::WGPUBufferUsage_Uniform;
    }
    if u.contains(wgpu::BufferUsages::STORAGE) {
        out |= native::WGPUBufferUsage_Storage;
    }
    if u.contains(wgpu::BufferUsages::INDIRECT) {
        out |= native::WGPUBufferUsage_Indirect;
    }
    if u.contains(wgpu::BufferUsages::QUERY_RESOLVE) {
        out |= native::WGPUBufferUsage_QueryResolve;
    }
    // BLAS_INPUT and TLAS_INPUT share their bit values with wgpu-types, and wgpu-native
    // uses the same wgpu-types, so we pass the raw bits directly.
    if u.contains(wgpu::BufferUsages::BLAS_INPUT) {
        out |= wgpu::BufferUsages::BLAS_INPUT.bits() as native::WGPUBufferUsage;
    }
    if u.contains(wgpu::BufferUsages::TLAS_INPUT) {
        out |= wgpu::BufferUsages::TLAS_INPUT.bits() as native::WGPUBufferUsage;
    }
    out
}

pub fn texture_usage_to_native(u: wgpu::TextureUsages) -> native::WGPUTextureUsage {
    let mut out: native::WGPUTextureUsage = 0;
    if u.contains(wgpu::TextureUsages::COPY_SRC) {
        out |= native::WGPUTextureUsage_CopySrc;
    }
    if u.contains(wgpu::TextureUsages::COPY_DST) {
        out |= native::WGPUTextureUsage_CopyDst;
    }
    if u.contains(wgpu::TextureUsages::TEXTURE_BINDING) {
        out |= native::WGPUTextureUsage_TextureBinding;
    }
    if u.contains(wgpu::TextureUsages::STORAGE_BINDING) {
        out |= native::WGPUTextureUsage_StorageBinding;
    }
    if u.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
        out |= native::WGPUTextureUsage_RenderAttachment;
    }
    if u.contains(wgpu::TextureUsages::STORAGE_ATOMIC) {
        // Pass STORAGE_ATOMIC's raw bit (1 << 16) directly; wgpu-native's
        // from_u64_bits recognizes it as wgpu_types::TextureUsages::STORAGE_ATOMIC.
        out |= wgpu::TextureUsages::STORAGE_ATOMIC.bits() as native::WGPUTextureUsage;
    }
    out
}

pub fn map_texture_usage(u: native::WGPUTextureUsage) -> wgpu::TextureUsages {
    let mut out = wgpu::TextureUsages::empty();
    if (u & native::WGPUTextureUsage_CopySrc) != 0 {
        out |= wgpu::TextureUsages::COPY_SRC;
    }
    if (u & native::WGPUTextureUsage_CopyDst) != 0 {
        out |= wgpu::TextureUsages::COPY_DST;
    }
    if (u & native::WGPUTextureUsage_TextureBinding) != 0 {
        out |= wgpu::TextureUsages::TEXTURE_BINDING;
    }
    if (u & native::WGPUTextureUsage_StorageBinding) != 0 {
        out |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    if (u & native::WGPUTextureUsage_RenderAttachment) != 0 {
        out |= wgpu::TextureUsages::RENDER_ATTACHMENT;
    }
    out
}

// ── Misc enums ────────────────────────────────────────────────────────────────

pub fn power_preference_to_native(p: wgpu::PowerPreference) -> native::WGPUPowerPreference {
    match p {
        wgpu::PowerPreference::None => native::WGPUPowerPreference_Undefined,
        wgpu::PowerPreference::LowPower => native::WGPUPowerPreference_LowPower,
        wgpu::PowerPreference::HighPerformance => native::WGPUPowerPreference_HighPerformance,
    }
}

pub fn texture_dimension_to_native(d: wgpu::TextureDimension) -> native::WGPUTextureDimension {
    match d {
        wgpu::TextureDimension::D1 => native::WGPUTextureDimension_1D,
        wgpu::TextureDimension::D2 => native::WGPUTextureDimension_2D,
        wgpu::TextureDimension::D3 => native::WGPUTextureDimension_3D,
    }
}

pub fn map_texture_dimension(d: native::WGPUTextureDimension) -> wgpu::TextureDimension {
    match d {
        native::WGPUTextureDimension_1D => wgpu::TextureDimension::D1,
        native::WGPUTextureDimension_3D => wgpu::TextureDimension::D3,
        _ => wgpu::TextureDimension::D2,
    }
}

pub fn texture_view_dimension_to_native(
    d: wgpu::TextureViewDimension,
) -> native::WGPUTextureViewDimension {
    match d {
        wgpu::TextureViewDimension::D1 => native::WGPUTextureViewDimension_1D,
        wgpu::TextureViewDimension::D2 => native::WGPUTextureViewDimension_2D,
        wgpu::TextureViewDimension::D2Array => native::WGPUTextureViewDimension_2DArray,
        wgpu::TextureViewDimension::Cube => native::WGPUTextureViewDimension_Cube,
        wgpu::TextureViewDimension::CubeArray => native::WGPUTextureViewDimension_CubeArray,
        wgpu::TextureViewDimension::D3 => native::WGPUTextureViewDimension_3D,
    }
}

pub fn texture_aspect_to_native(a: wgpu::TextureAspect) -> native::WGPUTextureAspect {
    match a {
        wgpu::TextureAspect::All => native::WGPUTextureAspect_All,
        wgpu::TextureAspect::StencilOnly => native::WGPUTextureAspect_StencilOnly,
        wgpu::TextureAspect::DepthOnly => native::WGPUTextureAspect_DepthOnly,
        _ => native::WGPUTextureAspect_All,
    }
}

pub fn index_format_to_native(f: wgpu::IndexFormat) -> native::WGPUIndexFormat {
    match f {
        wgpu::IndexFormat::Uint16 => native::WGPUIndexFormat_Uint16,
        wgpu::IndexFormat::Uint32 => native::WGPUIndexFormat_Uint32,
    }
}

pub fn compare_function_to_native(c: wgpu::CompareFunction) -> native::WGPUCompareFunction {
    match c {
        wgpu::CompareFunction::Never => native::WGPUCompareFunction_Never,
        wgpu::CompareFunction::Less => native::WGPUCompareFunction_Less,
        wgpu::CompareFunction::Equal => native::WGPUCompareFunction_Equal,
        wgpu::CompareFunction::LessEqual => native::WGPUCompareFunction_LessEqual,
        wgpu::CompareFunction::Greater => native::WGPUCompareFunction_Greater,
        wgpu::CompareFunction::NotEqual => native::WGPUCompareFunction_NotEqual,
        wgpu::CompareFunction::GreaterEqual => native::WGPUCompareFunction_GreaterEqual,
        wgpu::CompareFunction::Always => native::WGPUCompareFunction_Always,
    }
}

pub fn stencil_op_to_native(op: wgpu::StencilOperation) -> native::WGPUStencilOperation {
    match op {
        wgpu::StencilOperation::Keep => native::WGPUStencilOperation_Keep,
        wgpu::StencilOperation::Zero => native::WGPUStencilOperation_Zero,
        wgpu::StencilOperation::Replace => native::WGPUStencilOperation_Replace,
        wgpu::StencilOperation::Invert => native::WGPUStencilOperation_Invert,
        wgpu::StencilOperation::IncrementClamp => native::WGPUStencilOperation_IncrementClamp,
        wgpu::StencilOperation::DecrementClamp => native::WGPUStencilOperation_DecrementClamp,
        wgpu::StencilOperation::IncrementWrap => native::WGPUStencilOperation_IncrementWrap,
        wgpu::StencilOperation::DecrementWrap => native::WGPUStencilOperation_DecrementWrap,
    }
}

pub fn blend_factor_to_native(f: wgpu::BlendFactor) -> native::WGPUBlendFactor {
    match f {
        wgpu::BlendFactor::Zero => native::WGPUBlendFactor_Zero,
        wgpu::BlendFactor::One => native::WGPUBlendFactor_One,
        wgpu::BlendFactor::Src => native::WGPUBlendFactor_Src,
        wgpu::BlendFactor::OneMinusSrc => native::WGPUBlendFactor_OneMinusSrc,
        wgpu::BlendFactor::SrcAlpha => native::WGPUBlendFactor_SrcAlpha,
        wgpu::BlendFactor::OneMinusSrcAlpha => native::WGPUBlendFactor_OneMinusSrcAlpha,
        wgpu::BlendFactor::Dst => native::WGPUBlendFactor_Dst,
        wgpu::BlendFactor::OneMinusDst => native::WGPUBlendFactor_OneMinusDst,
        wgpu::BlendFactor::DstAlpha => native::WGPUBlendFactor_DstAlpha,
        wgpu::BlendFactor::OneMinusDstAlpha => native::WGPUBlendFactor_OneMinusDstAlpha,
        wgpu::BlendFactor::SrcAlphaSaturated => native::WGPUBlendFactor_SrcAlphaSaturated,
        wgpu::BlendFactor::Constant => native::WGPUBlendFactor_Constant,
        wgpu::BlendFactor::OneMinusConstant => native::WGPUBlendFactor_OneMinusConstant,
        wgpu::BlendFactor::Src1 => native::WGPUBlendFactor_Src1,
        wgpu::BlendFactor::OneMinusSrc1 => native::WGPUBlendFactor_OneMinusSrc1,
        wgpu::BlendFactor::Src1Alpha => native::WGPUBlendFactor_Src1Alpha,
        wgpu::BlendFactor::OneMinusSrc1Alpha => native::WGPUBlendFactor_OneMinusSrc1Alpha,
    }
}

pub fn blend_op_to_native(op: wgpu::BlendOperation) -> native::WGPUBlendOperation {
    match op {
        wgpu::BlendOperation::Add => native::WGPUBlendOperation_Add,
        wgpu::BlendOperation::Subtract => native::WGPUBlendOperation_Subtract,
        wgpu::BlendOperation::ReverseSubtract => native::WGPUBlendOperation_ReverseSubtract,
        wgpu::BlendOperation::Min => native::WGPUBlendOperation_Min,
        wgpu::BlendOperation::Max => native::WGPUBlendOperation_Max,
    }
}

pub fn color_to_native(c: wgpu::Color) -> native::WGPUColor {
    native::WGPUColor {
        r: c.r,
        g: c.g,
        b: c.b,
        a: c.a,
    }
}

pub fn extent3d_to_native(e: wgpu::Extent3d) -> native::WGPUExtent3D {
    native::WGPUExtent3D {
        width: e.width,
        height: e.height,
        depthOrArrayLayers: e.depth_or_array_layers,
    }
}

pub fn origin3d_to_native(o: wgpu::Origin3d) -> native::WGPUOrigin3D {
    native::WGPUOrigin3D {
        x: o.x,
        y: o.y,
        z: o.z,
    }
}

pub fn error_filter_to_native(f: wgpu::ErrorFilter) -> native::WGPUErrorFilter {
    match f {
        wgpu::ErrorFilter::Validation => native::WGPUErrorFilter_Validation,
        wgpu::ErrorFilter::OutOfMemory => native::WGPUErrorFilter_OutOfMemory,
        wgpu::ErrorFilter::Internal => native::WGPUErrorFilter_Internal,
    }
}

pub fn map_present_mode(m: native::WGPUPresentMode) -> wgpu::PresentMode {
    match m {
        native::WGPUPresentMode_Fifo => wgpu::PresentMode::Fifo,
        native::WGPUPresentMode_FifoRelaxed => wgpu::PresentMode::FifoRelaxed,
        native::WGPUPresentMode_Immediate => wgpu::PresentMode::Immediate,
        native::WGPUPresentMode_Mailbox => wgpu::PresentMode::Mailbox,
        _ => wgpu::PresentMode::Fifo,
    }
}

pub fn present_mode_to_native(m: wgpu::PresentMode) -> Option<native::WGPUPresentMode> {
    match m {
        wgpu::PresentMode::Fifo => Some(native::WGPUPresentMode_Fifo),
        wgpu::PresentMode::FifoRelaxed => Some(native::WGPUPresentMode_FifoRelaxed),
        wgpu::PresentMode::Immediate => Some(native::WGPUPresentMode_Immediate),
        wgpu::PresentMode::Mailbox => Some(native::WGPUPresentMode_Mailbox),
        _ => None,
    }
}

pub fn composite_alpha_to_native(m: wgpu::CompositeAlphaMode) -> native::WGPUCompositeAlphaMode {
    match m {
        wgpu::CompositeAlphaMode::Auto => native::WGPUCompositeAlphaMode_Auto,
        wgpu::CompositeAlphaMode::Opaque => native::WGPUCompositeAlphaMode_Opaque,
        wgpu::CompositeAlphaMode::PreMultiplied => native::WGPUCompositeAlphaMode_Premultiplied,
        wgpu::CompositeAlphaMode::PostMultiplied => native::WGPUCompositeAlphaMode_Unpremultiplied,
        wgpu::CompositeAlphaMode::Inherit => native::WGPUCompositeAlphaMode_Inherit,
    }
}

pub fn map_composite_alpha(m: native::WGPUCompositeAlphaMode) -> wgpu::CompositeAlphaMode {
    match m {
        native::WGPUCompositeAlphaMode_Opaque => wgpu::CompositeAlphaMode::Opaque,
        native::WGPUCompositeAlphaMode_Premultiplied => wgpu::CompositeAlphaMode::PreMultiplied,
        native::WGPUCompositeAlphaMode_Unpremultiplied => wgpu::CompositeAlphaMode::PostMultiplied,
        native::WGPUCompositeAlphaMode_Inherit => wgpu::CompositeAlphaMode::Inherit,
        _ => wgpu::CompositeAlphaMode::Auto,
    }
}

pub fn primitive_topology_to_native(t: wgpu::PrimitiveTopology) -> native::WGPUPrimitiveTopology {
    match t {
        wgpu::PrimitiveTopology::PointList => native::WGPUPrimitiveTopology_PointList,
        wgpu::PrimitiveTopology::LineList => native::WGPUPrimitiveTopology_LineList,
        wgpu::PrimitiveTopology::LineStrip => native::WGPUPrimitiveTopology_LineStrip,
        wgpu::PrimitiveTopology::TriangleList => native::WGPUPrimitiveTopology_TriangleList,
        wgpu::PrimitiveTopology::TriangleStrip => native::WGPUPrimitiveTopology_TriangleStrip,
    }
}

pub fn front_face_to_native(f: wgpu::FrontFace) -> native::WGPUFrontFace {
    match f {
        wgpu::FrontFace::Ccw => native::WGPUFrontFace_CCW,
        wgpu::FrontFace::Cw => native::WGPUFrontFace_CW,
    }
}

pub fn cull_mode_to_native(c: Option<wgpu::Face>) -> native::WGPUCullMode {
    match c {
        None => native::WGPUCullMode_None,
        Some(wgpu::Face::Front) => native::WGPUCullMode_Front,
        Some(wgpu::Face::Back) => native::WGPUCullMode_Back,
    }
}

pub fn vertex_format_to_native(f: wgpu::VertexFormat) -> native::WGPUVertexFormat {
    match f {
        wgpu::VertexFormat::Uint8 => native::WGPUVertexFormat_Uint8,
        wgpu::VertexFormat::Uint8x2 => native::WGPUVertexFormat_Uint8x2,
        wgpu::VertexFormat::Uint8x4 => native::WGPUVertexFormat_Uint8x4,
        wgpu::VertexFormat::Sint8 => native::WGPUVertexFormat_Sint8,
        wgpu::VertexFormat::Sint8x2 => native::WGPUVertexFormat_Sint8x2,
        wgpu::VertexFormat::Sint8x4 => native::WGPUVertexFormat_Sint8x4,
        wgpu::VertexFormat::Unorm8 => native::WGPUVertexFormat_Unorm8,
        wgpu::VertexFormat::Unorm8x2 => native::WGPUVertexFormat_Unorm8x2,
        wgpu::VertexFormat::Unorm8x4 => native::WGPUVertexFormat_Unorm8x4,
        wgpu::VertexFormat::Snorm8 => native::WGPUVertexFormat_Snorm8,
        wgpu::VertexFormat::Snorm8x2 => native::WGPUVertexFormat_Snorm8x2,
        wgpu::VertexFormat::Snorm8x4 => native::WGPUVertexFormat_Snorm8x4,
        wgpu::VertexFormat::Uint16 => native::WGPUVertexFormat_Uint16,
        wgpu::VertexFormat::Uint16x2 => native::WGPUVertexFormat_Uint16x2,
        wgpu::VertexFormat::Uint16x4 => native::WGPUVertexFormat_Uint16x4,
        wgpu::VertexFormat::Sint16 => native::WGPUVertexFormat_Sint16,
        wgpu::VertexFormat::Sint16x2 => native::WGPUVertexFormat_Sint16x2,
        wgpu::VertexFormat::Sint16x4 => native::WGPUVertexFormat_Sint16x4,
        wgpu::VertexFormat::Unorm16 => native::WGPUVertexFormat_Unorm16,
        wgpu::VertexFormat::Unorm16x2 => native::WGPUVertexFormat_Unorm16x2,
        wgpu::VertexFormat::Unorm16x4 => native::WGPUVertexFormat_Unorm16x4,
        wgpu::VertexFormat::Snorm16 => native::WGPUVertexFormat_Snorm16,
        wgpu::VertexFormat::Snorm16x2 => native::WGPUVertexFormat_Snorm16x2,
        wgpu::VertexFormat::Snorm16x4 => native::WGPUVertexFormat_Snorm16x4,
        wgpu::VertexFormat::Float16 => native::WGPUVertexFormat_Float16,
        wgpu::VertexFormat::Float16x2 => native::WGPUVertexFormat_Float16x2,
        wgpu::VertexFormat::Float16x4 => native::WGPUVertexFormat_Float16x4,
        wgpu::VertexFormat::Float32 => native::WGPUVertexFormat_Float32,
        wgpu::VertexFormat::Float32x2 => native::WGPUVertexFormat_Float32x2,
        wgpu::VertexFormat::Float32x3 => native::WGPUVertexFormat_Float32x3,
        wgpu::VertexFormat::Float32x4 => native::WGPUVertexFormat_Float32x4,
        wgpu::VertexFormat::Uint32 => native::WGPUVertexFormat_Uint32,
        wgpu::VertexFormat::Uint32x2 => native::WGPUVertexFormat_Uint32x2,
        wgpu::VertexFormat::Uint32x3 => native::WGPUVertexFormat_Uint32x3,
        wgpu::VertexFormat::Uint32x4 => native::WGPUVertexFormat_Uint32x4,
        wgpu::VertexFormat::Sint32 => native::WGPUVertexFormat_Sint32,
        wgpu::VertexFormat::Sint32x2 => native::WGPUVertexFormat_Sint32x2,
        wgpu::VertexFormat::Sint32x3 => native::WGPUVertexFormat_Sint32x3,
        wgpu::VertexFormat::Sint32x4 => native::WGPUVertexFormat_Sint32x4,
        wgpu::VertexFormat::Unorm10_10_10_2 => native::WGPUVertexFormat_Unorm10_10_10_2,
        wgpu::VertexFormat::Unorm8x4Bgra => native::WGPUVertexFormat_Unorm8x4BGRA,
        wgpu::VertexFormat::Float64
        | wgpu::VertexFormat::Float64x2
        | wgpu::VertexFormat::Float64x3
        | wgpu::VertexFormat::Float64x4 => {
            panic!("Float64 vertex formats are not supported by WebGPU")
        }
    }
}

pub fn vertex_step_mode_to_native(m: wgpu::VertexStepMode) -> native::WGPUVertexStepMode {
    match m {
        wgpu::VertexStepMode::Vertex => native::WGPUVertexStepMode_Vertex,
        wgpu::VertexStepMode::Instance => native::WGPUVertexStepMode_Instance,
    }
}

pub fn address_mode_to_native(m: wgpu::AddressMode) -> native::WGPUAddressMode {
    match m {
        wgpu::AddressMode::ClampToEdge => native::WGPUAddressMode_ClampToEdge,
        wgpu::AddressMode::Repeat => native::WGPUAddressMode_Repeat,
        wgpu::AddressMode::MirrorRepeat => native::WGPUAddressMode_MirrorRepeat,
        wgpu::AddressMode::ClampToBorder => {
            native::WGPUNativeAddressMode_ClampToBorder as native::WGPUAddressMode
        }
    }
}

pub fn border_color_to_native(c: wgpu::SamplerBorderColor) -> native::WGPUSamplerBorderColor {
    match c {
        wgpu::SamplerBorderColor::TransparentBlack => {
            native::WGPUSamplerBorderColor_TransparentBlack
        }
        wgpu::SamplerBorderColor::OpaqueBlack => native::WGPUSamplerBorderColor_OpaqueBlack,
        wgpu::SamplerBorderColor::OpaqueWhite => native::WGPUSamplerBorderColor_OpaqueWhite,
        wgpu::SamplerBorderColor::Zero => native::WGPUSamplerBorderColor_Zero,
    }
}

pub fn memory_hints_to_native(
    hints: &wgpu::MemoryHints,
) -> (native::WGPUMemoryHints, u64, u64) {
    match hints {
        wgpu::MemoryHints::Performance => (native::WGPUMemoryHints_Performance, 0, 0),
        wgpu::MemoryHints::MemoryUsage => (native::WGPUMemoryHints_MemoryUsage, 0, 0),
        wgpu::MemoryHints::Manual { suballocated_device_memory_block_size } => (
            native::WGPUMemoryHints_Manual,
            suballocated_device_memory_block_size.start,
            suballocated_device_memory_block_size.end,
        ),
    }
}

pub fn filter_mode_to_native(m: wgpu::FilterMode) -> native::WGPUFilterMode {
    match m {
        wgpu::FilterMode::Nearest => native::WGPUFilterMode_Nearest,
        wgpu::FilterMode::Linear => native::WGPUFilterMode_Linear,
    }
}

pub fn mipmap_filter_to_native(m: wgpu::MipmapFilterMode) -> native::WGPUMipmapFilterMode {
    match m {
        wgpu::MipmapFilterMode::Nearest => native::WGPUMipmapFilterMode_Nearest,
        wgpu::MipmapFilterMode::Linear => native::WGPUMipmapFilterMode_Linear,
    }
}

pub fn query_type_to_native(t: wgpu::QueryType) -> native::WGPUQueryType {
    match t {
        wgpu::QueryType::Occlusion => native::WGPUQueryType_Occlusion,
        wgpu::QueryType::Timestamp => native::WGPUQueryType_Timestamp,
        wgpu::QueryType::PipelineStatistics(_) => {
            native::WGPUNativeQueryType_PipelineStatistics as native::WGPUQueryType
        }
    }
}

pub fn image_copy_texture_to_native(
    ict: &wgpu::TexelCopyTextureInfo,
    tex_ptr: native::WGPUTexture,
) -> native::WGPUTexelCopyTextureInfo {
    native::WGPUTexelCopyTextureInfo {
        texture: tex_ptr,
        mipLevel: ict.mip_level,
        origin: origin3d_to_native(ict.origin),
        aspect: texture_aspect_to_native(ict.aspect),
    }
}

pub fn image_copy_buffer_to_native(
    icb: &wgpu::TexelCopyBufferInfo,
    buf_ptr: native::WGPUBuffer,
) -> native::WGPUTexelCopyBufferInfo {
    native::WGPUTexelCopyBufferInfo {
        layout: native::WGPUTexelCopyBufferLayout {
            offset: icb.layout.offset,
            bytesPerRow: icb.layout.bytes_per_row.unwrap_or(native::WGPU_COPY_STRIDE_UNDEFINED),
            rowsPerImage: icb.layout.rows_per_image.unwrap_or(native::WGPU_COPY_STRIDE_UNDEFINED),
        },
        buffer: buf_ptr,
    }
}

// ── Shader ────────────────────────────────────────────────────────────────────

pub fn shader_stages_to_native(s: wgpu::ShaderStages) -> native::WGPUShaderStage {
    let mut out: native::WGPUShaderStage = 0;
    if s.contains(wgpu::ShaderStages::VERTEX) {
        out |= native::WGPUShaderStage_Vertex;
    }
    if s.contains(wgpu::ShaderStages::FRAGMENT) {
        out |= native::WGPUShaderStage_Fragment;
    }
    if s.contains(wgpu::ShaderStages::COMPUTE) {
        out |= native::WGPUShaderStage_Compute;
    }
    out
}

// ── Bind group layout ─────────────────────────────────────────────────────────

pub fn buffer_binding_type_to_native(t: wgpu::BufferBindingType) -> native::WGPUBufferBindingType {
    match t {
        wgpu::BufferBindingType::Uniform => native::WGPUBufferBindingType_Uniform,
        wgpu::BufferBindingType::Storage { read_only: false } => {
            native::WGPUBufferBindingType_Storage
        }
        wgpu::BufferBindingType::Storage { read_only: true } => {
            native::WGPUBufferBindingType_ReadOnlyStorage
        }
    }
}

pub fn sampler_binding_type_to_native(
    t: wgpu::SamplerBindingType,
) -> native::WGPUSamplerBindingType {
    match t {
        wgpu::SamplerBindingType::Filtering => native::WGPUSamplerBindingType_Filtering,
        wgpu::SamplerBindingType::NonFiltering => native::WGPUSamplerBindingType_NonFiltering,
        wgpu::SamplerBindingType::Comparison => native::WGPUSamplerBindingType_Comparison,
    }
}

pub fn texture_sample_type_to_native(t: wgpu::TextureSampleType) -> native::WGPUTextureSampleType {
    match t {
        wgpu::TextureSampleType::Float { filterable: true } => native::WGPUTextureSampleType_Float,
        wgpu::TextureSampleType::Float { filterable: false } => {
            native::WGPUTextureSampleType_UnfilterableFloat
        }
        wgpu::TextureSampleType::Depth => native::WGPUTextureSampleType_Depth,
        wgpu::TextureSampleType::Sint => native::WGPUTextureSampleType_Sint,
        wgpu::TextureSampleType::Uint => native::WGPUTextureSampleType_Uint,
    }
}

pub fn storage_texture_access_to_native(
    a: wgpu::StorageTextureAccess,
) -> native::WGPUStorageTextureAccess {
    match a {
        wgpu::StorageTextureAccess::WriteOnly => native::WGPUStorageTextureAccess_WriteOnly,
        wgpu::StorageTextureAccess::ReadOnly => native::WGPUStorageTextureAccess_ReadOnly,
        wgpu::StorageTextureAccess::ReadWrite => native::WGPUStorageTextureAccess_ReadWrite,
        wgpu::StorageTextureAccess::Atomic => wgpu_native::conv::WGPU_NATIVE_STORAGE_TEXTURE_ACCESS_ATOMIC,
    }
}

// ── Color writes / load-store ops ─────────────────────────────────────────────

pub fn color_writes_to_native(w: wgpu::ColorWrites) -> native::WGPUColorWriteMask {
    let mut out: native::WGPUColorWriteMask = 0;
    if w.contains(wgpu::ColorWrites::RED) {
        out |= native::WGPUColorWriteMask_Red;
    }
    if w.contains(wgpu::ColorWrites::GREEN) {
        out |= native::WGPUColorWriteMask_Green;
    }
    if w.contains(wgpu::ColorWrites::BLUE) {
        out |= native::WGPUColorWriteMask_Blue;
    }
    if w.contains(wgpu::ColorWrites::ALPHA) {
        out |= native::WGPUColorWriteMask_Alpha;
    }
    out
}

pub fn load_op_color_to_native(
    op: &wgpu::LoadOp<wgpu::Color>,
) -> (native::WGPULoadOp, native::WGPUColor) {
    match op {
        wgpu::LoadOp::Load | wgpu::LoadOp::DontCare(_) => (
            native::WGPULoadOp_Load,
            native::WGPUColor {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
        ),
        wgpu::LoadOp::Clear(c) => (native::WGPULoadOp_Clear, color_to_native(*c)),
    }
}

pub fn load_op_f32_to_native(op: &wgpu::Operations<f32>) -> (native::WGPULoadOp, f32) {
    match op.load {
        wgpu::LoadOp::Load | wgpu::LoadOp::DontCare(_) => (native::WGPULoadOp_Load, f32::NAN),
        wgpu::LoadOp::Clear(v) => (native::WGPULoadOp_Clear, v),
    }
}

pub fn load_op_u32_to_native(op: &wgpu::Operations<u32>) -> (native::WGPULoadOp, u32) {
    match op.load {
        wgpu::LoadOp::Load | wgpu::LoadOp::DontCare(_) => (native::WGPULoadOp_Load, 0),
        wgpu::LoadOp::Clear(v) => (native::WGPULoadOp_Clear, v),
    }
}

pub fn store_op_to_native(op: wgpu::StoreOp) -> native::WGPUStoreOp {
    match op {
        wgpu::StoreOp::Store => native::WGPUStoreOp_Store,
        wgpu::StoreOp::Discard => native::WGPUStoreOp_Discard,
    }
}

// ── Polygon mode ──────────────────────────────────────────────────────────────

pub fn polygon_mode_to_native(m: wgpu::PolygonMode) -> native::WGPUPolygonMode {
    match m {
        wgpu::PolygonMode::Fill => native::WGPUPolygonMode_Fill,
        wgpu::PolygonMode::Line => native::WGPUPolygonMode_Line,
        wgpu::PolygonMode::Point => native::WGPUPolygonMode_Point,
    }
}

// ── Optional bool ─────────────────────────────────────────────────────────────

pub fn bool_to_optional_bool(b: bool) -> native::WGPUOptionalBool {
    if b {
        native::WGPUOptionalBool_True
    } else {
        native::WGPUOptionalBool_False
    }
}

// ── Surface status ────────────────────────────────────────────────────────────

pub fn surface_status_from_native(
    s: native::WGPUSurfaceGetCurrentTextureStatus,
) -> wgpu::SurfaceStatus {
    match s {
        native::WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal => wgpu::SurfaceStatus::Good,
        native::WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal => {
            wgpu::SurfaceStatus::Suboptimal
        }
        native::WGPUSurfaceGetCurrentTextureStatus_Timeout => wgpu::SurfaceStatus::Timeout,
        native::WGPUSurfaceGetCurrentTextureStatus_Outdated => wgpu::SurfaceStatus::Outdated,
        native::WGPUSurfaceGetCurrentTextureStatus_Lost => wgpu::SurfaceStatus::Lost,
        _ => wgpu::SurfaceStatus::Lost,
    }
}

pub fn pipeline_statistics_to_native(
    flags: wgpu::PipelineStatisticsTypes,
) -> Vec<native::WGPUPipelineStatisticName> {
    let mut out = Vec::new();
    if flags.contains(wgpu::PipelineStatisticsTypes::VERTEX_SHADER_INVOCATIONS) {
        out.push(native::WGPUPipelineStatisticName_VertexShaderInvocations);
    }
    if flags.contains(wgpu::PipelineStatisticsTypes::CLIPPER_INVOCATIONS) {
        out.push(native::WGPUPipelineStatisticName_ClipperInvocations);
    }
    if flags.contains(wgpu::PipelineStatisticsTypes::CLIPPER_PRIMITIVES_OUT) {
        out.push(native::WGPUPipelineStatisticName_ClipperPrimitivesOut);
    }
    if flags.contains(wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS) {
        out.push(native::WGPUPipelineStatisticName_FragmentShaderInvocations);
    }
    if flags.contains(wgpu::PipelineStatisticsTypes::COMPUTE_SHADER_INVOCATIONS) {
        out.push(native::WGPUPipelineStatisticName_ComputeShaderInvocations);
    }
    out
}

pub fn cooperative_scalar_type_from_native(
    t: native::WGPUNativeCooperativeScalarType,
) -> wgpu::wgt::CooperativeScalarType {
    match t {
        native::WGPUNativeCooperativeScalarType_F16 => wgpu::wgt::CooperativeScalarType::F16,
        native::WGPUNativeCooperativeScalarType_I32 => wgpu::wgt::CooperativeScalarType::I32,
        native::WGPUNativeCooperativeScalarType_U32 => wgpu::wgt::CooperativeScalarType::U32,
        _ => wgpu::wgt::CooperativeScalarType::F32,
    }
}

pub fn acceleration_structure_flags_to_native(
    f: wgpu::AccelerationStructureFlags,
) -> native::WGPUAccelerationStructureFlags {
    let mut out = native::WGPUAccelerationStructureFlags_None;
    if f.contains(wgpu::AccelerationStructureFlags::ALLOW_UPDATE) {
        out |= native::WGPUAccelerationStructureFlags_AllowUpdate;
    }
    if f.contains(wgpu::AccelerationStructureFlags::ALLOW_COMPACTION) {
        out |= native::WGPUAccelerationStructureFlags_AllowCompaction;
    }
    if f.contains(wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE) {
        out |= native::WGPUAccelerationStructureFlags_PreferFastTrace;
    }
    if f.contains(wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD) {
        out |= native::WGPUAccelerationStructureFlags_PreferFastBuild;
    }
    if f.contains(wgpu::AccelerationStructureFlags::LOW_MEMORY) {
        out |= native::WGPUAccelerationStructureFlags_LowMemory;
    }
    if f.contains(wgpu::AccelerationStructureFlags::USE_TRANSFORM) {
        out |= native::WGPUAccelerationStructureFlags_UseTransform;
    }
    if f.contains(wgpu::AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN) {
        out |= native::WGPUAccelerationStructureFlags_AllowRayHitVertexReturn;
    }
    out
}

pub fn acceleration_structure_update_mode_to_native(
    m: wgpu::AccelerationStructureUpdateMode,
) -> native::WGPUAccelerationStructureUpdateMode {
    match m {
        wgpu::AccelerationStructureUpdateMode::Build => {
            native::WGPUAccelerationStructureUpdateMode_Build
        }
        wgpu::AccelerationStructureUpdateMode::PreferUpdate => {
            native::WGPUAccelerationStructureUpdateMode_PreferUpdate
        }
    }
}

pub fn acceleration_structure_geometry_flags_to_native(
    f: wgpu::AccelerationStructureGeometryFlags,
) -> native::WGPUAccelerationStructureGeometryFlags {
    let mut out = native::WGPUAccelerationStructureGeometryFlags_None;
    if f.contains(wgpu::AccelerationStructureGeometryFlags::OPAQUE) {
        out |= native::WGPUAccelerationStructureGeometryFlags_Opaque;
    }
    if f.contains(wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION) {
        out |= native::WGPUAccelerationStructureGeometryFlags_NoDuplicateAnyHitInvocation;
    }
    out
}
