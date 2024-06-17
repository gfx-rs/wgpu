#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// use here so that the recursive RIDL macro can find the crate
use winapi::um::unknwnbase::{IUnknown, IUnknownVtbl};
use winapi::RIDL;

RIDL! {#[uuid(0x63aad0b8, 0x7c24, 0x40ff, 0x85, 0xa8, 0x64, 0x0d, 0x94, 0x4c, 0xc3, 0x25)]
interface ISwapChainPanelNative(ISwapChainPanelNativeVtbl): IUnknown(IUnknownVtbl) {
    fn SetSwapChain(swapChain: *const winapi::shared::dxgi1_2::IDXGISwapChain1,) -> winapi::um::winnt::HRESULT,
}}

winapi::ENUM! {
    enum D3D12_VIEW_INSTANCING_TIER {
        D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED  = 0,
        D3D12_VIEW_INSTANCING_TIER_1 = 1,
        D3D12_VIEW_INSTANCING_TIER_2 = 2,
        D3D12_VIEW_INSTANCING_TIER_3 = 3,
    }
}

winapi::ENUM! {
    enum D3D12_COMMAND_LIST_SUPPORT_FLAGS {
        D3D12_COMMAND_LIST_SUPPORT_FLAG_NONE = 0,
        // D3D12_COMMAND_LIST_SUPPORT_FLAG_DIRECT,
        // D3D12_COMMAND_LIST_SUPPORT_FLAG_BUNDLE,
        // D3D12_COMMAND_LIST_SUPPORT_FLAG_COMPUTE,
        // D3D12_COMMAND_LIST_SUPPORT_FLAG_COPY,
        // D3D12_COMMAND_LIST_SUPPORT_FLAG_VIDEO_DECODE,
        // D3D12_COMMAND_LIST_SUPPORT_FLAG_VIDEO_PROCESS,
        // D3D12_COMMAND_LIST_SUPPORT_FLAG_VIDEO_ENCODE,
    }
}

winapi::STRUCT! {
    struct D3D12_FEATURE_DATA_D3D12_OPTIONS3 {
        CopyQueueTimestampQueriesSupported: winapi::shared::minwindef::BOOL,
        CastingFullyTypedFormatSupported: winapi::shared::minwindef::BOOL,
        WriteBufferImmediateSupportFlags: D3D12_COMMAND_LIST_SUPPORT_FLAGS,
        ViewInstancingTier: D3D12_VIEW_INSTANCING_TIER,
        BarycentricsSupported: winapi::shared::minwindef::BOOL,
    }
}

winapi::ENUM! {
    enum D3D12_WAVE_MMA_TIER  {
        D3D12_WAVE_MMA_TIER_NOT_SUPPORTED = 0,
        D3D12_WAVE_MMA_TIER_1_0 = 10,
    }
}

winapi::STRUCT! {
    struct D3D12_FEATURE_DATA_D3D12_OPTIONS9 {
        MeshShaderPipelineStatsSupported: winapi::shared::minwindef::BOOL,
        MeshShaderSupportsFullRangeRenderTargetArrayIndex: winapi::shared::minwindef::BOOL,
        AtomicInt64OnTypedResourceSupported: winapi::shared::minwindef::BOOL,
        AtomicInt64OnGroupSharedSupported: winapi::shared::minwindef::BOOL,
        DerivativesInMeshAndAmplificationShadersSupported: winapi::shared::minwindef::BOOL,
        WaveMMATier: D3D12_WAVE_MMA_TIER,
    }
}

winapi::ENUM! {
    enum D3D_SHADER_MODEL {
        D3D_SHADER_MODEL_NONE = 0,
        D3D_SHADER_MODEL_5_1 = 0x51,
        D3D_SHADER_MODEL_6_0 = 0x60,
        D3D_SHADER_MODEL_6_1 = 0x61,
        D3D_SHADER_MODEL_6_2 = 0x62,
        D3D_SHADER_MODEL_6_3 = 0x63,
        D3D_SHADER_MODEL_6_4 = 0x64,
        D3D_SHADER_MODEL_6_5 = 0x65,
        D3D_SHADER_MODEL_6_6 = 0x66,
        D3D_SHADER_MODEL_6_7 = 0x67,
        D3D_HIGHEST_SHADER_MODEL = 0x67,
    }
}

winapi::STRUCT! {
    struct D3D12_FEATURE_DATA_SHADER_MODEL {
        HighestShaderModel: D3D_SHADER_MODEL,
    }
}
