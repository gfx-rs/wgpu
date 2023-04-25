#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

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
