use crate::com::ComPtr;
use winapi::um::d3d12;

pub type Heap = ComPtr<d3d12::ID3D12Heap>;

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum HeapType {
    Default = d3d12::D3D12_HEAP_TYPE_DEFAULT,
    Upload = d3d12::D3D12_HEAP_TYPE_UPLOAD,
    Readback = d3d12::D3D12_HEAP_TYPE_READBACK,
    Custom = d3d12::D3D12_HEAP_TYPE_CUSTOM,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum CpuPageProperty {
    Unknown = d3d12::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    NotAvailable = d3d12::D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
    WriteCombine = d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
    WriteBack = d3d12::D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
}

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum MemoryPool {
    Unknown = d3d12::D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
    L0 = d3d12::D3D12_MEMORY_POOL_L0,
    L1 = d3d12::D3D12_MEMORY_POOL_L1,
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
    pub struct HeapFlags: u32 {
        const NONE = d3d12::D3D12_HEAP_FLAG_NONE;
        const SHARED = d3d12::D3D12_HEAP_FLAG_SHARED;
        const DENY_BUFFERS = d3d12::D3D12_HEAP_FLAG_DENY_BUFFERS;
        const ALLOW_DISPLAY = d3d12::D3D12_HEAP_FLAG_ALLOW_DISPLAY;
        const SHARED_CROSS_ADAPTER = d3d12::D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
        const DENT_RT_DS_TEXTURES = d3d12::D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES;
        const DENY_NON_RT_DS_TEXTURES = d3d12::D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;
        const HARDWARE_PROTECTED = d3d12::D3D12_HEAP_FLAG_HARDWARE_PROTECTED;
        const ALLOW_WRITE_WATCH = d3d12::D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH;
        const ALLOW_ALL_BUFFERS_AND_TEXTURES = d3d12::D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES;
        const ALLOW_ONLY_BUFFERS = d3d12::D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        const ALLOW_ONLY_NON_RT_DS_TEXTURES = d3d12::D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        const ALLOW_ONLY_RT_DS_TEXTURES = d3d12::D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
    }
}

#[repr(transparent)]
pub struct HeapProperties(pub d3d12::D3D12_HEAP_PROPERTIES);
impl HeapProperties {
    pub fn new(
        heap_type: HeapType,
        cpu_page_property: CpuPageProperty,
        memory_pool_preference: MemoryPool,
        creation_node_mask: u32,
        visible_node_mask: u32,
    ) -> Self {
        HeapProperties(d3d12::D3D12_HEAP_PROPERTIES {
            Type: heap_type as _,
            CPUPageProperty: cpu_page_property as _,
            MemoryPoolPreference: memory_pool_preference as _,
            CreationNodeMask: creation_node_mask,
            VisibleNodeMask: visible_node_mask,
        })
    }
}

#[repr(transparent)]
pub struct HeapDesc(d3d12::D3D12_HEAP_DESC);
impl HeapDesc {
    pub fn new(
        size_in_bytes: u64,
        properties: HeapProperties,
        alignment: u64,
        flags: HeapFlags,
    ) -> Self {
        HeapDesc(d3d12::D3D12_HEAP_DESC {
            SizeInBytes: size_in_bytes,
            Properties: properties.0,
            Alignment: alignment,
            Flags: flags.bits(),
        })
    }
}
