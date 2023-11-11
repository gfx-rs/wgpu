use crate::com::ComPtr;
use winapi::um::d3d12;

#[repr(u32)]
#[derive(Debug, Copy, Clone)]
pub enum QueryHeapType {
    Occlusion = d3d12::D3D12_QUERY_HEAP_TYPE_OCCLUSION,
    Timestamp = d3d12::D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
    PipelineStatistics = d3d12::D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
    SOStatistics = d3d12::D3D12_QUERY_HEAP_TYPE_SO_STATISTICS,
    // VideoDecodeStatistcs = d3d12::D3D12_QUERY_HEAP_TYPE_VIDEO_DECODE_STATISTICS,
    // CopyQueueTimestamp = d3d12::D3D12_QUERY_HEAP_TYPE_COPY_QUEUE_TIMESTAMP,
}

pub type QueryHeap = ComPtr<d3d12::ID3D12QueryHeap>;
