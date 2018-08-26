//! Graphics command list

use com::WeakPtr;
use resource::DiscardRegion;
use std::mem;
use winapi::um::d3d12;
use {
    CommandAllocator, CpuDescriptor, DescriptorHeap, Format, GpuAddress, GpuDescriptor, IndexCount,
    InstanceCount, PipelineState, Rect, Resource, RootSignature, VertexCount, VertexOffset,
    WorkGroupCount, HRESULT,
};

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum CmdListType {
    Direct = d3d12::D3D12_COMMAND_LIST_TYPE_DIRECT,
    Bundle = d3d12::D3D12_COMMAND_LIST_TYPE_BUNDLE,
    Compute = d3d12::D3D12_COMMAND_LIST_TYPE_COMPUTE,
    Copy = d3d12::D3D12_COMMAND_LIST_TYPE_COPY,
    // VideoDecode = d3d12::D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE,
    // VideoProcess = d3d12::D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
}

bitflags! {
    pub struct ClearFlags: u32 {
        const DEPTH = d3d12::D3D12_CLEAR_FLAG_DEPTH;
        const STENCIL = d3d12::D3D12_CLEAR_FLAG_STENCIL;
    }
}

#[repr(transparent)]
pub struct IndirectArgument(d3d12::D3D12_INDIRECT_ARGUMENT_DESC);

impl IndirectArgument {
    pub fn draw() -> Self {
        IndirectArgument(d3d12::D3D12_INDIRECT_ARGUMENT_DESC {
            Type: d3d12::D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
            ..unsafe { mem::zeroed() }
        })
    }

    pub fn draw_indexed() -> Self {
        IndirectArgument(d3d12::D3D12_INDIRECT_ARGUMENT_DESC {
            Type: d3d12::D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            ..unsafe { mem::zeroed() }
        })
    }

    pub fn dispatch() -> Self {
        IndirectArgument(d3d12::D3D12_INDIRECT_ARGUMENT_DESC {
            Type: d3d12::D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..unsafe { mem::zeroed() }
        })
    }

    // TODO: missing variants
}

pub type CommandSignature = WeakPtr<d3d12::ID3D12CommandSignature>;
pub type GraphicsCommandList = WeakPtr<d3d12::ID3D12GraphicsCommandList>;

impl GraphicsCommandList {
    pub fn close(&self) -> HRESULT {
        unsafe { self.Close() }
    }

    pub fn reset(&self, allocator: CommandAllocator, initial_pso: PipelineState) -> HRESULT {
        unsafe { self.Reset(allocator.as_mut_ptr(), initial_pso.as_mut_ptr()) }
    }

    pub fn discard_resource(&self, resource: Resource, region: DiscardRegion) {
        debug_assert!(region.subregions.start < region.subregions.end);
        unsafe {
            self.DiscardResource(
                resource.as_mut_ptr(),
                &d3d12::D3D12_DISCARD_REGION {
                    NumRects: region.rects.len() as _,
                    pRects: region.rects.as_ptr(),
                    FirstSubresource: region.subregions.start,
                    NumSubresources: region.subregions.end - region.subregions.start - 1,
                },
            );
        }
    }

    pub fn clear_depth_stencil_view(
        &self,
        dsv: CpuDescriptor,
        flags: ClearFlags,
        depth: f32,
        stencil: u8,
        rects: &[Rect],
    ) {
        unsafe {
            self.ClearDepthStencilView(
                dsv,
                flags.bits(),
                depth,
                stencil,
                rects.len() as _,
                rects.as_ptr(),
            );
        }
    }

    pub fn dispatch(&self, count: WorkGroupCount) {
        unsafe {
            self.Dispatch(count[0], count[1], count[2]);
        }
    }

    pub fn draw(
        &self,
        num_vertices: VertexCount,
        num_instances: InstanceCount,
        start_vertex: VertexCount,
        start_instance: InstanceCount,
    ) {
        unsafe {
            self.DrawInstanced(num_vertices, num_instances, start_vertex, start_instance);
        }
    }

    pub fn draw_indexed(
        &self,
        num_indices: IndexCount,
        num_instances: InstanceCount,
        start_index: IndexCount,
        base_vertex: VertexOffset,
        start_instance: InstanceCount,
    ) {
        unsafe {
            self.DrawIndexedInstanced(
                num_indices,
                num_instances,
                start_index,
                base_vertex,
                start_instance,
            );
        }
    }

    pub fn set_index_buffer(&self, gpu_address: GpuAddress, size: u32, format: Format) {
        let mut ibv = d3d12::D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: gpu_address,
            SizeInBytes: size,
            Format: format,
        };
        unsafe {
            self.IASetIndexBuffer(&mut ibv);
        }
    }

    pub fn set_blend_factor(&self, factor: [f32; 4]) {
        unsafe {
            self.OMSetBlendFactor(&factor);
        }
    }

    pub fn set_stencil_reference(&self, reference: u32) {
        unsafe {
            self.OMSetStencilRef(reference);
        }
    }

    pub fn set_pipeline_state(&self, pso: PipelineState) {
        unsafe {
            self.SetPipelineState(pso.as_mut_ptr());
        }
    }

    pub fn execute_bundle(&self, bundle: GraphicsCommandList) {
        unsafe {
            self.ExecuteBundle(bundle.as_mut_ptr());
        }
    }

    pub fn set_descriptor_heaps(&self, heaps: &[DescriptorHeap]) {
        unsafe {
            self.SetDescriptorHeaps(
                heaps.len() as _,
                heaps.as_ptr() as *mut &DescriptorHeap as *mut _,
            );
        }
    }

    pub fn set_compute_root_signature(&self, signature: RootSignature) {
        unsafe {
            self.SetComputeRootSignature(signature.as_mut_ptr());
        }
    }

    pub fn set_graphics_root_signature(&self, signature: RootSignature) {
        unsafe {
            self.SetGraphicsRootSignature(signature.as_mut_ptr());
        }
    }

    pub fn set_compute_root_descriptor_table(
        &self,
        root_index: u32,
        base_descriptor: GpuDescriptor,
    ) {
        unsafe {
            self.SetComputeRootDescriptorTable(root_index, base_descriptor);
        }
    }

    pub fn set_graphics_root_descriptor_table(
        &self,
        root_index: u32,
        base_descriptor: GpuDescriptor,
    ) {
        unsafe {
            self.SetGraphicsRootDescriptorTable(root_index, base_descriptor);
        }
    }
}
