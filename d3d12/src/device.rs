//! Device

use crate::{
    com::ComPtr,
    command_list::{CmdListType, CommandSignature, IndirectArgument},
    descriptor::{CpuDescriptor, DescriptorHeapFlags, DescriptorHeapType, RenderTargetViewDesc},
    heap::{Heap, HeapFlags, HeapProperties},
    pso, query, queue, Blob, CachedPSO, CommandAllocator, CommandQueue, D3DResult, DescriptorHeap,
    Fence, GraphicsCommandList, NodeMask, PipelineState, QueryHeap, RootSignature, Shader,
    TextureAddressMode,
};
use std::ops::Range;
use winapi::{um::d3d12, Interface};

pub type Device = ComPtr<d3d12::ID3D12Device>;

#[cfg(feature = "libloading")]
impl crate::D3D12Lib {
    pub fn create_device<I: Interface>(
        &self,
        adapter: &ComPtr<I>,
        feature_level: crate::FeatureLevel,
    ) -> Result<D3DResult<Device>, libloading::Error> {
        type Fun = extern "system" fn(
            *mut winapi::um::unknwnbase::IUnknown,
            winapi::um::d3dcommon::D3D_FEATURE_LEVEL,
            winapi::shared::guiddef::REFGUID,
            *mut *mut winapi::ctypes::c_void,
        ) -> crate::HRESULT;

        let mut device = std::ptr::null_mut();
        let hr = unsafe {
            let func: libloading::Symbol<Fun> = self.lib.get(b"D3D12CreateDevice")?;
            func(
                adapter.as_unknown() as *const _ as *mut _,
                feature_level as _,
                &d3d12::ID3D12Device::uuidof(),
                &mut device,
            )
        };
        let device = unsafe { ComPtr::from_reffed(device.cast()) };

        Ok((device, hr))
    }
}

impl Device {
    #[cfg(feature = "implicit-link")]
    pub fn create<I: Interface>(
        adapter: ComPtr<I>,
        feature_level: crate::FeatureLevel,
    ) -> D3DResult<Self> {
        let mut device = std::ptr::null_mut();
        let hr = unsafe {
            d3d12::D3D12CreateDevice(
                adapter.as_unknown() as *const _ as *mut _,
                feature_level as _,
                &d3d12::ID3D12Device::uuidof(),
                &mut device,
            )
        };
        let device = unsafe { ComPtr::from_reffed(device.cast()) };

        (device, hr)
    }

    pub fn create_heap(
        &self,
        size_in_bytes: u64,
        properties: HeapProperties,
        alignment: u64,
        flags: HeapFlags,
    ) -> D3DResult<Heap> {
        let mut heap = std::ptr::null_mut();

        let desc = d3d12::D3D12_HEAP_DESC {
            SizeInBytes: size_in_bytes,
            Properties: properties.0,
            Alignment: alignment,
            Flags: flags.bits(),
        };

        let hr = unsafe { self.CreateHeap(&desc, &d3d12::ID3D12Heap::uuidof(), &mut heap) };
        let heap = unsafe { ComPtr::from_reffed(heap.cast()) };

        (heap, hr)
    }

    pub fn create_command_allocator(&self, list_type: CmdListType) -> D3DResult<CommandAllocator> {
        let mut allocator = std::ptr::null_mut();
        let hr = unsafe {
            self.CreateCommandAllocator(
                list_type as _,
                &d3d12::ID3D12CommandAllocator::uuidof(),
                &mut allocator,
            )
        };
        let allocator = unsafe { ComPtr::from_reffed(allocator.cast()) };

        (allocator, hr)
    }

    pub fn create_command_queue(
        &self,
        list_type: CmdListType,
        priority: queue::Priority,
        flags: queue::CommandQueueFlags,
        node_mask: NodeMask,
    ) -> D3DResult<CommandQueue> {
        let desc = d3d12::D3D12_COMMAND_QUEUE_DESC {
            Type: list_type as _,
            Priority: priority as _,
            Flags: flags.bits(),
            NodeMask: node_mask,
        };

        let mut queue = std::ptr::null_mut();
        let hr = unsafe {
            self.CreateCommandQueue(&desc, &d3d12::ID3D12CommandQueue::uuidof(), &mut queue)
        };
        let queue = unsafe { ComPtr::from_reffed(queue.cast()) };

        (queue, hr)
    }

    pub fn create_descriptor_heap(
        &self,
        num_descriptors: u32,
        heap_type: DescriptorHeapType,
        flags: DescriptorHeapFlags,
        node_mask: NodeMask,
    ) -> D3DResult<DescriptorHeap> {
        let desc = d3d12::D3D12_DESCRIPTOR_HEAP_DESC {
            Type: heap_type as _,
            NumDescriptors: num_descriptors,
            Flags: flags.bits(),
            NodeMask: node_mask,
        };

        let mut heap = std::ptr::null_mut();
        let hr = unsafe {
            self.CreateDescriptorHeap(&desc, &d3d12::ID3D12DescriptorHeap::uuidof(), &mut heap)
        };
        let heap = unsafe { ComPtr::from_reffed(heap.cast()) };

        (heap, hr)
    }

    pub fn get_descriptor_increment_size(&self, heap_type: DescriptorHeapType) -> u32 {
        unsafe { self.GetDescriptorHandleIncrementSize(heap_type as _) }
    }

    pub fn create_graphics_command_list(
        &self,
        list_type: CmdListType,
        allocator: &CommandAllocator,
        initial: Option<&PipelineState>,
        node_mask: NodeMask,
    ) -> D3DResult<GraphicsCommandList> {
        let mut command_list = std::ptr::null_mut();
        let initial = initial.map_or(std::ptr::null_mut(), |i| i.as_mut_ptr());
        let hr = unsafe {
            self.CreateCommandList(
                node_mask,
                list_type as _,
                allocator.as_mut_ptr(),
                initial,
                &d3d12::ID3D12GraphicsCommandList::uuidof(),
                &mut command_list,
            )
        };
        let command_list = unsafe { ComPtr::from_reffed(command_list.cast()) };

        (command_list, hr)
    }

    pub fn create_query_heap(
        &self,
        heap_ty: query::QueryHeapType,
        count: u32,
        node_mask: NodeMask,
    ) -> D3DResult<QueryHeap> {
        let desc = d3d12::D3D12_QUERY_HEAP_DESC {
            Type: heap_ty as _,
            Count: count,
            NodeMask: node_mask,
        };

        let mut query_heap = std::ptr::null_mut();
        let hr = unsafe {
            self.CreateQueryHeap(&desc, &d3d12::ID3D12QueryHeap::uuidof(), &mut query_heap)
        };
        let query_heap = unsafe { ComPtr::from_reffed(query_heap.cast()) };

        (query_heap, hr)
    }

    pub fn create_graphics_pipeline_state(
        &self,
        _root_signature: RootSignature,
        _vs: Shader,
        _ps: Shader,
        _gs: Shader,
        _hs: Shader,
        _ds: Shader,
        _node_mask: NodeMask,
        _cached_pso: CachedPSO,
        _flags: pso::PipelineStateFlags,
    ) -> D3DResult<PipelineState> {
        unimplemented!()
    }

    pub fn create_compute_pipeline_state(
        &self,
        root_signature: Option<&RootSignature>,
        cs: Shader,
        node_mask: NodeMask,
        cached_pso: CachedPSO,
        flags: pso::PipelineStateFlags,
    ) -> D3DResult<PipelineState> {
        let mut pipeline = std::ptr::null_mut();
        let root_signature = root_signature.map_or(std::ptr::null_mut(), |sig| sig.as_mut_ptr());
        let desc = d3d12::D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: root_signature,
            CS: *cs,
            NodeMask: node_mask,
            CachedPSO: *cached_pso,
            Flags: flags.bits(),
        };

        let hr = unsafe {
            self.CreateComputePipelineState(
                &desc,
                &d3d12::ID3D12PipelineState::uuidof(),
                &mut pipeline,
            )
        };
        let pipeline = unsafe { ComPtr::from_reffed(pipeline.cast()) };

        (pipeline, hr)
    }

    pub fn create_sampler(
        &self,
        sampler: CpuDescriptor,
        filter: d3d12::D3D12_FILTER,
        address_mode: TextureAddressMode,
        mip_lod_bias: f32,
        max_anisotropy: u32,
        comparison_op: d3d12::D3D12_COMPARISON_FUNC,
        border_color: [f32; 4],
        lod: Range<f32>,
    ) {
        let desc = d3d12::D3D12_SAMPLER_DESC {
            Filter: filter,
            AddressU: address_mode[0],
            AddressV: address_mode[1],
            AddressW: address_mode[2],
            MipLODBias: mip_lod_bias,
            MaxAnisotropy: max_anisotropy,
            ComparisonFunc: comparison_op,
            BorderColor: border_color,
            MinLOD: lod.start,
            MaxLOD: lod.end,
        };

        unsafe {
            self.CreateSampler(&desc, sampler);
        }
    }

    pub fn create_root_signature(
        &self,
        blob: Blob,
        node_mask: NodeMask,
    ) -> D3DResult<RootSignature> {
        let mut signature = std::ptr::null_mut();
        let hr = unsafe {
            self.CreateRootSignature(
                node_mask,
                blob.GetBufferPointer(),
                blob.GetBufferSize(),
                &d3d12::ID3D12RootSignature::uuidof(),
                &mut signature,
            )
        };
        let signature = unsafe { ComPtr::from_reffed(signature.cast()) };

        (signature, hr)
    }

    pub fn create_command_signature(
        &self,
        arguments: &[IndirectArgument],
        stride: u32,
        node_mask: NodeMask,
    ) -> D3DResult<CommandSignature> {
        let mut signature = std::ptr::null_mut();
        let desc = d3d12::D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: stride,
            NumArgumentDescs: arguments.len() as _,
            pArgumentDescs: arguments.as_ptr() as *const _,
            NodeMask: node_mask,
        };

        let hr = unsafe {
            self.CreateCommandSignature(
                &desc,
                std::ptr::null_mut(),
                &d3d12::ID3D12CommandSignature::uuidof(),
                &mut signature,
            )
        };
        let signature = unsafe { ComPtr::from_reffed(signature.cast()) };

        (signature, hr)
    }

    pub fn create_render_target_view_from_desc(
        &self,
        desc: &RenderTargetViewDesc,
        descriptor: CpuDescriptor,
    ) {
        // A null pResource is used to initialize a null descriptor,
        // which guarantees D3D11-like null binding behavior (reading 0s, writes are discarded)
        unsafe {
            self.CreateRenderTargetView(std::ptr::null_mut(), &desc.0 as *const _, descriptor);
        }
    }

    // TODO: interface not complete
    pub fn create_fence(&self, initial: u64) -> D3DResult<Fence> {
        let mut fence = std::ptr::null_mut();
        let hr = unsafe {
            self.CreateFence(
                initial,
                d3d12::D3D12_FENCE_FLAG_NONE,
                &d3d12::ID3D12Fence::uuidof(),
                &mut fence,
            )
        };
        let fence = unsafe { ComPtr::from_reffed(fence.cast()) };

        (fence, hr)
    }
}
