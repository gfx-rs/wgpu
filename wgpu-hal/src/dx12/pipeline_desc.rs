//! We try to use pipeline stream descriptors where possible, but this isn't allowed
//! on some older windows 10 versions. Therefore, we also must have some logic to
//! convert such descriptors to the "traditional" equivalent,
//! `D3D12_GRAPHICS_PIPELINE_STATE_DESC`.
//!
//! Stream descriptors allow extending the pipeline, enabling more advanced features,
//! including mesh shaders and multiview/view instancing. Using a stream descriptor
//! is like using a vulkan descriptor with a `pNext` chain. It doesn't have direct
//! benefits to all use cases, but allows new use cases.
//!
//! The code for pipeline stream descriptors is very complicated, and can have bad
//! consequences if it is written incorrectly. It has been isolated to this file for
//! that reason.

use core::mem::ManuallyDrop;

use alloc::vec::Vec;
use windows::Win32::Graphics::Direct3D12;
use windows::Win32::Graphics::Dxgi;

impl super::RenderPipelineStateStreamDesc {
    /// # Safety
    ///
    /// Must not outlive `self`, as it contains a pointer to the root signature as pointed to
    /// by this struct.
    pub unsafe fn to_bytes(&self) -> Vec<u8> {
        use Direct3D12::*;

        // Dynamic allocation is used here because the resulting stream can become very large.
        let mut bytes = Vec::new();

        // The thing to understand is that DX12 expects it to be laid out like any normal struct.
        // Therefore, everything must obey certain alignment rules. Otherwise, everything gets
        // corrupted. Unfortunately, we can't just use a normal struct because we can't push
        // subobjects that aren't being used, and we shouldn't try to give all permutations
        // of used subobjects their own struct.
        //
        // Essentially, each field is prefaced by its type identifier. This identifier is a 32
        // bit integer, but must be aligned to 64 bits. Then the actual subobject is aligned as
        // it is required to (not necessarily to 64 bits!). And if we don't use a field, we
        // shouldn't push it.
        //
        // Therefore, we "construct" a struct manually here. Adding fields should be done with
        // extreme caution.
        //
        // See <https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_pipeline_state_stream_desc>.
        macro_rules! push_subobject {
            ($subobject_type:expr, $data:expr) => {{
                // Ensure 8-byte alignment for the subobject start, even though
                // the tag is only a u32. I don't fully understand why.
                let alignment = 8;
                let aligned_length = bytes.len().next_multiple_of(alignment);
                // Pad with zeros
                bytes.resize(aligned_length, 0);

                // Append the type tag (u32)
                let tag: u32 = $subobject_type.0 as u32;
                bytes.extend_from_slice(&tag.to_ne_bytes());

                // Align the data
                let obj_align = align_of_val(&$data);
                let data_start = bytes.len().next_multiple_of(obj_align);
                // Pad with zeros
                bytes.resize(data_start, 0);

                // Append the data itself, as raw bytes
                #[allow(clippy::ptr_as_ptr, trivial_casts)]
                let data_ptr = &$data as *const _ as *const u8;
                let data_size = size_of_val(&$data);
                let slice = unsafe { core::slice::from_raw_parts(data_ptr, data_size) };
                bytes.extend_from_slice(slice);
            }};
        }
        push_subobject!(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            self.root_signature
        );
        push_subobject!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, self.blend_state);
        push_subobject!(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            self.sample_mask
        );
        push_subobject!(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
            self.rasterizer_state
        );
        push_subobject!(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
            self.depth_stencil_state
        );
        push_subobject!(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            self.primitive_topology_type
        );
        if self.rtv_formats.NumRenderTargets != 0 {
            push_subobject!(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
                self.rtv_formats
            );
        }
        if self.dsv_format != Dxgi::Common::DXGI_FORMAT_UNKNOWN {
            push_subobject!(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
                self.dsv_format
            );
        }
        push_subobject!(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
            self.sample_desc
        );
        if self.node_mask != 0 {
            push_subobject!(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_NODE_MASK,
                self.node_mask
            );
        }
        if !self.cached_pso.pCachedBlob.is_null() {
            push_subobject!(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,
                self.cached_pso
            );
        }
        push_subobject!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS, self.flags);

        if !self.pixel_shader.pShaderBytecode.is_null() {
            push_subobject!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, self.pixel_shader);
        }

        // Vertex pipeline stuff
        if !self.vertex_shader.pShaderBytecode.is_null() {
            push_subobject!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, self.vertex_shader);
        }
        if !self.vertex_shader.pShaderBytecode.is_null() {
            push_subobject!(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
                self.input_layout
            );
        }
        if !self.vertex_shader.pShaderBytecode.is_null() {
            push_subobject!(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
                self.index_buffer_strip_cut_value
            );
        }
        if !self.vertex_shader.pShaderBytecode.is_null() {
            push_subobject!(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT,
                self.stream_output
            );
        }

        // Mesh pipeline stuff
        if !self.task_shader.pShaderBytecode.is_null() {
            push_subobject!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS, self.task_shader);
        }
        if !self.mesh_shader.pShaderBytecode.is_null() {
            push_subobject!(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, self.mesh_shader);
        }

        bytes
    }

    pub fn to_graphics_pipeline_descriptor(
        &self,
    ) -> Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        Direct3D12::D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(if !self.root_signature.is_null() {
                Some(unsafe { (*self.root_signature).clone() })
            } else {
                None
            }),
            VS: self.vertex_shader,
            PS: self.pixel_shader,
            DS: Direct3D12::D3D12_SHADER_BYTECODE::default(),
            HS: Direct3D12::D3D12_SHADER_BYTECODE::default(),
            GS: Direct3D12::D3D12_SHADER_BYTECODE::default(),
            StreamOutput: self.stream_output,
            BlendState: self.blend_state,
            SampleMask: self.sample_mask,
            RasterizerState: self.rasterizer_state,
            DepthStencilState: self.depth_stencil_state,
            InputLayout: self.input_layout,
            IBStripCutValue: self.index_buffer_strip_cut_value,
            PrimitiveTopologyType: self.primitive_topology_type,
            NumRenderTargets: self.rtv_formats.NumRenderTargets,
            RTVFormats: self.rtv_formats.RTFormats,
            DSVFormat: self.dsv_format,
            SampleDesc: self.sample_desc,
            NodeMask: self.node_mask,
            CachedPSO: self.cached_pso,
            Flags: self.flags,
        }
    }
}
