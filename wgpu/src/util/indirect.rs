/// The structure expected in `indirect_buffer` for [`RenderEncoder::draw_indirect`](crate::util::RenderEncoder::draw_indirect).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DrawIndirect {
    /// The number of vertices to draw.
    pub vertex_count: u32,
    /// The number of instances to draw.
    pub instance_count: u32,
    /// The Index of the first vertex to draw.
    pub base_vertex: u32,
    /// The instance ID of the first instance to draw.
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    pub base_instance: u32,
}

impl DrawIndirect {
    /// Returns the bytes representation of the struct, ready to be written in a [`Buffer`](crate::Buffer).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::mem::transmute(std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<Self>(),
            ))
        }
    }
}

/// The structure expected in `indirect_buffer` for [`RenderEncoder::draw_indexed_indirect`](crate::util::RenderEncoder::draw_indexed_indirect).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DrawIndexedIndirect {
    /// The number of vertices to draw.
    pub vertex_count: u32,
    /// The number of instances to draw.
    pub instance_count: u32,
    /// The base index within the index buffer.
    pub base_index: u32,
    /// The value added to the vertex index before indexing into the vertex buffer.
    pub vertex_offset: i32,
    /// The instance ID of the first instance to draw.
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    pub base_instance: u32,
}

impl DrawIndexedIndirect {
    /// Returns the bytes representation of the struct, ready to be written in a [`Buffer`](crate::Buffer).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::mem::transmute(std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<Self>(),
            ))
        }
    }
}

/// The structure expected in `indirect_buffer` for [`ComputePass::dispatch_indirect`](crate::ComputePass::dispatch_indirect).
///
/// x, y and z denote the number of work groups to dispatch in each dimension.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DispatchIndirect {
    /// The number of work groups in X dimension.
    pub x: u32,
    /// The number of work groups in Y dimension.
    pub y: u32,
    /// The number of work groups in Z dimension.
    pub z: u32,
}

impl DispatchIndirect {
    /// Returns the bytes representation of the struct, ready to be written in a [`Buffer`](crate::Buffer).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::mem::transmute(std::slice::from_raw_parts(
                self as *const _ as *const u8,
                std::mem::size_of::<Self>(),
            ))
        }
    }
}
