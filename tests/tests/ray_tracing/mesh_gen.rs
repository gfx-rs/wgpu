use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    _pos: [f32; 4],
    _tex_coord: [f32; 2],
}

fn vertex(pos: [i8; 3], tc: [i8; 2]) -> Vertex {
    Vertex {
        _pos: [pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0],
        _tex_coord: [tc[0] as f32, tc[1] as f32],
    }
}

pub fn create_vertices() -> (Vec<Vertex>, Vec<u16>) {
    let vertex_data = [
        // top (0, 0, 1)
        vertex([-1, -1, 1], [0, 0]),
        vertex([1, -1, 1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([-1, 1, 1], [0, 1]),
        // bottom (0, 0, -1)
        vertex([-1, 1, -1], [1, 0]),
        vertex([1, 1, -1], [0, 0]),
        vertex([1, -1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // right (1, 0, 0)
        vertex([1, -1, -1], [0, 0]),
        vertex([1, 1, -1], [1, 0]),
        vertex([1, 1, 1], [1, 1]),
        vertex([1, -1, 1], [0, 1]),
        // left (-1, 0, 0)
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, 1, 1], [0, 0]),
        vertex([-1, 1, -1], [0, 1]),
        vertex([-1, -1, -1], [1, 1]),
        // front (0, 1, 0)
        vertex([1, 1, -1], [1, 0]),
        vertex([-1, 1, -1], [0, 0]),
        vertex([-1, 1, 1], [0, 1]),
        vertex([1, 1, 1], [1, 1]),
        // back (0, -1, 0)
        vertex([1, -1, 1], [0, 0]),
        vertex([-1, -1, 1], [1, 0]),
        vertex([-1, -1, -1], [1, 1]),
        vertex([1, -1, -1], [0, 1]),
    ];

    let index_data: &[u16] = &[
        0, 1, 2, 2, 3, 0, // top
        4, 5, 6, 6, 7, 4, // bottom
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // front
        20, 21, 22, 22, 23, 20, // back
    ];

    (vertex_data.to_vec(), index_data.to_vec())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct AccelerationStructureInstance {
    transform: [f32; 12],
    custom_index_and_mask: u32,
    shader_binding_table_record_offset_and_flags: u32,
    acceleration_structure_reference: u64,
}

impl std::fmt::Debug for AccelerationStructureInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("transform", &self.transform)
            .field("custom_index()", &self.custom_index())
            .field("mask()", &self.mask())
            .field(
                "shader_binding_table_record_offset()",
                &self.shader_binding_table_record_offset(),
            )
            .field("flags()", &self.flags())
            .field(
                "acceleration_structure_reference",
                &self.acceleration_structure_reference,
            )
            .finish()
    }
}

#[allow(dead_code)]
impl AccelerationStructureInstance {
    const LOW_24_MASK: u32 = 0x00ff_ffff;
    const MAX_U24: u32 = (1u32 << 24u32) - 1u32;

    #[inline]
    pub fn affine_to_rows(mat: &Affine3A) -> [f32; 12] {
        let row_0 = mat.matrix3.row(0);
        let row_1 = mat.matrix3.row(1);
        let row_2 = mat.matrix3.row(2);
        let translation = mat.translation;
        [
            row_0.x,
            row_0.y,
            row_0.z,
            translation.x,
            row_1.x,
            row_1.y,
            row_1.z,
            translation.y,
            row_2.x,
            row_2.y,
            row_2.z,
            translation.z,
        ]
    }

    #[inline]
    fn rows_to_affine(rows: &[f32; 12]) -> Affine3A {
        Affine3A::from_cols_array(&[
            rows[0], rows[3], rows[6], rows[9], rows[1], rows[4], rows[7], rows[10], rows[2],
            rows[5], rows[8], rows[11],
        ])
    }

    pub fn transform_as_affine(&self) -> Affine3A {
        Self::rows_to_affine(&self.transform)
    }
    pub fn set_transform(&mut self, transform: &Affine3A) {
        self.transform = Self::affine_to_rows(transform);
    }

    pub fn custom_index(&self) -> u32 {
        self.custom_index_and_mask & Self::LOW_24_MASK
    }

    pub fn mask(&self) -> u8 {
        (self.custom_index_and_mask >> 24) as u8
    }

    pub fn shader_binding_table_record_offset(&self) -> u32 {
        self.shader_binding_table_record_offset_and_flags & Self::LOW_24_MASK
    }

    pub fn flags(&self) -> u8 {
        (self.shader_binding_table_record_offset_and_flags >> 24) as u8
    }

    pub fn set_custom_index(&mut self, custom_index: u32) {
        debug_assert!(
            custom_index <= Self::MAX_U24,
            "custom_index uses more than 24 bits! {custom_index} > {}",
            Self::MAX_U24
        );
        self.custom_index_and_mask =
            (custom_index & Self::LOW_24_MASK) | (self.custom_index_and_mask & !Self::LOW_24_MASK)
    }

    pub fn set_mask(&mut self, mask: u8) {
        self.custom_index_and_mask =
            (self.custom_index_and_mask & Self::LOW_24_MASK) | (u32::from(mask) << 24)
    }

    pub fn set_shader_binding_table_record_offset(
        &mut self,
        shader_binding_table_record_offset: u32,
    ) {
        debug_assert!(shader_binding_table_record_offset <= Self::MAX_U24, "shader_binding_table_record_offset uses more than 24 bits! {shader_binding_table_record_offset} > {}", Self::MAX_U24);
        self.shader_binding_table_record_offset_and_flags = (shader_binding_table_record_offset
            & Self::LOW_24_MASK)
            | (self.shader_binding_table_record_offset_and_flags & !Self::LOW_24_MASK)
    }

    pub fn set_flags(&mut self, flags: u8) {
        self.shader_binding_table_record_offset_and_flags =
            (self.shader_binding_table_record_offset_and_flags & Self::LOW_24_MASK)
                | (u32::from(flags) << 24)
    }

    pub fn new(
        transform: &Affine3A,
        custom_index: u32,
        mask: u8,
        shader_binding_table_record_offset: u32,
        flags: u8,
        acceleration_structure_reference: u64,
    ) -> Self {
        debug_assert!(
            custom_index <= Self::MAX_U24,
            "custom_index uses more than 24 bits! {custom_index} > {}",
            Self::MAX_U24
        );
        debug_assert!(
            shader_binding_table_record_offset <= Self::MAX_U24,
            "shader_binding_table_record_offset uses more than 24 bits! {shader_binding_table_record_offset} > {}", Self::MAX_U24
        );
        AccelerationStructureInstance {
            transform: Self::affine_to_rows(transform),
            custom_index_and_mask: (custom_index & Self::MAX_U24) | (u32::from(mask) << 24),
            shader_binding_table_record_offset_and_flags: (shader_binding_table_record_offset
                & Self::MAX_U24)
                | (u32::from(flags) << 24),
            acceleration_structure_reference,
        }
    }
}
