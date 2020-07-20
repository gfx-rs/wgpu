//! Convenience macros

/// Macro to produce an array of [`VertexAttributeDescriptor`].
///
/// Output has type: `[VertexAttributeDescriptor; _]`. Usage is as follows:
/// ```
/// # use wgpu::vertex_attr_array;
/// let attrs = vertex_attr_array![0 => Float2, 1 => Float, 2 => Ushort4];
/// ```
/// This example specifies a list of three [`VertexAttributeDescriptor`],
/// each with the given `shader_location` and `format`.
/// Offsets are calculated automatically.
#[macro_export]
macro_rules! vertex_attr_array {
    ($($loc:expr => $fmt:ident),* $(,)?) => {
        $crate::vertex_attr_array!([] ; 0; $($loc => $fmt ,)*)
    };
    ([$($t:expr,)*] ; $off:expr ;) => { [$($t,)*] };
    ([$($t:expr,)*] ; $off:expr ; $loc:expr => $item:ident, $($ll:expr => $ii:ident ,)*) => {
        $crate::vertex_attr_array!(
            [$($t,)*
            $crate::VertexAttributeDescriptor {
                format: $crate::VertexFormat :: $item,
                offset: $off,
                shader_location: $loc,
            },];
            $off + $crate::vertex_format_size!($item);
            $($ll => $ii ,)*
        )
    };
}

/// Helper macro which turns a vertex attribute type into a size.
///
/// Mainly used as a helper for [`vertex_attr_array`] but might be externally useful.
#[macro_export]
macro_rules! vertex_format_size {
    (Uchar2) => {
        2
    };
    (Uchar4) => {
        4
    };
    (Char2) => {
        2
    };
    (Char4) => {
        4
    };
    (Uchar2Norm) => {
        2
    };
    (Uchar4Norm) => {
        4
    };
    (Char2Norm) => {
        2
    };
    (Char4Norm) => {
        4
    };
    (Ushort2) => {
        4
    };
    (Ushort4) => {
        8
    };
    (Short2) => {
        4
    };
    (Short4) => {
        8
    };
    (Ushort2Norm) => {
        4
    };
    (Ushort4Norm) => {
        8
    };
    (Short2Norm) => {
        4
    };
    (Short4Norm) => {
        8
    };
    (Half2) => {
        4
    };
    (Half4) => {
        8
    };
    (Float) => {
        4
    };
    (Float2) => {
        8
    };
    (Float3) => {
        12
    };
    (Float4) => {
        16
    };
    (Uint) => {
        4
    };
    (Uint2) => {
        8
    };
    (Uint3) => {
        12
    };
    (Uint4) => {
        16
    };
    (Int) => {
        4
    };
    (Int2) => {
        8
    };
    (Int3) => {
        12
    };
    (Int4) => {
        16
    };
}

#[test]
fn test_vertex_attr_array() {
    let attrs = vertex_attr_array![0 => Float2, 3 => Ushort4];
    // VertexAttributeDescriptor does not support PartialEq, so we cannot test directly
    assert_eq!(attrs.len(), 2);
    assert_eq!(attrs[0].offset, 0);
    assert_eq!(attrs[0].shader_location, 0);
    assert_eq!(attrs[1].offset, std::mem::size_of::<(f32, f32)>() as u64);
    assert_eq!(attrs[1].shader_location, 3);
}

/// Macro to load a SPIR-V module statically.
///
/// It ensures the word alignment as well as the magic number.
#[macro_export]
macro_rules! include_spirv {
    ($($token:tt)*) => {
        $crate::util::make_spirv(include_bytes!($($token)*))
    };
}
