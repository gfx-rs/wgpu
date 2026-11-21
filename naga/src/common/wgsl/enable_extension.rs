//! WGSL [enable-extensions](https://www.w3.org/TR/WGSL/#enable-extensions-sec).

use crate::valid::Capabilities as C;

macro_rules! enable_extensions {
    ($($(#[$attr:ident $($args:tt)*])* $name:ident = $ident:literal => $caps:expr;)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        enum Bit { $($name),* }

        bitflags::bitflags! {
            /// A set of [WGSL enable-extensions] implemented by Naga.
            ///
            /// Iteration and the WGSL backend's `enable` declarations follow declaration order.
            ///
            /// [WGSL enable-extensions]: https://www.w3.org/TR/WGSL/#enable-extensions-sec
            #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
            pub struct EnableExtensions: u32 {
                $($(#[$attr $($args)*])* const $name = 1 << Bit::$name as u32;)*
            }
        }

        impl EnableExtensions {
            const TABLE: &[(Self, &'static str, C)] = &[$((Self::$name, $ident, $caps)),*];
        }
    };
}

enable_extensions! {
    /// `enable f16;`: `f16` scalars.
    F16 = "f16" => C::SHADER_FLOAT16;
    /// `enable wgpu_f64;`: `f64` scalars. Native only.
    WGPU_F64 = "wgpu_f64" => C::FLOAT64;
    /// `enable wgpu_int16;`: `i16`/`u16` scalars. Native only.
    WGPU_INT16 = "wgpu_int16" => C::SHADER_INT16;
    /// `enable dual_source_blending;`: the `blend_src` attribute.
    DUAL_SOURCE_BLENDING = "dual_source_blending" => C::DUAL_SOURCE_BLENDING;
    /// `enable clip_distances;`: the `clip_distances` built-in.
    CLIP_DISTANCES = "clip_distances" => C::CLIP_DISTANCES;
    /// `enable wgpu_mesh_shader;`: task and mesh shaders. Native only.
    WGPU_MESH_SHADER = "wgpu_mesh_shader" => C::MESH_SHADER;
    /// `enable wgpu_binding_array;`: `binding_array`. Native only.
    WGPU_BINDING_ARRAY = "wgpu_binding_array" => C::BUFFER_BINDING_ARRAY
        .union(C::BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
        .union(C::STORAGE_BUFFER_BINDING_ARRAY)
        .union(C::STORAGE_BUFFER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
        .union(C::STORAGE_TEXTURE_BINDING_ARRAY)
        .union(C::STORAGE_TEXTURE_BINDING_ARRAY_NON_UNIFORM_INDEXING)
        .union(C::TEXTURE_AND_SAMPLER_BINDING_ARRAY)
        .union(C::TEXTURE_AND_SAMPLER_BINDING_ARRAY_NON_UNIFORM_INDEXING)
        .union(C::ACCELERATION_STRUCTURE_BINDING_ARRAY);
    /// `enable draw_index;`: the `draw_index` built-in.
    DRAW_INDEX = "draw_index" => C::DRAW_INDEX;
    /// `enable primitive_index;`: the `primitive_index` built-in.
    PRIMITIVE_INDEX = "primitive_index" => C::PRIMITIVE_INDEX;
    /// `enable wgpu_cooperative_matrix;`: cooperative matrices. Native only.
    WGPU_COOPERATIVE_MATRIX = "wgpu_cooperative_matrix" => C::COOPERATIVE_MATRIX;
    /// `enable wgpu_ray_query;`: ray queries. Native only.
    WGPU_RAY_QUERY = "wgpu_ray_query" => C::RAY_QUERY;
    /// `enable wgpu_ray_query_vertex_return;`: the `vertex_return` ray flag. Native only.
    WGPU_RAY_QUERY_VERTEX_RETURN = "wgpu_ray_query_vertex_return" => C::RAY_HIT_VERTEX_POSITION;
    /// `enable wgpu_ray_tracing_pipeline;`: ray tracing pipeline stages. Native only.
    WGPU_RAY_TRACING_PIPELINE = "wgpu_ray_tracing_pipeline" => C::RAY_TRACING_PIPELINE;
    /// `enable wgpu_ray_tracing_invocation_reorder;`: `hit_object`, the `hitObject*`
    /// built-in functions, and `reorderThread`. Native only.
    WGPU_RAY_TRACING_INVOCATION_REORDER = "wgpu_ray_tracing_invocation_reorder"
        => C::RAY_TRACING_INVOCATION_REORDER;
    /// `enable wgpu_per_vertex;`: `@interpolate(per_vertex)`. Native only.
    WGPU_PER_VERTEX = "wgpu_per_vertex" => C::PER_VERTEX;
    /// `enable wgpu_debug_printf;`: `debugPrintf`. Native only.
    WGPU_DEBUG_PRINTF = "wgpu_debug_printf" => C::DEBUG_PRINTF;
}

impl EnableExtensions {
    /// Look up the extension named `ident` in WGSL.
    pub fn from_ident(ident: &str) -> Option<Self> {
        Self::TABLE
            .iter()
            .find(|&&(_, name, _)| name == ident)
            .map(|&(ext, _, _)| ext)
    }

    /// The WGSL name of the first extension in this set.
    pub fn to_ident(self) -> &'static str {
        Self::TABLE
            .iter()
            .find(|&&(ext, _, _)| self.contains(ext))
            .map_or("", |&(_, name, _)| name)
    }

    /// The capabilities that each permit at least one extension in this set.
    pub fn capabilities(self) -> C {
        Self::TABLE
            .iter()
            .filter(|&&(ext, _, _)| self.contains(ext))
            .fold(C::empty(), |acc, &(_, _, caps)| acc | caps)
    }
}

#[test]
fn every_extension_has_unique_ident() {
    for &(ext, name, _) in EnableExtensions::TABLE {
        assert_eq!(EnableExtensions::from_ident(name), Some(ext));
        assert_eq!(ext.to_ident(), name);
    }
    assert_eq!(
        EnableExtensions::TABLE.len(),
        EnableExtensions::all().iter().count()
    );
}
