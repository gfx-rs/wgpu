//! Build [`ModuleReflection`] for a validated module.

use alloc::{string::String, vec::Vec};
use core::num::NonZeroU32;

pub use naga_types::reflect::*;

use crate::{
    arena::{Handle, UniqueArena},
    proc::{LayoutError, Layouter},
    valid::{ImmediateUsage, ImmediateUsageExt as _, ModuleInfo},
};

/// Describe the interface of `module`.
///
/// `info` must be the result of validating `module`.
pub fn reflect(module: &crate::Module, info: &ModuleInfo) -> Result<ModuleReflection, LayoutError> {
    let mut layouter = Layouter::default();
    layouter.update(module.to_ctx())?;

    let types = module
        .types
        .iter()
        .map(|(handle, ty)| Type {
            name: ty.name.clone(),
            size: layouter[handle].size,
            align: layouter[handle].alignment * 1,
            inner: reflect_type_inner(&ty.inner),
        })
        .collect();

    let mut resources = Vec::new();
    let mut resource_index = crate::FastHashMap::default();
    for (var_handle, var) in module.global_variables.iter() {
        let Some(binding) = var.binding else { continue };

        let (inner_ty, binding_array_size) = match module.types[var.ty].inner {
            crate::TypeInner::BindingArray { base, size } => {
                (&module.types[base].inner, Some(reflect_array_size(size)))
            }
            ref ty => (ty, None),
        };

        let kind = match *inner_ty {
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => ResourceKind::Texture {
                dim,
                arrayed,
                class,
            },
            crate::TypeInner::Sampler { comparison } => ResourceKind::Sampler { comparison },
            crate::TypeInner::AccelerationStructure { vertex_return } => {
                ResourceKind::AccelerationStructure { vertex_return }
            }
            ref other => ResourceKind::Buffer {
                min_binding_size: NonZeroU32::new(other.size(module.to_ctx()))
                    .expect("validated buffer types are never zero-sized"),
            },
        };

        resource_index.insert(var_handle, resources.len());
        resources.push(Resource {
            name: var.name.clone(),
            binding,
            space: var.space,
            ty: var.ty.index(),
            kind,
            binding_array_size,
        });
    }

    let overrides = module
        .overrides
        .iter()
        .map(|(_, o)| Override {
            name: o.name.clone(),
            id: o.id,
            ty: o.ty.index(),
            has_default: o.init.is_some(),
        })
        .collect();

    let entry_points = module
        .entry_points
        .iter()
        .enumerate()
        .map(|(index, ep)| reflect_entry_point(module, info, index, ep, &resource_index))
        .collect();

    Ok(ModuleReflection {
        types,
        resources,
        overrides,
        entry_points,
    })
}

const fn reflect_array_size(size: crate::ArraySize) -> ArraySize {
    match size {
        crate::ArraySize::Constant(n) => ArraySize::Constant(n),
        crate::ArraySize::Pending(o) => ArraySize::Override(o.index()),
        crate::ArraySize::Dynamic => ArraySize::Dynamic,
    }
}

fn reflect_type_inner(inner: &crate::TypeInner) -> TypeInner {
    use crate::TypeInner as Ti;
    match *inner {
        Ti::Scalar(scalar) => TypeInner::Scalar(scalar),
        Ti::Vector { size, scalar } => TypeInner::Vector { size, scalar },
        Ti::Matrix {
            columns,
            rows,
            scalar,
        } => TypeInner::Matrix {
            columns,
            rows,
            scalar,
        },
        Ti::Atomic(scalar) => TypeInner::Atomic(scalar),
        Ti::Pointer { base, space } => TypeInner::Pointer {
            base: base.index(),
            space,
        },
        Ti::Array { base, size, stride } => TypeInner::Array {
            base: base.index(),
            size: reflect_array_size(size),
            stride,
        },
        Ti::Struct { ref members, span } => TypeInner::Struct {
            members: members
                .iter()
                .map(|m| StructMember {
                    name: m.name.clone(),
                    ty: m.ty.index(),
                    offset: m.offset,
                })
                .collect(),
            span,
        },
        Ti::Image {
            dim,
            arrayed,
            class,
        } => TypeInner::Image {
            dim,
            arrayed,
            class,
        },
        Ti::Sampler { comparison } => TypeInner::Sampler { comparison },
        Ti::AccelerationStructure { vertex_return } => {
            TypeInner::AccelerationStructure { vertex_return }
        }
        Ti::RayQuery { vertex_return } => TypeInner::RayQuery { vertex_return },
        Ti::BindingArray { base, size } => TypeInner::BindingArray {
            base: base.index(),
            size: reflect_array_size(size),
        },
        Ti::ValuePointer { .. } | Ti::CooperativeMatrix { .. } | Ti::HitObject => TypeInner::Other,
    }
}

fn reflect_entry_point(
    module: &crate::Module,
    info: &ModuleInfo,
    index: usize,
    entry_point: &crate::EntryPoint,
    resource_index: &crate::FastHashMap<Handle<crate::GlobalVariable>, ResourceIndex>,
) -> EntryPoint {
    let func_info = info.get_entry_point(index);
    let function = &entry_point.function;

    let mut inputs = Vec::new();
    for arg in function.arguments.iter() {
        populate(
            &mut inputs,
            arg.name.as_ref(),
            arg.binding.as_ref(),
            arg.ty,
            &module.types,
        );
    }
    let mut outputs = Vec::new();
    if let Some(ref result) = function.result {
        populate(
            &mut outputs,
            None,
            result.binding.as_ref(),
            result.ty,
            &module.types,
        );
    }

    let mut resources = Vec::new();
    let mut workgroup_storage_size = 0u64;
    for (var_handle, var) in module.global_variables.iter() {
        if func_info[var_handle].is_empty() {
            continue;
        }
        if let Some(&index) = resource_index.get(&var_handle) {
            resources.push(index);
        }
        if var.space == crate::AddressSpace::WorkGroup {
            let size = module.types[var.ty].inner.size(module.to_ctx());
            workgroup_storage_size =
                workgroup_storage_size.saturating_add(u64::from(size).next_multiple_of(16));
        }
    }

    let mut sampling_pairs: Vec<_> = func_info
        .sampling_set
        .iter()
        .filter_map(|key| {
            Some((
                *resource_index.get(&key.image)?,
                *resource_index.get(&key.sampler)?,
            ))
        })
        .collect();
    sampling_pairs.sort_unstable();
    sampling_pairs.dedup();

    let mut used_immediates = module
        .global_variables
        .iter()
        .filter(|&(handle, var)| {
            var.space == crate::AddressSpace::Immediate && !func_info[handle].is_empty()
        })
        .map(|(_, var)| var);
    let immediate_usage = used_immediates
        .next()
        .map(|var| {
            ImmediateUsage::from_type(&module.types[var.ty].inner, &module.types, module.to_ctx())
        })
        .unwrap_or_default();
    debug_assert!(
        used_immediates.next().is_none(),
        "validation allows at most one immediate per entry point"
    );

    let task_payload_size = entry_point.task_payload.map(|task_payload| {
        module.types[module.global_variables[task_payload].ty]
            .inner
            .size(module.to_ctx())
    });

    let mesh_info = entry_point.mesh_info.as_ref().map(|mesh_info| {
        populate(
            &mut outputs,
            None,
            None,
            mesh_info.vertex_output_type,
            &module.types,
        );
        populate(
            &mut outputs,
            None,
            None,
            mesh_info.primitive_output_type,
            &module.types,
        );
        MeshInfo {
            max_vertices: mesh_info.max_vertices,
            max_primitives: mesh_info.max_primitives,
            topology: mesh_info.topology,
        }
    });

    let ray_tracing_payload = match entry_point.stage {
        crate::ShaderStage::ClosestHit | crate::ShaderStage::AnyHit | crate::ShaderStage::Miss => {
            entry_point
                .incoming_ray_payload
                .map(|payload| module.global_variables[payload].ty.index())
        }
        crate::ShaderStage::RayGeneration => module
            .global_variables
            .iter()
            .filter(|&(handle, var)| {
                !func_info[handle].is_empty() && var.space == crate::AddressSpace::RayPayload
            })
            .map(|(_, var)| var.ty.index())
            .next_back(),
        _ => None,
    };

    let workgroup_size_from_overrides = entry_point
        .workgroup_size_overrides
        .map(|dims| dims.map(|dim| dim.is_some()))
        .unwrap_or_default();

    EntryPoint {
        name: entry_point.name.clone(),
        stage: entry_point.stage,
        inputs,
        outputs,
        resources,
        sampling_pairs,
        workgroup_size: entry_point.workgroup_size,
        workgroup_size_from_overrides,
        workgroup_storage_size,
        dual_source_blending: func_info.dual_source_blending,
        task_payload_size,
        mesh_info,
        immediate_usage,
        ray_tracing_payload,
    }
}

const fn numeric_type(inner: &crate::TypeInner) -> Option<NumericType> {
    Some(match *inner {
        crate::TypeInner::Scalar(scalar) => NumericType::Scalar(scalar),
        crate::TypeInner::Vector { size, scalar } => NumericType::Vector { size, scalar },
        crate::TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => NumericType::Matrix {
            columns,
            rows,
            scalar,
        },
        _ => return None,
    })
}

/// Add the varyings described by an entry point argument or result to `list`,
/// flattening structs.
fn populate(
    list: &mut Vec<Varying>,
    name: Option<&String>,
    binding: Option<&crate::Binding>,
    ty: Handle<crate::Type>,
    types: &UniqueArena<crate::Type>,
) {
    let inner = &types[ty].inner;
    if let crate::TypeInner::Struct { ref members, .. } = *inner {
        for member in members {
            populate(
                list,
                member.name.as_ref(),
                member.binding.as_ref(),
                member.ty,
                types,
            );
        }
        return;
    }

    let name = name.cloned();
    match binding {
        Some(&crate::Binding::BuiltIn(crate::BuiltIn::ClipDistances)) => {
            let crate::TypeInner::Array {
                size: crate::ArraySize::Constant(array_size),
                ..
            } = *inner
            else {
                log::error!("Unexpected type for `clip_distances`: {inner:?}");
                return;
            };
            list.push(Varying::BuiltIn {
                name,
                built_in: crate::BuiltIn::ClipDistances,
                ty: None,
                array_size: Some(array_size.get()),
            });
        }
        Some(&crate::Binding::BuiltIn(built_in)) => {
            let Some(ty) = numeric_type(inner) else {
                log::error!("Unexpected varying type: {inner:?}");
                return;
            };
            list.push(Varying::BuiltIn {
                name,
                built_in,
                ty: Some(ty),
                array_size: None,
            });
        }
        Some(&crate::Binding::Location {
            location,
            interpolation,
            sampling,
            per_primitive,
            blend_src,
        }) => {
            let Some(ty) = numeric_type(inner) else {
                log::error!("Unexpected varying type: {inner:?}");
                return;
            };
            list.push(Varying::Location {
                name,
                location,
                ty,
                interpolation,
                sampling,
                per_primitive,
                blend_src,
            });
        }
        None => {
            log::error!("Missing binding for a varying");
        }
    }
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use super::*;

    fn reflect_wgsl(source: &str) -> ModuleReflection {
        let module = crate::front::wgsl::parse_str(source).unwrap();
        let info = crate::valid::Validator::new(
            crate::valid::ValidationFlags::all(),
            crate::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        reflect(&module, &info).unwrap()
    }

    #[test]
    fn resources_and_usage() {
        let r = reflect_wgsl(
            "
            struct U { a: vec4<f32>, b: f32 }
            @group(0) @binding(0) var<uniform> u: U;
            @group(0) @binding(1) var t: texture_2d<f32>;
            @group(0) @binding(2) var s: sampler;
            @group(1) @binding(0) var<storage, read_write> unused: array<u32>;
            @fragment fn fs() -> @location(0) vec4<f32> {
                return textureSample(t, s, vec2(0.0)) * u.a;
            }
            ",
        );
        assert_eq!(r.resources.len(), 4);
        assert_eq!(
            r.resources[0].kind,
            ResourceKind::Buffer {
                min_binding_size: NonZeroU32::new(32).unwrap()
            }
        );
        assert_eq!(
            r.resources[3].kind,
            ResourceKind::Buffer {
                min_binding_size: NonZeroU32::new(4).unwrap()
            }
        );
        let ep = &r.entry_points[0];
        assert_eq!(ep.resources, [0, 1, 2]);
        assert_eq!(ep.sampling_pairs, [(1, 2)]);
        assert!(matches!(
            ep.outputs[..],
            [Varying::Location { location: 0, .. }]
        ));
    }

    #[test]
    fn varyings_are_flattened() {
        let r = reflect_wgsl(
            "
            struct In { @location(1) @interpolate(flat) a: u32, @builtin(position) p: vec4<f32> }
            @fragment fn fs(i: In) -> @location(0) vec4<f32> { return i.p; }
            ",
        );
        let inputs = &r.entry_points[0].inputs;
        assert_eq!(inputs.len(), 2);
        assert!(matches!(
            inputs[0],
            Varying::Location {
                location: 1,
                interpolation: Some(crate::Interpolation::Flat),
                ..
            }
        ));
        assert!(matches!(
            inputs[1],
            Varying::BuiltIn {
                built_in: crate::BuiltIn::Position { .. },
                ..
            }
        ));
    }

    #[test]
    fn struct_layout_and_overrides() {
        let r = reflect_wgsl(
            "
            struct S { a: f32, b: vec3<f32> }
            @group(0) @binding(0) var<storage> s: S;
            override n: u32;
            override m: f32 = 1.0;
            @compute @workgroup_size(n, 2) fn cs() { _ = s.a; }
            ",
        );
        let s = r
            .types
            .iter()
            .find(|t| t.name.as_deref() == Some("S"))
            .unwrap();
        assert_eq!((s.size, s.align), (32, 16));
        let TypeInner::Struct { ref members, .. } = s.inner else {
            panic!()
        };
        assert_eq!(members[1].offset, 16);
        assert_eq!(r.overrides.len(), 2);
        assert!(!r.overrides[0].has_default);
        assert!(r.overrides[1].has_default);
        let ep = &r.entry_points[0];
        assert_eq!(ep.workgroup_size_from_overrides, [true, false, false]);
    }
}
