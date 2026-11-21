//! Give `var<uniform>` globals types that have std140-compatible layouts.

mod rewrite;

use alloc::{borrow::Cow, format, string::String, vec::Vec};

use super::PassError;
use crate::{
    proc::Alignment, AddressSpace, Expression, FastHashMap, FastHashSet, Handle, Module, Scalar,
    Span, StructMember, Type, TypeInner, UniqueArena, VectorSize,
};

/// An error from the uniform matrix layout pass.
#[derive(Clone, Debug, thiserror::Error)]
pub enum UniformMatrixLayoutError {
    #[error("unsupported use of a uniform pointer: {0:?}")]
    UnsupportedUse(Handle<Expression>),
}

/// Give each `var<uniform>` global that contains an affected matrix its
/// decomposed type, and rewrite the expressions that use it.
///
/// Conversion functions for arrays are appended to `module.functions`, after
/// the functions that call them; compaction puts them in order. Return
/// whether the pass changed `module`; it doesn't touch `module` if no uniform
/// contains an affected matrix.
pub(crate) fn run(module: &mut Cow<'_, Module>) -> Result<bool, PassError> {
    let affected = module.global_variables.iter().any(|(_, global)| {
        global.space == AddressSpace::Uniform && contains_affected(&module.types, global.ty)
    });
    if !affected {
        return Ok(false);
    }
    let module = module.to_mut();

    let mut layouts = Layouts::default();
    let mut globals = FastHashMap::default();
    let uniforms: Vec<_> = module
        .global_variables
        .iter()
        .filter(|&(_, global)| global.space == AddressSpace::Uniform)
        .map(|(handle, _)| handle)
        .collect();
    for handle in uniforms {
        let ty = module.global_variables[handle].ty;
        if let Some(decomposed) = layouts.decompose(module, ty) {
            module.global_variables[handle].ty = decomposed;
            globals.insert(handle, ty);
        }
    }
    if !globals.is_empty() {
        rewrite::rewrite(module, &layouts, &globals)?;
    }
    Ok(true)
}

/// Whether `ty` contains an affected matrix, which [`Layouts::decompose`]
/// would decompose.
fn contains_affected(types: &UniqueArena<Type>, ty: Handle<Type>) -> bool {
    match types[ty].inner {
        TypeInner::Matrix { rows, scalar, .. } => is_affected_matrix(rows, scalar),
        TypeInner::Array { base, .. } | TypeInner::BindingArray { base, .. } => {
            contains_affected(types, base)
        }
        TypeInner::Struct { ref members, .. } => members
            .iter()
            .any(|member| contains_affected(types, member.ty)),
        _ => false,
    }
}

/// Whether a matrix with `rows` rows has columns that std140 would pad to 16
/// bytes, but WGSL's uniform layout does not.
pub(super) fn is_affected_matrix(rows: VectorSize, scalar: Scalar) -> bool {
    Alignment::from(rows) * Alignment::from_width(scalar.width) < Alignment::MIN_UNIFORM
}

/// Where a member of a decomposed struct went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MemberMap {
    /// The member is the one at this index of the new struct.
    Same(u32),
    /// The matrix member became `columns` vector members, starting at `first`.
    Split { first: u32, columns: VectorSize },
}

/// A type's decomposition.
#[derive(Clone, Debug)]
pub(super) struct Decomposed {
    pub ty: Handle<Type>,
    /// For a struct, where each of its members went. Empty for other types.
    pub members: Vec<MemberMap>,
}

/// The decompositions of the types of a module.
#[derive(Default)]
pub(super) struct Layouts {
    pub map: FastHashMap<Handle<Type>, Decomposed>,
    unaffected: FastHashSet<Handle<Type>>,
}

impl Layouts {
    /// Return the std140-compatible counterpart of `ty`, or `None` if `ty` has
    /// no affected matrix.
    pub fn decompose(&mut self, module: &mut Module, ty: Handle<Type>) -> Option<Handle<Type>> {
        if let Some(decomposed) = self.map.get(&ty) {
            return Some(decomposed.ty);
        }
        if self.unaffected.contains(&ty) {
            return None;
        }

        let name = module.types[ty].name.clone();
        let decomposed = match module.types[ty].inner.clone() {
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } if is_affected_matrix(rows, scalar) => Some(Decomposed {
                ty: matrix_wrapper(module, columns, rows, scalar),
                members: Vec::new(),
            }),
            TypeInner::Array { base, size, stride } => {
                self.decompose(module, base).map(|base| Decomposed {
                    ty: module.types.insert(
                        Type {
                            name: None,
                            inner: TypeInner::Array { base, size, stride },
                        },
                        Span::UNDEFINED,
                    ),
                    members: Vec::new(),
                })
            }
            TypeInner::BindingArray { base, size } => {
                self.decompose(module, base).map(|base| Decomposed {
                    ty: module.types.insert(
                        Type {
                            name: None,
                            inner: TypeInner::BindingArray { base, size },
                        },
                        Span::UNDEFINED,
                    ),
                    members: Vec::new(),
                })
            }
            TypeInner::Struct { members, span } => {
                self.decompose_struct(module, name, &members, span)
            }
            _ => None,
        };

        match decomposed {
            Some(decomposed) => {
                let handle = decomposed.ty;
                self.map.insert(ty, decomposed);
                Some(handle)
            }
            None => {
                self.unaffected.insert(ty);
                None
            }
        }
    }

    fn decompose_struct(
        &mut self,
        module: &mut Module,
        name: Option<String>,
        members: &[StructMember],
        span: u32,
    ) -> Option<Decomposed> {
        let mut new_members = Vec::with_capacity(members.len());
        let mut maps = Vec::with_capacity(members.len());
        let mut changed = false;

        for member in members {
            let next = new_members.len() as u32;
            if let TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } = module.types[member.ty].inner
            {
                if is_affected_matrix(rows, scalar) {
                    changed = true;
                    maps.push(MemberMap::Split {
                        first: next,
                        columns,
                    });
                    let column_ty = vector(module, rows, scalar);
                    let stride = column_stride(rows, scalar);
                    for column in 0..columns as u32 {
                        new_members.push(StructMember {
                            name: member.name.as_ref().map(|name| format!("{name}_{column}")),
                            ty: column_ty,
                            binding: None,
                            offset: member.offset + column * stride,
                        });
                    }
                    continue;
                }
            }

            maps.push(MemberMap::Same(next));
            let ty = match self.decompose(module, member.ty) {
                Some(ty) => {
                    changed = true;
                    ty
                }
                None => member.ty,
            };
            new_members.push(StructMember {
                ty,
                ..member.clone()
            });
        }

        changed.then(|| Decomposed {
            ty: module.types.insert(
                Type {
                    name: name.map(|name| format!("{name}_std140")),
                    inner: TypeInner::Struct {
                        members: new_members,
                        span,
                    },
                },
                Span::UNDEFINED,
            ),
            members: maps,
        })
    }
}

fn column_stride(rows: VectorSize, scalar: Scalar) -> u32 {
    Alignment::from(rows) * u32::from(scalar.width)
}

fn vector(module: &mut Module, size: VectorSize, scalar: Scalar) -> Handle<Type> {
    module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector { size, scalar },
        },
        Span::UNDEFINED,
    )
}

fn matrix_wrapper(
    module: &mut Module,
    columns: VectorSize,
    rows: VectorSize,
    scalar: Scalar,
) -> Handle<Type> {
    let column_ty = vector(module, rows, scalar);
    let stride = column_stride(rows, scalar);
    let members = (0..columns as u32)
        .map(|column| StructMember {
            name: Some(format!("c_{column}")),
            ty: column_ty,
            binding: None,
            offset: column * stride,
        })
        .collect();
    module.types.insert(
        Type {
            name: Some(format!(
                "mat{}x{}f{}_std140",
                columns as u32,
                rows as u32,
                scalar.width * 8
            )),
            inner: TypeInner::Struct {
                members,
                span: columns as u32 * stride,
            },
        },
        Span::UNDEFINED,
    )
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use super::*;
    use crate::ArraySize;

    fn parse(source: &str) -> Module {
        crate::front::wgsl::parse_str(source).unwrap()
    }

    fn find_named(module: &Module, name: &str) -> Handle<Type> {
        module
            .types
            .iter()
            .find(|&(_, ty)| ty.name.as_deref() == Some(name))
            .unwrap()
            .0
    }

    fn find_inner(module: &Module, pred: impl Fn(&TypeInner) -> bool) -> Handle<Type> {
        module
            .types
            .iter()
            .find(|&(_, ty)| pred(&ty.inner))
            .unwrap()
            .0
    }

    fn leaves(module: &Module, ty: Handle<Type>, base: u32, out: &mut Vec<(u32, Scalar)>) {
        match module.types[ty].inner {
            TypeInner::Scalar(scalar) => out.push((base, scalar)),
            TypeInner::Vector { size, scalar } => {
                for i in 0..size as u32 {
                    out.push((base + i * u32::from(scalar.width), scalar));
                }
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                let stride = Alignment::from(rows) * u32::from(scalar.width);
                for column in 0..columns as u32 {
                    for row in 0..rows as u32 {
                        out.push((
                            base + column * stride + row * u32::from(scalar.width),
                            scalar,
                        ));
                    }
                }
            }
            TypeInner::Array {
                base: element,
                size: ArraySize::Constant(len),
                stride,
            } => {
                for i in 0..len.get() {
                    leaves(module, element, base + i * stride, out);
                }
            }
            TypeInner::Struct { ref members, .. } => {
                for member in members {
                    leaves(module, member.ty, base + member.offset, out);
                }
            }
            ref other => panic!("unexpected type {other:?}"),
        }
    }

    fn all_leaves(module: &Module, ty: Handle<Type>) -> Vec<(u32, Scalar)> {
        let mut out = Vec::new();
        leaves(module, ty, 0, &mut out);
        out
    }

    fn member_offsets(module: &Module, ty: Handle<Type>) -> Vec<u32> {
        match module.types[ty].inner {
            TypeInner::Struct { ref members, .. } => members.iter().map(|m| m.offset).collect(),
            ref other => panic!("not a struct: {other:?}"),
        }
    }

    fn span(module: &Module, ty: Handle<Type>) -> u32 {
        match module.types[ty].inner {
            TypeInner::Struct { span, .. } => span,
            ref other => panic!("not a struct: {other:?}"),
        }
    }

    #[test]
    fn affected_matrices() {
        use VectorSize::*;
        assert!(is_affected_matrix(Bi, Scalar::F32));
        assert!(is_affected_matrix(Bi, Scalar::F16));
        assert!(is_affected_matrix(Tri, Scalar::F16));
        assert!(is_affected_matrix(Quad, Scalar::F16));
        assert!(!is_affected_matrix(Tri, Scalar::F32));
        assert!(!is_affected_matrix(Quad, Scalar::F32));
        assert!(!is_affected_matrix(Bi, Scalar::F64));
    }

    #[test]
    fn struct_member_split() {
        let mut module = parse(
            "
            struct S { a: f32, m: mat3x2<f32>, b: f32 }
            @group(0) @binding(0) var<uniform> u: S;
            ",
        );
        let s = find_named(&module, "S");
        let mut layouts = Layouts::default();
        let decomposed = layouts.decompose(&mut module, s).unwrap();
        let members = layouts.map[&s].members.clone();

        assert_eq!(member_offsets(&module, decomposed), [0, 8, 16, 24, 32]);
        assert_eq!(span(&module, decomposed), span(&module, s));
        assert_eq!(all_leaves(&module, decomposed), all_leaves(&module, s));
        assert_eq!(
            members,
            [
                MemberMap::Same(0),
                MemberMap::Split {
                    first: 1,
                    columns: VectorSize::Tri
                },
                MemberMap::Same(4)
            ]
        );
        match module.types[decomposed].inner {
            TypeInner::Struct { ref members, .. } => {
                let names: Vec<_> = members.iter().map(|m| m.name.as_deref().unwrap()).collect();
                assert_eq!(names, ["a", "m_0", "m_1", "m_2", "b"]);
            }
            _ => unreachable!(),
        }
        assert_eq!(module.types[decomposed].name.as_deref(), Some("S_std140"));
    }

    #[test]
    fn unaffected_untouched() {
        let mut module = parse(
            "
            struct S { m: mat3x3<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            @group(0) @binding(1) var<uniform> v: mat3x3<f32>;
            ",
        );
        let s = find_named(&module, "S");
        let m = find_inner(&module, |inner| matches!(inner, TypeInner::Matrix { .. }));
        let mut layouts = Layouts::default();
        assert_eq!(layouts.decompose(&mut module, m), None);
        assert_eq!(layouts.decompose(&mut module, s), None);
    }

    #[test]
    fn array_of_matrix() {
        let mut module = parse(
            "
            @group(0) @binding(0) var<uniform> u: array<mat2x2<f32>, 4>;
            ",
        );
        let array = find_inner(&module, |inner| matches!(inner, TypeInner::Array { .. }));
        let mut layouts = Layouts::default();
        let decomposed = layouts.decompose(&mut module, array).unwrap();
        match module.types[decomposed].inner {
            TypeInner::Array { base, size, stride } => {
                assert_eq!(stride, 16);
                assert_eq!(size, ArraySize::Constant(4.try_into().unwrap()));
                assert_eq!(module.types[base].name.as_deref(), Some("mat2x2f32_std140"));
                assert_eq!(member_offsets(&module, base), [0, 8]);
            }
            ref other => panic!("{other:?}"),
        }
        assert_eq!(module.types[decomposed].name, None);
        assert_eq!(all_leaves(&module, decomposed), all_leaves(&module, array));
    }

    #[test]
    fn f16_cx3() {
        let mut module = parse(
            "
            enable f16;
            @group(0) @binding(0) var<uniform> u: mat4x3<f16>;
            ",
        );
        let matrix = find_inner(&module, |inner| matches!(inner, TypeInner::Matrix { .. }));
        let mut layouts = Layouts::default();
        let wrapper = layouts.decompose(&mut module, matrix).unwrap();
        assert_eq!(member_offsets(&module, wrapper), [0, 8, 16, 24]);
        assert_eq!(span(&module, wrapper), 32);
        assert_eq!(all_leaves(&module, wrapper), all_leaves(&module, matrix));
        assert!(layouts.map[&matrix].members.is_empty());
    }

    #[test]
    fn memoized() {
        let mut module = parse(
            "
            struct S { m: mat2x2<f32> }
            @group(0) @binding(0) var<uniform> u: S;
            ",
        );
        let s = find_named(&module, "S");
        let mut layouts = Layouts::default();
        let first = layouts.decompose(&mut module, s);
        let count = module.types.len();
        assert_eq!(layouts.decompose(&mut module, s), first);
        assert_eq!(module.types.len(), count);
    }

    #[test]
    fn nested_struct_and_array_members() {
        let mut module = parse(
            "
            struct Inner { m: mat2x2<f32> }
            struct Outer { a: f32, i: Inner, arr: array<mat2x2<f32>, 2> }
            @group(0) @binding(0) var<uniform> u: Outer;
            ",
        );
        let outer = find_named(&module, "Outer");
        let inner = find_named(&module, "Inner");
        let mut layouts = Layouts::default();
        let decomposed = layouts.decompose(&mut module, outer).unwrap();

        assert_eq!(
            layouts.map[&outer].members,
            [MemberMap::Same(0), MemberMap::Same(1), MemberMap::Same(2)]
        );
        assert_eq!(
            layouts.map[&inner].members,
            [MemberMap::Split {
                first: 0,
                columns: VectorSize::Bi
            }]
        );
        assert_eq!(span(&module, decomposed), span(&module, outer));
        assert_eq!(
            member_offsets(&module, decomposed),
            member_offsets(&module, outer)
        );
        assert_eq!(all_leaves(&module, decomposed), all_leaves(&module, outer));

        let TypeInner::Struct { ref members, .. } = module.types[decomposed].inner else {
            unreachable!()
        };
        assert_eq!(Some(members[1].ty), layouts.map.get(&inner).map(|d| d.ty));
        assert_ne!(members[1].ty, inner);
        let TypeInner::Array { base, .. } = module.types[members[2].ty].inner else {
            panic!("member is not an array")
        };
        assert_eq!(module.types[base].name.as_deref(), Some("mat2x2f32_std140"));
    }

    #[test]
    fn array_of_structs() {
        let mut module = parse(
            "
            struct Inner { a: vec4<f32>, m: mat2x2<f32> }
            @group(0) @binding(0) var<uniform> u: array<Inner, 2>;
            ",
        );
        let inner = find_named(&module, "Inner");
        let array = find_inner(&module, |inner| matches!(inner, TypeInner::Array { .. }));
        let mut layouts = Layouts::default();
        let decomposed = layouts.decompose(&mut module, array).unwrap();

        let TypeInner::Array { base, stride, .. } = module.types[decomposed].inner else {
            unreachable!()
        };
        let TypeInner::Array {
            stride: old_stride, ..
        } = module.types[array].inner
        else {
            unreachable!()
        };
        assert_eq!(stride, old_stride);
        assert_eq!(Some(base), layouts.map.get(&inner).map(|d| d.ty));
        assert_eq!(all_leaves(&module, decomposed), all_leaves(&module, array));
    }

    #[test]
    fn binding_array() {
        let mut module = parse(
            "
            enable wgpu_binding_array;
            @group(0) @binding(0) var<uniform> u: binding_array<mat2x2<f32>, 2>;
            ",
        );
        let array = find_inner(&module, |inner| {
            matches!(inner, TypeInner::BindingArray { .. })
        });
        let mut layouts = Layouts::default();
        let decomposed = layouts.decompose(&mut module, array).unwrap();
        let TypeInner::BindingArray { base, size } = module.types[decomposed].inner else {
            panic!("not a binding array")
        };
        let TypeInner::BindingArray { size: old_size, .. } = module.types[array].inner else {
            unreachable!()
        };
        assert_eq!(size, old_size);
        assert_eq!(module.types[base].name.as_deref(), Some("mat2x2f32_std140"));
    }

    #[test]
    fn unaffected_is_cached() {
        let mut module = parse(
            "
            struct S { a: vec4<f32>, b: array<vec4<f32>, 2> }
            @group(0) @binding(0) var<uniform> u: S;
            ",
        );
        let s = find_named(&module, "S");
        let mut layouts = Layouts::default();
        let count = module.types.len();
        assert_eq!(layouts.decompose(&mut module, s), None);
        assert!(layouts.unaffected.contains(&s));
        assert_eq!(layouts.decompose(&mut module, s), None);
        assert_eq!(module.types.len(), count);
        assert!(layouts.map.is_empty());
    }
}
