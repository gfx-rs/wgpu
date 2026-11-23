//! Store `var<immediate>` globals that contain 16-bit scalars as 32-bit words.

use alloc::{borrow::Cow, format, vec, vec::Vec};
use core::mem;

use super::{
    emit::{self, access_base, with_base, Emitting},
    PassError,
};
use crate::{
    arena::Arena, proc::Alignment, AddressSpace, Expression, FastHashMap, FastHashSet, Function,
    GlobalVariable, Handle, Module, Scalar, ScalarKind, Span, StructMember, Type, TypeInner,
    UniqueArena, VectorSize,
};

/// An error from the immediate words pass.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ImmediateWordsError {
    #[error("unsupported use of an immediate pointer: {0:?}")]
    UnsupportedUse(Handle<Expression>),
    #[error("immediate global of {size} bytes is larger than the {MAX_SIZE} bytes supported")]
    TooLarge { size: u32 },
}

/// The largest immediate global, in bytes, that the pass rewrites. This is far
/// above any device's immediate size, and keeps the words struct well within
/// the limits of the targets.
const MAX_SIZE: u32 = 0x10000;

use ImmediateWordsError::UnsupportedUse;

const U32: Type = Type {
    name: None,
    inner: TypeInner::Scalar(Scalar::U32),
};

/// Give each `var<immediate>` global that contains a 16-bit scalar a struct of
/// `u32` words as its type, and rewrite its loads to assemble the value from
/// those words. Return whether the pass changed `module`; it doesn't touch
/// `module` if no such global exists.
pub(crate) fn run(module: &mut Cow<'_, Module>) -> Result<bool, PassError> {
    let mut globals = FastHashMap::default();
    let candidates: Vec<_> = module
        .global_variables
        .iter()
        .filter(|&(_, global)| {
            global.space == AddressSpace::Immediate && has_16_bit(&module.types, global.ty)
        })
        .map(|(handle, global)| (handle, global.ty))
        .collect();
    if candidates.is_empty() {
        return Ok(false);
    }
    let module = module.to_mut();

    let u32_ty = module.types.insert(U32, Span::UNDEFINED);
    for (handle, ty) in candidates {
        let size = module.types[ty].inner.size(module.to_ctx());
        if size > MAX_SIZE {
            return Err(ImmediateWordsError::TooLarge { size }.into());
        }
        let words = size.div_ceil(4);
        let members = (0..words)
            .map(|word| StructMember {
                name: Some(format!("w_{word}")),
                ty: u32_ty,
                binding: None,
                offset: word * 4,
            })
            .collect();
        let name = module.types[ty]
            .name
            .as_deref()
            .map_or_else(|| "immediates_words".into(), |name| format!("{name}_words"));
        let words_ty = module.types.insert(
            Type {
                name: Some(name),
                inner: TypeInner::Struct {
                    members,
                    span: words * 4,
                },
            },
            Span::UNDEFINED,
        );
        module.global_variables[handle].ty = words_ty;
        globals.insert(handle, ty);
    }

    let mut rewriter = Rewriter {
        types: &mut module.types,
        globals: &globals,
    };
    for (_, function) in module.functions.iter_mut() {
        rewriter.function(function)?;
    }
    for entry_point in module.entry_points.iter_mut() {
        rewriter.function(&mut entry_point.function)?;
    }
    Ok(true)
}

fn has_16_bit(types: &UniqueArena<Type>, ty: Handle<Type>) -> bool {
    match types[ty].inner {
        TypeInner::Scalar(scalar)
        | TypeInner::Vector { scalar, .. }
        | TypeInner::Matrix { scalar, .. } => scalar.width == 2,
        TypeInner::Struct { ref members, .. } => {
            members.iter().any(|member| has_16_bit(types, member.ty))
        }
        TypeInner::Array { base, .. } => has_16_bit(types, base),
        _ => false,
    }
}

/// What a pointer expression rooted at a rewritten global refers to.
#[derive(Clone, Copy)]
enum Place {
    /// The value of type `ty` at byte `offset` of the words that the global
    /// expression `root` points to.
    Data {
        root: Handle<Expression>,
        offset: u32,
        ty: Handle<Type>,
    },
    /// An access into a dynamically indexed vector or matrix. It is rewritten
    /// as an access into the loaded value.
    Value,
}

type Places = FastHashMap<Handle<Expression>, Place>;

struct Rewriter<'a> {
    types: &'a mut UniqueArena<Type>,
    /// The original type of each rewritten global.
    globals: &'a FastHashMap<Handle<GlobalVariable>, Handle<Type>>,
}

impl Rewriter<'_> {
    /// Rewrite `function`'s uses of rewritten globals. Return whether it
    /// changed.
    fn function(&mut self, function: &mut Function) -> Result<bool, ImmediateWordsError> {
        let (places, loads) = self.places(&function.expressions)?;
        if places.is_empty() {
            return Ok(false);
        }
        function
            .named_expressions
            .retain(|handle, _| !places.contains_key(handle));
        let mut body = mem::take(&mut function.body);
        let result = emit::rewrite_loads(
            &mut body,
            &mut function.expressions,
            &loads,
            &mut |statement| {
                let mut place_operand = None;
                emit::for_each_direct_operand(statement, |operand| {
                    if places.contains_key(&operand) {
                        place_operand.get_or_insert(operand);
                    }
                });
                place_operand.map_or(Ok(()), |operand| Err(UnsupportedUse(operand)))
            },
            &mut |out, load| {
                let value = self.load(out, &places, load)?;
                out.expressions[load] = value;
                Ok(true)
            },
        );
        function.body = body;
        result.map(|()| true)
    }

    /// Find the place of each pointer expression rooted at a rewritten global,
    /// and the loads from these places.
    fn places(
        &mut self,
        expressions: &Arena<Expression>,
    ) -> Result<(Places, FastHashSet<Handle<Expression>>), ImmediateWordsError> {
        let mut places = Places::default();
        let mut loads = FastHashSet::default();
        for (handle, expr) in expressions.iter() {
            let place = match *expr {
                Expression::GlobalVariable(global) => {
                    self.globals.get(&global).map(|&ty| Place::Data {
                        root: handle,
                        offset: 0,
                        ty,
                    })
                }
                Expression::AccessIndex { base, index } => match places.get(&base).copied() {
                    None => None,
                    Some(Place::Data { root, offset, ty }) => {
                        let (member_offset, member_ty) =
                            self.member(ty, index).ok_or(UnsupportedUse(handle))?;
                        Some(Place::Data {
                            root,
                            offset: offset + member_offset,
                            ty: member_ty,
                        })
                    }
                    Some(Place::Value) => Some(Place::Value),
                },
                Expression::Access { base, .. } => match places.get(&base).copied() {
                    None => None,
                    Some(Place::Data { ty, .. }) => match self.types[ty].inner {
                        TypeInner::Vector { .. } | TypeInner::Matrix { .. } => Some(Place::Value),
                        _ => return Err(UnsupportedUse(handle)),
                    },
                    Some(Place::Value) => Some(Place::Value),
                },
                Expression::Load { pointer } => {
                    if places.contains_key(&pointer) {
                        loads.insert(handle);
                    }
                    None
                }
                ref other => {
                    let mut uses_place = false;
                    crate::compact::expressions::for_each_operand(other, |operand| {
                        uses_place |= places.contains_key(&operand);
                    });
                    if uses_place {
                        return Err(UnsupportedUse(handle));
                    }
                    None
                }
            };
            if let Some(place) = place {
                places.insert(handle, place);
            }
        }
        Ok((places, loads))
    }

    /// The byte offset and type of the member at `index` of a value of type
    /// `ty`.
    fn member(&mut self, ty: Handle<Type>, index: u32) -> Option<(u32, Handle<Type>)> {
        match self.types[ty].inner {
            TypeInner::Struct { ref members, .. } => {
                let member = members.get(index as usize)?;
                Some((member.offset, member.ty))
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } if index < columns as u32 => {
                let stride = Alignment::from(rows) * u32::from(scalar.width);
                let column = self.vector(rows, scalar);
                Some((index * stride, column))
            }
            TypeInner::Vector { size, scalar } if index < size as u32 => {
                let ty = self.scalar(scalar);
                Some((index * u32::from(scalar.width), ty))
            }
            _ => None,
        }
    }

    fn vector(&mut self, size: VectorSize, scalar: Scalar) -> Handle<Type> {
        self.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector { size, scalar },
            },
            Span::UNDEFINED,
        )
    }

    fn scalar(&mut self, scalar: Scalar) -> Handle<Type> {
        self.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(scalar),
            },
            Span::UNDEFINED,
        )
    }

    /// Return the expression that `load` must become.
    fn load(
        &mut self,
        out: &mut Emitting,
        places: &Places,
        load: Handle<Expression>,
    ) -> Result<Expression, ImmediateWordsError> {
        let Expression::Load { pointer } = out.expressions[load] else {
            return Err(UnsupportedUse(load));
        };
        let mut chain = Vec::new();
        let mut cursor = pointer;
        while let Some(&Place::Value) = places.get(&cursor) {
            chain.push(cursor);
            cursor = access_base(&out.expressions[cursor]);
        }
        let Some(&Place::Data { root, offset, ty }) = places.get(&cursor) else {
            return Err(UnsupportedUse(cursor));
        };
        let mut words = Words {
            root,
            halves: FastHashMap::default(),
        };
        let value = self.build(out, &mut words, offset, ty, load)?;
        let Some((&last, rest)) = chain.split_first() else {
            return Ok(value);
        };
        let mut value = out.expr(value);
        for &access in rest.iter().rev() {
            value = out.expr(with_base(&out.expressions[access], value));
        }
        Ok(with_base(&out.expressions[last], value))
    }

    /// Return an expression for the value of type `ty` at byte `offset` of
    /// `words`.
    ///
    /// `site` is the load being rewritten, for errors.
    fn build(
        &mut self,
        out: &mut Emitting,
        words: &mut Words,
        offset: u32,
        ty: Handle<Type>,
        site: Handle<Expression>,
    ) -> Result<Expression, ImmediateWordsError> {
        let components = match self.types[ty].inner.clone() {
            TypeInner::Scalar(scalar) => {
                return words.scalar(out, self.types, offset, scalar, site)
            }
            TypeInner::Vector { size, scalar } => {
                let mut components = Vec::with_capacity(size as usize);
                for index in 0..size as u32 {
                    let offset = offset + index * u32::from(scalar.width);
                    let component = words.scalar(out, self.types, offset, scalar, site)?;
                    components.push(out.expr(component));
                }
                components
            }
            TypeInner::Matrix { columns, .. } => {
                let mut components = Vec::with_capacity(columns as usize);
                for index in 0..columns as u32 {
                    let (column_offset, column_ty) = self.member(ty, index).unwrap();
                    let column = self.build(out, words, offset + column_offset, column_ty, site)?;
                    components.push(out.expr(column));
                }
                components
            }
            TypeInner::Struct { members, .. } => {
                let mut components = Vec::with_capacity(members.len());
                for member in members {
                    let member = self.build(out, words, offset + member.offset, member.ty, site)?;
                    components.push(out.expr(member));
                }
                components
            }
            _ => return Err(UnsupportedUse(site)),
        };
        Ok(Expression::Compose { ty, components })
    }
}

/// The words of one rewritten global, as read by one load.
struct Words {
    /// The global expression.
    root: Handle<Expression>,
    /// The `vec2<f16>` that each word holding `f16`s was reinterpreted as.
    halves: FastHashMap<u32, Handle<Expression>>,
}

impl Words {
    fn word(&self, out: &mut Emitting, index: u32) -> Handle<Expression> {
        let pointer = out.expr(Expression::AccessIndex {
            base: self.root,
            index,
        });
        out.expr(Expression::Load { pointer })
    }

    /// Return an expression for the scalar at byte `offset`.
    fn scalar(
        &mut self,
        out: &mut Emitting,
        types: &mut UniqueArena<Type>,
        offset: u32,
        scalar: Scalar,
        site: Handle<Expression>,
    ) -> Result<Expression, ImmediateWordsError> {
        let mut insert = |inner| types.insert(Type { name: None, inner }, Span::UNDEFINED);
        let index = offset / 4;
        Ok(match (scalar.kind, scalar.width) {
            (ScalarKind::Uint, 4) => {
                let pointer = out.expr(Expression::AccessIndex {
                    base: self.root,
                    index,
                });
                Expression::Load { pointer }
            }
            (ScalarKind::Sint | ScalarKind::Float, 4) => Expression::Bitcast {
                expr: self.word(out, index),
                ty: insert(TypeInner::Scalar(scalar)),
            },
            (ScalarKind::Float, 2) => {
                let halves = match self.halves.get(&index) {
                    Some(&halves) => halves,
                    None => {
                        let word = self.word(out, index);
                        let ty = insert(TypeInner::Vector {
                            size: VectorSize::Bi,
                            scalar,
                        });
                        let halves = out.expr(Expression::Bitcast { expr: word, ty });
                        self.halves.insert(index, halves);
                        halves
                    }
                };
                Expression::AccessIndex {
                    base: halves,
                    index: (offset % 4) / 2,
                }
            }
            (ScalarKind::Uint | ScalarKind::Sint | ScalarKind::Float, 8) => {
                let low = self.word(out, index);
                let high = self.word(out, index + 1);
                let ty = insert(TypeInner::Vector {
                    size: VectorSize::Bi,
                    scalar: Scalar::U32,
                });
                let pair = out.expr(Expression::Compose {
                    ty,
                    components: vec![low, high],
                });
                Expression::Bitcast {
                    expr: pair,
                    ty: insert(TypeInner::Scalar(scalar)),
                }
            }
            _ => return Err(UnsupportedUse(site)),
        })
    }
}

#[cfg(all(test, feature = "wgsl-in"))]
mod tests {
    use super::*;
    use crate::valid::{Capabilities, ValidationFlags, Validator};

    fn validator() -> Validator {
        Validator::new(ValidationFlags::all(), Capabilities::all())
    }

    /// Run the pass on `source`, then compact and validate it as the backend
    /// pass runner does. Also return whether the pass changed the module.
    fn prepare(source: &str) -> (Module, bool) {
        let module = crate::front::wgsl::parse_str(source).unwrap();
        validator()
            .validate(&module)
            .unwrap_or_else(|e| panic!("invalid input: {}", e.emit_to_string(source)));
        let mut prepared = Cow::Borrowed(&module);
        let changed = run(&mut prepared).unwrap();
        let (prepared, _) = super::super::finish(
            prepared,
            changed,
            false,
            &crate::compact::Dropped::default(),
        )
        .unwrap_or_else(|e| panic!("invalid output: {e:?}"));
        (prepared.into_owned(), changed)
    }

    fn has_bitcast(function: &Function) -> bool {
        function
            .expressions
            .iter()
            .any(|(_, expression)| matches!(*expression, Expression::Bitcast { .. }))
    }

    fn immediate_type(module: &Module) -> Handle<Type> {
        let (_, global) = module
            .global_variables
            .iter()
            .find(|&(_, global)| global.space == AddressSpace::Immediate)
            .unwrap();
        global.ty
    }

    /// Assert that the immediate global is a struct of `words` `u32` words.
    fn assert_words(module: &Module, words: usize) {
        let ty = immediate_type(module);
        assert!(!has_16_bit(&module.types, ty));
        let TypeInner::Struct {
            ref members, span, ..
        } = module.types[ty].inner
        else {
            panic!("not a struct");
        };
        assert_eq!(members.len(), words);
        assert_eq!(span as usize, words * 4);
        for (index, member) in members.iter().enumerate() {
            assert_eq!(member.offset as usize, index * 4);
            assert_eq!(
                module.types[member.ty].inner,
                TypeInner::Scalar(Scalar::U32)
            );
        }
    }

    #[test]
    fn struct_with_f16() {
        let (module, changed) = prepare(
            "
            enable f16;
            struct Inner { h: f16, i: i32 }
            struct Data { a: f16, b: f16, v3: vec3<f16>, m: mat2x3<f16>, inner: Inner, f: f32, u: u32 }
            var<immediate> data: Data;
            @compute @workgroup_size(1) fn main() {
                let whole = data;
                let b = data.b;
                let z = data.v3.z;
                let column = data.m[1];
                let h = data.inner.h;
                let f = data.f + f32(data.u);
            }
            ",
        );
        assert_words(&module, 12);
        assert_eq!(
            module.types[immediate_type(&module)].name.as_deref(),
            Some("Data_words")
        );
        assert!(changed);
    }

    #[test]
    fn dynamic_indices() {
        let (module, _) = prepare(
            "
            enable f16;
            struct Data { v: vec3<f16>, m: mat3x3<f16> }
            var<immediate> data: Data;
            @compute @workgroup_size(1) fn main(@builtin(local_invocation_index) i: u32) {
                let a = data.v[i];
                let b = data.m[i][i];
                let c = data.m[i].y;
                let d = data.m[2][i];
            }
            ",
        );
        assert_words(&module, 8);
    }

    #[test]
    fn top_level_scalar() {
        let (module, _) = prepare(
            "
            enable f16;
            var<immediate> h: f16;
            @compute @workgroup_size(1) fn main() {
                let x = h;
            }
            ",
        );
        assert_words(&module, 1);
    }

    #[test]
    fn sixty_four_bit_members() {
        let (module, _) = prepare(
            "
            enable f16;
            enable wgpu_f64;
            struct Data { h: f16, u: u64, i: i64, f: f64 }
            var<immediate> data: Data;
            @compute @workgroup_size(1) fn main() {
                let whole = data;
                let u = data.u;
            }
            ",
        );
        assert_words(&module, 8);
    }

    #[test]
    fn helper_function() {
        let (module, changed) = prepare(
            "
            enable f16;
            var<immediate> v: vec2<f16>;
            fn second() -> f16 { return v.y; }
            @compute @workgroup_size(1) fn main() {
                let x = second();
            }
            ",
        );
        assert_words(&module, 1);
        assert!(changed);
        // Only the helper reads the immediate, so only it assembles words.
        let (_, helper) = module.functions.iter().next().unwrap();
        assert!(has_bitcast(helper));
        assert!(!has_bitcast(&module.entry_points[0].function));
    }

    #[test]
    fn too_large() {
        let module = crate::front::wgsl::parse_str(
            "
            enable f16;
            struct S { h: f16, @size(262200) x: u32 }
            var<immediate> s: S;
            @compute @workgroup_size(1) fn main() {
                let h = s.h;
            }
            ",
        )
        .unwrap();
        let result = run(&mut Cow::Borrowed(&module));
        assert!(matches!(
            result,
            Err(PassError::ImmediateWords(ImmediateWordsError::TooLarge {
                size: 262204
            }))
        ));
    }

    #[test]
    fn untouched_without_16_bit() {
        let source = "
            struct Data { a: u32, v: vec3<f32> }
            var<immediate> data: Data;
            @compute @workgroup_size(1) fn main() {
                let x = data.v.y;
            }
            ";
        let before = crate::front::wgsl::parse_str(source).unwrap();
        let (module, changed) = prepare(source);
        assert_eq!(immediate_type(&module), immediate_type(&before));
        assert_eq!(module.types.len(), before.types.len());
        assert!(!changed);
    }
}
