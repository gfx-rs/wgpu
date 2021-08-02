use crate::{
    Binding, Block, BuiltIn, Constant, Expression, GlobalVariable, Handle, Interpolation,
    LocalVariable, ScalarKind, StorageAccess, StorageClass, SwizzleComponent, Type, TypeInner,
    VectorSize,
};

use super::ast::*;
use super::error::ErrorKind;
use super::token::SourceMetadata;

macro_rules! qualifier_arm {
    ($src:expr, $tgt:expr, $meta:expr, $msg:literal $(,)?) => {{
        if $tgt.is_some() {
            return Err(ErrorKind::SemanticError($meta, $msg.into()));
        }

        $tgt = Some($src);
    }};
}

pub struct VarDeclaration<'a> {
    pub qualifiers: &'a [(TypeQualifier, SourceMetadata)],
    pub ty: Handle<Type>,
    pub name: Option<String>,
    pub init: Option<Handle<Constant>>,
    pub meta: SourceMetadata,
}

pub enum GlobalOrConstant {
    Global(Handle<GlobalVariable>),
    Constant(Handle<Constant>),
}

impl Program {
    pub fn lookup_variable(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        name: &str,
    ) -> Result<Option<VariableReference>, ErrorKind> {
        if let Some(local_var) = ctx.lookup_local_var(name) {
            return Ok(Some(local_var));
        }
        if let Some(global_var) = ctx.lookup_global_var(name) {
            return Ok(Some(global_var));
        }

        let mut add_builtin = |inner, builtin, mutable, storage| {
            let ty = self
                .module
                .types
                .fetch_or_append(Type { name: None, inner });

            let handle = self.module.global_variables.append(GlobalVariable {
                name: Some(name.into()),
                class: StorageClass::Private,
                binding: None,
                ty,
                init: None,
            });

            let idx = self.entry_args.len();
            self.entry_args.push(EntryArg {
                name: None,
                binding: Binding::BuiltIn(builtin),
                handle,
                storage,
            });

            self.global_variables.push((
                name.into(),
                GlobalLookup {
                    kind: GlobalLookupKind::Variable(handle),
                    entry_arg: Some(idx),
                    mutable,
                },
            ));

            let expr = ctx.add_expression(Expression::GlobalVariable(handle), body);
            ctx.lookup_global_var_exps.insert(
                name.into(),
                VariableReference {
                    expr,
                    load: true,
                    mutable,
                    entry_arg: Some(idx),
                },
            );

            Ok(ctx.lookup_global_var(name))
        };
        match name {
            "gl_Position" => add_builtin(
                TypeInner::Vector {
                    size: VectorSize::Quad,
                    kind: ScalarKind::Float,
                    width: 4,
                },
                BuiltIn::Position,
                true,
                StorageQualifier::Output,
            ),
            "gl_FragCoord" => add_builtin(
                TypeInner::Vector {
                    size: VectorSize::Quad,
                    kind: ScalarKind::Float,
                    width: 4,
                },
                BuiltIn::Position,
                false,
                StorageQualifier::Input,
            ),
            "gl_FragDepth" => add_builtin(
                TypeInner::Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                },
                BuiltIn::FragDepth,
                true,
                StorageQualifier::Output,
            ),
            "gl_VertexIndex" => add_builtin(
                TypeInner::Scalar {
                    kind: ScalarKind::Uint,
                    width: 4,
                },
                BuiltIn::VertexIndex,
                false,
                StorageQualifier::Input,
            ),
            "gl_InstanceIndex" => add_builtin(
                TypeInner::Scalar {
                    kind: ScalarKind::Uint,
                    width: 4,
                },
                BuiltIn::InstanceIndex,
                false,
                StorageQualifier::Input,
            ),
            "gl_GlobalInvocationID" => add_builtin(
                TypeInner::Vector {
                    size: VectorSize::Tri,
                    kind: ScalarKind::Uint,
                    width: 4,
                },
                BuiltIn::GlobalInvocationId,
                false,
                StorageQualifier::Input,
            ),
            "gl_FrontFacing" => add_builtin(
                TypeInner::Scalar {
                    kind: ScalarKind::Bool,
                    width: crate::BOOL_WIDTH,
                },
                BuiltIn::FrontFacing,
                false,
                StorageQualifier::Input,
            ),
            "gl_PrimitiveID" => add_builtin(
                TypeInner::Scalar {
                    kind: ScalarKind::Uint,
                    width: 4,
                },
                BuiltIn::PrimitiveIndex,
                false,
                StorageQualifier::Input,
            ),
            _ => Ok(None),
        }
    }

    pub fn field_selection(
        &mut self,
        ctx: &mut Context,
        lhs: bool,
        body: &mut Block,
        expression: Handle<Expression>,
        name: &str,
        meta: SourceMetadata,
    ) -> Result<Handle<Expression>, ErrorKind> {
        let (ty, is_pointer) = match *self.resolve_type(ctx, expression, meta)? {
            TypeInner::Pointer { base, .. } => (&self.module.types[base].inner, true),
            ref ty => (ty, false),
        };
        match *ty {
            TypeInner::Struct { ref members, .. } => {
                let index = members
                    .iter()
                    .position(|m| m.name == Some(name.into()))
                    .ok_or_else(|| ErrorKind::UnknownField(meta, name.into()))?;
                Ok(ctx.add_expression(
                    Expression::AccessIndex {
                        base: expression,
                        index: index as u32,
                    },
                    body,
                ))
            }
            // swizzles (xyzw, rgba, stpq)
            TypeInner::Vector { size, .. } => {
                let check_swizzle_components = |comps: &str| {
                    name.chars()
                        .map(|c| {
                            comps
                                .find(c)
                                .filter(|i| *i < size as usize)
                                .map(|i| SwizzleComponent::from_index(i as u32))
                        })
                        .collect::<Option<Vec<SwizzleComponent>>>()
                };

                let components = check_swizzle_components("xyzw")
                    .or_else(|| check_swizzle_components("rgba"))
                    .or_else(|| check_swizzle_components("stpq"));

                if let Some(components) = components {
                    if lhs {
                        let not_unique = (1..components.len())
                            .any(|i| components[i..].contains(&components[i - 1]));
                        if not_unique {
                            return Err(ErrorKind::SemanticError(
                                meta,
                                format!(
                                    "swizzle cannot have duplicate components in left-hand-side expression for \"{:?}\"",
                                    name
                                )
                                .into(),
                            ));
                        }
                    }

                    let mut pattern = [SwizzleComponent::X; 4];
                    for (pat, component) in pattern.iter_mut().zip(&components) {
                        *pat = *component;
                    }

                    // flatten nested swizzles (vec.zyx.xy.x => vec.z)
                    let mut expression = expression;
                    if let Expression::Swizzle {
                        size: _,
                        vector,
                        pattern: ref src_pattern,
                    } = *ctx.get_expression(expression)
                    {
                        expression = vector;
                        for pat in &mut pattern {
                            *pat = src_pattern[pat.index() as usize];
                        }
                    }

                    let size = match components.len() {
                        1 => {
                            // only single element swizzle, like pos.y, just return that component.
                            if lhs {
                                // Because of possible nested swizzles, like pos.xy.x, we have to unwrap the potential load expr.
                                if let Expression::Load { ref pointer } =
                                    *ctx.get_expression(expression)
                                {
                                    expression = *pointer;
                                }
                            }
                            return Ok(ctx.add_expression(
                                Expression::AccessIndex {
                                    base: expression,
                                    index: pattern[0].index(),
                                },
                                body,
                            ));
                        }
                        2 => VectorSize::Bi,
                        3 => VectorSize::Tri,
                        4 => VectorSize::Quad,
                        _ => {
                            return Err(ErrorKind::SemanticError(
                                meta,
                                format!("Bad swizzle size for \"{:?}\"", name).into(),
                            ));
                        }
                    };

                    if is_pointer {
                        // NOTE: for lhs expression, this extra load ends up as an unused expr, because the
                        // assignment will extract the pointer and use it directly anyway. Unfortunately we
                        // need it for validation to pass, as swizzles cannot operate on pointer values.
                        expression = ctx.add_expression(
                            Expression::Load {
                                pointer: expression,
                            },
                            body,
                        );
                    }

                    Ok(ctx.add_expression(
                        Expression::Swizzle {
                            size,
                            vector: expression,
                            pattern,
                        },
                        body,
                    ))
                } else {
                    Err(ErrorKind::SemanticError(
                        meta,
                        format!("Invalid swizzle for vector \"{}\"", name).into(),
                    ))
                }
            }
            _ => Err(ErrorKind::SemanticError(
                meta,
                format!("Can't lookup field on this type \"{}\"", name).into(),
            )),
        }
    }

    pub fn add_global_var(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        VarDeclaration {
            qualifiers,
            ty,
            name,
            init,
            meta,
        }: VarDeclaration,
    ) -> Result<GlobalOrConstant, ErrorKind> {
        let mut storage = StorageQualifier::StorageClass(StorageClass::Private);
        let mut interpolation = None;
        let mut binding = None;
        let mut location = None;
        let mut sampling = None;
        let mut layout = None;
        let mut precision = None;
        let mut access = StorageAccess::all();

        for &(ref qualifier, meta) in qualifiers {
            match *qualifier {
                TypeQualifier::StorageQualifier(s) => {
                    if StorageQualifier::StorageClass(StorageClass::PushConstant) == storage
                        && s == StorageQualifier::StorageClass(StorageClass::Uniform)
                    {
                        // Ignore the Uniform qualifier if the class was already set to PushConstant
                        continue;
                    } else if StorageQualifier::StorageClass(StorageClass::Private) != storage {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    storage = s;
                }
                TypeQualifier::Interpolation(i) => qualifier_arm!(
                    i,
                    interpolation,
                    meta,
                    "Cannot use more than one interpolation qualifier per declaration"
                ),
                TypeQualifier::ResourceBinding(ref r) => qualifier_arm!(
                    r.clone(),
                    binding,
                    meta,
                    "Cannot use more than one binding per declaration"
                ),
                TypeQualifier::Location(l) => qualifier_arm!(
                    l,
                    location,
                    meta,
                    "Cannot use more than one binding per declaration"
                ),
                TypeQualifier::Sampling(s) => qualifier_arm!(
                    s,
                    sampling,
                    meta,
                    "Cannot use more than one sampling qualifier per declaration"
                ),
                TypeQualifier::Layout(ref l) => qualifier_arm!(
                    l,
                    layout,
                    meta,
                    "Cannot use more than one layout qualifier per declaration"
                ),
                TypeQualifier::Precision(ref p) => qualifier_arm!(
                    p,
                    precision,
                    meta,
                    "Cannot use more than one precision qualifier per declaration"
                ),
                TypeQualifier::StorageAccess(a) => access &= a,
                _ => {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "Qualifier not supported in globals".into(),
                    ));
                }
            }
        }

        if binding.is_some() && storage != StorageQualifier::StorageClass(StorageClass::Uniform) {
            match storage {
                StorageQualifier::StorageClass(StorageClass::PushConstant)
                | StorageQualifier::StorageClass(StorageClass::Uniform)
                | StorageQualifier::StorageClass(StorageClass::Storage { .. }) => {}
                _ => {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "binding requires uniform or buffer storage qualifier".into(),
                    ))
                }
            }
        }

        if (sampling.is_some() || interpolation.is_some()) && location.is_none() {
            return Err(ErrorKind::SemanticError(
                meta,
                "Sampling and interpolation qualifiers can only be used in in/out variables".into(),
            ));
        }

        if let Some(location) = location {
            let input = storage == StorageQualifier::Input;
            let interpolation = interpolation.or_else(|| {
                let kind = self.module.types[ty].inner.scalar_kind()?;
                Some(match kind {
                    ScalarKind::Float => Interpolation::Perspective,
                    _ => Interpolation::Flat,
                })
            });

            let handle = self.module.global_variables.append(GlobalVariable {
                name: name.clone(),
                class: StorageClass::Private,
                binding: None,
                ty,
                init,
            });

            let idx = self.entry_args.len();
            self.entry_args.push(EntryArg {
                name: name.clone(),
                binding: Binding::Location {
                    location,
                    interpolation,
                    sampling,
                },
                handle,
                storage,
            });

            if let Some(name) = name {
                let lookup = GlobalLookup {
                    kind: GlobalLookupKind::Variable(handle),
                    entry_arg: Some(idx),
                    mutable: !input,
                };
                ctx.add_global(&name, lookup, self, body);

                self.global_variables.push((name, lookup));
            }

            return Ok(GlobalOrConstant::Global(handle));
        } else if let StorageQualifier::Const = storage {
            let init = init.ok_or_else(|| {
                ErrorKind::SemanticError(meta, "const values must have an initializer".into())
            })?;
            if let Some(name) = name {
                let lookup = GlobalLookup {
                    kind: GlobalLookupKind::Constant(init),
                    entry_arg: None,
                    mutable: false,
                };
                ctx.add_global(&name, lookup, self, body);

                self.global_variables.push((name, lookup));
            }
            return Ok(GlobalOrConstant::Constant(init));
        }

        let class = match self.module.types[ty].inner {
            TypeInner::Image { .. } => StorageClass::Handle,
            TypeInner::Sampler { .. } => StorageClass::Handle,
            _ => {
                if let StorageQualifier::StorageClass(StorageClass::Storage { .. }) = storage {
                    StorageClass::Storage { access }
                } else {
                    match storage {
                        StorageQualifier::StorageClass(class) => class,
                        _ => StorageClass::Private,
                    }
                }
            }
        };

        let handle = self.module.global_variables.append(GlobalVariable {
            name: name.clone(),
            class,
            binding,
            ty,
            init,
        });

        if let Some(name) = name {
            let lookup = GlobalLookup {
                kind: GlobalLookupKind::Variable(handle),
                entry_arg: None,
                mutable: true,
            };
            ctx.add_global(&name, lookup, self, body);

            self.global_variables.push((name, lookup));
        }

        Ok(GlobalOrConstant::Global(handle))
    }

    pub fn add_local_var(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        #[cfg_attr(not(feature = "glsl-validate"), allow(unused_variables))]
        VarDeclaration {
            qualifiers,
            ty,
            name,
            init,
            meta,
        }: VarDeclaration,
    ) -> Result<Handle<Expression>, ErrorKind> {
        #[cfg(feature = "glsl-validate")]
        if let Some(ref name) = name {
            if ctx.lookup_local_var_current_scope(name).is_some() {
                return Err(ErrorKind::VariableAlreadyDeclared(meta, name.clone()));
            }
        }

        let mut mutable = true;
        let mut precision = None;

        for &(ref qualifier, meta) in qualifiers {
            match *qualifier {
                TypeQualifier::StorageQualifier(StorageQualifier::Const) => {
                    if !mutable {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one constant qualifier per declaration".into(),
                        ));
                    }

                    mutable = false;
                }
                TypeQualifier::Precision(ref p) => qualifier_arm!(
                    p,
                    precision,
                    meta,
                    "Cannot use more than one precision qualifier per declaration"
                ),
                _ => {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "Qualifier not supported in locals".into(),
                    ));
                }
            }
        }

        let handle = ctx.locals.append(LocalVariable {
            name: name.clone(),
            ty,
            init,
        });
        let expr = ctx.add_expression(Expression::LocalVariable(handle), body);

        if let Some(name) = name {
            ctx.add_local_var(name, expr, mutable);
        }

        Ok(expr)
    }
}
