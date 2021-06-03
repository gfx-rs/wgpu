use crate::{
    Binding, Block, BuiltIn, Constant, Expression, GlobalVariable, Handle, ImageClass,
    Interpolation, LocalVariable, ScalarKind, StorageAccess, StorageClass, Type, TypeInner,
    VectorSize,
};

use super::ast::*;
use super::error::ErrorKind;
use super::token::SourceMetadata;

pub struct VarDeclaration<'a> {
    pub qualifiers: &'a [TypeQualifier],
    pub ty: Handle<Type>,
    pub name: String,
    pub init: Option<Handle<Constant>>,
    pub meta: SourceMetadata,
}

impl Program<'_> {
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

        let mut add_builtin = |inner, builtin, mutable, prologue| {
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
                storage_access: StorageAccess::empty(),
            });

            let idx = self.entry_args.len();
            self.entry_args.push(EntryArg {
                binding: Binding::BuiltIn(builtin),
                handle,
                prologue,
            });

            self.global_variables.push((
                name.into(),
                GlobalLookup {
                    kind: GlobalLookupKind::Variable(handle),
                    entry_arg: Some(idx),
                },
            ));
            ctx.arg_use.push(EntryArgUse::empty());

            let expr = ctx.add_expression(Expression::GlobalVariable(handle), body);
            let load = ctx.add_expression(Expression::Load { pointer: expr }, body);
            ctx.lookup_global_var_exps.insert(
                name.into(),
                VariableReference {
                    expr,
                    load: Some(load),
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
                PrologueStage::FRAGMENT,
            ),
            "gl_VertexIndex" => add_builtin(
                TypeInner::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                },
                BuiltIn::VertexIndex,
                false,
                PrologueStage::VERTEX,
            ),
            "gl_InstanceIndex" => add_builtin(
                TypeInner::Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                },
                BuiltIn::InstanceIndex,
                false,
                PrologueStage::VERTEX,
            ),
            _ => Ok(None),
        }
    }

    pub fn field_selection(
        &mut self,
        ctx: &mut Context,
        body: &mut Block,
        expression: Handle<Expression>,
        name: &str,
        meta: SourceMetadata,
    ) -> Result<Handle<Expression>, ErrorKind> {
        match *self.resolve_type(ctx, expression, meta)? {
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
            TypeInner::Vector { size, kind, width } => {
                let check_swizzle_components = |comps: &str| {
                    name.chars()
                        .map(|c| {
                            comps
                                .find(c)
                                .and_then(|i| if i < size as usize { Some(i) } else { None })
                        })
                        .fold(Some(Vec::<usize>::new()), |acc, cur| {
                            cur.and_then(|i| {
                                acc.map(|mut v| {
                                    v.push(i);
                                    v
                                })
                            })
                        })
                };

                let indices = check_swizzle_components("xyzw")
                    .or_else(|| check_swizzle_components("rgba"))
                    .or_else(|| check_swizzle_components("stpq"));

                if let Some(v) = indices {
                    let components: Vec<Handle<Expression>> = v
                        .iter()
                        .map(|idx| {
                            ctx.add_expression(
                                Expression::AccessIndex {
                                    base: expression,
                                    index: *idx as u32,
                                },
                                body,
                            )
                        })
                        .collect();
                    if components.len() == 1 {
                        // only single element swizzle, like pos.y, just return that component
                        Ok(components[0])
                    } else {
                        let size = match components.len() {
                            2 => VectorSize::Bi,
                            3 => VectorSize::Tri,
                            4 => VectorSize::Quad,
                            _ => {
                                return Err(ErrorKind::SemanticError(
                                    meta,
                                    format!("Bad swizzle size for \"{:?}\": {:?}", name, v).into(),
                                ));
                            }
                        };
                        Ok(ctx.add_expression(
                            Expression::Compose {
                                ty: self.module.types.fetch_or_append(Type {
                                    name: None,
                                    inner: TypeInner::Vector { kind, width, size },
                                }),
                                components,
                            },
                            body,
                        ))
                    }
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
    ) -> Result<Handle<Expression>, ErrorKind> {
        let mut storage = StorageQualifier::StorageClass(StorageClass::Private);
        let mut interpolation = None;
        let mut binding = None;
        let mut location = None;
        let mut sampling = None;
        let mut layout = None;

        for qualifier in qualifiers {
            match *qualifier {
                TypeQualifier::StorageQualifier(s) => {
                    if StorageQualifier::StorageClass(StorageClass::Private) != storage {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    storage = s;
                }
                TypeQualifier::Interpolation(i) => {
                    if interpolation.is_some() {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    interpolation = Some(i);
                }
                TypeQualifier::ResourceBinding(ref r) => {
                    if binding.is_some() {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    binding = Some(r.clone());
                }
                TypeQualifier::Location(l) => {
                    if location.is_some() {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    location = Some(l);
                }
                TypeQualifier::Sampling(s) => {
                    if sampling.is_some() {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    sampling = Some(s);
                }
                TypeQualifier::Layout(ref l) => {
                    if layout.is_some() {
                        return Err(ErrorKind::SemanticError(
                            meta,
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    layout = Some(l);
                }
                TypeQualifier::EarlyFragmentTests => {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "Cannot set early fragment tests on a declaration".into(),
                    ));
                }
            }
        }

        if binding.is_some() && storage != StorageQualifier::StorageClass(StorageClass::Uniform) {
            return Err(ErrorKind::SemanticError(
                meta,
                "binding requires uniform or buffer storage qualifier".into(),
            ));
        }

        if (sampling.is_some() || interpolation.is_some()) && location.is_none() {
            return Err(ErrorKind::SemanticError(
                meta,
                "Sampling and interpolation qualifiers can only be used in in/out variables".into(),
            ));
        }

        if let Some(location) = location {
            let prologue = if let StorageQualifier::Input = storage {
                PrologueStage::all()
            } else {
                PrologueStage::empty()
            };
            let interpolation = self.module.types[ty].inner.scalar_kind().map(|kind| {
                if let ScalarKind::Float = kind {
                    Interpolation::Perspective
                } else {
                    Interpolation::Flat
                }
            });

            let handle = self.module.global_variables.append(GlobalVariable {
                name: Some(name.clone()),
                class: StorageClass::Private,
                binding: None,
                ty,
                init,
                storage_access: StorageAccess::empty(),
            });

            let idx = self.entry_args.len();
            self.entry_args.push(EntryArg {
                binding: Binding::Location {
                    location,
                    interpolation,
                    sampling,
                },
                handle,
                prologue,
            });

            self.global_variables.push((
                name,
                GlobalLookup {
                    kind: GlobalLookupKind::Variable(handle),
                    entry_arg: Some(idx),
                },
            ));

            return Ok(ctx.add_expression(Expression::GlobalVariable(handle), body));
        } else if let StorageQualifier::Const = storage {
            let handle = init.ok_or_else(|| {
                ErrorKind::SemanticError(meta, "Constant must have a initializer".into())
            })?;

            self.constants.push((name, handle));

            return Ok(ctx.add_expression(Expression::Constant(handle), body));
        }

        let (class, storage_access) = match self.module.types[ty].inner {
            TypeInner::Image { class, .. } => (
                StorageClass::Handle,
                if let ImageClass::Storage(_) = class {
                    // TODO: Add support for qualifiers such as readonly,
                    // writeonly and readwrite
                    StorageAccess::all()
                } else {
                    StorageAccess::empty()
                },
            ),
            TypeInner::Sampler { .. } => (StorageClass::Handle, StorageAccess::empty()),
            _ => (
                match storage {
                    StorageQualifier::StorageClass(class) => class,
                    _ => StorageClass::Private,
                },
                StorageAccess::empty(),
            ),
        };

        let handle = self.module.global_variables.append(GlobalVariable {
            name: Some(name.clone()),
            class,
            binding,
            ty,
            init,
            storage_access,
        });

        self.global_variables.push((
            name,
            GlobalLookup {
                kind: GlobalLookupKind::Variable(handle),
                entry_arg: None,
            },
        ));

        Ok(ctx.add_expression(Expression::GlobalVariable(handle), body))
    }

    pub fn add_local_var(
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
    ) -> Result<Handle<Expression>, ErrorKind> {
        #[cfg(feature = "glsl-validate")]
        if ctx.lookup_local_var_current_scope(&name).is_some() {
            return Err(ErrorKind::VariableAlreadyDeclared(name));
        }

        let mut mutable = true;

        for qualifier in qualifiers {
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
                _ => {
                    return Err(ErrorKind::SemanticError(
                        meta,
                        "Qualifier not supported in locals".into(),
                    ));
                }
            }
        }

        let handle = ctx.locals.append(LocalVariable {
            name: Some(name.clone()),
            ty,
            init,
        });
        let expr = ctx.add_expression(Expression::LocalVariable(handle), body);

        ctx.add_local_var(name, expr, mutable);

        Ok(expr)
    }
}
