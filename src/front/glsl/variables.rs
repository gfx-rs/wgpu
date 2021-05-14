use crate::{
    Binding, BuiltIn, Constant, Expression, GlobalVariable, Handle, LocalVariable, ScalarKind,
    StorageAccess, StorageClass, StructMember, Type, TypeInner, VectorSize,
};

use super::ast::*;
use super::error::ErrorKind;
use super::token::TokenMetadata;

impl Program<'_> {
    pub fn lookup_variable(
        &mut self,
        context: &mut Context,
        name: &str,
    ) -> Result<Option<VariableReference>, ErrorKind> {
        if let Some(local_var) = context.lookup_local_var(name) {
            return Ok(Some(local_var));
        }
        if let Some(global_var) = context.lookup_global_var(self, name) {
            return Ok(Some(global_var));
        }
        if let Some(constant) = context.lookup_constants_var(self, name) {
            return Ok(Some(constant));
        }
        match name {
            "gl_Position" => {
                let ty = self.module.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Vector {
                        size: VectorSize::Quad,
                        kind: ScalarKind::Float,
                        width: 4,
                    },
                });

                let handle = self.module.global_variables.append(GlobalVariable {
                    name: Some(name.into()),
                    class: StorageClass::Function,
                    binding: None,
                    ty,
                    init: None,
                    storage_access: StorageAccess::all(),
                });

                self.built_ins.push((BuiltIn::Position, handle));

                self.lookup_global_variables
                    .insert(name.into(), GlobalLookup::Variable(handle));

                Ok(context.lookup_global_var(self, name))
            }
            "gl_VertexIndex" => {
                let ty = self.module.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Scalar {
                        kind: ScalarKind::Sint,
                        width: 4,
                    },
                });

                let handle = self.module.global_variables.append(GlobalVariable {
                    name: Some(name.into()),
                    class: StorageClass::Function,
                    binding: None,
                    ty,
                    init: None,
                    storage_access: StorageAccess::all(),
                });

                self.built_ins.push((BuiltIn::VertexIndex, handle));

                self.lookup_global_variables
                    .insert(name.into(), GlobalLookup::Variable(handle));

                Ok(context.lookup_global_var(self, name))
            }
            "gl_InstanceIndex" => {
                let ty = self.module.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Scalar {
                        kind: ScalarKind::Sint,
                        width: 4,
                    },
                });

                let handle = self.module.global_variables.append(GlobalVariable {
                    name: Some(name.into()),
                    class: StorageClass::Function,
                    binding: None,
                    ty,
                    init: None,
                    storage_access: StorageAccess::all(),
                });

                self.built_ins.push((BuiltIn::InstanceIndex, handle));

                self.lookup_global_variables
                    .insert(name.into(), GlobalLookup::Variable(handle));

                Ok(context.lookup_global_var(self, name))
            }
            _ => Ok(None),
        }
    }

    pub fn field_selection(
        &mut self,
        context: &mut Context,
        expression: Handle<Expression>,
        name: &str,
        meta: TokenMetadata,
    ) -> Result<Handle<Expression>, ErrorKind> {
        match *self.resolve_type(context, expression)? {
            TypeInner::Struct { ref members, .. } => {
                let index = members
                    .iter()
                    .position(|m| m.name == Some(name.into()))
                    .ok_or_else(|| ErrorKind::UnknownField(meta, name.into()))?;
                Ok(context.expressions.append(Expression::AccessIndex {
                    base: expression,
                    index: index as u32,
                }))
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
                            context.expressions.append(Expression::AccessIndex {
                                base: expression,
                                index: *idx as u32,
                            })
                        })
                        .collect();
                    if components.len() == 1 {
                        // only single element swizzle, like pos.y, just return that component
                        Ok(components[0])
                    } else {
                        Ok(context.expressions.append(Expression::Compose {
                            ty: self.module.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Vector {
                                    kind,
                                    width,
                                    size: match components.len() {
                                        2 => VectorSize::Bi,
                                        3 => VectorSize::Tri,
                                        4 => VectorSize::Quad,
                                        _ => {
                                            return Err(ErrorKind::SemanticError(
                                                format!(
                                                    "Bad swizzle size for \"{:?}\": {:?}",
                                                    name, v
                                                )
                                                .into(),
                                            ));
                                        }
                                    },
                                },
                            }),
                            components,
                        }))
                    }
                } else {
                    Err(ErrorKind::SemanticError(
                        format!("Invalid swizzle for vector \"{}\"", name).into(),
                    ))
                }
            }
            _ => Err(ErrorKind::SemanticError(
                format!("Can't lookup field on this type \"{}\"", name).into(),
            )),
        }
    }

    pub fn add_global_var(
        &mut self,
        ctx: &mut Context,
        qualifiers: &[TypeQualifier],
        ty: Handle<Type>,
        name: String,
        init: Option<Handle<Constant>>,
    ) -> Result<Handle<Expression>, ErrorKind> {
        let mut storage = StorageQualifier::StorageClass(StorageClass::Function);
        let mut interpolation = None;
        let mut binding = None;
        let mut location = None;
        let mut sampling = None;
        let mut layout = None;

        for qualifier in qualifiers {
            match *qualifier {
                TypeQualifier::StorageQualifier(s) => {
                    if StorageQualifier::StorageClass(StorageClass::Function) != storage {
                        return Err(ErrorKind::SemanticError(
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    storage = s;
                }
                TypeQualifier::Interpolation(i) => {
                    if interpolation.is_some() {
                        return Err(ErrorKind::SemanticError(
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    interpolation = Some(i);
                }
                TypeQualifier::ResourceBinding(ref r) => {
                    if binding.is_some() {
                        return Err(ErrorKind::SemanticError(
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    binding = Some(r.clone());
                }
                TypeQualifier::Location(l) => {
                    if location.is_some() {
                        return Err(ErrorKind::SemanticError(
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    location = Some(l);
                }
                TypeQualifier::Sampling(s) => {
                    if sampling.is_some() {
                        return Err(ErrorKind::SemanticError(
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    sampling = Some(s);
                }
                TypeQualifier::Layout(ref l) => {
                    if layout.is_some() {
                        return Err(ErrorKind::SemanticError(
                            "Cannot use more than one storage qualifier per declaration".into(),
                        ));
                    }

                    layout = Some(l);
                }
                TypeQualifier::EarlyFragmentTests => {
                    return Err(ErrorKind::SemanticError(
                        "Cannot set early fragment tests on a declaration".into(),
                    ));
                }
            }
        }

        if binding.is_some() && storage != StorageQualifier::StorageClass(StorageClass::Uniform) {
            return Err(ErrorKind::SemanticError(
                "binding requires uniform or buffer storage qualifier".into(),
            ));
        }

        if (sampling.is_some() || interpolation.is_some()) && location.is_none() {
            return Err(ErrorKind::SemanticError(
                "Sampling and interpolation qualifiers can only be used in in/out variables".into(),
            ));
        }

        if let Some(location) = location {
            let input = StorageQualifier::Input == storage;

            let index = self.add_member(
                input,
                Some(name.clone()),
                ty,
                Some(Binding::Location {
                    location,
                    interpolation,
                    sampling,
                }),
            );

            self.lookup_global_variables
                .insert(name, GlobalLookup::Select(index));

            let base = ctx.expressions.append(Expression::FunctionArgument(0));
            return Ok(ctx
                .expressions
                .append(Expression::AccessIndex { base, index }));
        } else if let StorageQualifier::Const = storage {
            let handle = init.ok_or_else(|| {
                ErrorKind::SemanticError("Constant must have a initializer".into())
            })?;

            self.lookup_constants.insert(name, handle);

            return Ok(ctx.expressions.append(Expression::Constant(handle)));
        }

        let class = match storage {
            StorageQualifier::StorageClass(class) => class,
            _ => StorageClass::Function,
        };

        let handle = self.module.global_variables.append(GlobalVariable {
            name: Some(name.clone()),
            class,
            binding,
            ty,
            init,
            // TODO
            storage_access: StorageAccess::all(),
        });

        self.lookup_global_variables
            .insert(name, GlobalLookup::Variable(handle));

        Ok(ctx.expressions.append(Expression::GlobalVariable(handle)))
    }

    pub fn add_local_var(
        &mut self,
        ctx: &mut Context,
        qualifiers: &[TypeQualifier],
        ty: Handle<Type>,
        name: String,
        init: Option<Handle<Constant>>,
    ) -> Result<Handle<Expression>, ErrorKind> {
        #[cfg(feature = "glsl-validate")]
        if ctx.lookup_local_var_current_scope(&name).is_some() {
            return Err(ErrorKind::VariableAlreadyDeclared(name));
        }

        // FIXME: this should be removed after
        // fixing the lack of constant qualifier support
        #[allow(clippy::never_loop)]
        for qualifier in qualifiers {
            match *qualifier {
                TypeQualifier::StorageQualifier(StorageQualifier::Const) => {
                    return Err(ErrorKind::NotImplemented("const qualifier"));
                }
                _ => {
                    return Err(ErrorKind::SemanticError(
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
        let expr = ctx.expressions.append(Expression::LocalVariable(handle));

        ctx.add_local_var(name, expr);

        Ok(expr)
    }

    fn add_member(
        &mut self,
        input: bool,
        name: Option<String>,
        ty: Handle<Type>,
        binding: Option<Binding>,
    ) -> u32 {
        let handle = match input {
            true => self.input_struct,
            false => self.output_struct,
        };

        let offset = if let TypeInner::Struct { ref members, .. } = self.module.types[handle].inner
        {
            members
                .last()
                .map(|member| {
                    member.offset
                        + self.module.types[member.ty]
                            .inner
                            .span(&self.module.constants)
                })
                .unwrap_or(0)
        } else {
            0
        };

        if let TypeInner::Struct {
            ref mut members, ..
        } = self.module.types.get_mut(handle).inner
        {
            members.push(StructMember {
                name,
                ty,
                binding,
                offset,
            });

            members.len() as u32 - 1
        } else {
            unreachable!()
        }
    }
}
