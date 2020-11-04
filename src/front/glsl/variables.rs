use crate::{
    Binding, BuiltIn, Expression, GlobalVariable, Handle, ScalarKind, ShaderStage, StorageAccess,
    StorageClass, Type, TypeInner, VectorSize,
};

use super::ast::*;
use super::error::ErrorKind;
use super::token::TokenMetadata;

impl Program {
    pub fn lookup_variable(&mut self, name: &str) -> Result<Option<Handle<Expression>>, ErrorKind> {
        let mut expression: Option<Handle<Expression>> = None;
        match name {
            "gl_Position" => {
                #[cfg(feature = "glsl-validate")]
                match self.shader_stage {
                    ShaderStage::Vertex | ShaderStage::Fragment { .. } => {}
                    _ => {
                        return Err(ErrorKind::VariableNotAvailable(name.into()));
                    }
                };
                let h = self
                    .module
                    .global_variables
                    .fetch_or_append(GlobalVariable {
                        name: Some(name.into()),
                        class: if self.shader_stage == ShaderStage::Vertex {
                            StorageClass::Output
                        } else {
                            StorageClass::Input
                        },
                        binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                        ty: self.module.types.fetch_or_append(Type {
                            name: None,
                            inner: TypeInner::Vector {
                                size: VectorSize::Quad,
                                kind: ScalarKind::Float,
                                width: 4,
                            },
                        }),
                        init: None,
                        interpolation: None,
                        storage_access: StorageAccess::empty(),
                    });
                self.lookup_global_variables.insert(name.into(), h);
                let exp = self
                    .context
                    .expressions
                    .append(Expression::GlobalVariable(h));
                self.context.lookup_global_var_exps.insert(name.into(), exp);

                expression = Some(exp);
            }
            "gl_VertexIndex" => {
                #[cfg(feature = "glsl-validate")]
                match self.shader_stage {
                    ShaderStage::Vertex => {}
                    _ => {
                        return Err(ErrorKind::VariableNotAvailable(name.into()));
                    }
                };
                let h = self
                    .module
                    .global_variables
                    .fetch_or_append(GlobalVariable {
                        name: Some(name.into()),
                        class: StorageClass::Input,
                        binding: Some(Binding::BuiltIn(BuiltIn::VertexIndex)),
                        ty: self.module.types.fetch_or_append(Type {
                            name: None,
                            inner: TypeInner::Scalar {
                                kind: ScalarKind::Uint,
                                width: 4,
                            },
                        }),
                        init: None,
                        interpolation: None,
                        storage_access: StorageAccess::empty(),
                    });
                self.lookup_global_variables.insert(name.into(), h);
                let exp = self
                    .context
                    .expressions
                    .append(Expression::GlobalVariable(h));
                self.context.lookup_global_var_exps.insert(name.into(), exp);

                expression = Some(exp);
            }
            _ => {}
        }

        if let Some(expression) = expression {
            Ok(Some(expression))
        } else if let Some(local_var) = self.context.lookup_local_var(name) {
            Ok(Some(local_var))
        } else if let Some(global_var) = self.context.lookup_global_var_exps.get(name) {
            Ok(Some(*global_var))
        } else {
            Ok(None)
        }
    }

    pub fn field_selection(
        &mut self,
        expression: Handle<Expression>,
        name: &str,
        meta: TokenMetadata,
    ) -> Result<Handle<Expression>, ErrorKind> {
        match *self.resolve_type(expression)? {
            TypeInner::Struct { ref members } => {
                let index = members
                    .iter()
                    .position(|m| m.name == Some(name.into()))
                    .ok_or_else(|| ErrorKind::UnknownField(meta, name.into()))?;
                Ok(self.context.expressions.append(Expression::AccessIndex {
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
                            self.context.expressions.append(Expression::AccessIndex {
                                base: expression,
                                index: *idx as u32,
                            })
                        })
                        .collect();
                    if components.len() == 1 {
                        // only single element swizzle, like pos.y, just return that component
                        Ok(components[0])
                    } else {
                        Ok(self.context.expressions.append(Expression::Compose {
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
                                                "Bad swizzle size",
                                            ));
                                        }
                                    },
                                },
                            }),
                            components,
                        }))
                    }
                } else {
                    Err(ErrorKind::SemanticError("Invalid swizzle for vector"))
                }
            }
            _ => Err(ErrorKind::SemanticError("Can't lookup field on this type")),
        }
    }
}
