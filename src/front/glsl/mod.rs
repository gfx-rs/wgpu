#![allow(clippy::panic)]
use crate::{
    Arena, ArraySize, BinaryOperator, Binding, BuiltIn, Constant, ConstantInner, EntryPoint,
    Expression, FastHashMap, Function, GlobalVariable, Handle, Header, Interpolation,
    LocalVariable, Module, ScalarKind, ShaderStage, StorageClass, StructMember, Type, TypeInner,
    VectorSize,
};
use glsl::{
    parser::{Parse, ParseError},
    syntax::*,
};
use parser::{Token, TokenMetadata};

mod helpers;
mod parser;

#[derive(Debug, thiserror::Error)]
pub enum ErrorKind {
    #[error("Unexpected token:\nexpected: {}\ngot: {}", expected.iter().map(|t| t.type_to_string()).collect::<Vec<_>>().join(" |"), got.token)]
    UnexpectedToken {
        expected: Vec<Token>,
        got: TokenMetadata,
    },
    #[error("Unexpected word:\nexpected: {}\ngot: {got}", expected.join("|"))]
    UnexpectedWord {
        expected: Vec<&'static str>,
        got: String,
    },
    #[error("Expected end of line:\ngot: {}", got.token)]
    ExpectedEOL { got: TokenMetadata },
    #[error("Unknown pragma: {pragma}")]
    UnknownPragma { pragma: String },
    #[error("The extension \"{extension}\" is not supported")]
    ExtensionNotSupported { extension: String },
    #[error("All extensions can't be require or enable")]
    AllExtensionsEnabled,
    #[error("The extension behavior must be one of require|enable|warn|disable got: {behavior}")]
    ExtensionUnknownBehavior { behavior: String },
    #[error("The version {version} isn't supported; use either 450 or 460")]
    UnsupportedVersion { version: usize },
    #[error("The profile {profile} isn't supported; use core")]
    UnsupportedProfile { profile: String },
    #[error("The profile {profile} isn't defined; use core")]
    UnknownProfile { profile: String },
    #[error("The preprocessor directive {directive} isn't defined")]
    UnknownPreprocessorDirective { directive: String },
    #[error("The preprocessor directives \"else\", \"elif\" or \"endif\" must be preceded by an \"if\", token: {}", token.token)]
    UnboundedIfCloserOrVariant { token: TokenMetadata },
    #[error("The preprocessor \"if\" directive can only contain integrals found: {}", token.token)]
    NonIntegralType { token: TokenMetadata },
    #[error("Type resolver error: {kind}")]
    TypeResolverError {
        #[from]
        kind: crate::proc::ResolveError,
    },
    #[error("Parser error: {error}")]
    ParseError {
        #[from]
        error: ParseError,
    },
    #[error("Macro can't begin with GL_")]
    ReservedMacro,
    #[error("End of line")]
    EOL,
    #[error("End of file")]
    EOF,
    #[error("Non constant expression encountered where a constant expression was expected")]
    NonConstantExpr,
}

#[derive(Debug, thiserror::Error)]
#[error("{kind}")]
pub struct Error {
    #[from]
    kind: ErrorKind,
}

#[derive(Debug, Copy, Clone)]
enum Global {
    Variable(Handle<GlobalVariable>),
    StructShorthand(Handle<GlobalVariable>, u32),
}

struct Parser<'a> {
    source: &'a str,
    types: Arena<Type>,
    globals: Arena<GlobalVariable>,
    globals_lookup: FastHashMap<String, Global>,
    globals_constants: FastHashMap<String, Handle<Constant>>,
    constants: Arena<Constant>,
    functions: Arena<Function>,
    shader_stage: ShaderStage,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str, shader_stage: ShaderStage) -> Self {
        Self {
            source,
            types: Arena::new(),
            globals: Arena::new(),
            globals_lookup: FastHashMap::default(),
            globals_constants: FastHashMap::default(),
            constants: Arena::new(),
            functions: Arena::new(),
            shader_stage,
        }
    }

    pub fn parse(mut self, entry: String) -> Result<crate::Module, Error> {
        let ast = TranslationUnit::parse(self.source).map_err(|e| Error { kind: e.into() })?;

        //println!("{:#?}", ast);

        let mut entry_point = None;
        let parameter_lookup = FastHashMap::default();
        let mut locals = Arena::<LocalVariable>::new();
        let mut locals_map = FastHashMap::default();
        let mut expressions = Arena::<Expression>::new();

        for declaration in ast {
            match declaration {
                ExternalDeclaration::Preprocessor(_) =>
                {
                    #[cfg(feature = "glsl_preprocessor")]
                    unreachable!()
                }
                ExternalDeclaration::FunctionDefinition(function) => {
                    let function = self.parse_function_definition(function)?;

                    if *self.functions[function].name.as_ref().unwrap() == entry {
                        assert!(entry_point.is_none());

                        entry_point = Some(function);
                    }
                }
                ExternalDeclaration::Declaration(decl) => match decl {
                    Declaration::InitDeclaratorList(mut init) => {
                        // Get initializer out for lifetime reasons. Maybe self.parse_global needs
                        // to take a reference and clone what it needs?
                        let mut initializer = None;
                        std::mem::swap(&mut initializer, &mut init.head.initializer);

                        let handle = self.parse_global(init.head)?;
                        let name = self.globals[handle].name.clone().unwrap();
                        if let Some(initializer) = initializer {
                            match initializer {
                                Initializer::Simple(expr) => {
                                    let expr = self.parse_expression(
                                        *expr,
                                        &mut expressions,
                                        &mut locals,
                                        &mut locals_map,
                                        &parameter_lookup,
                                        &[],
                                    )?;
                                    let handle = expressions.append(expr);
                                    let val = self.eval_const_expr(handle, &expressions)?;
                                    self.globals_constants.insert(name.clone(), val);
                                }
                                _ => todo!(),
                            }
                        }

                        self.globals_lookup.insert(name, Global::Variable(handle));
                    }
                    Declaration::Block(block) => {
                        let (class, binding, interpolation) =
                            Self::parse_type_qualifier(block.qualifier);
                        let ty_name = block.name.0;

                        let name = block.identifier.clone().map(|ident| ident.ident.0);

                        let mut fields = Vec::new();
                        let mut reexports = Vec::new();
                        let mut index = 0;

                        for field in block.fields {
                            let ty = self.parse_type(field.ty, &[]).unwrap();

                            for ident in field.identifiers {
                                let field_name = ident.ident.0;
                                let origin = crate::MemberOrigin::Offset(0); //TODO

                                fields.push(StructMember {
                                    name: Some(field_name.clone()),
                                    origin,
                                    ty: if let Some(array_spec) = ident.array_spec {
                                        let size = self.parse_array_size(array_spec, &[])?;
                                        self.types.fetch_or_append(Type {
                                            name: None,
                                            inner: TypeInner::Array {
                                                base: ty,
                                                size,
                                                stride: None,
                                            },
                                        })
                                    } else {
                                        ty
                                    },
                                });

                                if name.is_none() {
                                    reexports.push((field_name, index));
                                    index += 1;
                                }
                            }
                        }

                        let ty = if let Some(array_spec) =
                            block.identifier.and_then(|ident| ident.array_spec)
                        {
                            let base = self.types.fetch_or_append(Type {
                                name: Some(ty_name),
                                inner: TypeInner::Struct { members: fields },
                            });

                            let size = self.parse_array_size(array_spec, &[])?;
                            self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Array {
                                    base,
                                    size,
                                    stride: None,
                                },
                            })
                        } else {
                            self.types.fetch_or_append(Type {
                                name: Some(ty_name),
                                inner: TypeInner::Struct { members: fields },
                            })
                        };

                        let handle = self.globals.append(GlobalVariable {
                            binding,
                            class,
                            name,
                            ty,
                            interpolation,
                        });

                        for (name, index) in reexports {
                            self.globals_lookup
                                .insert(name, Global::StructShorthand(handle, index));
                        }
                    }
                    _ => unimplemented!(),
                },
            }
        }

        Ok(Module {
            header: Header {
                version: (1, 0, 0),
                generator: 0,
            },
            types: self.types,
            constants: self.constants,
            global_variables: self.globals,
            functions: self.functions,
            entry_points: vec![EntryPoint {
                stage: self.shader_stage,
                function: entry_point.unwrap(),
                name: entry,
            }],
        })
    }

    fn parse_function_definition(
        &mut self,
        function: FunctionDefinition,
    ) -> Result<Handle<Function>, Error> {
        let name = function.prototype.name.0;

        // Parse return type
        let ty = self.parse_type(function.prototype.ty.ty, &[]);

        let mut parameter_types = Vec::with_capacity(function.prototype.parameters.len());
        let mut parameter_lookup = FastHashMap::default();

        let mut local_variables = Arena::<LocalVariable>::new();
        let mut locals_map = FastHashMap::default();
        let mut expressions = Arena::<Expression>::new();
        let mut body = Vec::new();

        // TODO: Parse Qualifiers
        for (index, parameter) in function.prototype.parameters.into_iter().enumerate() {
            match parameter {
                FunctionParameterDeclaration::Named(_ /* TODO */, decl) => {
                    let ty = self.parse_type(decl.ty, &[]).unwrap();

                    let ty = if let Some(array_spec) = decl.ident.array_spec {
                        let size = self.parse_array_size(array_spec, &[])?;
                        self.types.fetch_or_append(Type {
                            name: None,
                            inner: TypeInner::Array {
                                base: ty,
                                size,
                                stride: None,
                            },
                        })
                    } else {
                        ty
                    };

                    parameter_types.push(ty);
                    parameter_lookup.insert(
                        decl.ident.ident.0,
                        Expression::FunctionParameter(index as u32),
                    );
                }
                FunctionParameterDeclaration::Unnamed(_, ty) => {
                    parameter_types.push(self.parse_type(ty, &[]).unwrap());
                }
            }
        }

        for statement in function.statement.statement_list {
            match statement {
                Statement::Compound(_) => unimplemented!(),
                Statement::Simple(statement) => match *statement {
                    SimpleStatement::Declaration(declaration) => match declaration {
                        Declaration::InitDeclaratorList(init) => {
                            self.parse_local_variable(
                                init,
                                &mut expressions,
                                &mut local_variables,
                                &mut locals_map,
                                &parameter_lookup,
                                &parameter_types,
                            )?;
                        }
                        _ => unimplemented!(),
                    },
                    SimpleStatement::Expression(Some(expr)) => {
                        body.push(self.parse_statement(
                            expr,
                            &mut expressions,
                            &mut local_variables,
                            &mut locals_map,
                            &parameter_lookup,
                            &parameter_types,
                        )?);
                    }
                    SimpleStatement::Expression(None) => (),
                    SimpleStatement::Selection(_) => unimplemented!(),
                    SimpleStatement::Switch(_) => unimplemented!(),
                    SimpleStatement::CaseLabel(_) => unimplemented!(),
                    SimpleStatement::Iteration(_) => unimplemented!(),
                    SimpleStatement::Jump(op) => body.push(match op {
                        JumpStatement::Continue => crate::Statement::Continue,
                        JumpStatement::Break => crate::Statement::Break,
                        JumpStatement::Return(expr) => crate::Statement::Return {
                            value: expr.map(|expr| {
                                let expr = self
                                    .parse_expression(
                                        *expr,
                                        &mut expressions,
                                        &mut local_variables,
                                        &mut locals_map,
                                        &parameter_lookup,
                                        &parameter_types,
                                    )
                                    .unwrap();
                                expressions.append(expr)
                            }),
                        },
                        JumpStatement::Discard => crate::Statement::Kill,
                    }),
                },
            }
        }

        let handle = self.functions.append(Function {
            name: Some(name),
            parameter_types,
            return_type: ty,
            global_usage: vec![],
            local_variables,
            expressions,
            body,
        });
        Ok(handle)
    }

    fn parse_local_variable(
        &mut self,
        init: InitDeclaratorList,
        expressions: &mut Arena<Expression>,
        locals: &mut Arena<LocalVariable>,
        locals_map: &mut FastHashMap<String, Handle<LocalVariable>>,
        parameter_lookup: &FastHashMap<String, Expression>,
        parameter_types: &[Handle<Type>],
    ) -> Result<Handle<LocalVariable>, Error> {
        let name = init.head.name.map(|d| d.0);
        let ty = {
            let ty = self.parse_type(init.head.ty.ty, parameter_types).unwrap();

            if let Some(array_spec) = init.head.array_specifier {
                let size = self.parse_array_size(array_spec, parameter_types)?;
                self.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Array {
                        base: ty,
                        size,
                        stride: None,
                    },
                })
            } else {
                ty
            }
        };

        let initializer = if let Some(initializer) = init.head.initializer {
            Some(self.parse_initializer(
                initializer,
                expressions,
                locals,
                locals_map,
                parameter_lookup,
                parameter_types,
            )?)
        } else {
            None
        };

        let handle = locals.append(LocalVariable {
            name: name.clone(),
            ty,
            init: initializer,
        });

        locals_map.insert(name.unwrap(), handle);

        Ok(handle)
    }

    fn parse_initializer(
        &mut self,
        initializer: Initializer,
        expressions: &mut Arena<Expression>,
        locals: &mut Arena<LocalVariable>,
        locals_map: &mut FastHashMap<String, Handle<LocalVariable>>,
        parameter_lookup: &FastHashMap<String, Expression>,
        parameter_types: &[Handle<Type>],
    ) -> Result<Handle<Expression>, Error> {
        match initializer {
            Initializer::Simple(expr) => {
                let handle = self.parse_expression(
                    *expr,
                    expressions,
                    locals,
                    locals_map,
                    parameter_lookup,
                    parameter_types,
                )?;

                Ok(expressions.append(handle))
            }
            Initializer::List(_exprs) => unimplemented!(),
        }
    }

    fn parse_statement(
        &mut self,
        expr: Expr,
        expressions: &mut Arena<Expression>,
        locals: &mut Arena<LocalVariable>,
        locals_map: &mut FastHashMap<String, Handle<LocalVariable>>,
        parameter_lookup: &FastHashMap<String, Expression>,
        parameter_types: &[Handle<Type>],
    ) -> Result<crate::Statement, Error> {
        match expr {
            Expr::Assignment(reg, op, value) => {
                let pointer = {
                    let pointer = self.parse_expression(
                        *reg,
                        expressions,
                        locals,
                        locals_map,
                        parameter_lookup,
                        parameter_types,
                    )?;
                    expressions.append(pointer)
                };

                let right = self.parse_expression(
                    *value,
                    expressions,
                    locals,
                    locals_map,
                    parameter_lookup,
                    parameter_types,
                )?;
                let value = match op {
                    AssignmentOp::Equal => right,
                    AssignmentOp::Mult => Expression::Binary {
                        op: BinaryOperator::Multiply,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::Div => Expression::Binary {
                        op: BinaryOperator::Divide,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::Mod => Expression::Binary {
                        op: BinaryOperator::Modulo,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::Add => Expression::Binary {
                        op: BinaryOperator::Add,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::Sub => Expression::Binary {
                        op: BinaryOperator::Subtract,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::LShift => Expression::Binary {
                        op: BinaryOperator::ShiftLeftLogical,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::RShift => {
                        Expression::Binary {
                            op: BinaryOperator::ShiftRightArithmetic, /* ??? */
                            left: pointer,
                            right: expressions.append(right),
                        }
                    }
                    AssignmentOp::And => Expression::Binary {
                        op: BinaryOperator::And,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::Xor => Expression::Binary {
                        op: BinaryOperator::ExclusiveOr,
                        left: pointer,
                        right: expressions.append(right),
                    },
                    AssignmentOp::Or => Expression::Binary {
                        op: BinaryOperator::InclusiveOr,
                        left: pointer,
                        right: expressions.append(right),
                    },
                };

                Ok(crate::Statement::Store {
                    pointer,
                    value: expressions.append(value),
                })
            }
            Expr::FunCall(_, _) => unimplemented!(),
            Expr::PostInc(_) => unimplemented!(),
            Expr::PostDec(_) => unimplemented!(),
            _ => panic!(),
        }
    }

    fn parse_expression(
        &mut self,
        expr: Expr,
        expressions: &mut Arena<Expression>,
        locals: &mut Arena<LocalVariable>,
        locals_map: &mut FastHashMap<String, Handle<LocalVariable>>,
        parameter_lookup: &FastHashMap<String, Expression>,
        parameter_types: &[Handle<Type>],
    ) -> Result<Expression, Error> {
        match expr {
            Expr::Variable(ident) => {
                let name = ident.0;

                match name.as_str() {
                    "gl_VertexIndex" => Ok(Expression::GlobalVariable(
                        self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::VertexIndex)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 4,
                                },
                            }),
                            interpolation: None,
                        }),
                    )),
                    "gl_InstanceIndex" => Ok(Expression::GlobalVariable(
                        self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::InstanceIndex)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 4,
                                },
                            }),
                            interpolation: None,
                        }),
                    )),
                    "gl_BaseVertex" => Ok(Expression::GlobalVariable(
                        self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::BaseVertex)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 4,
                                },
                            }),
                            interpolation: None,
                        }),
                    )),
                    "gl_BaseInstance" => Ok(Expression::GlobalVariable(
                        self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::BaseInstance)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 4,
                                },
                            }),
                            interpolation: None,
                        }),
                    )),
                    "gl_Position" => Ok(Expression::GlobalVariable(self.globals.fetch_or_append(
                        GlobalVariable {
                            name: Some(name),
                            class: match self.shader_stage {
                                ShaderStage::Vertex => StorageClass::Output,
                                ShaderStage::Fragment => StorageClass::Input,
                                _ => panic!(),
                            },
                            binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Vector {
                                    size: VectorSize::Quad,
                                    kind: ScalarKind::Float,
                                    width: 4,
                                },
                            }),
                            interpolation: None,
                        },
                    ))),
                    "gl_PointSize" => Ok(Expression::GlobalVariable(self.globals.fetch_or_append(
                        GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Output,
                            binding: Some(Binding::BuiltIn(BuiltIn::PointSize)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Float,
                                    width: 4,
                                },
                            }),
                            interpolation: None,
                        },
                    ))),
                    "gl_ClipDistance" => Ok(Expression::GlobalVariable(
                        self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Output,
                            binding: Some(Binding::BuiltIn(BuiltIn::ClipDistance)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Float,
                                    width: 4,
                                },
                            }),
                            interpolation: None,
                        }),
                    )),
                    other => {
                        if let Some(global) = self.globals_lookup.get(other) {
                            match *global {
                                Global::Variable(handle) => Ok(Expression::GlobalVariable(handle)),
                                Global::StructShorthand(struct_handle, index) => {
                                    Ok(Expression::AccessIndex {
                                        base: expressions
                                            .append(Expression::GlobalVariable(struct_handle)),
                                        index,
                                    })
                                }
                            }
                        } else if let Some(expr) = parameter_lookup.get(other) {
                            Ok(expr.clone())
                        } else if let Some(local) = locals_map.get(other) {
                            Ok(Expression::LocalVariable(*local))
                        } else {
                            println!("{}", other);
                            panic!()
                        }
                    }
                }
            }
            Expr::IntConst(value) => Ok(Expression::Constant(self.constants.fetch_or_append(
                Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Sint(value as i64),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Sint,
                            width: 4,
                        },
                    }),
                },
            ))),
            Expr::UIntConst(value) => Ok(Expression::Constant(self.constants.fetch_or_append(
                Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Uint(value as u64),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Uint,
                            width: 4,
                        },
                    }),
                },
            ))),
            Expr::BoolConst(value) => Ok(Expression::Constant(self.constants.fetch_or_append(
                Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Bool(value),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Bool,
                            width: 1,
                        },
                    }),
                },
            ))),
            Expr::FloatConst(value) => Ok(Expression::Constant(self.constants.fetch_or_append(
                Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Float(value as f64),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Float,
                            width: 4,
                        },
                    }),
                },
            ))),
            Expr::DoubleConst(value) => Ok(Expression::Constant(self.constants.fetch_or_append(
                Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Float(value),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Float,
                            width: 8,
                        },
                    }),
                },
            ))),
            Expr::Unary(op, reg) => {
                let expr = self.parse_expression(
                    *reg,
                    expressions,
                    locals,
                    locals_map,
                    parameter_lookup,
                    parameter_types,
                )?;
                Ok(Expression::Unary {
                    op: helpers::glsl_to_spirv_unary_op(op),
                    expr: expressions.append(expr),
                })
            }
            Expr::Binary(op, left, right) => {
                let left = self.parse_expression(
                    *left,
                    expressions,
                    locals,
                    locals_map,
                    parameter_lookup,
                    parameter_types,
                )?;
                let right = self.parse_expression(
                    *right,
                    expressions,
                    locals,
                    locals_map,
                    parameter_lookup,
                    parameter_types,
                )?;

                Ok(Expression::Binary {
                    op: helpers::glsl_to_spirv_binary_op(op),
                    left: expressions.append(left),
                    right: expressions.append(right),
                })
            }
            Expr::Ternary(_condition, _accept, _reject) => unimplemented!(),
            Expr::Assignment(_, _, _) => panic!(),
            Expr::Bracket(_reg, _index) => unimplemented!(),
            Expr::FunCall(ident, mut args) => {
                let name = match ident {
                    FunIdentifier::Identifier(ident) => ident.0,
                    FunIdentifier::Expr(_expr) => todo!(),
                };

                match name.as_str() {
                    "vec2" | "vec3" | "vec4" => Ok(Expression::Compose {
                        ty: self.types.fetch_or_append(Type {
                            name: None,
                            inner: TypeInner::Vector {
                                size: match name.chars().last().unwrap() {
                                    '2' => VectorSize::Bi,
                                    '3' => VectorSize::Tri,
                                    '4' => VectorSize::Quad,
                                    _ => panic!(),
                                },
                                kind: ScalarKind::Float,
                                width: 4,
                            },
                        }),
                        components: args
                            .into_iter()
                            .map(|arg| {
                                let expr = self
                                    .parse_expression(
                                        arg,
                                        expressions,
                                        locals,
                                        locals_map,
                                        parameter_lookup,
                                        parameter_types,
                                    )
                                    .unwrap();
                                expressions.append(expr)
                            })
                            .collect(),
                    }),
                    "texture" => {
                        let (image, sampler) =
                            if let Expr::FunCall(ident, mut sample_args) = args.remove(0) {
                                let name = match ident {
                                    FunIdentifier::Expr(_) => unimplemented!(),
                                    FunIdentifier::Identifier(ident) => ident.0,
                                };

                                match name.as_str() {
                                    "sampler2D" => (
                                        self.parse_expression(
                                            sample_args.remove(0),
                                            expressions,
                                            locals,
                                            locals_map,
                                            parameter_lookup,
                                            parameter_types,
                                        )?,
                                        self.parse_expression(
                                            sample_args.remove(0),
                                            expressions,
                                            locals,
                                            locals_map,
                                            parameter_lookup,
                                            parameter_types,
                                        )?,
                                    ),
                                    _ => unimplemented!(),
                                }
                            } else {
                                panic!()
                            };

                        let coordinate = self.parse_expression(
                            args.remove(0),
                            expressions,
                            locals,
                            locals_map,
                            parameter_lookup,
                            parameter_types,
                        )?;

                        Ok(Expression::ImageSample {
                            image: expressions.append(image),
                            sampler: expressions.append(sampler),
                            coordinate: expressions.append(coordinate),
                            level: crate::SampleLevel::Auto,
                            depth_ref: None, //TODO
                        })
                    }
                    _ => Ok(Expression::Call {
                        origin: crate::FunctionOrigin::External(name),
                        arguments: args
                            .into_iter()
                            .map(|arg| {
                                let expr = self
                                    .parse_expression(
                                        arg,
                                        expressions,
                                        locals,
                                        locals_map,
                                        parameter_lookup,
                                        parameter_types,
                                    )
                                    .unwrap();
                                expressions.append(expr)
                            })
                            .collect(),
                    }),
                }
            }
            Expr::Dot(reg, ident) => {
                let handle = {
                    let expr = self.parse_expression(
                        *reg,
                        expressions,
                        locals,
                        locals_map,
                        parameter_lookup,
                        parameter_types,
                    )?;
                    expressions.append(expr)
                };

                let mut typefier = crate::proc::Typifier::new();
                let name = ident.0.as_str();
                let type_handle = typefier
                    .resolve(
                        handle,
                        expressions,
                        &mut self.types,
                        &self.constants,
                        &self.globals,
                        locals,
                        &self.functions,
                        parameter_types,
                    )
                    .map_err(|e| Error { kind: e.into() })?;
                let base_type = &self.types[type_handle];
                match base_type.inner {
                    crate::TypeInner::Struct { ref members } => {
                        let index = members
                            .iter()
                            .position(|m| m.name.as_deref() == Some(name))
                            .unwrap() as u32;
                        Ok(crate::Expression::AccessIndex {
                            base: handle,
                            index,
                        })
                    }
                    crate::TypeInner::Vector { size, kind, width }
                    | crate::TypeInner::Matrix {
                        columns: size,
                        kind,
                        width,
                        ..
                    } => {
                        const MEMBERS: [char; 4] = ['x', 'y', 'z', 'w'];
                        if name.len() > 1 {
                            let mut components = Vec::with_capacity(name.len());

                            for ch in name.chars() {
                                let expr = crate::Expression::AccessIndex {
                                    base: handle,
                                    index: MEMBERS[..size as usize]
                                        .iter()
                                        .position(|&m| m == ch)
                                        .unwrap() as u32,
                                };
                                components.push(expressions.append(expr));
                            }
                            let size = match name.len() {
                                2 => crate::VectorSize::Bi,
                                3 => crate::VectorSize::Tri,
                                4 => crate::VectorSize::Quad,
                                _ => panic!(),
                            };
                            let inner =
                                if let crate::TypeInner::Matrix { rows, .. } = base_type.inner {
                                    crate::TypeInner::Matrix {
                                        columns: size,
                                        rows,
                                        kind,
                                        width,
                                    }
                                } else {
                                    crate::TypeInner::Vector { size, kind, width }
                                };
                            Ok(crate::Expression::Compose {
                                ty: self.types.fetch_or_append(Type { name: None, inner }),
                                components,
                            })
                        } else {
                            let ch = name.chars().next().unwrap();
                            let index = MEMBERS[..size as usize]
                                .iter()
                                .position(|&m| m == ch)
                                .unwrap() as u32;
                            Ok(crate::Expression::AccessIndex {
                                base: handle,
                                index,
                            })
                        }
                    }
                    _ => panic!(),
                }
            }
            Expr::PostInc(_reg) => unimplemented!(),
            Expr::PostDec(_reg) => unimplemented!(),
            Expr::Comma(_, _) => unimplemented!(),
        }
    }

    // None = void
    fn parse_type(
        &mut self,
        ty: TypeSpecifier,
        parameter_types: &[Handle<Type>],
    ) -> Option<Handle<Type>> {
        let base_ty = helpers::glsl_to_spirv_type(ty.ty)?;

        let ty = if let Some(array_spec) = ty.array_specifier {
            let handle = self.types.fetch_or_append(Type {
                name: None,
                inner: base_ty,
            });
            let size = self.parse_array_size(array_spec, parameter_types).unwrap();

            TypeInner::Array {
                base: handle,
                size,
                stride: None,
            }
        } else {
            base_ty
        };

        Some(self.types.fetch_or_append(Type {
            name: None,
            inner: ty,
        }))
    }

    fn parse_global(&mut self, head: SingleDeclaration) -> Result<Handle<GlobalVariable>, Error> {
        let name = head.name.map(|d| d.0);
        let ty = {
            let ty = self.parse_type(head.ty.ty, &[]).unwrap();

            if let Some(array_spec) = head.array_specifier {
                let size = self.parse_array_size(array_spec, &[])?;
                self.types.fetch_or_append(Type {
                    name: None,
                    inner: TypeInner::Array {
                        base: ty,
                        size,
                        stride: None,
                    },
                })
            } else {
                ty
            }
        };

        let (class, binding, interpolation) = head
            .ty
            .qualifier
            .map(Self::parse_type_qualifier)
            .unwrap_or((StorageClass::Private, None, None));

        Ok(self.globals.append(GlobalVariable {
            name,
            class,
            binding,
            ty,
            interpolation,
        }))
    }

    /// https://www.khronos.org/opengl/wiki/Core_Language_(GLSL)#Constant_expression
    pub fn eval_const_expr(
        &mut self,
        expr: Handle<Expression>,
        expressions: &Arena<Expression>,
    ) -> Result<Handle<Constant>, Error> {
        match &expressions[expr] {
            Expression::Constant(handle) => Ok(*handle),
            Expression::Call { .. } => todo!(),
            Expression::GlobalVariable(handle) => {
                let name = self.globals[*handle].name.as_ref().unwrap();
                if let Some(handle) = self.globals_constants.get(name) {
                    Ok(*handle)
                } else {
                    todo!("Global const error")
                }
            }
            Expression::Binary { left, right, op } => {
                let left = self.eval_const_expr(*left, expressions)?;
                let right = self.eval_const_expr(*right, expressions)?;
                let inner: ConstantInner;
                let ty;
                match op {
                    BinaryOperator::Add => {
                        match (&self.constants[left].inner, &self.constants[right].inner) {
                            (ConstantInner::Sint(left), ConstantInner::Sint(right)) => {
                                inner = ConstantInner::Sint(left + right);
                                ty = self.types.fetch_or_append(Type {
                                    name: None,
                                    inner: TypeInner::Scalar {
                                        kind: ScalarKind::Sint,
                                        width: 4,
                                    },
                                })
                            }
                            (ConstantInner::Uint(left), ConstantInner::Uint(right)) => {
                                inner = ConstantInner::Uint(left + right);
                                ty = self.types.fetch_or_append(Type {
                                    name: None,
                                    inner: TypeInner::Scalar {
                                        kind: ScalarKind::Uint,
                                        width: 4,
                                    },
                                })
                            }
                            (ConstantInner::Float(left), ConstantInner::Float(right)) => {
                                inner = ConstantInner::Float(left + right);
                                ty = self.types.fetch_or_append(Type {
                                    name: None,
                                    inner: TypeInner::Scalar {
                                        kind: ScalarKind::Float,
                                        width: 4,
                                    },
                                })
                            }
                            _ => todo!(),
                        }
                    }

                    _ => todo!(),
                }
                Ok(self.constants.fetch_or_append(Constant {
                    name: None,
                    specialization: None,
                    inner,
                    ty,
                }))
            }
            expr => todo!("Const eval for {:?}", expr),
        }
    }

    pub fn parse_array_size(
        &mut self,
        array_spec: ArraySpecifier,
        parameter_types: &[Handle<Type>],
    ) -> Result<ArraySize, Error> {
        let parameter_lookup = FastHashMap::default();
        let mut locals = Arena::<LocalVariable>::new();
        let mut locals_map = FastHashMap::default();
        let mut expressions = Arena::<Expression>::new();
        let size = match array_spec {
            ArraySpecifier::Unsized => ArraySize::Dynamic,
            ArraySpecifier::ExplicitlySized(expr) => {
                let expr = self.parse_expression(
                    *expr,
                    &mut expressions,
                    &mut locals,
                    &mut locals_map,
                    &parameter_lookup,
                    parameter_types,
                )?;
                let handle = expressions.append(expr);

                let const_handle = self.eval_const_expr(handle, &expressions)?;

                match &self.constants[const_handle].inner {
                    ConstantInner::Sint(val) => ArraySize::Static(*val as u32),
                    ConstantInner::Uint(val) => ArraySize::Static(*val as u32),
                    val => panic!(
                        "Array size must be an integral constant expression, got: {:?}",
                        val
                    ),
                }
            }
        };

        Ok(size)
    }

    fn parse_type_qualifier(
        qualifier: TypeQualifier,
    ) -> (StorageClass, Option<Binding>, Option<Interpolation>) {
        let mut storage = None;
        let mut binding = None;
        let mut interpolation = None;

        for qualifier in qualifier.qualifiers {
            match qualifier {
                TypeQualifierSpec::Storage(storage_qualifier) => {
                    assert!(storage.is_none());

                    storage = Some(match storage_qualifier {
                        StorageQualifier::Const => StorageClass::Constant,
                        StorageQualifier::In => StorageClass::Input,
                        StorageQualifier::Out => StorageClass::Output,
                        StorageQualifier::Uniform => StorageClass::Uniform,
                        StorageQualifier::Buffer => StorageClass::StorageBuffer,
                        StorageQualifier::Shared => StorageClass::WorkGroup,
                        StorageQualifier::Coherent => StorageClass::WorkGroup,
                        _ => panic!(),
                    });
                }
                TypeQualifierSpec::Layout(layout_qualifier) => {
                    assert!(binding.is_none());

                    let mut set = None;
                    let mut bind = None;
                    let mut location = None;

                    for identifier in layout_qualifier.ids {
                        match identifier {
                            LayoutQualifierSpec::Identifier(identifier, Some(expr)) => {
                                if let Expr::IntConst(word) = *expr {
                                    match identifier.as_str() {
                                        "location" => {
                                            assert!(set.is_none(),);
                                            assert!(bind.is_none(),);
                                            assert!(location.is_none());

                                            location = Some(word);
                                        }
                                        "binding" => {
                                            assert!(bind.is_none(),);
                                            assert!(location.is_none());

                                            bind = Some(word);
                                        }
                                        "set" => {
                                            assert!(set.is_none(),);
                                            assert!(location.is_none());

                                            set = Some(word);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            _ => unimplemented!(),
                        }
                    }

                    if let (Some(set), Some(bind)) = (set, bind) {
                        binding = Some(Binding::Descriptor {
                            set: set as u32,
                            binding: bind as u32,
                        })
                    } else if let Some(location) = location {
                        binding = Some(Binding::Location(location as u32))
                    } else {
                        panic!()
                    }
                }
                TypeQualifierSpec::Interpolation(interpolation_qualifier) => {
                    interpolation = Some(match interpolation_qualifier {
                        InterpolationQualifier::NoPerspective => Interpolation::Linear,
                        InterpolationQualifier::Flat => Interpolation::Flat,
                        InterpolationQualifier::Smooth => Interpolation::Perspective,
                    });
                }
                _ => unimplemented!(),
            }
        }

        (
            storage.unwrap_or(StorageClass::Private),
            binding,
            interpolation,
        )
    }
}

pub fn parse_str(source: &str, entry: String, stage: ShaderStage) -> Result<crate::Module, Error> {
    let input = parser::parse(source)?;

    log::debug!("------GLSL PREPROCESSOR------");
    log::debug!("\n{}", input);
    log::debug!("-----------------------------");

    Parser::new(&input, stage).parse(entry)
}

#[cfg(test)]
mod tests {
    use super::parse_str;

    #[test]
    fn test_vertex() {
        let data = include_str!("../../../test-data/glsl_vertex_test_shader.vert");

        println!(
            "{:#?}",
            parse_str(data, String::from("main"), crate::ShaderStage::Vertex)
        );
    }

    #[test]
    fn test_frag() {
        let _ = env_logger::try_init();

        let data = include_str!("../../../test-data/glsl_phong_lighting.frag");

        println!(
            "{:#?}",
            parse_str(data, String::from("main"), crate::ShaderStage::Fragment)
        );
    }

    #[cfg(feature = "glsl_preprocessor")]
    #[test]
    fn test_preprocess() {
        let _ = env_logger::try_init();

        let data = include_str!("../../../test-data/glsl_preprocessor_abuse.vert");

        println!(
            "{:#?}",
            parse_str(data, String::from("main"), crate::ShaderStage::Vertex)
        );
    }

    #[cfg(feature = "glsl_preprocessor")]
    #[test]
    #[should_panic]
    fn test_preprocess_ifs() {
        let _ = env_logger::try_init();

        let data = include_str!("../../../test-data/glsl_if_preprocessor.vert");

        println!(
            "{:#?}",
            parse_str(data, String::from("main"), crate::ShaderStage::Vertex)
        );
    }
}
