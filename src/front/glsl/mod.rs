use crate::{
    Arena, ArraySize, BinaryOperator, Binding, Constant, ConstantInner, EntryPoint, Expression,
    FastHashMap, Function, GlobalVariable, Handle, Header, LocalVariable, Module, ScalarKind,
    StructMember, Type, TypeInner, VectorSize,
};
use glsl::{
    parser::{Parse, ParseError},
    syntax::*,
};
use parser::{Token, TokenMetadata};
use spirv::{BuiltIn, ExecutionModel, StorageClass};
use std::fmt;

mod helpers;
mod parser;

#[derive(Debug)]
pub enum ErrorKind {
    UnexpectedToken {
        expected: Vec<Token>,
        got: TokenMetadata,
    },
    UnexpectedWord {
        expected: Vec<&'static str>,
        got: String,
    },
    ExpectedEOL {
        got: TokenMetadata,
    },
    UnknownPragma {
        pragma: String,
    },
    ExtensionNotSupported {
        extension: String,
    },
    AllExtensionsEnabled,
    ExtensionUnknownBehavior {
        behavior: String,
    },
    UnsupportedVersion {
        version: usize,
    },
    UnsupportedProfile {
        profile: String,
    },
    UnknownProfile {
        profile: String,
    },
    UnknownPreprocessorDirective {
        directive: String,
    },
    UnboundedIfCloserOrVariant {
        token: TokenMetadata,
    },
    NonIntegralType {
        token: TokenMetadata,
    },
    ReservedMacro,
    EOL,
    EOF,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorKind::UnexpectedToken { expected, got } => write!(
                f,
                "Unexpected token:\nexpected: {}\ngot: {}",
                expected
                    .iter()
                    .map(|token| {
                        let mut type_string = token.type_to_string();
                        type_string.push_str(" |");
                        type_string
                    })
                    .collect::<String>(),
                got.token
            ),
            ErrorKind::UnexpectedWord { expected, got } => write!(
                f,
                "Unexpected word:\nexpected: {}\ngot: {}",
                expected.iter().fold(String::new(), |mut acc, word| {
                    acc.push_str(*word);

                    acc.push_str("|");

                    acc
                }),
                got
            ),
            ErrorKind::ExpectedEOL { got } => {
                write!(f, "Expected end of line:\ngot: {}", got.token)
            }
            ErrorKind::UnknownPragma { pragma } => write!(f, "Unknown pragma: {}", pragma),
            ErrorKind::ExtensionNotSupported { extension } => {
                write!(f, "The extension \"{}\" is not supported", extension)
            }
            ErrorKind::AllExtensionsEnabled => {
                write!(f, "All extensions can't be require or enable")
            }
            ErrorKind::ExtensionUnknownBehavior { behavior } => write!(
                f,
                "The extension behavior must be one of require|enable|warn|disable got: {}",
                behavior
            ),
            ErrorKind::UnsupportedVersion { version } => write!(
                f,
                "The version {} isn't supported use either 450 or 460",
                version
            ),
            ErrorKind::UnsupportedProfile { profile } => {
                write!(f, "The profile {} isn't supported use core", profile)
            }
            ErrorKind::UnknownProfile { profile } => {
                write!(f, "The profile {} isn't defined use core", profile)
            }
            ErrorKind::UnknownPreprocessorDirective { directive } => {
                write!(f, "The preprocessor directive {} isn't defined", directive)
            }
            ErrorKind::UnboundedIfCloserOrVariant { token } => {
                write!(f, "The preprocessor directives \"else\", \"elif\" or \"endif\" must be preceded by an \"if\", token: {}", token.token)
            }
            ErrorKind::NonIntegralType { token } => {
                write!(f, "The preprocessor \"if\" directive can only contain integrals found: {}", token.token)
            }
            ErrorKind::ReservedMacro => write!(f, "Macro can't begin with GL_"),
            ErrorKind::EOL => write!(f, "End of line"),
            ErrorKind::EOF => write!(f, "End of file"),
        }
    }
}

#[derive(Debug)]
pub struct Error {
    pub kind: ErrorKind,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for Error {}

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
    constants: Arena<Constant>,
    functions: Arena<Function>,
    function_lookup: FastHashMap<String, Handle<Function>>,
    exec_model: ExecutionModel,
}

impl<'a> Parser<'a> {
    pub fn new(source: &'a str, exec_model: ExecutionModel) -> Result<Self, ParseError> {
        Ok(Self {
            source,
            types: Arena::new(),
            globals: Arena::new(),
            globals_lookup: FastHashMap::default(),
            constants: Arena::new(),
            functions: Arena::new(),
            function_lookup: FastHashMap::default(),
            exec_model,
        })
    }

    pub fn parse(mut self, entry: String) -> Result<crate::Module, ParseError> {
        let ast = TranslationUnit::parse(self.source)?;

        //println!("{:#?}", ast);

        let mut entry_point = None;

        for declaration in ast {
            match declaration {
                ExternalDeclaration::Preprocessor(_) =>
                {
                    #[cfg(feature = "glsl_preprocessor")]
                    unreachable!()
                }
                ExternalDeclaration::FunctionDefinition(function) => {
                    let function = self.parse_function_definition(function);

                    if *self.functions[function].name.as_ref().unwrap() == entry {
                        assert!(entry_point.is_none());

                        entry_point = Some(function);
                    }
                }
                ExternalDeclaration::Declaration(decl) => match decl {
                    Declaration::InitDeclaratorList(init) => {
                        let handle = self.parse_global(init);
                        let name = self.globals[handle].name.clone().unwrap();
                        self.globals_lookup.insert(name, Global::Variable(handle));
                    }
                    Declaration::Block(block) => {
                        let (class, binding) = Self::parse_type_qualifier(block.qualifier);
                        let ty_name = block.name.0;

                        let name = block.identifier.clone().map(|ident| ident.ident.0);

                        let mut fields = Vec::new();
                        let mut reexports = Vec::new();
                        let mut index = 0;

                        for field in block.fields {
                            let binding = field
                                .qualifier
                                .and_then(|qualifier| Self::parse_type_qualifier(qualifier).1);

                            let ty = self.parse_type(field.ty).unwrap();

                            for ident in field.identifiers {
                                let field_name = ident.ident.0;

                                fields.push(StructMember {
                                    name: Some(field_name.clone()),
                                    binding: binding.clone(),
                                    ty: if let Some(array_spec) = ident.array_spec {
                                        self.types.fetch_or_append(Type {
                                            name: None,
                                            inner: TypeInner::Array {
                                                base: ty,
                                                size: match array_spec {
                                                    ArraySpecifier::Unsized => ArraySize::Dynamic,
                                                    ArraySpecifier::ExplicitlySized(_expr) => {
                                                        unimplemented!()
                                                    }
                                                },
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

                            self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Array {
                                    base,
                                    size: match array_spec {
                                        ArraySpecifier::Unsized => ArraySize::Dynamic,
                                        ArraySpecifier::ExplicitlySized(_expr) => unimplemented!(),
                                    },
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
                exec_model: self.exec_model,
                function: entry_point.unwrap(),
                name: entry,
            }],
        })
    }

    fn parse_function_definition(&mut self, function: FunctionDefinition) -> Handle<Function> {
        let name = function.prototype.name.0;

        // Parse return type
        let ty = self.parse_type(function.prototype.ty.ty);

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
                    let ty = self.parse_type(decl.ty).unwrap();

                    let ty = if let Some(specifier) = decl.ident.array_spec {
                        match specifier {
                            ArraySpecifier::Unsized => self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Array {
                                    base: ty,
                                    size: ArraySize::Dynamic,
                                },
                            }),
                            ArraySpecifier::ExplicitlySized(_) => unimplemented!(),
                        }
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
                    parameter_types.push(self.parse_type(ty).unwrap());
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
                            );
                        }
                        _ => unimplemented!(),
                    },
                    SimpleStatement::Expression(expr) => {
                        let expr = match expr {
                            Some(expr) => expr,
                            None => continue,
                        };

                        body.push(self.parse_statement(
                            expr,
                            &mut expressions,
                            &mut local_variables,
                            &mut locals_map,
                            &parameter_lookup,
                        ));
                    }
                    SimpleStatement::Selection(_) => unimplemented!(),
                    SimpleStatement::Switch(_) => unimplemented!(),
                    SimpleStatement::CaseLabel(_) => unimplemented!(),
                    SimpleStatement::Iteration(_) => unimplemented!(),
                    SimpleStatement::Jump(op) => body.push(match op {
                        JumpStatement::Continue => crate::Statement::Continue,
                        JumpStatement::Break => crate::Statement::Break,
                        JumpStatement::Return(expr) => crate::Statement::Return {
                            value: expr.map(|expr| {
                                let expr = self.parse_expression(
                                    *expr,
                                    &mut expressions,
                                    &mut local_variables,
                                    &mut locals_map,
                                    &parameter_lookup,
                                );
                                expressions.append(expr)
                            }),
                        },
                        JumpStatement::Discard => crate::Statement::Kill,
                    }),
                },
            }
        }

        let handle = self.functions.append(Function {
            name: Some(name.clone()),
            control: spirv::FunctionControl::NONE,
            parameter_types,
            return_type: ty,
            global_usage: vec![],
            local_variables,
            expressions,
            body,
        });

        self.function_lookup.insert(name, handle);

        handle
    }

    fn parse_local_variable(
        &mut self,
        init: InitDeclaratorList,
        expressions: &mut Arena<Expression>,
        locals: &mut Arena<LocalVariable>,
        locals_map: &mut FastHashMap<String, Handle<LocalVariable>>,
        parameter_lookup: &FastHashMap<String, Expression>,
    ) -> Handle<LocalVariable> {
        let name = init.head.name.map(|d| d.0);
        let ty = {
            let ty = self.parse_type(init.head.ty.ty).unwrap();

            if let Some(specifier) = init.head.array_specifier {
                match specifier {
                    ArraySpecifier::Unsized => self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Array {
                            base: ty,
                            size: ArraySize::Dynamic,
                        },
                    }),
                    ArraySpecifier::ExplicitlySized(_) => unimplemented!(),
                }
            } else {
                ty
            }
        };

        let initializer = init.head.initializer.map(|initializer| {
            self.parse_initializer(
                initializer,
                expressions,
                locals,
                locals_map,
                parameter_lookup,
            )
        });

        let handle = locals.append(LocalVariable {
            name: name.clone(),
            ty,
            init: initializer,
        });

        locals_map.insert(name.unwrap(), handle);

        handle
    }

    fn parse_initializer(
        &mut self,
        initializer: Initializer,
        expressions: &mut Arena<Expression>,
        locals: &mut Arena<LocalVariable>,
        locals_map: &mut FastHashMap<String, Handle<LocalVariable>>,
        parameter_lookup: &FastHashMap<String, Expression>,
    ) -> Handle<Expression> {
        match initializer {
            Initializer::Simple(expr) => {
                let handle =
                    self.parse_expression(*expr, expressions, locals, locals_map, parameter_lookup);

                expressions.append(handle)
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
    ) -> crate::Statement {
        match expr {
            Expr::Assignment(reg, op, value) => {
                let pointer = {
                    let pointer = self.parse_expression(
                        *reg,
                        expressions,
                        locals,
                        locals_map,
                        parameter_lookup,
                    );
                    expressions.append(pointer)
                };

                let right = self.parse_expression(
                    *value,
                    expressions,
                    locals,
                    locals_map,
                    parameter_lookup,
                );
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

                crate::Statement::Store {
                    pointer,
                    value: expressions.append(value),
                }
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
    ) -> Expression {
        match expr {
            Expr::Variable(ident) => {
                let name = ident.0;

                match name.as_str() {
                    "gl_VertexIndex" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::VertexIndex)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_InstanceIndex" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::InstanceIndex)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_DrawID" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::DrawIndex)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_BaseVertex" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::BaseVertex)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_BaseInstance" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Input,
                            binding: Some(Binding::BuiltIn(BuiltIn::BaseInstance)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Sint,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_Position" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: match self.exec_model {
                                ExecutionModel::Vertex => StorageClass::Output,
                                ExecutionModel::Fragment => StorageClass::Input,
                                _ => panic!(),
                            },
                            binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Vector {
                                    size: VectorSize::Quad,
                                    kind: ScalarKind::Float,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_PointSize" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Output,
                            binding: Some(Binding::BuiltIn(BuiltIn::PointSize)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Float,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_ClipDistance" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Output,
                            binding: Some(Binding::BuiltIn(BuiltIn::ClipDistance)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Float,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    "gl_CullDistance" => {
                        Expression::GlobalVariable(self.globals.fetch_or_append(GlobalVariable {
                            name: Some(name),
                            class: StorageClass::Output,
                            binding: Some(Binding::BuiltIn(BuiltIn::CullDistance)),
                            ty: self.types.fetch_or_append(Type {
                                name: None,
                                inner: TypeInner::Scalar {
                                    kind: ScalarKind::Float,
                                    width: 32,
                                },
                            }),
                        }))
                    }
                    other => {
                        if let Some(global) = self.globals_lookup.get(other) {
                            match *global {
                                Global::Variable(handle) => Expression::GlobalVariable(handle),
                                Global::StructShorthand(struct_handle, index) => {
                                    Expression::AccessIndex {
                                        base: expressions
                                            .append(Expression::GlobalVariable(struct_handle)),
                                        index,
                                    }
                                }
                            }
                        } else if let Some(expr) = parameter_lookup.get(other) {
                            expr.clone()
                        } else if let Some(local) = locals_map.get(other) {
                            Expression::LocalVariable(*local)
                        } else {
                            println!("{}", other);
                            panic!()
                        }
                    }
                }
            }
            Expr::IntConst(value) => {
                Expression::Constant(self.constants.fetch_or_append(Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Sint(value as i64),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Sint,
                            width: 32,
                        },
                    }),
                }))
            }
            Expr::UIntConst(value) => {
                Expression::Constant(self.constants.fetch_or_append(Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Uint(value as u64),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Uint,
                            width: 32,
                        },
                    }),
                }))
            }
            Expr::BoolConst(value) => {
                Expression::Constant(self.constants.fetch_or_append(Constant {
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
                }))
            }
            Expr::FloatConst(value) => {
                Expression::Constant(self.constants.fetch_or_append(Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Float(value as f64),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Float,
                            width: 32,
                        },
                    }),
                }))
            }
            Expr::DoubleConst(value) => {
                Expression::Constant(self.constants.fetch_or_append(Constant {
                    name: None,
                    specialization: None,
                    inner: ConstantInner::Float(value),
                    ty: self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Scalar {
                            kind: ScalarKind::Float,
                            width: 64,
                        },
                    }),
                }))
            }
            Expr::Unary(_op, _reg) => unimplemented!(),
            Expr::Binary(op, left, right) => {
                let left =
                    self.parse_expression(*left, expressions, locals, locals_map, parameter_lookup);
                let right = self.parse_expression(
                    *right,
                    expressions,
                    locals,
                    locals_map,
                    parameter_lookup,
                );

                Expression::Binary {
                    op: helpers::glsl_to_spirv_binary_op(op),
                    left: expressions.append(left),
                    right: expressions.append(right),
                }
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
                    "vec2" | "vec3" | "vec4" => Expression::Compose {
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
                                width: 32,
                            },
                        }),
                        components: args
                            .into_iter()
                            .map(|arg| {
                                let expr = self.parse_expression(
                                    arg,
                                    expressions,
                                    locals,
                                    locals_map,
                                    parameter_lookup,
                                );
                                expressions.append(expr)
                            })
                            .collect(),
                    },
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
                                        ),
                                        self.parse_expression(
                                            sample_args.remove(0),
                                            expressions,
                                            locals,
                                            locals_map,
                                            parameter_lookup,
                                        ),
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
                        );

                        Expression::ImageSample {
                            image: expressions.append(image),
                            sampler: expressions.append(sampler),
                            coordinate: expressions.append(coordinate),
                        }
                    }
                    _ => Expression::Call {
                        name,
                        arguments: args
                            .into_iter()
                            .map(|arg| {
                                let expr = self.parse_expression(
                                    arg,
                                    expressions,
                                    locals,
                                    locals_map,
                                    parameter_lookup,
                                );
                                expressions.append(expr)
                            })
                            .collect(),
                    },
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
                    );
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
                        &self.function_lookup,
                    )
                    .unwrap();
                let base_type = &self.types[type_handle];
                match base_type.inner {
                    crate::TypeInner::Struct { ref members } => {
                        let index = members
                            .iter()
                            .position(|m| m.name.as_deref() == Some(name))
                            .unwrap() as u32;
                        crate::Expression::AccessIndex {
                            base: handle,
                            index,
                        }
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
                            crate::Expression::Compose {
                                ty: crate::proc::Typifier::deduce_type_handle(
                                    inner,
                                    &mut self.types,
                                ),
                                components,
                            }
                        } else {
                            let ch = name.chars().next().unwrap();
                            let index = MEMBERS[..size as usize]
                                .iter()
                                .position(|&m| m == ch)
                                .unwrap() as u32;
                            crate::Expression::AccessIndex {
                                base: handle,
                                index,
                            }
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
    fn parse_type(&mut self, ty: TypeSpecifier) -> Option<Handle<Type>> {
        let base_ty = helpers::glsl_to_spirv_type(ty.ty, &mut self.types)?;

        let ty = if let Some(specifier) = ty.array_specifier {
            let handle = self.types.fetch_or_append(Type {
                name: None,
                inner: base_ty,
            });

            match specifier {
                ArraySpecifier::Unsized => TypeInner::Array {
                    base: handle,
                    size: ArraySize::Dynamic,
                },
                ArraySpecifier::ExplicitlySized(_) => unimplemented!(),
            }
        } else {
            base_ty
        };

        Some(self.types.fetch_or_append(Type {
            name: None,
            inner: ty,
        }))
    }

    fn parse_global(&mut self, init: InitDeclaratorList) -> Handle<GlobalVariable> {
        let name = init.head.name.map(|d| d.0);
        let ty = {
            let ty = self.parse_type(init.head.ty.ty).unwrap();

            if let Some(specifier) = init.head.array_specifier {
                match specifier {
                    ArraySpecifier::Unsized => self.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Array {
                            base: ty,
                            size: ArraySize::Dynamic,
                        },
                    }),
                    ArraySpecifier::ExplicitlySized(_) => unimplemented!(),
                }
            } else {
                ty
            }
        };

        let (class, binding) = init
            .head
            .ty
            .qualifier
            .map(Self::parse_type_qualifier)
            .unwrap_or((StorageClass::Private, None));

        self.globals.append(GlobalVariable {
            name,
            class,
            binding,
            ty,
        })
    }

    fn parse_type_qualifier(qualifier: TypeQualifier) -> (StorageClass, Option<Binding>) {
        let mut storage = None;
        let mut binding = None;

        for qualifier in qualifier.qualifiers {
            match qualifier {
                TypeQualifierSpec::Storage(storage_qualifier) => {
                    assert!(storage.is_none());

                    match storage_qualifier {
                        StorageQualifier::Const => storage = Some(StorageClass::UniformConstant),
                        StorageQualifier::In => storage = Some(StorageClass::Input),
                        StorageQualifier::Out => storage = Some(StorageClass::Output),
                        StorageQualifier::Uniform => storage = Some(StorageClass::Uniform),
                        StorageQualifier::Buffer => storage = Some(StorageClass::StorageBuffer),
                        StorageQualifier::Shared => storage = Some(StorageClass::Workgroup),
                        StorageQualifier::Coherent => storage = Some(StorageClass::Workgroup),
                        _ => panic!(),
                    }
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
                _ => unimplemented!(),
            }
        }

        (storage.unwrap_or(StorageClass::Private), binding)
    }
}

pub fn parse_str(
    source: &str,
    entry: String,
    exec: ExecutionModel,
) -> Result<crate::Module, ParseError> {
    let input = parser::parse(source).unwrap();

    log::debug!("------GLSL PREPROCESSOR------");
    log::debug!("\n{}", input);
    log::debug!("-----------------------------");

    Parser::new(&input, exec)?.parse(entry)
}

#[cfg(test)]
mod tests {
    use super::parse_str;

    #[test]
    fn test_vertex() {
        let data = include_str!("../../../test-data/glsl_vertex_test_shader.vert");

        println!(
            "{:#?}",
            parse_str(data, String::from("main"), spirv::ExecutionModel::Vertex)
        );
    }

    #[test]
    fn test_frag() {
        let _ = env_logger::try_init();

        let data = include_str!("../../../test-data/glsl_phong_lighting.frag");

        println!(
            "{:#?}",
            parse_str(data, String::from("main"), spirv::ExecutionModel::Fragment)
        );
    }

    #[cfg(feature = "glsl_preprocessor")]
    #[test]
    fn test_preprocess() {
        let _ = env_logger::try_init();

        let data = include_str!("../../../test-data/glsl_preprocessor_abuse.vert");

        println!(
            "{:#?}",
            parse_str(data, String::from("main"), spirv::ExecutionModel::Vertex)
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
            parse_str(data, String::from("main"), spirv::ExecutionModel::Vertex)
        );
    }
}
