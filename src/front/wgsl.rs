use crate::{
    storage::{Storage, Token},
    FastHashMap,
};


#[derive(Parser)]
#[grammar = "../grammars/wgsl.pest"]
struct Tokenizer;

#[derive(Debug)]
pub enum Error {
    Pest(pest::error::Error<Rule>),
    BadInt(std::num::ParseIntError),
    BadFloat(std::num::ParseFloatError),
    BadStorageClass(String),
    BadBool(String),
    BadDecoration(String),
    UnknownIdent(String),
    UnknownType(String),
    UnknownFunction(String),
    InvalidVariableClass(spirv::StorageClass),
}
impl From<pest::error::Error<Rule>> for Error {
    fn from(error: pest::error::Error<Rule>) -> Self {
        Error::Pest(error)
    }
}
impl From<std::num::ParseIntError> for Error {
    fn from(error: std::num::ParseIntError) -> Self {
        Error::BadInt(error)
    }
}
impl From<std::num::ParseFloatError> for Error {
    fn from(error: std::num::ParseFloatError) -> Self {
        Error::BadFloat(error)
    }
}

trait StringValueLookup {
    type Value;
    fn lookup(&self, key: &str) -> Result<Self::Value, Error>;
}
impl StringValueLookup for FastHashMap<String, Token<crate::Expression>> {
    type Value = Token<crate::Expression>;
    fn lookup(&self, key: &str) -> Result<Self::Value, Error> {
        self.get(key)
            .cloned()
            .ok_or(Error::UnknownIdent(key.to_owned()))
    }
}

type ExpressionResult = Result<Token<crate::Expression>, Error>;

struct ExpressionContext<'a> {
    function: &'a mut crate::Function,
    lookup_ident: &'a FastHashMap<String, Token<crate::Expression>>,
    types: &'a mut Storage<crate::Type>,
    constants: &'a mut Storage<crate::Constant>,
}

impl<'a> ExpressionContext<'a> {
    fn reborrow(&mut self) -> ExpressionContext {
        ExpressionContext {
            function: self.function,
            lookup_ident: self.lookup_ident,
            types: self.types,
            constants: self.constants,
        }
    }

    fn parse_binary(
        &mut self,
        expression_pair: pest::iterators::Pair<Rule>,
        rule: Rule,
        op: crate::BinaryOperator,
        mut parser: impl FnMut(pest::iterators::Pair<Rule>, ExpressionContext<'_>) -> ExpressionResult,
    ) -> ExpressionResult {
        assert_eq!(expression_pair.as_rule(), rule);
        let mut expression_pairs = expression_pair.into_inner();
        let base_pair = expression_pairs.next().unwrap();
        let mut left = parser(base_pair, self.reborrow())?;
        for next_pair in expression_pairs {
            let expression = crate::Expression::Binary {
                op,
                left,
                right: parser(next_pair, self.reborrow())?,
            };
            left = self.function.expressions.append(expression);
        }
        Ok(left)
    }
}

pub struct Parser {
    lookup_type: FastHashMap<String, Token<crate::Type>>,
}

impl Parser {
    pub fn new() -> Self {
        Parser {
            lookup_type: FastHashMap::default(),
        }
    }

    fn parse_uint_literal(pair: pest::iterators::Pair<Rule>) -> Result<u32, Error> {
        Ok(pair.as_str().parse()?)
    }

    fn parse_int_literal(pair: pest::iterators::Pair<Rule>) -> Result<i32, Error> {
        Ok(pair.as_str().parse()?)
    }

    fn parse_float_literal(pair: pest::iterators::Pair<Rule>) -> Result<f32, Error> {
        Ok(pair.as_str().parse()?)
    }

    fn parse_decoration_list(variable_decoration_list: pest::iterators::Pair<Rule>) -> Result<Option<crate::Binding>, Error> {
        assert_eq!(variable_decoration_list.as_rule(), Rule::variable_decoration_list);
        for variable_decoration in variable_decoration_list.into_inner() {
            match variable_decoration.as_rule() {
                Rule::location_decoration => {
                    let location_pair = variable_decoration.into_inner().next().unwrap();
                    let location = Self::parse_uint_literal(location_pair)?;
                    return Ok(Some(crate::Binding::Location(location)));
                }
                Rule::builtin_decoration => {
                    let builtin = match variable_decoration.as_str() {
                        "position" => spirv::BuiltIn::Position,
                        "vertex_idx" => spirv::BuiltIn::VertexIndex,
                        other => return Err(Error::BadDecoration(other.to_owned())),
                    };
                    return Ok(Some(crate::Binding::BuiltIn(builtin)));
                }
                unknown => panic!("Unexpected decoration: {:?}", unknown),
            }
        }
        unimplemented!()
    }

    fn parse_storage_class(storage_class: pest::iterators::Pair<Rule>) -> Result<spirv::StorageClass, Error> {
        match storage_class.as_str() {
            "in" => Ok(spirv::StorageClass::Input),
            "out" => Ok(spirv::StorageClass::Output),
            other => Err(Error::BadStorageClass(other.to_owned())),
        }
    }

    fn parse_type_decl(
        &self,
        type_decl: pest::iterators::Pair<Rule>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<Token<crate::Type>, Error> {
        assert_eq!(type_decl.as_rule(), Rule::type_decl);
        let mut type_decl_pairs = type_decl.into_inner();
        let type_kind = type_decl_pairs.next().unwrap();
        let inner = match type_kind.as_rule() {
            Rule::scalar_type => {
                let kind = match type_kind.as_str() {
                    "f32" => crate::ScalarKind::Float,
                    "i32" => crate::ScalarKind::Sint,
                    "u32" => crate::ScalarKind::Uint,
                    other => panic!("Unexpected scalar kind {:?}", other),
                };
                crate::TypeInner::Scalar { kind, width: 32 }
            }
            Rule::type_array_kind => {
                let base = self.parse_type_decl(type_decl_pairs.next().unwrap(), type_store)?;
                let size = match type_decl_pairs.next() {
                    Some(pair) => crate::ArraySize::Static(Self::parse_uint_literal(pair)?),
                    None => crate::ArraySize::Dynamic,
                };
                crate::TypeInner::Array { base, size }
            }
            Rule::type_vec_kind => {
                let size = match type_kind.as_str() {
                    "vec2" => crate::VectorSize::Bi,
                    "vec3" => crate::VectorSize::Tri,
                    "vec4" => crate::VectorSize::Quad,
                    other => panic!("Unexpected vec kind {:?}", other),
                };
                crate::TypeInner::Vector { size, kind: crate::ScalarKind::Float, width: 32 }
            }
            Rule::ident => {
                return self.lookup_type
                    .get(type_kind.as_str())
                    .cloned()
                    .ok_or(Error::UnknownType(type_kind.as_str().to_owned()));
            }
            other => panic!("Unexpected type {:?}", other),
        };
        if let Some((token, _)) = type_store
            .iter()
            .find(|(_, ty)| ty.inner == inner)
        {
            return Ok(token);
        }
        Ok(type_store.append(crate::Type {
            name: None,
            inner,
        }))
    }

    fn parse_variable_ident_decl(
        &self,
        variable_ident_decl: pest::iterators::Pair<Rule>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<(String, Token<crate::Type>), Error> {
        assert_eq!(variable_ident_decl.as_rule(), Rule::variable_ident_decl);
        let mut pairs = variable_ident_decl.into_inner();
        let name = pairs.next().unwrap().as_str().to_owned();
        let ty = self.parse_type_decl(pairs.next().unwrap(), type_store)?;
        Ok((name, ty))
    }

    fn parse_variable_decl(
        &self,
        variable_decl: pest::iterators::Pair<Rule>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<(String, Option<spirv::StorageClass>, Token<crate::Type>), Error> {
        assert_eq!(variable_decl.as_rule(), Rule::variable_decl);
        let mut var_decl_pairs = variable_decl.into_inner();
        let mut body = var_decl_pairs.next().unwrap();
        let class = if body.as_rule() == Rule::storage_class {
            let class = Self::parse_storage_class(body)?;
            body = var_decl_pairs.next().unwrap();
            Some(class)
        } else {
            None
        };
        let (name, ty) = self.parse_variable_ident_decl(body, type_store)?;
        Ok((name, class, ty))
    }

    fn parse_struct_decl(
        &self,
        struct_body_decl: pest::iterators::Pair<Rule>,
        type_store: &mut Storage<crate::Type>,
    ) -> Result<crate::TypeInner, Error> {
        assert_eq!(struct_body_decl.as_rule(), Rule::struct_body_decl);
        let mut members = Vec::new();
        for member_decl in struct_body_decl.into_inner() {
            assert_eq!(member_decl.as_rule(), Rule::struct_member);
            let mut member_decl_pairs = member_decl.into_inner();
            let mut body = member_decl_pairs.next().unwrap();
            let binding = None;
            if body.as_rule() == Rule::struct_member_decoration_decl {
                body = member_decl_pairs.next().unwrap(); //TODO: parse properly
            }
            let (member_name, ty) = self.parse_variable_ident_decl(body, type_store)?;
            members.push(crate::StructMember {
                name: Some(member_name),
                binding,
                ty,
            });
        }
        Ok(crate::TypeInner::Struct { members })
    }

    fn parse_const_literal(
        const_literal: pest::iterators::Pair<Rule>,
    ) -> Result<crate::ConstantInner, Error> {
        let inner = match const_literal.as_rule() {
            Rule::int_literal => {
                let value = Self::parse_int_literal(const_literal)?;
                crate::ConstantInner::Sint(value as i64)
            }
            Rule::uint_literal => {
                let value = Self::parse_uint_literal(const_literal)?;
                crate::ConstantInner::Uint(value as u64)
            }
            Rule::bool_literal => {
                let value = match const_literal.as_str() {
                    "true" => true,
                    "false" => false,
                    other => return Err(Error::BadBool(other.to_owned())),
                };
                crate::ConstantInner::Bool(value)
            }
            Rule::float_literal => {
                let value = Self::parse_float_literal(const_literal)?;
                crate::ConstantInner::Float(value as f64)
            }
            ref other => panic!("Unknown const literal {:?}", other),
        };
        Ok(inner)
    }

    fn parse_const_expression(
        const_expression: pest::iterators::Pair<Rule>,
        _const_store: &mut Storage<crate::Constant>,
    ) -> Result<crate::ConstantInner, Error> {
        let mut const_expr_pairs = const_expression.into_inner();
        let first_pair = const_expr_pairs.next().unwrap();
        let inner = match first_pair.as_rule() {
            Rule::const_literal => {
                let const_literal = first_pair.into_inner().next().unwrap();
                Self::parse_const_literal(const_literal)?
            }
            Rule::type_decl => unimplemented!(),
            _ => panic!("Unknown const expr {:?}", first_pair),
        };
        Ok(inner)
    }

    fn parse_primary_expression(
        &self,
        primary_expression: pest::iterators::Pair<Rule>,
        mut ctx: ExpressionContext,
    ) -> ExpressionResult {
        match primary_expression.as_rule() {
            Rule::typed_expression => {
                let mut expr_pairs = primary_expression.into_inner();
                let ty = self.parse_type_decl(expr_pairs.next().unwrap(), ctx.types)?;
                let mut components = Vec::new();
                for argument_pair in expr_pairs {
                    let expr_token = self.parse_primary_expression(argument_pair, ctx.reborrow())?;
                    components.push(expr_token);
                }
                let expression = crate::Expression::Compose { ty, components };
                Ok(ctx.function.expressions.append(expression))
            }
            Rule::const_expr => {
                let inner = Self::parse_const_expression(primary_expression, ctx.constants)?;
                let token = ctx.constants.append(crate::Constant {
                    name: None,
                    specialization: None,
                    inner,
                });
                let expression = crate::Expression::Constant(token);
                Ok(ctx.function.expressions.append(expression))
            }
            Rule::logical_or_expression => {
                self.parse_logical_or_expression(primary_expression, ctx)
            }
            Rule::ident => {
                ctx.lookup_ident.lookup(primary_expression.as_str())
            }
            _ => panic!("Unknown expression {:?}", primary_expression),
        }
    }

    fn parse_relational_expression(
        &self,
        pair: pest::iterators::Pair<Rule>,
        mut context: ExpressionContext,
    ) -> ExpressionResult {
        context.parse_binary(
            pair,
            Rule::additive_expression,
            crate::BinaryOperator::Add,
            |pair, mut context| context.parse_binary(
                pair,
                Rule::multiplicative_expression,
                crate::BinaryOperator::Multiply,
                |pair, context| self.parse_primary_expression(pair, context),
            ),
        )
    }

    fn parse_logical_or_expression(
        &self,
        pair: pest::iterators::Pair<Rule>,
        mut context: ExpressionContext,
    ) -> ExpressionResult {
        context.parse_binary(
            pair,
            Rule::logical_or_expression,
            crate::BinaryOperator::LogicalOr,
            |pair, mut context| context.parse_binary(
                pair,
                Rule::logical_and_expression,
                crate::BinaryOperator::LogicalAnd,
                |pair, mut context| context.parse_binary(
                    pair,
                    Rule::inclusive_or_expression,
                    crate::BinaryOperator::InclusiveOr,
                    |pair, mut context| context.parse_binary(
                        pair,
                        Rule::exclusive_or_expression,
                        crate::BinaryOperator::ExclusiveOr,
                        |pair, mut context| context.parse_binary(
                            pair,
                            Rule::and_expression,
                            crate::BinaryOperator::And,
                            |pair, mut context| context.parse_binary(
                                pair,
                                Rule::equality_expression,
                                crate::BinaryOperator::Equals,
                                |pair, context| self.parse_relational_expression(pair, context),
                            ),
                        ),
                    ),
                ),
            ),
        )
    }

    fn parse_function_decl(
        &self,
        function_decl: pest::iterators::Pair<Rule>,
        module: &mut crate::Module,
    ) -> Result<Token<crate::Function>, Error> {
        let mut lookup_ident = FastHashMap::default();
        assert_eq!(function_decl.as_rule(), Rule::function_decl);
        let mut function_decl_pairs = function_decl.into_inner();

        let function_header = function_decl_pairs.next().unwrap();
        assert_eq!(function_header.as_rule(), Rule::function_header);
        let mut function_header_pairs = function_header.into_inner();
        let fun_name = function_header_pairs.next().unwrap().as_str().to_owned();
        let mut fun = crate::Function {
            name: Some(fun_name),
            control: spirv::FunctionControl::empty(),
            parameter_types: Vec::new(),
            return_type: None,
            expressions: Storage::new(),
            body: Vec::new(),
        };
        for (const_token, constant) in module.constants.iter() {
            if let Some(ref name) = constant.name {
                let expr_token = fun.expressions.append(crate::Expression::Constant(const_token));
                lookup_ident.insert(name.clone(), expr_token);
            }
        }
        for (var_token, variable) in module.global_variables.iter() {
            if let Some(ref name) = variable.name {
                let expr_token = fun.expressions.append(crate::Expression::GlobalVariable(var_token));
                lookup_ident.insert(name.clone(), expr_token);
            }
        }

        let param_list = function_header_pairs.next().unwrap();
        assert_eq!(param_list.as_rule(), Rule::param_list);
        for (i, variable_ident_decl) in param_list.into_inner().enumerate() {
            assert_eq!(variable_ident_decl.as_rule(), Rule::variable_ident_decl);
            let mut variable_ident_decl_pairs = variable_ident_decl.into_inner();
            let param_name = variable_ident_decl_pairs.next().unwrap().as_str().to_owned();
            let expression_token = fun.expressions.append(crate::Expression::FunctionParameter(i as u32));
            lookup_ident.insert(param_name, expression_token);
            let param_type_decl = variable_ident_decl_pairs.next().unwrap();
            let ty = self.parse_type_decl(param_type_decl, &mut module.types)?;
            fun.parameter_types.push(ty);
        }
        if let Some(function_type_decl) = function_header_pairs.next() {
            let ty = self.parse_type_decl(function_type_decl, &mut module.types)?;
            fun.return_type = Some(ty);
        }

        let function_body = function_decl_pairs.next().unwrap();
        assert_eq!(function_body.as_rule(), Rule::body_stmt);
        for statement in function_body.into_inner() {
            assert_eq!(statement.as_rule(), Rule::statement);
            let mut statement_pairs = statement.into_inner();
            let first_statement = match statement_pairs.next() {
                Some(st) => st,
                None => continue,
            };
            let context = ExpressionContext {
                function: &mut fun,
                lookup_ident: &lookup_ident,
                types: &mut module.types,
                constants: &mut module.constants,
            };
            let stmt = match first_statement.as_rule() {
                Rule::return_statement => {
                    let mut return_pairs = first_statement.into_inner();
                    let value = match return_pairs.next() {
                        Some(exp) => Some(self.parse_primary_expression(exp, context)?),
                        None => None,
                    };
                    crate::Statement::Return { value }
                }
                Rule::variable_statement => {
                    let mut variable_pairs = first_statement.into_inner();
                    let variable_decl = variable_pairs.next().unwrap();
                    match variable_decl.as_rule() {
                        Rule::variable_decl => {
                            let (name, class, ty) = self.parse_variable_decl(variable_decl, context.types)?;
                            if let Some(class) = class {
                                return Err(Error::InvalidVariableClass(class));
                            }
                            let value = if let Some(value_pair) = variable_pairs.next() {
                                let value_token = self.parse_primary_expression(value_pair, context)?;
                                lookup_ident.insert(name.clone(), value_token);
                                Some(value_token)
                            } else {
                                None
                            };
                            crate::Statement::VariableDeclaration {
                                name,
                                ty,
                                value,
                            }
                        }
                        Rule::variable_ident_decl => {
                            let (name, ty) = self.parse_variable_ident_decl(variable_decl, context.types)?;
                            let value_pair = variable_pairs.next().unwrap();
                            let value_token = self.parse_primary_expression(value_pair, context)?;
                            lookup_ident.insert(name.clone(), value_token);
                            crate::Statement::VariableDeclaration {
                                name,
                                ty,
                                value: Some(value_token),
                            }
                        }
                        _ => panic!("Unexpected variable decl {:?}", variable_decl)
                    }
                }
                Rule::assignment_statement => {
                    let mut assignment_pairs = first_statement.into_inner();
                    let left_token = lookup_ident.lookup(assignment_pairs.next().unwrap().as_str())?;
                    let right_pair = assignment_pairs.next().unwrap();
                    let right_token = self.parse_primary_expression(right_pair, context)?;
                    crate::Statement::Store {
                        pointer: left_token,
                        value: right_token,
                    }
                }
                _ => panic!("Unknown statement {:?}", first_statement),
            };
            fun.body.push(stmt);
        }
        Ok(module.functions.append(fun))
    }

    pub fn parse(&mut self, source: &str) -> Result<crate::Module, Error> {
        use pest::Parser as _;
        let pairs = Tokenizer::parse(Rule::translation_unit, source)?;
        let mut module = crate::Module::generate_empty();
        for global_decl_maybe in pairs {
            match global_decl_maybe.as_rule() {
                Rule::global_decl => {
                    let global_decl = global_decl_maybe.into_inner().next().unwrap();
                    match global_decl.as_rule() {
                        Rule::import_decl => {
                            let mut import_decl = global_decl.into_inner();
                            let path = import_decl.next().unwrap().as_str();
                            log::warn!("Ignoring import {:?}", path);
                        }
                        Rule::global_variable_decl => {
                            let mut global_decl_pairs = global_decl.into_inner();
                            let binding = Self::parse_decoration_list(global_decl_pairs.next().unwrap())?;
                            let var_decl = global_decl_pairs.next().unwrap();
                            let (name, class, ty) = self.parse_variable_decl(var_decl, &mut module.types)?;
                            module.global_variables.append(crate::GlobalVariable {
                                name: Some(name),
                                class: class.unwrap_or(spirv::StorageClass::Private),
                                binding,
                                ty,
                            });
                        }
                        Rule::global_constant_decl => {
                            let mut global_constant_pairs = global_decl.into_inner();
                            let variable_ident_decl = global_constant_pairs.next().unwrap();
                            let (name, _ty) = self.parse_variable_ident_decl(variable_ident_decl, &mut module.types)?;
                            let const_expr_decl = global_constant_pairs.next().unwrap();
                            let inner = Self::parse_const_expression(const_expr_decl, &mut module.constants)?;
                            module.constants.append(crate::Constant {
                                name: Some(name),
                                specialization: None,
                                inner,
                            });
                        }
                        Rule::type_alias => {
                            let mut type_alias_pairs = global_decl.into_inner();
                            let name = type_alias_pairs.next().unwrap().as_str().to_owned();
                            let something_decl = type_alias_pairs.next().unwrap();
                            match something_decl.as_rule() {
                                Rule::type_decl => {
                                    let token = self.parse_type_decl(something_decl, &mut module.types)?;
                                    self.lookup_type.insert(name, token);
                                }
                                Rule::struct_decl => {
                                    let mut struct_decl_pairs = something_decl.into_inner();
                                    let mut body = struct_decl_pairs.next().unwrap();
                                    while body.as_rule() == Rule::struct_decoration_decl {
                                        body = struct_decl_pairs.next().unwrap(); //skip
                                    }
                                    let inner = self.parse_struct_decl(body, &mut module.types)?;
                                    module.types.append(crate::Type {
                                        name: Some(name),
                                        inner,
                                    });
                                }
                                other => panic!("Unexpected type alias rule {:?}", other),
                            };
                        }
                        Rule::function_decl => {
                            self.parse_function_decl(global_decl, &mut module)?;
                        }
                        Rule::entry_point_decl => {
                            let mut ep_decl_pairs = global_decl.into_inner();
                            let pipeline_stage_pair = ep_decl_pairs.next().unwrap();
                            assert_eq!(pipeline_stage_pair.as_rule(), Rule::pipeline_stage);
                            let mut fun_name_pair = ep_decl_pairs.next().unwrap();
                            let name = fun_name_pair.as_str().to_owned();
                            if fun_name_pair.as_rule() == Rule::string_literal {
                                fun_name_pair = ep_decl_pairs.next().unwrap();
                            }
                            let fun_ident = fun_name_pair.as_str();
                            let function = module.functions
                                .iter()
                                .find(|(_, fun)| fun.name.as_ref().map(|s| s.as_str()) == Some(fun_ident))
                                .map(|(token, _)| token)
                                .ok_or(Error::UnknownFunction(fun_ident.to_owned()))?;
                            module.entry_points.push(crate::EntryPoint {
                                exec_model: match pipeline_stage_pair.as_str() {
                                    "vertex" => spirv::ExecutionModel::Vertex,
                                    "fragment" => spirv::ExecutionModel::Fragment,
                                    "compute" =>  spirv::ExecutionModel::GLCompute,
                                    other => panic!("Unknown execution model {:?}", other),
                                },
                                name,
                                inputs: Vec::new(), //TODO
                                outputs: Vec::new(), //TODO
                                function,
                            });
                        }
                        unknown => panic!("Unexpected global decl: {:?}", unknown),
                    }
                }
                Rule::EOI => break,
                unknown => panic!("Unexpected: {:?}", unknown),
            }
        }
        Ok(module)
    }
}

pub fn parse_str(source: &str) -> Result<crate::Module, Error> {
    Parser::new().parse(source)
}
