use crate::front::wgsl::error::{Error, ExpectedToken};
use crate::front::wgsl::parse::lexer::{Lexer, Token};
use crate::front::wgsl::parse::number::Number;
use crate::front::SymbolTable;
use crate::{Arena, FastIndexSet, Handle, ShaderStage, Span};

pub mod ast;
pub mod conv;
pub mod lexer;
pub mod number;

/// State for constructing an AST expression.
///
/// Not to be confused with [`lower::ExpressionContext`], which is for producing
/// Naga IR from the AST we produce here.
///
/// [`lower::ExpressionContext`]: super::lower::ExpressionContext
struct ExpressionContext<'input, 'temp, 'out> {
    /// The [`TranslationUnit::expressions`] arena to which we should contribute
    /// expressions.
    ///
    /// [`TranslationUnit::expressions`]: ast::TranslationUnit::expressions
    expressions: &'out mut Arena<ast::Expression<'input>>,

    /// The [`TranslationUnit::types`] arena to which we should contribute new
    /// types.
    ///
    /// [`TranslationUnit::types`]: ast::TranslationUnit::types
    types: &'out mut Arena<ast::Type<'input>>,

    /// A map from identifiers in scope to the locals/arguments they represent.
    ///
    /// The handles refer to the [`Function::locals`] area; see that field's
    /// documentation for details.
    ///
    /// [`Function::locals`]: ast::Function::locals
    local_table: &'temp mut SymbolTable<&'input str, Handle<ast::Local>>,

    /// The [`Function::locals`] arena for the function we're building.
    ///
    /// [`Function::locals`]: ast::Function::locals
    locals: &'out mut Arena<ast::Local>,

    /// Identifiers used by the current global declaration that have no local definition.
    ///
    /// This becomes the [`GlobalDecl`]'s [`dependencies`] set.
    ///
    /// Note that we don't know at parse time what kind of [`GlobalDecl`] the
    /// name refers to. We can't look up names until we've seen the entire
    /// translation unit.
    ///
    /// [`GlobalDecl`]: ast::GlobalDecl
    /// [`dependencies`]: ast::GlobalDecl::dependencies
    unresolved: &'out mut FastIndexSet<ast::Dependency<'input>>,
}

impl<'a> ExpressionContext<'a, '_, '_> {
    fn parse_binary_op(
        &mut self,
        lexer: &mut Lexer<'a>,
        classifier: impl Fn(Token<'a>) -> Option<crate::BinaryOperator>,
        mut parser: impl FnMut(
            &mut Lexer<'a>,
            &mut Self,
        ) -> Result<Handle<ast::Expression<'a>>, Error<'a>>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        let start = lexer.start_byte_offset();
        let mut accumulator = parser(lexer, self)?;
        while let Some(op) = classifier(lexer.peek().0) {
            let _ = lexer.next();
            let left = accumulator;
            let right = parser(lexer, self)?;
            accumulator = self.expressions.append(
                ast::Expression::Binary { op, left, right },
                lexer.span_from(start),
            );
        }
        Ok(accumulator)
    }

    fn declare_local(&mut self, name: ast::Ident<'a>) -> Result<Handle<ast::Local>, Error<'a>> {
        let handle = self.locals.append(ast::Local, name.span);
        if let Some(old) = self.local_table.add(name.name, handle) {
            Err(Error::Redefinition {
                previous: self.locals.get_span(old),
                current: name.span,
            })
        } else {
            Ok(handle)
        }
    }
}

/// Which grammar rule we are in the midst of parsing.
///
/// This is used for error checking. `Parser` maintains a stack of
/// these and (occasionally) checks that it is being pushed and popped
/// as expected.
#[derive(Clone, Debug, PartialEq)]
enum Rule {
    Attribute,
    VariableDecl,
    TypeDecl,
    FunctionDecl,
    Block,
    Statement,
    PrimaryExpr,
    SingularExpr,
    UnaryExpr,
    GeneralExpr,
}

struct ParsedAttribute<T> {
    value: Option<T>,
}

impl<T> Default for ParsedAttribute<T> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<T> ParsedAttribute<T> {
    fn set(&mut self, value: T, name_span: Span) -> Result<(), Error<'static>> {
        if self.value.is_some() {
            return Err(Error::RepeatedAttribute(name_span));
        }
        self.value = Some(value);
        Ok(())
    }
}

#[derive(Default)]
struct BindingParser<'a> {
    location: ParsedAttribute<Handle<ast::Expression<'a>>>,
    second_blend_source: ParsedAttribute<bool>,
    built_in: ParsedAttribute<crate::BuiltIn>,
    interpolation: ParsedAttribute<crate::Interpolation>,
    sampling: ParsedAttribute<crate::Sampling>,
    invariant: ParsedAttribute<bool>,
}

impl<'a> BindingParser<'a> {
    fn parse(
        &mut self,
        parser: &mut Parser,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<(), Error<'a>> {
        match name {
            "location" => {
                lexer.expect(Token::Paren('('))?;
                self.location
                    .set(parser.general_expression(lexer, ctx)?, name_span)?;
                lexer.expect(Token::Paren(')'))?;
            }
            "builtin" => {
                lexer.expect(Token::Paren('('))?;
                let (raw, span) = lexer.next_ident_with_span()?;
                self.built_in
                    .set(conv::map_built_in(raw, span)?, name_span)?;
                lexer.expect(Token::Paren(')'))?;
            }
            "interpolate" => {
                lexer.expect(Token::Paren('('))?;
                let (raw, span) = lexer.next_ident_with_span()?;
                self.interpolation
                    .set(conv::map_interpolation(raw, span)?, name_span)?;
                if lexer.skip(Token::Separator(',')) {
                    let (raw, span) = lexer.next_ident_with_span()?;
                    self.sampling
                        .set(conv::map_sampling(raw, span)?, name_span)?;
                }
                lexer.expect(Token::Paren(')'))?;
            }
            "second_blend_source" => {
                self.second_blend_source.set(true, name_span)?;
            }
            "invariant" => {
                self.invariant.set(true, name_span)?;
            }
            _ => return Err(Error::UnknownAttribute(name_span)),
        }
        Ok(())
    }

    fn finish(self, span: Span) -> Result<Option<ast::Binding<'a>>, Error<'a>> {
        match (
            self.location.value,
            self.built_in.value,
            self.interpolation.value,
            self.sampling.value,
            self.invariant.value.unwrap_or_default(),
        ) {
            (None, None, None, None, false) => Ok(None),
            (Some(location), None, interpolation, sampling, false) => {
                // Before handing over the completed `Module`, we call
                // `apply_default_interpolation` to ensure that the interpolation and
                // sampling have been explicitly specified on all vertex shader output and fragment
                // shader input user bindings, so leaving them potentially `None` here is fine.
                Ok(Some(ast::Binding::Location {
                    location,
                    interpolation,
                    sampling,
                    second_blend_source: self.second_blend_source.value.unwrap_or(false),
                }))
            }
            (None, Some(crate::BuiltIn::Position { .. }), None, None, invariant) => {
                Ok(Some(ast::Binding::BuiltIn(crate::BuiltIn::Position {
                    invariant,
                })))
            }
            (None, Some(built_in), None, None, false) => Ok(Some(ast::Binding::BuiltIn(built_in))),
            (_, _, _, _, _) => Err(Error::InconsistentBinding(span)),
        }
    }
}

pub struct Parser {
    rules: Vec<(Rule, usize)>,
}

impl Parser {
    pub const fn new() -> Self {
        Parser { rules: Vec::new() }
    }

    fn reset(&mut self) {
        self.rules.clear();
    }

    fn push_rule_span(&mut self, rule: Rule, lexer: &mut Lexer<'_>) {
        self.rules.push((rule, lexer.start_byte_offset()));
    }

    fn pop_rule_span(&mut self, lexer: &Lexer<'_>) -> Span {
        let (_, initial) = self.rules.pop().unwrap();
        lexer.span_from(initial)
    }

    fn peek_rule_span(&mut self, lexer: &Lexer<'_>) -> Span {
        let &(_, initial) = self.rules.last().unwrap();
        lexer.span_from(initial)
    }

    fn switch_value<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<ast::SwitchValue<'a>, Error<'a>> {
        if let Token::Word("default") = lexer.peek().0 {
            let _ = lexer.next();
            return Ok(ast::SwitchValue::Default);
        }

        let expr = self.general_expression(lexer, ctx)?;
        Ok(ast::SwitchValue::Expr(expr))
    }

    /// Decide if we're looking at a construction expression, and return its
    /// type if so.
    ///
    /// If the identifier `word` is a [type-defining keyword], then return a
    /// [`ConstructorType`] value describing the type to build. Return an error
    /// if the type is not constructible (like `sampler`).
    ///
    /// If `word` isn't a type name, then return `None`.
    ///
    /// [type-defining keyword]: https://gpuweb.github.io/gpuweb/wgsl/#type-defining-keywords
    /// [`ConstructorType`]: ast::ConstructorType
    fn constructor_type<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        word: &'a str,
        span: Span,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<ast::ConstructorType<'a>>, Error<'a>> {
        if let Some((kind, width)) = conv::get_scalar_type(word) {
            return Ok(Some(ast::ConstructorType::Scalar { kind, width }));
        }

        let partial = match word {
            "vec2" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Bi,
            },
            "vec2i" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Bi,
                    kind: crate::ScalarKind::Sint,
                    width: 4,
                }))
            }
            "vec2u" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Bi,
                    kind: crate::ScalarKind::Uint,
                    width: 4,
                }))
            }
            "vec2f" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Bi,
                    kind: crate::ScalarKind::Float,
                    width: 4,
                }))
            }
            "vec3" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Tri,
            },
            "vec3i" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Tri,
                    kind: crate::ScalarKind::Sint,
                    width: 4,
                }))
            }
            "vec3u" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Tri,
                    kind: crate::ScalarKind::Uint,
                    width: 4,
                }))
            }
            "vec3f" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Tri,
                    kind: crate::ScalarKind::Float,
                    width: 4,
                }))
            }
            "vec4" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Quad,
            },
            "vec4i" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Quad,
                    kind: crate::ScalarKind::Sint,
                    width: 4,
                }))
            }
            "vec4u" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Quad,
                    kind: crate::ScalarKind::Uint,
                    width: 4,
                }))
            }
            "vec4f" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Quad,
                    kind: crate::ScalarKind::Float,
                    width: 4,
                }))
            }
            "mat2x2" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Bi,
            },
            "mat2x2f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Bi,
                    width: 4,
                }))
            }
            "mat2x3" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Tri,
            },
            "mat2x3f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Tri,
                    width: 4,
                }))
            }
            "mat2x4" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Quad,
            },
            "mat2x4f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Quad,
                    width: 4,
                }))
            }
            "mat3x2" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Bi,
            },
            "mat3x2f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Bi,
                    width: 4,
                }))
            }
            "mat3x3" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Tri,
            },
            "mat3x3f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Tri,
                    width: 4,
                }))
            }
            "mat3x4" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Quad,
            },
            "mat3x4f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Quad,
                    width: 4,
                }))
            }
            "mat4x2" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Bi,
            },
            "mat4x2f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Bi,
                    width: 4,
                }))
            }
            "mat4x3" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Tri,
            },
            "mat4x3f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Tri,
                    width: 4,
                }))
            }
            "mat4x4" => ast::ConstructorType::PartialMatrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Quad,
            },
            "mat4x4f" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Quad,
                    width: 4,
                }))
            }
            "array" => ast::ConstructorType::PartialArray,
            "atomic"
            | "binding_array"
            | "sampler"
            | "sampler_comparison"
            | "texture_1d"
            | "texture_1d_array"
            | "texture_2d"
            | "texture_2d_array"
            | "texture_3d"
            | "texture_cube"
            | "texture_cube_array"
            | "texture_multisampled_2d"
            | "texture_multisampled_2d_array"
            | "texture_depth_2d"
            | "texture_depth_2d_array"
            | "texture_depth_cube"
            | "texture_depth_cube_array"
            | "texture_depth_multisampled_2d"
            | "texture_storage_1d"
            | "texture_storage_1d_array"
            | "texture_storage_2d"
            | "texture_storage_2d_array"
            | "texture_storage_3d" => return Err(Error::TypeNotConstructible(span)),
            _ => return Ok(None),
        };

        // parse component type if present
        match (lexer.peek().0, partial) {
            (Token::Paren('<'), ast::ConstructorType::PartialVector { size }) => {
                let (kind, width) = lexer.next_scalar_generic()?;
                Ok(Some(ast::ConstructorType::Vector { size, kind, width }))
            }
            (Token::Paren('<'), ast::ConstructorType::PartialMatrix { columns, rows }) => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                match kind {
                    crate::ScalarKind::Float => Ok(Some(ast::ConstructorType::Matrix {
                        columns,
                        rows,
                        width,
                    })),
                    _ => Err(Error::BadMatrixScalarKind(span, kind, width)),
                }
            }
            (Token::Paren('<'), ast::ConstructorType::PartialArray) => {
                lexer.expect_generic_paren('<')?;
                let base = self.type_decl(lexer, ctx)?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let expr = self.unary_expression(lexer, ctx)?;
                    ast::ArraySize::Constant(expr)
                } else {
                    ast::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                Ok(Some(ast::ConstructorType::Array { base, size }))
            }
            (_, partial) => Ok(Some(partial)),
        }
    }

    /// Expects `name` to be consumed (not in lexer).
    fn arguments<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Vec<Handle<ast::Expression<'a>>>, Error<'a>> {
        lexer.open_arguments()?;
        let mut arguments = Vec::new();
        loop {
            if !arguments.is_empty() {
                if !lexer.next_argument()? {
                    break;
                }
            } else if lexer.skip(Token::Paren(')')) {
                break;
            }
            let arg = self.general_expression(lexer, ctx)?;
            arguments.push(arg);
        }

        Ok(arguments)
    }

    /// Expects [`Rule::PrimaryExpr`] or [`Rule::SingularExpr`] on top; does not pop it.
    /// Expects `name` to be consumed (not in lexer).
    fn function_call<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        assert!(self.rules.last().is_some());

        let expr = match name {
            // bitcast looks like a function call, but it's an operator and must be handled differently.
            "bitcast" => {
                lexer.expect_generic_paren('<')?;
                let start = lexer.start_byte_offset();
                let to = self.type_decl(lexer, ctx)?;
                let span = lexer.span_from(start);
                lexer.expect_generic_paren('>')?;

                lexer.open_arguments()?;
                let expr = self.general_expression(lexer, ctx)?;
                lexer.close_arguments()?;

                ast::Expression::Bitcast {
                    expr,
                    to,
                    ty_span: span,
                }
            }
            // everything else must be handled later, since they can be hidden by user-defined functions.
            _ => {
                let arguments = self.arguments(lexer, ctx)?;
                ctx.unresolved.insert(ast::Dependency {
                    ident: name,
                    usage: name_span,
                });
                ast::Expression::Call {
                    function: ast::Ident {
                        name,
                        span: name_span,
                    },
                    arguments,
                }
            }
        };

        let span = self.peek_rule_span(lexer);
        let expr = ctx.expressions.append(expr, span);
        Ok(expr)
    }

    fn ident_expr<'a>(
        &mut self,
        name: &'a str,
        name_span: Span,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> ast::IdentExpr<'a> {
        match ctx.local_table.lookup(name) {
            Some(&local) => ast::IdentExpr::Local(local),
            None => {
                ctx.unresolved.insert(ast::Dependency {
                    ident: name,
                    usage: name_span,
                });
                ast::IdentExpr::Unresolved(name)
            }
        }
    }

    fn primary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        self.push_rule_span(Rule::PrimaryExpr, lexer);

        let expr = match lexer.peek() {
            (Token::Paren('('), _) => {
                let _ = lexer.next();
                let expr = self.general_expression(lexer, ctx)?;
                lexer.expect(Token::Paren(')'))?;
                self.pop_rule_span(lexer);
                return Ok(expr);
            }
            (Token::Word("true"), _) => {
                let _ = lexer.next();
                ast::Expression::Literal(ast::Literal::Bool(true))
            }
            (Token::Word("false"), _) => {
                let _ = lexer.next();
                ast::Expression::Literal(ast::Literal::Bool(false))
            }
            (Token::Number(res), span) => {
                let _ = lexer.next();
                let num = res.map_err(|err| Error::BadNumber(span, err))?;
                ast::Expression::Literal(ast::Literal::Number(num))
            }
            (Token::Word("RAY_FLAG_NONE"), _) => {
                let _ = lexer.next();
                ast::Expression::Literal(ast::Literal::Number(Number::U32(0)))
            }
            (Token::Word("RAY_FLAG_TERMINATE_ON_FIRST_HIT"), _) => {
                let _ = lexer.next();
                ast::Expression::Literal(ast::Literal::Number(Number::U32(4)))
            }
            (Token::Word("RAY_QUERY_INTERSECTION_NONE"), _) => {
                let _ = lexer.next();
                ast::Expression::Literal(ast::Literal::Number(Number::U32(0)))
            }
            (Token::Word(word), span) => {
                let start = lexer.start_byte_offset();
                let _ = lexer.next();

                if let Some(ty) = self.constructor_type(lexer, word, span, ctx)? {
                    let ty_span = lexer.span_from(start);
                    let components = self.arguments(lexer, ctx)?;
                    ast::Expression::Construct {
                        ty,
                        ty_span,
                        components,
                    }
                } else if let Token::Paren('(') = lexer.peek().0 {
                    self.pop_rule_span(lexer);
                    return self.function_call(lexer, word, span, ctx);
                } else if word == "bitcast" {
                    self.pop_rule_span(lexer);
                    return self.function_call(lexer, word, span, ctx);
                } else {
                    let ident = self.ident_expr(word, span, ctx);
                    ast::Expression::Ident(ident)
                }
            }
            other => return Err(Error::Unexpected(other.1, ExpectedToken::PrimaryExpression)),
        };

        let span = self.pop_rule_span(lexer);
        let expr = ctx.expressions.append(expr, span);
        Ok(expr)
    }

    fn postfix<'a>(
        &mut self,
        span_start: usize,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
        expr: Handle<ast::Expression<'a>>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        let mut expr = expr;

        loop {
            let expression = match lexer.peek().0 {
                Token::Separator('.') => {
                    let _ = lexer.next();
                    let field = lexer.next_ident()?;

                    ast::Expression::Member { base: expr, field }
                }
                Token::Paren('[') => {
                    let _ = lexer.next();
                    let index = self.general_expression(lexer, ctx)?;
                    lexer.expect(Token::Paren(']'))?;

                    ast::Expression::Index { base: expr, index }
                }
                _ => break,
            };

            let span = lexer.span_from(span_start);
            expr = ctx.expressions.append(expression, span);
        }

        Ok(expr)
    }

    /// Parse a `unary_expression`.
    fn unary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        self.push_rule_span(Rule::UnaryExpr, lexer);
        //TODO: refactor this to avoid backing up
        let expr = match lexer.peek().0 {
            Token::Operation('-') => {
                let _ = lexer.next();
                let expr = self.unary_expression(lexer, ctx)?;
                let expr = ast::Expression::Unary {
                    op: crate::UnaryOperator::Negate,
                    expr,
                };
                let span = self.peek_rule_span(lexer);
                ctx.expressions.append(expr, span)
            }
            Token::Operation('!') => {
                let _ = lexer.next();
                let expr = self.unary_expression(lexer, ctx)?;
                let expr = ast::Expression::Unary {
                    op: crate::UnaryOperator::LogicalNot,
                    expr,
                };
                let span = self.peek_rule_span(lexer);
                ctx.expressions.append(expr, span)
            }
            Token::Operation('~') => {
                let _ = lexer.next();
                let expr = self.unary_expression(lexer, ctx)?;
                let expr = ast::Expression::Unary {
                    op: crate::UnaryOperator::BitwiseNot,
                    expr,
                };
                let span = self.peek_rule_span(lexer);
                ctx.expressions.append(expr, span)
            }
            Token::Operation('*') => {
                let _ = lexer.next();
                let expr = self.unary_expression(lexer, ctx)?;
                let expr = ast::Expression::Deref(expr);
                let span = self.peek_rule_span(lexer);
                ctx.expressions.append(expr, span)
            }
            Token::Operation('&') => {
                let _ = lexer.next();
                let expr = self.unary_expression(lexer, ctx)?;
                let expr = ast::Expression::AddrOf(expr);
                let span = self.peek_rule_span(lexer);
                ctx.expressions.append(expr, span)
            }
            _ => self.singular_expression(lexer, ctx)?,
        };

        self.pop_rule_span(lexer);
        Ok(expr)
    }

    /// Parse a `singular_expression`.
    fn singular_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        let start = lexer.start_byte_offset();
        self.push_rule_span(Rule::SingularExpr, lexer);
        let primary_expr = self.primary_expression(lexer, ctx)?;
        let singular_expr = self.postfix(start, lexer, ctx, primary_expr)?;
        self.pop_rule_span(lexer);

        Ok(singular_expr)
    }

    fn equality_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        context: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        // equality_expression
        context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('=') => Some(crate::BinaryOperator::Equal),
                Token::LogicalOperation('!') => Some(crate::BinaryOperator::NotEqual),
                _ => None,
            },
            // relational_expression
            |lexer, context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::Paren('<') => Some(crate::BinaryOperator::Less),
                        Token::Paren('>') => Some(crate::BinaryOperator::Greater),
                        Token::LogicalOperation('<') => Some(crate::BinaryOperator::LessEqual),
                        Token::LogicalOperation('>') => Some(crate::BinaryOperator::GreaterEqual),
                        _ => None,
                    },
                    // shift_expression
                    |lexer, context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::ShiftOperation('<') => {
                                    Some(crate::BinaryOperator::ShiftLeft)
                                }
                                Token::ShiftOperation('>') => {
                                    Some(crate::BinaryOperator::ShiftRight)
                                }
                                _ => None,
                            },
                            // additive_expression
                            |lexer, context| {
                                context.parse_binary_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('+') => Some(crate::BinaryOperator::Add),
                                        Token::Operation('-') => {
                                            Some(crate::BinaryOperator::Subtract)
                                        }
                                        _ => None,
                                    },
                                    // multiplicative_expression
                                    |lexer, context| {
                                        context.parse_binary_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('*') => {
                                                    Some(crate::BinaryOperator::Multiply)
                                                }
                                                Token::Operation('/') => {
                                                    Some(crate::BinaryOperator::Divide)
                                                }
                                                Token::Operation('%') => {
                                                    Some(crate::BinaryOperator::Modulo)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| self.unary_expression(lexer, context),
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )
    }

    fn general_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Expression<'a>>, Error<'a>> {
        self.general_expression_with_span(lexer, ctx)
            .map(|(expr, _)| expr)
    }

    fn general_expression_with_span<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        context: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<(Handle<ast::Expression<'a>>, Span), Error<'a>> {
        self.push_rule_span(Rule::GeneralExpr, lexer);
        // logical_or_expression
        let handle = context.parse_binary_op(
            lexer,
            |token| match token {
                Token::LogicalOperation('|') => Some(crate::BinaryOperator::LogicalOr),
                _ => None,
            },
            // logical_and_expression
            |lexer, context| {
                context.parse_binary_op(
                    lexer,
                    |token| match token {
                        Token::LogicalOperation('&') => Some(crate::BinaryOperator::LogicalAnd),
                        _ => None,
                    },
                    // inclusive_or_expression
                    |lexer, context| {
                        context.parse_binary_op(
                            lexer,
                            |token| match token {
                                Token::Operation('|') => Some(crate::BinaryOperator::InclusiveOr),
                                _ => None,
                            },
                            // exclusive_or_expression
                            |lexer, context| {
                                context.parse_binary_op(
                                    lexer,
                                    |token| match token {
                                        Token::Operation('^') => {
                                            Some(crate::BinaryOperator::ExclusiveOr)
                                        }
                                        _ => None,
                                    },
                                    // and_expression
                                    |lexer, context| {
                                        context.parse_binary_op(
                                            lexer,
                                            |token| match token {
                                                Token::Operation('&') => {
                                                    Some(crate::BinaryOperator::And)
                                                }
                                                _ => None,
                                            },
                                            |lexer, context| {
                                                self.equality_expression(lexer, context)
                                            },
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )?;
        Ok((handle, self.pop_rule_span(lexer)))
    }

    fn variable_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<ast::GlobalVariable<'a>, Error<'a>> {
        self.push_rule_span(Rule::VariableDecl, lexer);
        let mut space = crate::AddressSpace::Handle;

        if lexer.skip(Token::Paren('<')) {
            let (class_str, span) = lexer.next_ident_with_span()?;
            space = match class_str {
                "storage" => {
                    let access = if lexer.skip(Token::Separator(',')) {
                        lexer.next_storage_access()?
                    } else {
                        // defaulting to `read`
                        crate::StorageAccess::LOAD
                    };
                    crate::AddressSpace::Storage { access }
                }
                _ => conv::map_address_space(class_str, span)?,
            };
            lexer.expect(Token::Paren('>'))?;
        }
        let name = lexer.next_ident()?;
        lexer.expect(Token::Separator(':'))?;
        let ty = self.type_decl(lexer, ctx)?;

        let init = if lexer.skip(Token::Operation('=')) {
            let handle = self.general_expression(lexer, ctx)?;
            Some(handle)
        } else {
            None
        };
        lexer.expect(Token::Separator(';'))?;
        self.pop_rule_span(lexer);

        Ok(ast::GlobalVariable {
            name,
            space,
            binding: None,
            ty,
            init,
        })
    }

    fn struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Vec<ast::StructMember<'a>>, Error<'a>> {
        let mut members = Vec::new();

        lexer.expect(Token::Paren('{'))?;
        let mut ready = true;
        while !lexer.skip(Token::Paren('}')) {
            if !ready {
                return Err(Error::Unexpected(
                    lexer.next().1,
                    ExpectedToken::Token(Token::Separator(',')),
                ));
            }
            let (mut size, mut align) = (ParsedAttribute::default(), ParsedAttribute::default());
            self.push_rule_span(Rule::Attribute, lexer);
            let mut bind_parser = BindingParser::default();
            while lexer.skip(Token::Attribute) {
                match lexer.next_ident_with_span()? {
                    ("size", name_span) => {
                        lexer.expect(Token::Paren('('))?;
                        let expr = self.general_expression(lexer, ctx)?;
                        lexer.expect(Token::Paren(')'))?;
                        size.set(expr, name_span)?;
                    }
                    ("align", name_span) => {
                        lexer.expect(Token::Paren('('))?;
                        let expr = self.general_expression(lexer, ctx)?;
                        lexer.expect(Token::Paren(')'))?;
                        align.set(expr, name_span)?;
                    }
                    (word, word_span) => bind_parser.parse(self, lexer, word, word_span, ctx)?,
                }
            }

            let bind_span = self.pop_rule_span(lexer);
            let binding = bind_parser.finish(bind_span)?;

            let name = lexer.next_ident()?;
            lexer.expect(Token::Separator(':'))?;
            let ty = self.type_decl(lexer, ctx)?;
            ready = lexer.skip(Token::Separator(','));

            members.push(ast::StructMember {
                name,
                ty,
                binding,
                size: size.value,
                align: align.value,
            });
        }

        Ok(members)
    }

    fn matrix_scalar_type<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    ) -> Result<ast::Type<'a>, Error<'a>> {
        let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
        match kind {
            crate::ScalarKind::Float => Ok(ast::Type::Matrix {
                columns,
                rows,
                width,
            }),
            _ => Err(Error::BadMatrixScalarKind(span, kind, width)),
        }
    }

    fn type_decl_impl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        word: &'a str,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<ast::Type<'a>>, Error<'a>> {
        if let Some((kind, width)) = conv::get_scalar_type(word) {
            return Ok(Some(ast::Type::Scalar { kind, width }));
        }

        Ok(Some(match word {
            "vec2" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::Type::Vector {
                    size: crate::VectorSize::Bi,
                    kind,
                    width,
                }
            }
            "vec2i" => ast::Type::Vector {
                size: crate::VectorSize::Bi,
                kind: crate::ScalarKind::Sint,
                width: 4,
            },
            "vec2u" => ast::Type::Vector {
                size: crate::VectorSize::Bi,
                kind: crate::ScalarKind::Uint,
                width: 4,
            },
            "vec2f" => ast::Type::Vector {
                size: crate::VectorSize::Bi,
                kind: crate::ScalarKind::Float,
                width: 4,
            },
            "vec3" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::Type::Vector {
                    size: crate::VectorSize::Tri,
                    kind,
                    width,
                }
            }
            "vec3i" => ast::Type::Vector {
                size: crate::VectorSize::Tri,
                kind: crate::ScalarKind::Sint,
                width: 4,
            },
            "vec3u" => ast::Type::Vector {
                size: crate::VectorSize::Tri,
                kind: crate::ScalarKind::Uint,
                width: 4,
            },
            "vec3f" => ast::Type::Vector {
                size: crate::VectorSize::Tri,
                kind: crate::ScalarKind::Float,
                width: 4,
            },
            "vec4" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::Type::Vector {
                    size: crate::VectorSize::Quad,
                    kind,
                    width,
                }
            }
            "vec4i" => ast::Type::Vector {
                size: crate::VectorSize::Quad,
                kind: crate::ScalarKind::Sint,
                width: 4,
            },
            "vec4u" => ast::Type::Vector {
                size: crate::VectorSize::Quad,
                kind: crate::ScalarKind::Uint,
                width: 4,
            },
            "vec4f" => ast::Type::Vector {
                size: crate::VectorSize::Quad,
                kind: crate::ScalarKind::Float,
                width: 4,
            },
            "mat2x2" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Bi, crate::VectorSize::Bi)?
            }
            "mat2x2f" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Bi,
                width: 4,
            },
            "mat2x3" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Bi, crate::VectorSize::Tri)?
            }
            "mat2x3f" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Tri,
                width: 4,
            },
            "mat2x4" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Bi, crate::VectorSize::Quad)?
            }
            "mat2x4f" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Quad,
                width: 4,
            },
            "mat3x2" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Tri, crate::VectorSize::Bi)?
            }
            "mat3x2f" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Bi,
                width: 4,
            },
            "mat3x3" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Tri, crate::VectorSize::Tri)?
            }
            "mat3x3f" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Tri,
                width: 4,
            },
            "mat3x4" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Tri, crate::VectorSize::Quad)?
            }
            "mat3x4f" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Quad,
                width: 4,
            },
            "mat4x2" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Quad, crate::VectorSize::Bi)?
            }
            "mat4x2f" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Bi,
                width: 4,
            },
            "mat4x3" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Quad, crate::VectorSize::Tri)?
            }
            "mat4x3f" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Tri,
                width: 4,
            },
            "mat4x4" => {
                self.matrix_scalar_type(lexer, crate::VectorSize::Quad, crate::VectorSize::Quad)?
            }
            "mat4x4f" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Quad,
                width: 4,
            },
            "atomic" => {
                let (kind, width) = lexer.next_scalar_generic()?;
                ast::Type::Atomic { kind, width }
            }
            "ptr" => {
                lexer.expect_generic_paren('<')?;
                let (ident, span) = lexer.next_ident_with_span()?;
                let mut space = conv::map_address_space(ident, span)?;
                lexer.expect(Token::Separator(','))?;
                let base = self.type_decl(lexer, ctx)?;
                if let crate::AddressSpace::Storage { ref mut access } = space {
                    *access = if lexer.skip(Token::Separator(',')) {
                        lexer.next_storage_access()?
                    } else {
                        crate::StorageAccess::LOAD
                    };
                }
                lexer.expect_generic_paren('>')?;
                ast::Type::Pointer { base, space }
            }
            "array" => {
                lexer.expect_generic_paren('<')?;
                let base = self.type_decl(lexer, ctx)?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let size = self.unary_expression(lexer, ctx)?;
                    ast::ArraySize::Constant(size)
                } else {
                    ast::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                ast::Type::Array { base, size }
            }
            "binding_array" => {
                lexer.expect_generic_paren('<')?;
                let base = self.type_decl(lexer, ctx)?;
                let size = if lexer.skip(Token::Separator(',')) {
                    let size = self.unary_expression(lexer, ctx)?;
                    ast::ArraySize::Constant(size)
                } else {
                    ast::ArraySize::Dynamic
                };
                lexer.expect_generic_paren('>')?;

                ast::Type::BindingArray { base, size }
            }
            "sampler" => ast::Type::Sampler { comparison: false },
            "sampler_comparison" => ast::Type::Sampler { comparison: true },
            "texture_1d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_1d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_3d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_cube_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: false },
                }
            }
            "texture_multisampled_2d" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_multisampled_2d_array" => {
                let (kind, width, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(kind, width, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled { kind, multi: true },
                }
            }
            "texture_depth_2d" => ast::Type::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_2d_array" => ast::Type::Image {
                dim: crate::ImageDimension::D2,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube" => ast::Type::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_cube_array" => ast::Type::Image {
                dim: crate::ImageDimension::Cube,
                arrayed: true,
                class: crate::ImageClass::Depth { multi: false },
            },
            "texture_depth_multisampled_2d" => ast::Type::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::Depth { multi: true },
            },
            "texture_storage_1d" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_1d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_2d_array" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "texture_storage_3d" => {
                let (format, access) = lexer.next_format_generic()?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Storage { format, access },
                }
            }
            "acceleration_structure" => ast::Type::AccelerationStructure,
            "ray_query" => ast::Type::RayQuery,
            "RayDesc" => ast::Type::RayDesc,
            "RayIntersection" => ast::Type::RayIntersection,
            _ => return Ok(None),
        }))
    }

    const fn check_texture_sample_type(
        kind: crate::ScalarKind,
        width: u8,
        span: Span,
    ) -> Result<(), Error<'static>> {
        use crate::ScalarKind::*;
        // Validate according to https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
        match (kind, width) {
            (Float | Sint | Uint, 4) => Ok(()),
            _ => Err(Error::BadTextureSampleType { span, kind, width }),
        }
    }

    /// Parse type declaration of a given name.
    fn type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Handle<ast::Type<'a>>, Error<'a>> {
        self.push_rule_span(Rule::TypeDecl, lexer);

        let (name, span) = lexer.next_ident_with_span()?;

        let ty = match self.type_decl_impl(lexer, name, ctx)? {
            Some(ty) => ty,
            None => {
                ctx.unresolved.insert(ast::Dependency {
                    ident: name,
                    usage: span,
                });
                ast::Type::User(ast::Ident { name, span })
            }
        };

        self.pop_rule_span(lexer);

        let handle = ctx.types.append(ty, Span::UNDEFINED);
        Ok(handle)
    }

    fn assignment_op_and_rhs<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
        block: &mut ast::Block<'a>,
        target: Handle<ast::Expression<'a>>,
        span_start: usize,
    ) -> Result<(), Error<'a>> {
        use crate::BinaryOperator as Bo;

        let op = lexer.next();
        let (op, value) = match op {
            (Token::Operation('='), _) => {
                let value = self.general_expression(lexer, ctx)?;
                (None, value)
            }
            (Token::AssignmentOperation(c), _) => {
                let op = match c {
                    '<' => Bo::ShiftLeft,
                    '>' => Bo::ShiftRight,
                    '+' => Bo::Add,
                    '-' => Bo::Subtract,
                    '*' => Bo::Multiply,
                    '/' => Bo::Divide,
                    '%' => Bo::Modulo,
                    '&' => Bo::And,
                    '|' => Bo::InclusiveOr,
                    '^' => Bo::ExclusiveOr,
                    // Note: `consume_token` shouldn't produce any other assignment ops
                    _ => unreachable!(),
                };

                let value = self.general_expression(lexer, ctx)?;
                (Some(op), value)
            }
            token @ (Token::IncrementOperation | Token::DecrementOperation, _) => {
                let op = match token.0 {
                    Token::IncrementOperation => ast::StatementKind::Increment,
                    Token::DecrementOperation => ast::StatementKind::Decrement,
                    _ => unreachable!(),
                };

                let span = lexer.span_from(span_start);
                block.stmts.push(ast::Statement {
                    kind: op(target),
                    span,
                });
                return Ok(());
            }
            _ => return Err(Error::Unexpected(op.1, ExpectedToken::Assignment)),
        };

        let span = lexer.span_from(span_start);
        block.stmts.push(ast::Statement {
            kind: ast::StatementKind::Assign { target, op, value },
            span,
        });
        Ok(())
    }

    /// Parse an assignment statement (will also parse increment and decrement statements)
    fn assignment_statement<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
        block: &mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        let span_start = lexer.start_byte_offset();
        let target = self.general_expression(lexer, ctx)?;
        self.assignment_op_and_rhs(lexer, ctx, block, target, span_start)
    }

    /// Parse a function call statement.
    /// Expects `ident` to be consumed (not in the lexer).
    fn function_statement<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ident: &'a str,
        ident_span: Span,
        span_start: usize,
        context: &mut ExpressionContext<'a, '_, '_>,
        block: &mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        self.push_rule_span(Rule::SingularExpr, lexer);

        context.unresolved.insert(ast::Dependency {
            ident,
            usage: ident_span,
        });
        let arguments = self.arguments(lexer, context)?;
        let span = lexer.span_from(span_start);

        block.stmts.push(ast::Statement {
            kind: ast::StatementKind::Call {
                function: ast::Ident {
                    name: ident,
                    span: ident_span,
                },
                arguments,
            },
            span,
        });

        self.pop_rule_span(lexer);

        Ok(())
    }

    fn function_call_or_assignment_statement<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        context: &mut ExpressionContext<'a, '_, '_>,
        block: &mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        let span_start = lexer.start_byte_offset();
        match lexer.peek() {
            (Token::Word(name), span) => {
                // A little hack for 2 token lookahead.
                let cloned = lexer.clone();
                let _ = lexer.next();
                match lexer.peek() {
                    (Token::Paren('('), _) => {
                        self.function_statement(lexer, name, span, span_start, context, block)
                    }
                    _ => {
                        *lexer = cloned;
                        self.assignment_statement(lexer, context, block)
                    }
                }
            }
            _ => self.assignment_statement(lexer, context, block),
        }
    }

    fn statement<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
        block: &mut ast::Block<'a>,
    ) -> Result<(), Error<'a>> {
        self.push_rule_span(Rule::Statement, lexer);
        match lexer.peek() {
            (Token::Separator(';'), _) => {
                let _ = lexer.next();
                self.pop_rule_span(lexer);
                return Ok(());
            }
            (Token::Paren('{'), _) => {
                let (inner, span) = self.block(lexer, ctx)?;
                block.stmts.push(ast::Statement {
                    kind: ast::StatementKind::Block(inner),
                    span,
                });
                self.pop_rule_span(lexer);
                return Ok(());
            }
            (Token::Word(word), _) => {
                let kind = match word {
                    "_" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Operation('='))?;
                        let expr = self.general_expression(lexer, ctx)?;
                        lexer.expect(Token::Separator(';'))?;

                        ast::StatementKind::Ignore(expr)
                    }
                    "let" => {
                        let _ = lexer.next();
                        let name = lexer.next_ident()?;

                        let given_ty = if lexer.skip(Token::Separator(':')) {
                            let ty = self.type_decl(lexer, ctx)?;
                            Some(ty)
                        } else {
                            None
                        };
                        lexer.expect(Token::Operation('='))?;
                        let expr_id = self.general_expression(lexer, ctx)?;
                        lexer.expect(Token::Separator(';'))?;

                        let handle = ctx.declare_local(name)?;
                        ast::StatementKind::LocalDecl(ast::LocalDecl::Let(ast::Let {
                            name,
                            ty: given_ty,
                            init: expr_id,
                            handle,
                        }))
                    }
                    "var" => {
                        let _ = lexer.next();

                        let name = lexer.next_ident()?;
                        let ty = if lexer.skip(Token::Separator(':')) {
                            let ty = self.type_decl(lexer, ctx)?;
                            Some(ty)
                        } else {
                            None
                        };

                        let init = if lexer.skip(Token::Operation('=')) {
                            let init = self.general_expression(lexer, ctx)?;
                            Some(init)
                        } else {
                            None
                        };

                        lexer.expect(Token::Separator(';'))?;

                        let handle = ctx.declare_local(name)?;
                        ast::StatementKind::LocalDecl(ast::LocalDecl::Var(ast::LocalVariable {
                            name,
                            ty,
                            init,
                            handle,
                        }))
                    }
                    "return" => {
                        let _ = lexer.next();
                        let value = if lexer.peek().0 != Token::Separator(';') {
                            let handle = self.general_expression(lexer, ctx)?;
                            Some(handle)
                        } else {
                            None
                        };
                        lexer.expect(Token::Separator(';'))?;
                        ast::StatementKind::Return { value }
                    }
                    "if" => {
                        let _ = lexer.next();
                        let condition = self.general_expression(lexer, ctx)?;

                        let accept = self.block(lexer, ctx)?.0;

                        let mut elsif_stack = Vec::new();
                        let mut elseif_span_start = lexer.start_byte_offset();
                        let mut reject = loop {
                            if !lexer.skip(Token::Word("else")) {
                                break ast::Block::default();
                            }

                            if !lexer.skip(Token::Word("if")) {
                                // ... else { ... }
                                break self.block(lexer, ctx)?.0;
                            }

                            // ... else if (...) { ... }
                            let other_condition = self.general_expression(lexer, ctx)?;
                            let other_block = self.block(lexer, ctx)?;
                            elsif_stack.push((elseif_span_start, other_condition, other_block));
                            elseif_span_start = lexer.start_byte_offset();
                        };

                        // reverse-fold the else-if blocks
                        //Note: we may consider uplifting this to the IR
                        for (other_span_start, other_cond, other_block) in
                            elsif_stack.into_iter().rev()
                        {
                            let sub_stmt = ast::StatementKind::If {
                                condition: other_cond,
                                accept: other_block.0,
                                reject,
                            };
                            reject = ast::Block::default();
                            let span = lexer.span_from(other_span_start);
                            reject.stmts.push(ast::Statement {
                                kind: sub_stmt,
                                span,
                            })
                        }

                        ast::StatementKind::If {
                            condition,
                            accept,
                            reject,
                        }
                    }
                    "switch" => {
                        let _ = lexer.next();
                        let selector = self.general_expression(lexer, ctx)?;
                        lexer.expect(Token::Paren('{'))?;
                        let mut cases = Vec::new();

                        loop {
                            // cases + default
                            match lexer.next() {
                                (Token::Word("case"), _) => {
                                    // parse a list of values
                                    let value = loop {
                                        let value = self.switch_value(lexer, ctx)?;
                                        if lexer.skip(Token::Separator(',')) {
                                            if lexer.skip(Token::Separator(':')) {
                                                break value;
                                            }
                                        } else {
                                            lexer.skip(Token::Separator(':'));
                                            break value;
                                        }
                                        cases.push(ast::SwitchCase {
                                            value,
                                            body: ast::Block::default(),
                                            fall_through: true,
                                        });
                                    };

                                    let body = self.block(lexer, ctx)?.0;

                                    cases.push(ast::SwitchCase {
                                        value,
                                        body,
                                        fall_through: false,
                                    });
                                }
                                (Token::Word("default"), _) => {
                                    lexer.skip(Token::Separator(':'));
                                    let body = self.block(lexer, ctx)?.0;
                                    cases.push(ast::SwitchCase {
                                        value: ast::SwitchValue::Default,
                                        body,
                                        fall_through: false,
                                    });
                                }
                                (Token::Paren('}'), _) => break,
                                (_, span) => {
                                    return Err(Error::Unexpected(span, ExpectedToken::SwitchItem))
                                }
                            }
                        }

                        ast::StatementKind::Switch { selector, cases }
                    }
                    "loop" => self.r#loop(lexer, ctx)?,
                    "while" => {
                        let _ = lexer.next();
                        let mut body = ast::Block::default();

                        let (condition, span) = lexer.capture_span(|lexer| {
                            let condition = self.general_expression(lexer, ctx)?;
                            Ok(condition)
                        })?;
                        let mut reject = ast::Block::default();
                        reject.stmts.push(ast::Statement {
                            kind: ast::StatementKind::Break,
                            span,
                        });

                        body.stmts.push(ast::Statement {
                            kind: ast::StatementKind::If {
                                condition,
                                accept: ast::Block::default(),
                                reject,
                            },
                            span,
                        });

                        let (block, span) = self.block(lexer, ctx)?;
                        body.stmts.push(ast::Statement {
                            kind: ast::StatementKind::Block(block),
                            span,
                        });

                        ast::StatementKind::Loop {
                            body,
                            continuing: ast::Block::default(),
                            break_if: None,
                        }
                    }
                    "for" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Paren('('))?;

                        ctx.local_table.push_scope();

                        if !lexer.skip(Token::Separator(';')) {
                            let num_statements = block.stmts.len();
                            let (_, span) = {
                                let ctx = &mut *ctx;
                                let block = &mut *block;
                                lexer.capture_span(|lexer| self.statement(lexer, ctx, block))?
                            };

                            if block.stmts.len() != num_statements {
                                match block.stmts.last().unwrap().kind {
                                    ast::StatementKind::Call { .. }
                                    | ast::StatementKind::Assign { .. }
                                    | ast::StatementKind::LocalDecl(_) => {}
                                    _ => return Err(Error::InvalidForInitializer(span)),
                                }
                            }
                        };

                        let mut body = ast::Block::default();
                        if !lexer.skip(Token::Separator(';')) {
                            let (condition, span) = lexer.capture_span(|lexer| {
                                let condition = self.general_expression(lexer, ctx)?;
                                lexer.expect(Token::Separator(';'))?;
                                Ok(condition)
                            })?;
                            let mut reject = ast::Block::default();
                            reject.stmts.push(ast::Statement {
                                kind: ast::StatementKind::Break,
                                span,
                            });
                            body.stmts.push(ast::Statement {
                                kind: ast::StatementKind::If {
                                    condition,
                                    accept: ast::Block::default(),
                                    reject,
                                },
                                span,
                            });
                        };

                        let mut continuing = ast::Block::default();
                        if !lexer.skip(Token::Paren(')')) {
                            self.function_call_or_assignment_statement(
                                lexer,
                                ctx,
                                &mut continuing,
                            )?;
                            lexer.expect(Token::Paren(')'))?;
                        }

                        let (block, span) = self.block(lexer, ctx)?;
                        body.stmts.push(ast::Statement {
                            kind: ast::StatementKind::Block(block),
                            span,
                        });

                        ctx.local_table.pop_scope();

                        ast::StatementKind::Loop {
                            body,
                            continuing,
                            break_if: None,
                        }
                    }
                    "break" => {
                        let (_, span) = lexer.next();
                        // Check if the next token is an `if`, this indicates
                        // that the user tried to type out a `break if` which
                        // is illegal in this position.
                        let (peeked_token, peeked_span) = lexer.peek();
                        if let Token::Word("if") = peeked_token {
                            let span = span.until(&peeked_span);
                            return Err(Error::InvalidBreakIf(span));
                        }
                        lexer.expect(Token::Separator(';'))?;
                        ast::StatementKind::Break
                    }
                    "continue" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Separator(';'))?;
                        ast::StatementKind::Continue
                    }
                    "discard" => {
                        let _ = lexer.next();
                        lexer.expect(Token::Separator(';'))?;
                        ast::StatementKind::Kill
                    }
                    // assignment or a function call
                    _ => {
                        self.function_call_or_assignment_statement(lexer, ctx, block)?;
                        lexer.expect(Token::Separator(';'))?;
                        self.pop_rule_span(lexer);
                        return Ok(());
                    }
                };

                let span = self.pop_rule_span(lexer);
                block.stmts.push(ast::Statement { kind, span });
            }
            _ => {
                self.assignment_statement(lexer, ctx, block)?;
                lexer.expect(Token::Separator(';'))?;
                self.pop_rule_span(lexer);
            }
        }
        Ok(())
    }

    fn r#loop<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<ast::StatementKind<'a>, Error<'a>> {
        let _ = lexer.next();
        let mut body = ast::Block::default();
        let mut continuing = ast::Block::default();
        let mut break_if = None;

        lexer.expect(Token::Paren('{'))?;

        ctx.local_table.push_scope();

        loop {
            if lexer.skip(Token::Word("continuing")) {
                // Branch for the `continuing` block, this must be
                // the last thing in the loop body

                // Expect a opening brace to start the continuing block
                lexer.expect(Token::Paren('{'))?;
                loop {
                    if lexer.skip(Token::Word("break")) {
                        // Branch for the `break if` statement, this statement
                        // has the form `break if <expr>;` and must be the last
                        // statement in a continuing block

                        // The break must be followed by an `if` to form
                        // the break if
                        lexer.expect(Token::Word("if"))?;

                        let condition = self.general_expression(lexer, ctx)?;
                        // Set the condition of the break if to the newly parsed
                        // expression
                        break_if = Some(condition);

                        // Expect a semicolon to close the statement
                        lexer.expect(Token::Separator(';'))?;
                        // Expect a closing brace to close the continuing block,
                        // since the break if must be the last statement
                        lexer.expect(Token::Paren('}'))?;
                        // Stop parsing the continuing block
                        break;
                    } else if lexer.skip(Token::Paren('}')) {
                        // If we encounter a closing brace it means we have reached
                        // the end of the continuing block and should stop processing
                        break;
                    } else {
                        // Otherwise try to parse a statement
                        self.statement(lexer, ctx, &mut continuing)?;
                    }
                }
                // Since the continuing block must be the last part of the loop body,
                // we expect to see a closing brace to end the loop body
                lexer.expect(Token::Paren('}'))?;
                break;
            }
            if lexer.skip(Token::Paren('}')) {
                // If we encounter a closing brace it means we have reached
                // the end of the loop body and should stop processing
                break;
            }
            // Otherwise try to parse a statement
            self.statement(lexer, ctx, &mut body)?;
        }

        ctx.local_table.pop_scope();

        Ok(ast::StatementKind::Loop {
            body,
            continuing,
            break_if,
        })
    }

    /// compound_statement
    fn block<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<(ast::Block<'a>, Span), Error<'a>> {
        self.push_rule_span(Rule::Block, lexer);

        ctx.local_table.push_scope();

        lexer.expect(Token::Paren('{'))?;
        let mut block = ast::Block::default();
        while !lexer.skip(Token::Paren('}')) {
            self.statement(lexer, ctx, &mut block)?;
        }

        ctx.local_table.pop_scope();

        let span = self.pop_rule_span(lexer);
        Ok((block, span))
    }

    fn varying_binding<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<Option<ast::Binding<'a>>, Error<'a>> {
        let mut bind_parser = BindingParser::default();
        self.push_rule_span(Rule::Attribute, lexer);

        while lexer.skip(Token::Attribute) {
            let (word, span) = lexer.next_ident_with_span()?;
            bind_parser.parse(self, lexer, word, span, ctx)?;
        }

        let span = self.pop_rule_span(lexer);
        bind_parser.finish(span)
    }

    fn function_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        out: &mut ast::TranslationUnit<'a>,
        dependencies: &mut FastIndexSet<ast::Dependency<'a>>,
    ) -> Result<ast::Function<'a>, Error<'a>> {
        self.push_rule_span(Rule::FunctionDecl, lexer);
        // read function name
        let fun_name = lexer.next_ident()?;

        let mut locals = Arena::new();

        let mut ctx = ExpressionContext {
            expressions: &mut out.expressions,
            local_table: &mut SymbolTable::default(),
            locals: &mut locals,
            types: &mut out.types,
            unresolved: dependencies,
        };

        // start a scope that contains arguments as well as the function body
        ctx.local_table.push_scope();

        // read parameter list
        let mut arguments = Vec::new();
        lexer.expect(Token::Paren('('))?;
        let mut ready = true;
        while !lexer.skip(Token::Paren(')')) {
            if !ready {
                return Err(Error::Unexpected(
                    lexer.next().1,
                    ExpectedToken::Token(Token::Separator(',')),
                ));
            }
            let binding = self.varying_binding(lexer, &mut ctx)?;

            let param_name = lexer.next_ident()?;

            lexer.expect(Token::Separator(':'))?;
            let param_type = self.type_decl(lexer, &mut ctx)?;

            let handle = ctx.declare_local(param_name)?;
            arguments.push(ast::FunctionArgument {
                name: param_name,
                ty: param_type,
                binding,
                handle,
            });
            ready = lexer.skip(Token::Separator(','));
        }
        // read return type
        let result = if lexer.skip(Token::Arrow) && !lexer.skip(Token::Word("void")) {
            let binding = self.varying_binding(lexer, &mut ctx)?;
            let ty = self.type_decl(lexer, &mut ctx)?;
            Some(ast::FunctionResult { ty, binding })
        } else {
            None
        };

        // do not use `self.block` here, since we must not push a new scope
        lexer.expect(Token::Paren('{'))?;
        let mut body = ast::Block::default();
        while !lexer.skip(Token::Paren('}')) {
            self.statement(lexer, &mut ctx, &mut body)?;
        }

        ctx.local_table.pop_scope();

        let fun = ast::Function {
            entry_point: None,
            name: fun_name,
            arguments,
            result,
            body,
            locals,
        };

        // done
        self.pop_rule_span(lexer);

        Ok(fun)
    }

    fn global_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        out: &mut ast::TranslationUnit<'a>,
    ) -> Result<(), Error<'a>> {
        // read attributes
        let mut binding = None;
        let mut stage = ParsedAttribute::default();
        let mut compute_span = Span::new(0, 0);
        let mut workgroup_size = ParsedAttribute::default();
        let mut early_depth_test = ParsedAttribute::default();
        let (mut bind_index, mut bind_group) =
            (ParsedAttribute::default(), ParsedAttribute::default());

        let mut dependencies = FastIndexSet::default();
        let mut ctx = ExpressionContext {
            expressions: &mut out.expressions,
            local_table: &mut SymbolTable::default(),
            locals: &mut Arena::new(),
            types: &mut out.types,
            unresolved: &mut dependencies,
        };

        self.push_rule_span(Rule::Attribute, lexer);
        while lexer.skip(Token::Attribute) {
            match lexer.next_ident_with_span()? {
                ("binding", name_span) => {
                    lexer.expect(Token::Paren('('))?;
                    bind_index.set(self.general_expression(lexer, &mut ctx)?, name_span)?;
                    lexer.expect(Token::Paren(')'))?;
                }
                ("group", name_span) => {
                    lexer.expect(Token::Paren('('))?;
                    bind_group.set(self.general_expression(lexer, &mut ctx)?, name_span)?;
                    lexer.expect(Token::Paren(')'))?;
                }
                ("vertex", name_span) => {
                    stage.set(crate::ShaderStage::Vertex, name_span)?;
                }
                ("fragment", name_span) => {
                    stage.set(crate::ShaderStage::Fragment, name_span)?;
                }
                ("compute", name_span) => {
                    stage.set(crate::ShaderStage::Compute, name_span)?;
                    compute_span = name_span;
                }
                ("workgroup_size", name_span) => {
                    lexer.expect(Token::Paren('('))?;
                    let mut new_workgroup_size = [None; 3];
                    for (i, size) in new_workgroup_size.iter_mut().enumerate() {
                        *size = Some(self.general_expression(lexer, &mut ctx)?);
                        match lexer.next() {
                            (Token::Paren(')'), _) => break,
                            (Token::Separator(','), _) if i != 2 => (),
                            other => {
                                return Err(Error::Unexpected(
                                    other.1,
                                    ExpectedToken::WorkgroupSizeSeparator,
                                ))
                            }
                        }
                    }
                    workgroup_size.set(new_workgroup_size, name_span)?;
                }
                ("early_depth_test", name_span) => {
                    let conservative = if lexer.skip(Token::Paren('(')) {
                        let (ident, ident_span) = lexer.next_ident_with_span()?;
                        let value = conv::map_conservative_depth(ident, ident_span)?;
                        lexer.expect(Token::Paren(')'))?;
                        Some(value)
                    } else {
                        None
                    };
                    early_depth_test.set(crate::EarlyDepthTest { conservative }, name_span)?;
                }
                (_, word_span) => return Err(Error::UnknownAttribute(word_span)),
            }
        }

        let attrib_span = self.pop_rule_span(lexer);
        match (bind_group.value, bind_index.value) {
            (Some(group), Some(index)) => {
                binding = Some(ast::ResourceBinding {
                    group,
                    binding: index,
                });
            }
            (Some(_), None) => return Err(Error::MissingAttribute("binding", attrib_span)),
            (None, Some(_)) => return Err(Error::MissingAttribute("group", attrib_span)),
            (None, None) => {}
        }

        // read item
        let start = lexer.start_byte_offset();
        let kind = match lexer.next() {
            (Token::Separator(';'), _) => None,
            (Token::Word("struct"), _) => {
                let name = lexer.next_ident()?;

                let members = self.struct_body(lexer, &mut ctx)?;
                Some(ast::GlobalDeclKind::Struct(ast::Struct { name, members }))
            }
            (Token::Word("alias"), _) => {
                let name = lexer.next_ident()?;

                lexer.expect(Token::Operation('='))?;
                let ty = self.type_decl(lexer, &mut ctx)?;
                lexer.expect(Token::Separator(';'))?;
                Some(ast::GlobalDeclKind::Type(ast::TypeAlias { name, ty }))
            }
            (Token::Word("const"), _) => {
                let name = lexer.next_ident()?;

                let ty = if lexer.skip(Token::Separator(':')) {
                    let ty = self.type_decl(lexer, &mut ctx)?;
                    Some(ty)
                } else {
                    None
                };

                lexer.expect(Token::Operation('='))?;
                let init = self.general_expression(lexer, &mut ctx)?;
                lexer.expect(Token::Separator(';'))?;

                Some(ast::GlobalDeclKind::Const(ast::Const { name, ty, init }))
            }
            (Token::Word("var"), _) => {
                let mut var = self.variable_decl(lexer, &mut ctx)?;
                var.binding = binding.take();
                Some(ast::GlobalDeclKind::Var(var))
            }
            (Token::Word("fn"), _) => {
                let function = self.function_decl(lexer, out, &mut dependencies)?;
                Some(ast::GlobalDeclKind::Fn(ast::Function {
                    entry_point: if let Some(stage) = stage.value {
                        if stage == ShaderStage::Compute && workgroup_size.value.is_none() {
                            return Err(Error::MissingWorkgroupSize(compute_span));
                        }
                        Some(ast::EntryPoint {
                            stage,
                            early_depth_test: early_depth_test.value,
                            workgroup_size: workgroup_size.value,
                        })
                    } else {
                        None
                    },
                    ..function
                }))
            }
            (Token::End, _) => return Ok(()),
            other => return Err(Error::Unexpected(other.1, ExpectedToken::GlobalItem)),
        };

        if let Some(kind) = kind {
            out.decls.append(
                ast::GlobalDecl { kind, dependencies },
                lexer.span_from(start),
            );
        }

        if !self.rules.is_empty() {
            log::error!("Reached the end of global decl, but rule stack is not empty");
            log::error!("Rules: {:?}", self.rules);
            return Err(Error::Internal("rule stack is not empty"));
        };

        match binding {
            None => Ok(()),
            Some(_) => Err(Error::Internal("we had the attribute but no var?")),
        }
    }

    pub fn parse<'a>(&mut self, source: &'a str) -> Result<ast::TranslationUnit<'a>, Error<'a>> {
        self.reset();

        let mut lexer = Lexer::new(source);
        let mut tu = ast::TranslationUnit::default();
        loop {
            match self.global_decl(&mut lexer, &mut tu) {
                Err(error) => return Err(error),
                Ok(()) => {
                    if lexer.peek().0 == Token::End {
                        break;
                    }
                }
            }
        }

        Ok(tu)
    }
}
