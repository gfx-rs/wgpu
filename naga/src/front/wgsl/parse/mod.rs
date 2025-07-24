use alloc::{boxed::Box, vec::Vec};
use directive::enable_extension::ImplementedEnableExtension;

use crate::diagnostic_filter::{
    self, DiagnosticFilter, DiagnosticFilterMap, DiagnosticFilterNode, FilterableTriggeringRule,
    ShouldConflictOnFullDuplicate, StandardFilterableTriggeringRule,
};
use crate::front::wgsl::error::{DiagnosticAttributeNotSupportedPosition, Error, ExpectedToken};
use crate::front::wgsl::parse::directive::enable_extension::{EnableExtension, EnableExtensions};
use crate::front::wgsl::parse::directive::language_extension::LanguageExtension;
use crate::front::wgsl::parse::directive::DirectiveKind;
use crate::front::wgsl::parse::lexer::{Lexer, Token};
use crate::front::wgsl::parse::number::Number;
use crate::front::wgsl::{Result, Scalar};
use crate::front::SymbolTable;
use crate::{Arena, FastHashSet, FastIndexSet, Handle, ShaderStage, Span};

pub mod ast;
pub mod conv;
pub mod directive;
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
    /// The handles refer to the [`locals`] arena; see that field's
    /// documentation for details.
    ///
    /// [`locals`]: ExpressionContext::locals
    local_table: &'temp mut SymbolTable<&'input str, Handle<ast::Local>>,

    /// Local variable and function argument arena for the function we're building.
    ///
    /// Note that the [`ast::Local`] here is actually a zero-sized type. This
    /// `Arena`'s only role is to assign a unique `Handle` to each local
    /// identifier, and track its definition's span for use in diagnostics. All
    /// the detailed information about locals - names, types, etc. - is kept in
    /// the [`LocalDecl`] statements we parsed from their declarations. For
    /// arguments, that information is kept in [`arguments`].
    ///
    /// In the AST, when an [`Ident`] expression refers to a local variable or
    /// argument, its [`IdentExpr`] holds the referent's `Handle<Local>` in this
    /// arena.
    ///
    /// During lowering, [`LocalDecl`] statements add entries to a per-function
    /// table that maps `Handle<Local>` values to their Naga representations,
    /// accessed via [`StatementContext::local_table`] and
    /// [`LocalExpressionContext::local_table`]. This table is then consulted when
    /// lowering subsequent [`Ident`] expressions.
    ///
    /// [`LocalDecl`]: ast::StatementKind::LocalDecl
    /// [`arguments`]: ast::Function::arguments
    /// [`Ident`]: ast::Expression::Ident
    /// [`IdentExpr`]: ast::IdentExpr
    /// [`StatementContext::local_table`]: super::lower::StatementContext::local_table
    /// [`LocalExpressionContext::local_table`]: super::lower::LocalExpressionContext::local_table
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
        mut parser: impl FnMut(&mut Lexer<'a>, &mut Self) -> Result<'a, Handle<ast::Expression<'a>>>,
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
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

    fn declare_local(&mut self, name: ast::Ident<'a>) -> Result<'a, Handle<ast::Local>> {
        let handle = self.locals.append(ast::Local, name.span);
        if let Some(old) = self.local_table.add(name.name, handle) {
            Err(Box::new(Error::Redefinition {
                previous: self.locals.get_span(old),
                current: name.span,
            }))
        } else {
            Ok(handle)
        }
    }

    fn new_scalar(&mut self, scalar: Scalar) -> Handle<ast::Type<'a>> {
        self.types
            .append(ast::Type::Scalar(scalar), Span::UNDEFINED)
    }
}

/// Which grammar rule we are in the midst of parsing.
///
/// This is used for error checking. `Parser` maintains a stack of
/// these and (occasionally) checks that it is being pushed and popped
/// as expected.
#[derive(Copy, Clone, Debug, PartialEq)]
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
    Directive,
    GenericExpr,
    EnclosedExpr,
    LhsExpr,
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
    fn set(&mut self, value: T, name_span: Span) -> Result<'static, ()> {
        if self.value.is_some() {
            return Err(Box::new(Error::RepeatedAttribute(name_span)));
        }
        self.value = Some(value);
        Ok(())
    }
}

#[derive(Default)]
struct BindingParser<'a> {
    location: ParsedAttribute<Handle<ast::Expression<'a>>>,
    built_in: ParsedAttribute<crate::BuiltIn>,
    interpolation: ParsedAttribute<crate::Interpolation>,
    sampling: ParsedAttribute<crate::Sampling>,
    invariant: ParsedAttribute<bool>,
    blend_src: ParsedAttribute<Handle<ast::Expression<'a>>>,
}

impl<'a> BindingParser<'a> {
    fn parse(
        &mut self,
        parser: &mut Parser,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, ()> {
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
                self.built_in.set(
                    conv::map_built_in(&lexer.enable_extensions, raw, span)?,
                    name_span,
                )?;
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

            "invariant" => {
                self.invariant.set(true, name_span)?;
            }
            "blend_src" => {
                if !lexer
                    .enable_extensions
                    .contains(ImplementedEnableExtension::DualSourceBlending)
                {
                    return Err(Box::new(Error::EnableExtensionNotEnabled {
                        span: name_span,
                        kind: ImplementedEnableExtension::DualSourceBlending.into(),
                    }));
                }

                lexer.expect(Token::Paren('('))?;
                self.blend_src
                    .set(parser.general_expression(lexer, ctx)?, name_span)?;
                lexer.expect(Token::Paren(')'))?;
            }
            _ => return Err(Box::new(Error::UnknownAttribute(name_span))),
        }
        Ok(())
    }

    fn finish(self, span: Span) -> Result<'a, Option<ast::Binding<'a>>> {
        match (
            self.location.value,
            self.built_in.value,
            self.interpolation.value,
            self.sampling.value,
            self.invariant.value.unwrap_or_default(),
            self.blend_src.value,
        ) {
            (None, None, None, None, false, None) => Ok(None),
            (Some(location), None, interpolation, sampling, false, blend_src) => {
                // Before handing over the completed `Module`, we call
                // `apply_default_interpolation` to ensure that the interpolation and
                // sampling have been explicitly specified on all vertex shader output and fragment
                // shader input user bindings, so leaving them potentially `None` here is fine.
                Ok(Some(ast::Binding::Location {
                    location,
                    interpolation,
                    sampling,
                    blend_src,
                }))
            }
            (None, Some(crate::BuiltIn::Position { .. }), None, None, invariant, None) => {
                Ok(Some(ast::Binding::BuiltIn(crate::BuiltIn::Position {
                    invariant,
                })))
            }
            (None, Some(built_in), None, None, false, None) => {
                Ok(Some(ast::Binding::BuiltIn(built_in)))
            }
            (_, _, _, _, _, _) => Err(Box::new(Error::InconsistentBinding(span))),
        }
    }
}

/// Configuration for the whole parser run.
pub struct Options {
    /// Controls whether the parser should parse doc comments.
    pub parse_doc_comments: bool,
}

impl Options {
    /// Creates a new [`Options`] without doc comments parsing.
    pub const fn new() -> Self {
        Options {
            parse_doc_comments: false,
        }
    }
}

pub struct Parser {
    rules: Vec<(Rule, usize)>,
    recursion_depth: u32,
}

impl Parser {
    pub const fn new() -> Self {
        Parser {
            rules: Vec::new(),
            recursion_depth: 0,
        }
    }

    fn reset(&mut self) {
        self.rules.clear();
        self.recursion_depth = 0;
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

    fn race_rules(&self, rule0: Rule, rule1: Rule) -> Option<Rule> {
        Some(
            self.rules
                .iter()
                .rev()
                .find(|&x| x.0 == rule0 || x.0 == rule1)?
                .0,
        )
    }

    fn track_recursion<'a, F, R>(&mut self, f: F) -> Result<'a, R>
    where
        F: FnOnce(&mut Self) -> Result<'a, R>,
    {
        self.recursion_depth += 1;
        if self.recursion_depth >= 256 {
            return Err(Box::new(Error::Internal("Parser recursion limit exceeded")));
        }
        let ret = f(self);
        self.recursion_depth -= 1;
        ret
    }

    fn switch_value<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, ast::SwitchValue<'a>> {
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
    ) -> Result<'a, Option<ast::ConstructorType<'a>>> {
        if let Some(scalar) = conv::get_scalar_type(&lexer.enable_extensions, span, word)? {
            return Ok(Some(ast::ConstructorType::Scalar(scalar)));
        }

        let partial = match word {
            "vec2" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Bi,
            },
            "vec2i" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Bi,
                    ty: ctx.new_scalar(Scalar::I32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec2u" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Bi,
                    ty: ctx.new_scalar(Scalar::U32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec2f" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Bi,
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec2h" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Bi,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec3" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Tri,
            },
            "vec3i" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Tri,
                    ty: ctx.new_scalar(Scalar::I32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec3u" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Tri,
                    ty: ctx.new_scalar(Scalar::U32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec3f" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Tri,
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec3h" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Tri,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec4" => ast::ConstructorType::PartialVector {
                size: crate::VectorSize::Quad,
            },
            "vec4i" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Quad,
                    ty: ctx.new_scalar(Scalar::I32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec4u" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Quad,
                    ty: ctx.new_scalar(Scalar::U32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec4f" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Quad,
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "vec4h" => {
                return Ok(Some(ast::ConstructorType::Vector {
                    size: crate::VectorSize::Quad,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat2x2h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Bi,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat2x3h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Tri,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat2x4h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Bi,
                    rows: crate::VectorSize::Quad,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat3x2h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Bi,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat3x3h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Tri,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat3x4h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Tri,
                    rows: crate::VectorSize::Quad,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat4x2h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Bi,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat4x3h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Tri,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
                    ty: ctx.new_scalar(Scalar::F32),
                    ty_span: Span::UNDEFINED,
                }))
            }
            "mat4x4h" => {
                return Ok(Some(ast::ConstructorType::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Quad,
                    ty: ctx.new_scalar(Scalar::F16),
                    ty_span: Span::UNDEFINED,
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
            | "texture_external"
            | "texture_storage_1d"
            | "texture_storage_1d_array"
            | "texture_storage_2d"
            | "texture_storage_2d_array"
            | "texture_storage_3d" => return Err(Box::new(Error::TypeNotConstructible(span))),
            _ => return Ok(None),
        };

        // parse component type if present
        match (lexer.peek().0, partial) {
            (Token::Paren('<'), ast::ConstructorType::PartialVector { size }) => {
                let (ty, ty_span) = self.singular_generic(lexer, ctx)?;
                Ok(Some(ast::ConstructorType::Vector { size, ty, ty_span }))
            }
            (Token::Paren('<'), ast::ConstructorType::PartialMatrix { columns, rows }) => {
                let (ty, ty_span) = self.singular_generic(lexer, ctx)?;
                Ok(Some(ast::ConstructorType::Matrix {
                    columns,
                    rows,
                    ty,
                    ty_span,
                }))
            }
            (Token::Paren('<'), ast::ConstructorType::PartialArray) => {
                lexer.expect_generic_paren('<')?;
                let base = self.type_decl(lexer, ctx)?;
                let size = if lexer.end_of_generic_arguments() {
                    let expr = self.const_generic_expression(lexer, ctx)?;
                    lexer.skip(Token::Separator(','));
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
    ) -> Result<'a, Vec<Handle<ast::Expression<'a>>>> {
        self.push_rule_span(Rule::EnclosedExpr, lexer);
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

        self.pop_rule_span(lexer);
        Ok(arguments)
    }

    fn enclosed_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
        self.push_rule_span(Rule::EnclosedExpr, lexer);
        let expr = self.general_expression(lexer, ctx)?;
        self.pop_rule_span(lexer);
        Ok(expr)
    }

    /// Expects [`Rule::PrimaryExpr`] or [`Rule::SingularExpr`] on top; does not pop it.
    /// Expects `name` to be consumed (not in lexer).
    fn function_call<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        name: &'a str,
        name_span: Span,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
        assert!(self.rules.last().is_some());

        let expr = match name {
            // bitcast looks like a function call, but it's an operator and must be handled differently.
            "bitcast" => {
                let (to, span) = self.singular_generic(lexer, ctx)?;

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
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
        self.push_rule_span(Rule::PrimaryExpr, lexer);
        const fn literal_ray_flag<'b>(flag: crate::RayFlag) -> ast::Expression<'b> {
            ast::Expression::Literal(ast::Literal::Number(Number::U32(flag.bits())))
        }
        const fn literal_ray_intersection<'b>(
            intersection: crate::RayQueryIntersection,
        ) -> ast::Expression<'b> {
            ast::Expression::Literal(ast::Literal::Number(Number::U32(intersection as u32)))
        }

        let expr = match lexer.peek() {
            (Token::Paren('('), _) => {
                let _ = lexer.next();
                let expr = self.enclosed_expression(lexer, ctx)?;
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

                if let Some(enable_extension) = num.requires_enable_extension() {
                    if !lexer.enable_extensions.contains(enable_extension) {
                        return Err(Box::new(Error::EnableExtensionNotEnabled {
                            kind: enable_extension.into(),
                            span,
                        }));
                    }
                }

                ast::Expression::Literal(ast::Literal::Number(num))
            }
            (Token::Word("RAY_FLAG_NONE"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::empty())
            }
            (Token::Word("RAY_FLAG_FORCE_OPAQUE"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::FORCE_OPAQUE)
            }
            (Token::Word("RAY_FLAG_FORCE_NO_OPAQUE"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::FORCE_NO_OPAQUE)
            }
            (Token::Word("RAY_FLAG_TERMINATE_ON_FIRST_HIT"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::TERMINATE_ON_FIRST_HIT)
            }
            (Token::Word("RAY_FLAG_SKIP_CLOSEST_HIT_SHADER"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::SKIP_CLOSEST_HIT_SHADER)
            }
            (Token::Word("RAY_FLAG_CULL_BACK_FACING"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::CULL_BACK_FACING)
            }
            (Token::Word("RAY_FLAG_CULL_FRONT_FACING"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::CULL_FRONT_FACING)
            }
            (Token::Word("RAY_FLAG_CULL_OPAQUE"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::CULL_OPAQUE)
            }
            (Token::Word("RAY_FLAG_CULL_NO_OPAQUE"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::CULL_NO_OPAQUE)
            }
            (Token::Word("RAY_FLAG_SKIP_TRIANGLES"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::SKIP_TRIANGLES)
            }
            (Token::Word("RAY_FLAG_SKIP_AABBS"), _) => {
                let _ = lexer.next();
                literal_ray_flag(crate::RayFlag::SKIP_AABBS)
            }
            (Token::Word("RAY_QUERY_INTERSECTION_NONE"), _) => {
                let _ = lexer.next();
                literal_ray_intersection(crate::RayQueryIntersection::None)
            }
            (Token::Word("RAY_QUERY_INTERSECTION_TRIANGLE"), _) => {
                let _ = lexer.next();
                literal_ray_intersection(crate::RayQueryIntersection::Triangle)
            }
            (Token::Word("RAY_QUERY_INTERSECTION_GENERATED"), _) => {
                let _ = lexer.next();
                literal_ray_intersection(crate::RayQueryIntersection::Generated)
            }
            (Token::Word("RAY_QUERY_INTERSECTION_AABB"), _) => {
                let _ = lexer.next();
                literal_ray_intersection(crate::RayQueryIntersection::Aabb)
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
            other => {
                return Err(Box::new(Error::Unexpected(
                    other.1,
                    ExpectedToken::PrimaryExpression,
                )))
            }
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
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
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
                    let index = self.enclosed_expression(lexer, ctx)?;
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

    fn const_generic_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
        self.push_rule_span(Rule::GenericExpr, lexer);
        let expr = self.general_expression(lexer, ctx)?;
        self.pop_rule_span(lexer);
        Ok(expr)
    }

    /// Parse a `unary_expression`.
    fn unary_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
        self.track_recursion(|this| {
            this.push_rule_span(Rule::UnaryExpr, lexer);
            //TODO: refactor this to avoid backing up
            let expr = match lexer.peek().0 {
                Token::Operation('-') => {
                    let _ = lexer.next();
                    let expr = this.unary_expression(lexer, ctx)?;
                    let expr = ast::Expression::Unary {
                        op: crate::UnaryOperator::Negate,
                        expr,
                    };
                    let span = this.peek_rule_span(lexer);
                    ctx.expressions.append(expr, span)
                }
                Token::Operation('!') => {
                    let _ = lexer.next();
                    let expr = this.unary_expression(lexer, ctx)?;
                    let expr = ast::Expression::Unary {
                        op: crate::UnaryOperator::LogicalNot,
                        expr,
                    };
                    let span = this.peek_rule_span(lexer);
                    ctx.expressions.append(expr, span)
                }
                Token::Operation('~') => {
                    let _ = lexer.next();
                    let expr = this.unary_expression(lexer, ctx)?;
                    let expr = ast::Expression::Unary {
                        op: crate::UnaryOperator::BitwiseNot,
                        expr,
                    };
                    let span = this.peek_rule_span(lexer);
                    ctx.expressions.append(expr, span)
                }
                Token::Operation('*') => {
                    let _ = lexer.next();
                    let expr = this.unary_expression(lexer, ctx)?;
                    let expr = ast::Expression::Deref(expr);
                    let span = this.peek_rule_span(lexer);
                    ctx.expressions.append(expr, span)
                }
                Token::Operation('&') => {
                    let _ = lexer.next();
                    let expr = this.unary_expression(lexer, ctx)?;
                    let expr = ast::Expression::AddrOf(expr);
                    let span = this.peek_rule_span(lexer);
                    ctx.expressions.append(expr, span)
                }
                _ => this.singular_expression(lexer, ctx)?,
            };

            this.pop_rule_span(lexer);
            Ok(expr)
        })
    }

    /// Parse a `lhs_expression`.
    ///
    /// LHS expressions only support the `&` and `*` operators and
    /// the `[]` and `.` postfix selectors.
    fn lhs_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
        self.track_recursion(|this| {
            this.push_rule_span(Rule::LhsExpr, lexer);
            let start = lexer.start_byte_offset();
            let expr = match lexer.peek() {
                (Token::Operation('*'), _) => {
                    let _ = lexer.next();
                    let expr = this.lhs_expression(lexer, ctx)?;
                    let expr = ast::Expression::Deref(expr);
                    let span = this.peek_rule_span(lexer);
                    ctx.expressions.append(expr, span)
                }
                (Token::Operation('&'), _) => {
                    let _ = lexer.next();
                    let expr = this.lhs_expression(lexer, ctx)?;
                    let expr = ast::Expression::AddrOf(expr);
                    let span = this.peek_rule_span(lexer);
                    ctx.expressions.append(expr, span)
                }
                (Token::Operation('('), _) => {
                    let _ = lexer.next();
                    let primary_expr = this.lhs_expression(lexer, ctx)?;
                    lexer.expect(Token::Paren(')'))?;
                    this.postfix(start, lexer, ctx, primary_expr)?
                }
                (Token::Word(word), span) => {
                    let _ = lexer.next();
                    let ident = this.ident_expr(word, span, ctx);
                    let primary_expr = ctx.expressions.append(ast::Expression::Ident(ident), span);
                    this.postfix(start, lexer, ctx, primary_expr)?
                }
                _ => this.singular_expression(lexer, ctx)?,
            };

            this.pop_rule_span(lexer);
            Ok(expr)
        })
    }

    /// Parse a `singular_expression`.
    fn singular_expression<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
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
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
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
                let enclosing = self.race_rules(Rule::GenericExpr, Rule::EnclosedExpr);
                context.parse_binary_op(
                    lexer,
                    match enclosing {
                        Some(Rule::GenericExpr) => |token| match token {
                            Token::LogicalOperation('<') => Some(crate::BinaryOperator::LessEqual),
                            _ => None,
                        },
                        _ => |token| match token {
                            Token::Paren('<') => Some(crate::BinaryOperator::Less),
                            Token::Paren('>') => Some(crate::BinaryOperator::Greater),
                            Token::LogicalOperation('<') => Some(crate::BinaryOperator::LessEqual),
                            Token::LogicalOperation('>') => {
                                Some(crate::BinaryOperator::GreaterEqual)
                            }
                            _ => None,
                        },
                    },
                    // shift_expression
                    |lexer, context| {
                        context.parse_binary_op(
                            lexer,
                            match enclosing {
                                Some(Rule::GenericExpr) => |token| match token {
                                    Token::ShiftOperation('<') => {
                                        Some(crate::BinaryOperator::ShiftLeft)
                                    }
                                    _ => None,
                                },
                                _ => |token| match token {
                                    Token::ShiftOperation('<') => {
                                        Some(crate::BinaryOperator::ShiftLeft)
                                    }
                                    Token::ShiftOperation('>') => {
                                        Some(crate::BinaryOperator::ShiftRight)
                                    }
                                    _ => None,
                                },
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
    ) -> Result<'a, Handle<ast::Expression<'a>>> {
        self.general_expression_with_span(lexer, ctx)
            .map(|(expr, _)| expr)
    }

    fn general_expression_with_span<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        context: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, (Handle<ast::Expression<'a>>, Span)> {
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
    ) -> Result<'a, ast::GlobalVariable<'a>> {
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

        let ty = if lexer.skip(Token::Separator(':')) {
            Some(self.type_decl(lexer, ctx)?)
        } else {
            None
        };

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
            doc_comments: Vec::new(),
        })
    }

    fn struct_body<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Vec<ast::StructMember<'a>>> {
        let mut members = Vec::new();
        let mut member_names = FastHashSet::default();

        lexer.expect(Token::Paren('{'))?;
        let mut ready = true;
        while !lexer.skip(Token::Paren('}')) {
            if !ready {
                return Err(Box::new(Error::Unexpected(
                    lexer.next().1,
                    ExpectedToken::Token(Token::Separator(',')),
                )));
            }

            let doc_comments = lexer.accumulate_doc_comments();

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
                doc_comments,
            });

            if !member_names.insert(name.name) {
                return Err(Box::new(Error::Redefinition {
                    previous: members
                        .iter()
                        .find(|x| x.name.name == name.name)
                        .map(|x| x.name.span)
                        .unwrap(),
                    current: name.span,
                }));
            }
        }

        Ok(members)
    }

    /// Parses `<T>`, returning T and span of T
    fn singular_generic<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, (Handle<ast::Type<'a>>, Span)> {
        lexer.expect_generic_paren('<')?;
        let start = lexer.start_byte_offset();
        let ty = self.type_decl(lexer, ctx)?;
        let span = lexer.span_from(start);
        lexer.skip(Token::Separator(','));
        lexer.expect_generic_paren('>')?;
        Ok((ty, span))
    }

    fn matrix_with_type<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
        columns: crate::VectorSize,
        rows: crate::VectorSize,
    ) -> Result<'a, ast::Type<'a>> {
        let (ty, ty_span) = self.singular_generic(lexer, ctx)?;
        Ok(ast::Type::Matrix {
            columns,
            rows,
            ty,
            ty_span,
        })
    }

    fn type_decl_impl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        word: &'a str,
        span: Span,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Option<ast::Type<'a>>> {
        if let Some(scalar) = conv::get_scalar_type(&lexer.enable_extensions, span, word)? {
            return Ok(Some(ast::Type::Scalar(scalar)));
        }

        Ok(Some(match word {
            "vec2" => {
                let (ty, ty_span) = self.singular_generic(lexer, ctx)?;
                ast::Type::Vector {
                    size: crate::VectorSize::Bi,
                    ty,
                    ty_span,
                }
            }
            "vec2i" => ast::Type::Vector {
                size: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::I32),
                ty_span: Span::UNDEFINED,
            },
            "vec2u" => ast::Type::Vector {
                size: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::U32),
                ty_span: Span::UNDEFINED,
            },
            "vec2f" => ast::Type::Vector {
                size: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "vec2h" => ast::Type::Vector {
                size: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "vec3" => {
                let (ty, ty_span) = self.singular_generic(lexer, ctx)?;
                ast::Type::Vector {
                    size: crate::VectorSize::Tri,
                    ty,
                    ty_span,
                }
            }
            "vec3i" => ast::Type::Vector {
                size: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::I32),
                ty_span: Span::UNDEFINED,
            },
            "vec3u" => ast::Type::Vector {
                size: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::U32),
                ty_span: Span::UNDEFINED,
            },
            "vec3f" => ast::Type::Vector {
                size: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "vec3h" => ast::Type::Vector {
                size: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "vec4" => {
                let (ty, ty_span) = self.singular_generic(lexer, ctx)?;
                ast::Type::Vector {
                    size: crate::VectorSize::Quad,
                    ty,
                    ty_span,
                }
            }
            "vec4i" => ast::Type::Vector {
                size: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::I32),
                ty_span: Span::UNDEFINED,
            },
            "vec4u" => ast::Type::Vector {
                size: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::U32),
                ty_span: Span::UNDEFINED,
            },
            "vec4f" => ast::Type::Vector {
                size: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "vec4h" => ast::Type::Vector {
                size: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat2x2" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Bi, crate::VectorSize::Bi)?
            }
            "mat2x2f" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat2x2h" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat2x3" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Bi, crate::VectorSize::Tri)?
            }
            "mat2x3f" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat2x3h" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat2x4" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Bi, crate::VectorSize::Quad)?
            }
            "mat2x4f" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat2x4h" => ast::Type::Matrix {
                columns: crate::VectorSize::Bi,
                rows: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat3x2" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Tri, crate::VectorSize::Bi)?
            }
            "mat3x2f" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat3x2h" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat3x3" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Tri, crate::VectorSize::Tri)?
            }
            "mat3x3f" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat3x3h" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat3x4" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Tri, crate::VectorSize::Quad)?
            }
            "mat3x4f" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat3x4h" => ast::Type::Matrix {
                columns: crate::VectorSize::Tri,
                rows: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat4x2" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Quad, crate::VectorSize::Bi)?
            }
            "mat4x2f" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat4x2h" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Bi,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat4x3" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Quad, crate::VectorSize::Tri)?
            }
            "mat4x3f" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat4x3h" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Tri,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "mat4x4" => {
                self.matrix_with_type(lexer, ctx, crate::VectorSize::Quad, crate::VectorSize::Quad)?
            }
            "mat4x4f" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F32),
                ty_span: Span::UNDEFINED,
            },
            "mat4x4h" => ast::Type::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Quad,
                ty: ctx.new_scalar(Scalar::F16),
                ty_span: Span::UNDEFINED,
            },
            "atomic" => {
                let scalar = lexer.next_scalar_generic()?;
                ast::Type::Atomic(scalar)
            }
            "ptr" => {
                lexer.expect_generic_paren('<')?;
                let (ident, span) = lexer.next_ident_with_span()?;
                let mut space = conv::map_address_space(ident, span)?;
                lexer.expect(Token::Separator(','))?;
                let base = self.type_decl(lexer, ctx)?;
                if let crate::AddressSpace::Storage { ref mut access } = space {
                    *access = if lexer.end_of_generic_arguments() {
                        let result = lexer.next_storage_access()?;
                        lexer.skip(Token::Separator(','));
                        result
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
                let size = if lexer.end_of_generic_arguments() {
                    let size = self.const_generic_expression(lexer, ctx)?;
                    lexer.skip(Token::Separator(','));
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
                let size = if lexer.end_of_generic_arguments() {
                    let size = self.unary_expression(lexer, ctx)?;
                    lexer.skip(Token::Separator(','));
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
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: false,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: false,
                    },
                }
            }
            "texture_1d_array" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D1,
                    arrayed: true,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: false,
                    },
                }
            }
            "texture_2d" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: false,
                    },
                }
            }
            "texture_2d_array" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: false,
                    },
                }
            }
            "texture_3d" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D3,
                    arrayed: false,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: false,
                    },
                }
            }
            "texture_cube" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: false,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: false,
                    },
                }
            }
            "texture_cube_array" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::Cube,
                    arrayed: true,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: false,
                    },
                }
            }
            "texture_multisampled_2d" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: false,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: true,
                    },
                }
            }
            "texture_multisampled_2d_array" => {
                let (scalar, span) = lexer.next_scalar_generic_with_span()?;
                Self::check_texture_sample_type(scalar, span)?;
                ast::Type::Image {
                    dim: crate::ImageDimension::D2,
                    arrayed: true,
                    class: crate::ImageClass::Sampled {
                        kind: scalar.kind,
                        multi: true,
                    },
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
            "texture_external" => ast::Type::Image {
                dim: crate::ImageDimension::D2,
                arrayed: false,
                class: crate::ImageClass::External,
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
            "acceleration_structure" => {
                let vertex_return = lexer.next_acceleration_structure_flags()?;
                ast::Type::AccelerationStructure { vertex_return }
            }
            "ray_query" => {
                let vertex_return = lexer.next_acceleration_structure_flags()?;
                ast::Type::RayQuery { vertex_return }
            }
            "RayDesc" => ast::Type::RayDesc,
            "RayIntersection" => ast::Type::RayIntersection,
            _ => return Ok(None),
        }))
    }

    fn check_texture_sample_type(scalar: Scalar, span: Span) -> Result<'static, ()> {
        use crate::ScalarKind::*;
        // Validate according to https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
        match scalar {
            Scalar {
                kind: Float | Sint | Uint,
                width: 4,
            } => Ok(()),
            Scalar {
                kind: Uint,
                width: 8,
            } => Ok(()),
            _ => Err(Box::new(Error::BadTextureSampleType { span, scalar })),
        }
    }

    /// Parse type declaration of a given name.
    fn type_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Handle<ast::Type<'a>>> {
        self.track_recursion(|this| {
            this.push_rule_span(Rule::TypeDecl, lexer);

            let (name, span) = lexer.next_ident_with_span()?;

            let ty = match this.type_decl_impl(lexer, name, span, ctx)? {
                Some(ty) => ty,
                None => {
                    ctx.unresolved.insert(ast::Dependency {
                        ident: name,
                        usage: span,
                    });
                    ast::Type::User(ast::Ident { name, span })
                }
            };

            this.pop_rule_span(lexer);

            let handle = ctx.types.append(ty, Span::UNDEFINED);
            Ok(handle)
        })
    }

    fn assignment_op_and_rhs<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
        block: &mut ast::Block<'a>,
        target: Handle<ast::Expression<'a>>,
        span_start: usize,
    ) -> Result<'a, ()> {
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
            _ => return Err(Box::new(Error::Unexpected(op.1, ExpectedToken::Assignment))),
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
    ) -> Result<'a, ()> {
        let span_start = lexer.start_byte_offset();
        let target = self.lhs_expression(lexer, ctx)?;
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
    ) -> Result<'a, ()> {
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
    ) -> Result<'a, ()> {
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
        brace_nesting_level: u8,
    ) -> Result<'a, ()> {
        self.track_recursion(|this| {
            this.push_rule_span(Rule::Statement, lexer);
            match lexer.peek() {
                (Token::Separator(';'), _) => {
                    let _ = lexer.next();
                    this.pop_rule_span(lexer);
                }
                (Token::Paren('{') | Token::Attribute, _) => {
                    let (inner, span) = this.block(lexer, ctx, brace_nesting_level)?;
                    block.stmts.push(ast::Statement {
                        kind: ast::StatementKind::Block(inner),
                        span,
                    });
                    this.pop_rule_span(lexer);
                }
                (Token::Word(word), _) => {
                    let kind = match word {
                        "_" => {
                            let _ = lexer.next();
                            lexer.expect(Token::Operation('='))?;
                            let expr = this.general_expression(lexer, ctx)?;
                            lexer.expect(Token::Separator(';'))?;

                            ast::StatementKind::Phony(expr)
                        }
                        "let" => {
                            let _ = lexer.next();
                            let name = lexer.next_ident()?;

                            let given_ty = if lexer.skip(Token::Separator(':')) {
                                let ty = this.type_decl(lexer, ctx)?;
                                Some(ty)
                            } else {
                                None
                            };
                            lexer.expect(Token::Operation('='))?;
                            let expr_id = this.general_expression(lexer, ctx)?;
                            lexer.expect(Token::Separator(';'))?;

                            let handle = ctx.declare_local(name)?;
                            ast::StatementKind::LocalDecl(ast::LocalDecl::Let(ast::Let {
                                name,
                                ty: given_ty,
                                init: expr_id,
                                handle,
                            }))
                        }
                        "const" => {
                            let _ = lexer.next();
                            let name = lexer.next_ident()?;

                            let given_ty = if lexer.skip(Token::Separator(':')) {
                                let ty = this.type_decl(lexer, ctx)?;
                                Some(ty)
                            } else {
                                None
                            };
                            lexer.expect(Token::Operation('='))?;
                            let expr_id = this.general_expression(lexer, ctx)?;
                            lexer.expect(Token::Separator(';'))?;

                            let handle = ctx.declare_local(name)?;
                            ast::StatementKind::LocalDecl(ast::LocalDecl::Const(ast::LocalConst {
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
                                let ty = this.type_decl(lexer, ctx)?;
                                Some(ty)
                            } else {
                                None
                            };

                            let init = if lexer.skip(Token::Operation('=')) {
                                let init = this.general_expression(lexer, ctx)?;
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
                                let handle = this.general_expression(lexer, ctx)?;
                                Some(handle)
                            } else {
                                None
                            };
                            lexer.expect(Token::Separator(';'))?;
                            ast::StatementKind::Return { value }
                        }
                        "if" => {
                            let _ = lexer.next();
                            let condition = this.general_expression(lexer, ctx)?;

                            let accept = this.block(lexer, ctx, brace_nesting_level)?.0;

                            let mut elsif_stack = Vec::new();
                            let mut elseif_span_start = lexer.start_byte_offset();
                            let mut reject = loop {
                                if !lexer.skip(Token::Word("else")) {
                                    break ast::Block::default();
                                }

                                if !lexer.skip(Token::Word("if")) {
                                    // ... else { ... }
                                    break this.block(lexer, ctx, brace_nesting_level)?.0;
                                }

                                // ... else if (...) { ... }
                                let other_condition = this.general_expression(lexer, ctx)?;
                                let other_block = this.block(lexer, ctx, brace_nesting_level)?;
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
                            let selector = this.general_expression(lexer, ctx)?;
                            let brace_span = lexer.expect_span(Token::Paren('{'))?;
                            let brace_nesting_level =
                                Self::increase_brace_nesting(brace_nesting_level, brace_span)?;
                            let mut cases = Vec::new();

                            loop {
                                // cases + default
                                match lexer.next() {
                                    (Token::Word("case"), _) => {
                                        // parse a list of values
                                        let value = loop {
                                            let value = this.switch_value(lexer, ctx)?;
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

                                        let body = this.block(lexer, ctx, brace_nesting_level)?.0;

                                        cases.push(ast::SwitchCase {
                                            value,
                                            body,
                                            fall_through: false,
                                        });
                                    }
                                    (Token::Word("default"), _) => {
                                        lexer.skip(Token::Separator(':'));
                                        let body = this.block(lexer, ctx, brace_nesting_level)?.0;
                                        cases.push(ast::SwitchCase {
                                            value: ast::SwitchValue::Default,
                                            body,
                                            fall_through: false,
                                        });
                                    }
                                    (Token::Paren('}'), _) => break,
                                    (_, span) => {
                                        return Err(Box::new(Error::Unexpected(
                                            span,
                                            ExpectedToken::SwitchItem,
                                        )))
                                    }
                                }
                            }

                            ast::StatementKind::Switch { selector, cases }
                        }
                        "loop" => this.r#loop(lexer, ctx, brace_nesting_level)?,
                        "while" => {
                            let _ = lexer.next();
                            let mut body = ast::Block::default();

                            let (condition, span) =
                                lexer.capture_span(|lexer| this.general_expression(lexer, ctx))?;
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

                            let (block, span) = this.block(lexer, ctx, brace_nesting_level)?;
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
                                    lexer.capture_span(|lexer| {
                                        this.statement(lexer, ctx, block, brace_nesting_level)
                                    })?
                                };

                                if block.stmts.len() != num_statements {
                                    match block.stmts.last().unwrap().kind {
                                        ast::StatementKind::Call { .. }
                                        | ast::StatementKind::Assign { .. }
                                        | ast::StatementKind::LocalDecl(_) => {}
                                        _ => {
                                            return Err(Box::new(Error::InvalidForInitializer(
                                                span,
                                            )))
                                        }
                                    }
                                }
                            };

                            let mut body = ast::Block::default();
                            if !lexer.skip(Token::Separator(';')) {
                                let (condition, span) =
                                    lexer.capture_span(|lexer| -> Result<'_, _> {
                                        let condition = this.general_expression(lexer, ctx)?;
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
                                this.function_call_or_assignment_statement(
                                    lexer,
                                    ctx,
                                    &mut continuing,
                                )?;
                                lexer.expect(Token::Paren(')'))?;
                            }

                            let (block, span) = this.block(lexer, ctx, brace_nesting_level)?;
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
                                return Err(Box::new(Error::InvalidBreakIf(span)));
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
                        // https://www.w3.org/TR/WGSL/#const-assert-statement
                        "const_assert" => {
                            let _ = lexer.next();
                            // parentheses are optional
                            let paren = lexer.skip(Token::Paren('('));

                            let condition = this.general_expression(lexer, ctx)?;

                            if paren {
                                lexer.expect(Token::Paren(')'))?;
                            }
                            lexer.expect(Token::Separator(';'))?;
                            ast::StatementKind::ConstAssert(condition)
                        }
                        // assignment or a function call
                        _ => {
                            this.function_call_or_assignment_statement(lexer, ctx, block)?;
                            lexer.expect(Token::Separator(';'))?;
                            this.pop_rule_span(lexer);
                            return Ok(());
                        }
                    };

                    let span = this.pop_rule_span(lexer);
                    block.stmts.push(ast::Statement { kind, span });
                }
                _ => {
                    this.assignment_statement(lexer, ctx, block)?;
                    lexer.expect(Token::Separator(';'))?;
                    this.pop_rule_span(lexer);
                }
            }
            Ok(())
        })
    }

    fn r#loop<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
        brace_nesting_level: u8,
    ) -> Result<'a, ast::StatementKind<'a>> {
        let _ = lexer.next();
        let mut body = ast::Block::default();
        let mut continuing = ast::Block::default();
        let mut break_if = None;

        let brace_span = lexer.expect_span(Token::Paren('{'))?;
        let brace_nesting_level = Self::increase_brace_nesting(brace_nesting_level, brace_span)?;

        ctx.local_table.push_scope();

        loop {
            if lexer.skip(Token::Word("continuing")) {
                // Branch for the `continuing` block, this must be
                // the last thing in the loop body

                // Expect a opening brace to start the continuing block
                let brace_span = lexer.expect_span(Token::Paren('{'))?;
                let brace_nesting_level =
                    Self::increase_brace_nesting(brace_nesting_level, brace_span)?;
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
                        self.statement(lexer, ctx, &mut continuing, brace_nesting_level)?;
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
            self.statement(lexer, ctx, &mut body, brace_nesting_level)?;
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
        brace_nesting_level: u8,
    ) -> Result<'a, (ast::Block<'a>, Span)> {
        self.push_rule_span(Rule::Block, lexer);

        ctx.local_table.push_scope();

        let mut diagnostic_filters = DiagnosticFilterMap::new();

        self.push_rule_span(Rule::Attribute, lexer);
        while lexer.skip(Token::Attribute) {
            let (name, name_span) = lexer.next_ident_with_span()?;
            if let Some(DirectiveKind::Diagnostic) = DirectiveKind::from_ident(name) {
                let filter = self.diagnostic_filter(lexer)?;
                let span = self.peek_rule_span(lexer);
                diagnostic_filters
                    .add(filter, span, ShouldConflictOnFullDuplicate::Yes)
                    .map_err(|e| Box::new(e.into()))?;
            } else {
                return Err(Box::new(Error::Unexpected(
                    name_span,
                    ExpectedToken::DiagnosticAttribute,
                )));
            }
        }
        self.pop_rule_span(lexer);

        if !diagnostic_filters.is_empty() {
            return Err(Box::new(
                Error::DiagnosticAttributeNotYetImplementedAtParseSite {
                    site_name_plural: "compound statements",
                    spans: diagnostic_filters.spans().collect(),
                },
            ));
        }

        let brace_span = lexer.expect_span(Token::Paren('{'))?;
        let brace_nesting_level = Self::increase_brace_nesting(brace_nesting_level, brace_span)?;
        let mut block = ast::Block::default();
        while !lexer.skip(Token::Paren('}')) {
            self.statement(lexer, ctx, &mut block, brace_nesting_level)?;
        }

        ctx.local_table.pop_scope();

        let span = self.pop_rule_span(lexer);
        Ok((block, span))
    }

    fn varying_binding<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        ctx: &mut ExpressionContext<'a, '_, '_>,
    ) -> Result<'a, Option<ast::Binding<'a>>> {
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
        diagnostic_filter_leaf: Option<Handle<DiagnosticFilterNode>>,
        must_use: Option<Span>,
        out: &mut ast::TranslationUnit<'a>,
        dependencies: &mut FastIndexSet<ast::Dependency<'a>>,
    ) -> Result<'a, ast::Function<'a>> {
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
                return Err(Box::new(Error::Unexpected(
                    lexer.next().1,
                    ExpectedToken::Token(Token::Separator(',')),
                )));
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
        let result = if lexer.skip(Token::Arrow) {
            let binding = self.varying_binding(lexer, &mut ctx)?;
            let ty = self.type_decl(lexer, &mut ctx)?;
            let must_use = must_use.is_some();
            Some(ast::FunctionResult {
                ty,
                binding,
                must_use,
            })
        } else if let Some(must_use) = must_use {
            return Err(Box::new(Error::FunctionMustUseReturnsVoid(
                must_use,
                self.peek_rule_span(lexer),
            )));
        } else {
            None
        };

        // do not use `self.block` here, since we must not push a new scope
        lexer.expect(Token::Paren('{'))?;
        let brace_nesting_level = 1;
        let mut body = ast::Block::default();
        while !lexer.skip(Token::Paren('}')) {
            self.statement(lexer, &mut ctx, &mut body, brace_nesting_level)?;
        }

        ctx.local_table.pop_scope();

        let fun = ast::Function {
            entry_point: None,
            name: fun_name,
            arguments,
            result,
            body,
            diagnostic_filter_leaf,
            doc_comments: Vec::new(),
        };

        // done
        self.pop_rule_span(lexer);

        Ok(fun)
    }

    fn directive_ident_list<'a>(
        &self,
        lexer: &mut Lexer<'a>,
        handler: impl FnMut(&'a str, Span) -> Result<'a, ()>,
    ) -> Result<'a, ()> {
        let mut handler = handler;
        'next_arg: loop {
            let (ident, span) = lexer.next_ident_with_span()?;
            handler(ident, span)?;

            let expected_token = match lexer.peek().0 {
                Token::Separator(',') => {
                    let _ = lexer.next();
                    if matches!(lexer.peek().0, Token::Word(..)) {
                        continue 'next_arg;
                    }
                    ExpectedToken::AfterIdentListComma
                }
                _ => ExpectedToken::AfterIdentListArg,
            };

            if !matches!(lexer.next().0, Token::Separator(';')) {
                return Err(Box::new(Error::Unexpected(span, expected_token)));
            }

            break Ok(());
        }
    }

    fn global_decl<'a>(
        &mut self,
        lexer: &mut Lexer<'a>,
        out: &mut ast::TranslationUnit<'a>,
    ) -> Result<'a, ()> {
        let doc_comments = lexer.accumulate_doc_comments();

        // read attributes
        let mut binding = None;
        let mut stage = ParsedAttribute::default();
        let mut compute_span = Span::new(0, 0);
        let mut workgroup_size = ParsedAttribute::default();
        let mut early_depth_test = ParsedAttribute::default();
        let (mut bind_index, mut bind_group) =
            (ParsedAttribute::default(), ParsedAttribute::default());
        let mut id = ParsedAttribute::default();

        let mut must_use: ParsedAttribute<Span> = ParsedAttribute::default();

        let mut dependencies = FastIndexSet::default();
        let mut ctx = ExpressionContext {
            expressions: &mut out.expressions,
            local_table: &mut SymbolTable::default(),
            locals: &mut Arena::new(),
            types: &mut out.types,
            unresolved: &mut dependencies,
        };
        let mut diagnostic_filters = DiagnosticFilterMap::new();
        let ensure_no_diag_attrs = |on_what, filters: DiagnosticFilterMap| -> Result<()> {
            if filters.is_empty() {
                Ok(())
            } else {
                Err(Box::new(Error::DiagnosticAttributeNotSupported {
                    on_what,
                    spans: filters.spans().collect(),
                }))
            }
        };

        self.push_rule_span(Rule::Attribute, lexer);
        while lexer.skip(Token::Attribute) {
            let (name, name_span) = lexer.next_ident_with_span()?;
            if let Some(DirectiveKind::Diagnostic) = DirectiveKind::from_ident(name) {
                let filter = self.diagnostic_filter(lexer)?;
                let span = self.peek_rule_span(lexer);
                diagnostic_filters
                    .add(filter, span, ShouldConflictOnFullDuplicate::Yes)
                    .map_err(|e| Box::new(e.into()))?;
                continue;
            }
            match name {
                "binding" => {
                    lexer.expect(Token::Paren('('))?;
                    bind_index.set(self.general_expression(lexer, &mut ctx)?, name_span)?;
                    lexer.expect(Token::Paren(')'))?;
                }
                "group" => {
                    lexer.expect(Token::Paren('('))?;
                    bind_group.set(self.general_expression(lexer, &mut ctx)?, name_span)?;
                    lexer.expect(Token::Paren(')'))?;
                }
                "id" => {
                    lexer.expect(Token::Paren('('))?;
                    id.set(self.general_expression(lexer, &mut ctx)?, name_span)?;
                    lexer.expect(Token::Paren(')'))?;
                }
                "vertex" => {
                    stage.set(ShaderStage::Vertex, name_span)?;
                }
                "fragment" => {
                    stage.set(ShaderStage::Fragment, name_span)?;
                }
                "compute" => {
                    stage.set(ShaderStage::Compute, name_span)?;
                    compute_span = name_span;
                }
                "workgroup_size" => {
                    lexer.expect(Token::Paren('('))?;
                    let mut new_workgroup_size = [None; 3];
                    for (i, size) in new_workgroup_size.iter_mut().enumerate() {
                        *size = Some(self.general_expression(lexer, &mut ctx)?);
                        match lexer.next() {
                            (Token::Paren(')'), _) => break,
                            (Token::Separator(','), _) if i != 2 => (),
                            other => {
                                return Err(Box::new(Error::Unexpected(
                                    other.1,
                                    ExpectedToken::WorkgroupSizeSeparator,
                                )))
                            }
                        }
                    }
                    workgroup_size.set(new_workgroup_size, name_span)?;
                }
                "early_depth_test" => {
                    lexer.expect(Token::Paren('('))?;
                    let (ident, ident_span) = lexer.next_ident_with_span()?;
                    let value = if ident == "force" {
                        crate::EarlyDepthTest::Force
                    } else {
                        crate::EarlyDepthTest::Allow {
                            conservative: conv::map_conservative_depth(ident, ident_span)?,
                        }
                    };
                    lexer.expect(Token::Paren(')'))?;
                    early_depth_test.set(value, name_span)?;
                }
                "must_use" => {
                    must_use.set(name_span, name_span)?;
                }
                _ => return Err(Box::new(Error::UnknownAttribute(name_span))),
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
            (Some(_), None) => {
                return Err(Box::new(Error::MissingAttribute("binding", attrib_span)))
            }
            (None, Some(_)) => return Err(Box::new(Error::MissingAttribute("group", attrib_span))),
            (None, None) => {}
        }

        // read item
        let start = lexer.start_byte_offset();
        let kind = match lexer.next() {
            (Token::Separator(';'), _) => {
                ensure_no_diag_attrs(
                    DiagnosticAttributeNotSupportedPosition::SemicolonInModulePosition,
                    diagnostic_filters,
                )?;
                None
            }
            (Token::Word(word), directive_span) if DirectiveKind::from_ident(word).is_some() => {
                return Err(Box::new(Error::DirectiveAfterFirstGlobalDecl {
                    directive_span,
                }));
            }
            (Token::Word("struct"), _) => {
                ensure_no_diag_attrs("`struct`s".into(), diagnostic_filters)?;

                let name = lexer.next_ident()?;

                let members = self.struct_body(lexer, &mut ctx)?;

                Some(ast::GlobalDeclKind::Struct(ast::Struct {
                    name,
                    members,
                    doc_comments,
                }))
            }
            (Token::Word("alias"), _) => {
                ensure_no_diag_attrs("`alias`es".into(), diagnostic_filters)?;

                let name = lexer.next_ident()?;

                lexer.expect(Token::Operation('='))?;
                let ty = self.type_decl(lexer, &mut ctx)?;
                lexer.expect(Token::Separator(';'))?;
                Some(ast::GlobalDeclKind::Type(ast::TypeAlias { name, ty }))
            }
            (Token::Word("const"), _) => {
                ensure_no_diag_attrs("`const`s".into(), diagnostic_filters)?;

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

                Some(ast::GlobalDeclKind::Const(ast::Const {
                    name,
                    ty,
                    init,
                    doc_comments,
                }))
            }
            (Token::Word("override"), _) => {
                ensure_no_diag_attrs("`override`s".into(), diagnostic_filters)?;

                let name = lexer.next_ident()?;

                let ty = if lexer.skip(Token::Separator(':')) {
                    Some(self.type_decl(lexer, &mut ctx)?)
                } else {
                    None
                };

                let init = if lexer.skip(Token::Operation('=')) {
                    Some(self.general_expression(lexer, &mut ctx)?)
                } else {
                    None
                };

                lexer.expect(Token::Separator(';'))?;

                Some(ast::GlobalDeclKind::Override(ast::Override {
                    name,
                    id: id.value,
                    ty,
                    init,
                }))
            }
            (Token::Word("var"), _) => {
                ensure_no_diag_attrs("`var`s".into(), diagnostic_filters)?;

                let mut var = self.variable_decl(lexer, &mut ctx)?;
                var.binding = binding.take();
                var.doc_comments = doc_comments;
                Some(ast::GlobalDeclKind::Var(var))
            }
            (Token::Word("fn"), _) => {
                let diagnostic_filter_leaf = Self::write_diagnostic_filters(
                    &mut out.diagnostic_filters,
                    diagnostic_filters,
                    out.diagnostic_filter_leaf,
                );

                let function = self.function_decl(
                    lexer,
                    diagnostic_filter_leaf,
                    must_use.value,
                    out,
                    &mut dependencies,
                )?;
                Some(ast::GlobalDeclKind::Fn(ast::Function {
                    entry_point: if let Some(stage) = stage.value {
                        if stage == ShaderStage::Compute && workgroup_size.value.is_none() {
                            return Err(Box::new(Error::MissingWorkgroupSize(compute_span)));
                        }
                        Some(ast::EntryPoint {
                            stage,
                            early_depth_test: early_depth_test.value,
                            workgroup_size: workgroup_size.value,
                        })
                    } else {
                        None
                    },
                    doc_comments,
                    ..function
                }))
            }
            (Token::Word("const_assert"), _) => {
                ensure_no_diag_attrs("`const_assert`s".into(), diagnostic_filters)?;

                // parentheses are optional
                let paren = lexer.skip(Token::Paren('('));

                let condition = self.general_expression(lexer, &mut ctx)?;

                if paren {
                    lexer.expect(Token::Paren(')'))?;
                }
                lexer.expect(Token::Separator(';'))?;
                Some(ast::GlobalDeclKind::ConstAssert(condition))
            }
            (Token::End, _) => return Ok(()),
            other => {
                return Err(Box::new(Error::Unexpected(
                    other.1,
                    ExpectedToken::GlobalItem,
                )))
            }
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
            return Err(Box::new(Error::Internal("rule stack is not empty")));
        };

        match binding {
            None => Ok(()),
            Some(_) => Err(Box::new(Error::Internal(
                "we had the attribute but no var?",
            ))),
        }
    }

    pub fn parse<'a>(
        &mut self,
        source: &'a str,
        options: &Options,
    ) -> Result<'a, ast::TranslationUnit<'a>> {
        self.reset();

        let mut lexer = Lexer::new(source, !options.parse_doc_comments);
        let mut tu = ast::TranslationUnit::default();
        let mut enable_extensions = EnableExtensions::empty();
        let mut diagnostic_filters = DiagnosticFilterMap::new();

        // Parse module doc comments.
        tu.doc_comments = lexer.accumulate_module_doc_comments();

        // Parse directives.
        while let Ok((ident, _directive_ident_span)) = lexer.peek_ident_with_span() {
            if let Some(kind) = DirectiveKind::from_ident(ident) {
                self.push_rule_span(Rule::Directive, &mut lexer);
                let _ = lexer.next_ident_with_span().unwrap();
                match kind {
                    DirectiveKind::Diagnostic => {
                        let diagnostic_filter = self.diagnostic_filter(&mut lexer)?;
                        let span = self.peek_rule_span(&lexer);
                        diagnostic_filters
                            .add(diagnostic_filter, span, ShouldConflictOnFullDuplicate::No)
                            .map_err(|e| Box::new(e.into()))?;
                        lexer.expect(Token::Separator(';'))?;
                    }
                    DirectiveKind::Enable => {
                        self.directive_ident_list(&mut lexer, |ident, span| {
                            let kind = EnableExtension::from_ident(ident, span)?;
                            let extension = match kind {
                                EnableExtension::Implemented(kind) => kind,
                                EnableExtension::Unimplemented(kind) => {
                                    return Err(Box::new(Error::EnableExtensionNotYetImplemented {
                                        kind,
                                        span,
                                    }))
                                }
                            };
                            enable_extensions.add(extension);
                            Ok(())
                        })?;
                    }
                    DirectiveKind::Requires => {
                        self.directive_ident_list(&mut lexer, |ident, span| {
                            match LanguageExtension::from_ident(ident) {
                                Some(LanguageExtension::Implemented(_kind)) => {
                                    // NOTE: No further validation is needed for an extension, so
                                    // just throw parsed information away. If we ever want to apply
                                    // what we've parsed to diagnostics, maybe we'll want to refer
                                    // to enabled extensions later?
                                    Ok(())
                                }
                                Some(LanguageExtension::Unimplemented(kind)) => {
                                    Err(Box::new(Error::LanguageExtensionNotYetImplemented {
                                        kind,
                                        span,
                                    }))
                                }
                                None => Err(Box::new(Error::UnknownLanguageExtension(span, ident))),
                            }
                        })?;
                    }
                }
                self.pop_rule_span(&lexer);
            } else {
                break;
            }
        }

        lexer.enable_extensions = enable_extensions.clone();
        tu.enable_extensions = enable_extensions;
        tu.diagnostic_filter_leaf =
            Self::write_diagnostic_filters(&mut tu.diagnostic_filters, diagnostic_filters, None);

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

    fn increase_brace_nesting(brace_nesting_level: u8, brace_span: Span) -> Result<'static, u8> {
        // From [spec.](https://gpuweb.github.io/gpuweb/wgsl/#limits):
        //
        // > § 2.4. Limits
        // >
        // > …
        // >
        // > Maximum nesting depth of brace-enclosed statements in a function[:] 127
        const BRACE_NESTING_MAXIMUM: u8 = 127;
        if brace_nesting_level + 1 > BRACE_NESTING_MAXIMUM {
            return Err(Box::new(Error::ExceededLimitForNestedBraces {
                span: brace_span,
                limit: BRACE_NESTING_MAXIMUM,
            }));
        }
        Ok(brace_nesting_level + 1)
    }

    fn diagnostic_filter<'a>(&self, lexer: &mut Lexer<'a>) -> Result<'a, DiagnosticFilter> {
        lexer.expect(Token::Paren('('))?;

        let (severity_control_name, severity_control_name_span) = lexer.next_ident_with_span()?;
        let new_severity = diagnostic_filter::Severity::from_wgsl_ident(severity_control_name)
            .ok_or(Error::DiagnosticInvalidSeverity {
                severity_control_name_span,
            })?;

        lexer.expect(Token::Separator(','))?;

        let (diagnostic_name_token, diagnostic_name_token_span) = lexer.next_ident_with_span()?;
        let triggering_rule = if lexer.skip(Token::Separator('.')) {
            let (ident, _span) = lexer.next_ident_with_span()?;
            FilterableTriggeringRule::User(Box::new([diagnostic_name_token.into(), ident.into()]))
        } else {
            let diagnostic_rule_name = diagnostic_name_token;
            let diagnostic_rule_name_span = diagnostic_name_token_span;
            if let Some(triggering_rule) =
                StandardFilterableTriggeringRule::from_wgsl_ident(diagnostic_rule_name)
            {
                FilterableTriggeringRule::Standard(triggering_rule)
            } else {
                diagnostic_filter::Severity::Warning.report_wgsl_parse_diag(
                    Box::new(Error::UnknownDiagnosticRuleName(diagnostic_rule_name_span)),
                    lexer.source,
                )?;
                FilterableTriggeringRule::Unknown(diagnostic_rule_name.into())
            }
        };
        let filter = DiagnosticFilter {
            triggering_rule,
            new_severity,
        };
        lexer.skip(Token::Separator(','));
        lexer.expect(Token::Paren(')'))?;

        Ok(filter)
    }

    pub(crate) fn write_diagnostic_filters(
        arena: &mut Arena<DiagnosticFilterNode>,
        filters: DiagnosticFilterMap,
        parent: Option<Handle<DiagnosticFilterNode>>,
    ) -> Option<Handle<DiagnosticFilterNode>> {
        filters
            .into_iter()
            .fold(parent, |parent, (triggering_rule, (new_severity, span))| {
                Some(arena.append(
                    DiagnosticFilterNode {
                        inner: DiagnosticFilter {
                            new_severity,
                            triggering_rule,
                        },
                        parent,
                    },
                    span,
                ))
            })
    }
}
