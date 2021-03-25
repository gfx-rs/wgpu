#![allow(clippy::panic, clippy::needless_lifetimes, clippy::upper_case_acronyms)]
use pomelo::pomelo;
pomelo! {
    //%verbose;
    %include {
        use super::super::{error::ErrorKind, token::*, ast::*};
        use crate::{
            BOOL_WIDTH,
            Arena, BinaryOperator, Binding, Block, Constant,
            ConstantInner, Expression,
            Function, FunctionArgument, FunctionResult,
            GlobalVariable, Handle, Interpolation,
            LocalVariable, ResourceBinding, ScalarValue, ScalarKind,
            Statement, StorageAccess, StorageClass, StructMember,
            SwitchCase, Type, TypeInner, UnaryOperator,
        };
        use pp_rs::token::PreprocessorError;
    }
    %token #[derive(Debug)] #[cfg_attr(test, derive(PartialEq))] pub enum Token {};
    %parser pub struct Parser<'a, 'b> {};
    %extra_argument &'a mut Program<'b>;
    %extra_token TokenMetadata;
    %error ErrorKind;
    %syntax_error {
        match token {
            Some(token) => Err(ErrorKind::InvalidToken(token)),
            None => Err(ErrorKind::EndOfFile),
        }
    }
    %parse_fail {
        ErrorKind::ParserFail
    }
    %stack_overflow {
        ErrorKind::ParserStackOverflow
    }

    %type Unknown PreprocessorError;
    %type Pragma ();
    %type Extension ();

    %type Identifier String;
    // constants
    %type IntConstant i64;
    %type UintConstant u64;
    %type FloatConstant f32;
    %type BoolConstant bool;
    %type DoubleConstant f64;
    %type String String;
    // function
    %type function_prototype Function;
    %type function_declarator Function;
    %type function_header Function;
    %type function_header_with_parameters (Function, Vec<FunctionArgument>);
    %type function_definition Function;

    // statements
    %type compound_statement Block;
    %type compound_statement_no_new_scope Block;
    %type statement_list Block;
    %type statement Statement;
    %type simple_statement Statement;
    %type expression_statement Statement;
    %type declaration_statement Statement;
    %type jump_statement Statement;
    %type iteration_statement Statement;
    %type selection_statement Statement;
    %type switch_statement_list Vec<(Option<i32>, Block, bool)>;
    %type switch_statement (Option<i32>, Block, bool);
    %type for_init_statement Statement;
    %type for_rest_statement (Option<ExpressionRule>, Option<ExpressionRule>);
    %type condition_opt Option<ExpressionRule>;

    // expressions
    %type unary_expression ExpressionRule;
    %type postfix_expression ExpressionRule;
    %type primary_expression ExpressionRule;
    %type variable_identifier ExpressionRule;

    %type function_call ExpressionRule;
    %type function_call_or_method FunctionCall;
    %type function_call_generic FunctionCall;
    %type function_call_header_no_parameters FunctionCall;
    %type function_call_header_with_parameters FunctionCall;
    %type function_call_header FunctionCall;
    %type function_identifier FunctionCallKind;

    %type parameter_declarator FunctionArgument;
    %type parameter_declaration FunctionArgument;
    %type parameter_type_specifier Handle<Type>;

    %type multiplicative_expression ExpressionRule;
    %type additive_expression ExpressionRule;
    %type shift_expression ExpressionRule;
    %type relational_expression ExpressionRule;
    %type equality_expression ExpressionRule;
    %type and_expression ExpressionRule;
    %type exclusive_or_expression ExpressionRule;
    %type inclusive_or_expression ExpressionRule;
    %type logical_and_expression ExpressionRule;
    %type logical_xor_expression ExpressionRule;
    %type logical_or_expression ExpressionRule;
    %type conditional_expression ExpressionRule;

    %type assignment_expression ExpressionRule;
    %type assignment_operator BinaryOperator;
    %type expression ExpressionRule;
    %type constant_expression Handle<Constant>;

    %type initializer ExpressionRule;

    // declarations
    %type declaration Option<VarDeclaration>;
    %type init_declarator_list VarDeclaration;
    %type single_declaration VarDeclaration;
    %type layout_qualifier StructLayout;
    %type layout_qualifier_id_list Vec<(String, u32)>;
    %type layout_qualifier_id (String, u32);
    %type type_qualifier Vec<TypeQualifier>;
    %type single_type_qualifier TypeQualifier;
    %type storage_qualifier StorageQualifier;
    %type interpolation_qualifier Interpolation;
    %type Interpolation Interpolation;

    // types
    %type fully_specified_type (Vec<TypeQualifier>, Option<Handle<Type>>);
    %type type_specifier Option<Handle<Type>>;
    %type type_specifier_nonarray Option<Type>;
    %type struct_specifier Type;
    %type struct_declaration_list Vec<StructMember>;
    %type struct_declaration Vec<StructMember>;
    %type struct_declarator_list Vec<String>;
    %type struct_declarator String;

    %type TypeName Type;

    // precedence
    %right Else;

    root ::= version_pragma translation_unit;
    version_pragma ::= Version IntConstant(V) Identifier?(P) {
        match V.1 {
            440 => (),
            450 => (),
            460 => (),
            _ => return Err(ErrorKind::InvalidVersion(V.0, V.1))
        }
        extra.version = V.1 as u16;
        extra.profile = match P {
            Some((meta, profile)) => {
                match profile.as_str() {
                    "core" => Profile::Core,
                    _ => return Err(ErrorKind::InvalidProfile(meta, profile))
                }
            },
            None => Profile::Core,
        }
    };

    // expression
    variable_identifier ::= Identifier(v) {
        let var = extra.lookup_variable(&v.1)?;
        match var {
            Some(expression) => {
                ExpressionRule::from_expression(expression)
            },
            None => {
                return Err(ErrorKind::UnknownVariable(v.0, v.1));
            }
        }
    }

    primary_expression ::= variable_identifier;
    primary_expression ::= IntConstant(i) {
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Sint(i.1),
            },
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    primary_expression ::= UintConstant(i) {
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Uint(i.1),
            },
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    primary_expression ::= FloatConstant(f) {
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 4,
                value: ScalarValue::Float(f.1 as f64),
            },
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    primary_expression ::= BoolConstant(b) {
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: BOOL_WIDTH,
                value: ScalarValue::Bool(b.1)
            },
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    primary_expression ::= DoubleConstant(f) {
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            inner: ConstantInner::Scalar {
                width: 8,
                value: ScalarValue::Float(f.1),
            },
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    primary_expression ::= LeftParen expression(e) RightParen {
        e
    }

    postfix_expression ::= primary_expression;
    postfix_expression ::= postfix_expression LeftBracket integer_expression RightBracket {
        //TODO
        return Err(ErrorKind::NotImplemented("[]"))
    }
    postfix_expression ::= function_call;
    postfix_expression ::= postfix_expression(e) Dot Identifier(i) /* FieldSelection in spec */ {
        //TODO: how will this work as l-value?
        let expression = extra.field_selection(e.expression, &*i.1, i.0)?;
        ExpressionRule { expression, statements: e.statements, sampler: None }
    }
    postfix_expression ::= postfix_expression(pe) IncOp {
        //TODO
        return Err(ErrorKind::NotImplemented("post++"))
    }
    postfix_expression ::= postfix_expression(pe) DecOp {
        //TODO
        return Err(ErrorKind::NotImplemented("post--"))
    }

    integer_expression ::= expression;

    function_call ::= function_call_or_method(fc) {
       extra.function_call(fc)?
    }
    function_call_or_method ::= function_call_generic;
    function_call_generic ::= function_call_header_with_parameters(h) RightParen {
        h
    }
    function_call_generic ::= function_call_header_no_parameters(h) RightParen {
        h
    }
    function_call_header_no_parameters ::= function_call_header(h) Void {
        h
    }
    function_call_header_no_parameters ::= function_call_header;
    function_call_header_with_parameters ::= function_call_header(mut h) assignment_expression(ae) {
        h.args.push(ae);
        h
    }
    function_call_header_with_parameters ::= function_call_header_with_parameters(mut h) Comma assignment_expression(ae) {
        h.args.push(ae);
        h
    }
    function_call_header ::= function_identifier(i) LeftParen {
        FunctionCall {
            kind: i,
            args: vec![],
        }
    }

    // Grammar Note: Constructors look like functions, but lexical analysis recognized most of them as
    // keywords. They are now recognized through “type_specifier”.
    function_identifier ::= type_specifier(t) {
        if let Some(ty) = t {
            FunctionCallKind::TypeConstructor(ty)
        } else {
            return Err(ErrorKind::NotImplemented("bad type ctor"))
        }
    }

    //TODO
    // Methods (.length), subroutine array calls, and identifiers are recognized through postfix_expression.
    // function_identifier ::= postfix_expression(e) {
    //     FunctionCallKind::Function(e.expression)
    // }

    // Simplification of above
    function_identifier ::= Identifier(i) {
        FunctionCallKind::Function(i.1)
    }


    unary_expression ::= postfix_expression;

    unary_expression ::= IncOp unary_expression {
        //TODO
        return Err(ErrorKind::NotImplemented("++pre"))
    }
    unary_expression ::= DecOp unary_expression {
        //TODO
        return Err(ErrorKind::NotImplemented("--pre"))
    }
    unary_expression ::= Plus unary_expression(tgt) {
        tgt
    }
    unary_expression ::= Dash unary_expression(tgt) {
        extra.unary_expr(UnaryOperator::Negate, &tgt)
    }
    unary_expression ::= Bang unary_expression(tgt) {
        if let TypeInner::Scalar { kind: ScalarKind::Bool, .. } = *extra.resolve_type(tgt.expression)? {
            extra.unary_expr(UnaryOperator::Not, &tgt)
        } else {
            return Err(ErrorKind::SemanticError("Cannot apply '!' to non bool type".into()))
        }
    }
    unary_expression ::= Tilde unary_expression(tgt) {
        if extra.resolve_type(tgt.expression)?.scalar_kind() != Some(ScalarKind::Bool) {
            extra.unary_expr(UnaryOperator::Not, &tgt)
        } else {
            return Err(ErrorKind::SemanticError("Cannot apply '~' to type".into()))
        }
    }

    multiplicative_expression ::= unary_expression;
    multiplicative_expression ::= multiplicative_expression(left) Star unary_expression(right) {
        extra.binary_expr(BinaryOperator::Multiply, &left, &right)
    }
    multiplicative_expression ::= multiplicative_expression(left) Slash unary_expression(right) {
        extra.binary_expr(BinaryOperator::Divide, &left, &right)
    }
    multiplicative_expression ::= multiplicative_expression(left) Percent unary_expression(right) {
        extra.binary_expr(BinaryOperator::Modulo, &left, &right)
    }
    additive_expression ::= multiplicative_expression;
    additive_expression ::= additive_expression(left) Plus multiplicative_expression(right) {
        extra.binary_expr(BinaryOperator::Add, &left, &right)
    }
    additive_expression ::= additive_expression(left) Dash multiplicative_expression(right) {
        extra.binary_expr(BinaryOperator::Subtract, &left, &right)
    }
    shift_expression ::= additive_expression;
    shift_expression ::= shift_expression(left) LeftOp additive_expression(right) {
        extra.binary_expr(BinaryOperator::ShiftLeft, &left, &right)
    }
    shift_expression ::= shift_expression(left) RightOp additive_expression(right) {
        extra.binary_expr(BinaryOperator::ShiftRight, &left, &right)
    }
    relational_expression ::= shift_expression;
    relational_expression ::= relational_expression(left) LeftAngle shift_expression(right) {
        extra.binary_expr(BinaryOperator::Less, &left, &right)
    }
    relational_expression ::= relational_expression(left) RightAngle shift_expression(right) {
        extra.binary_expr(BinaryOperator::Greater, &left, &right)
    }
    relational_expression ::= relational_expression(left) LeOp shift_expression(right) {
        extra.binary_expr(BinaryOperator::LessEqual, &left, &right)
    }
    relational_expression ::= relational_expression(left) GeOp shift_expression(right) {
        extra.binary_expr(BinaryOperator::GreaterEqual, &left, &right)
    }
    equality_expression ::= relational_expression;
    equality_expression ::= equality_expression(left) EqOp relational_expression(right) {
        extra.equality_expr(true, &left, &right)?
    }
    equality_expression ::= equality_expression(left) NeOp relational_expression(right) {
        extra.equality_expr(false, &left, &right)?
    }
    and_expression ::= equality_expression;
    and_expression ::= and_expression(left) Ampersand equality_expression(right) {
        extra.binary_expr(BinaryOperator::And, &left, &right)
    }
    exclusive_or_expression ::= and_expression;
    exclusive_or_expression ::= exclusive_or_expression(left) Caret and_expression(right) {
        extra.binary_expr(BinaryOperator::ExclusiveOr, &left, &right)
    }
    inclusive_or_expression ::= exclusive_or_expression;
    inclusive_or_expression ::= inclusive_or_expression(left) VerticalBar exclusive_or_expression(right) {
        extra.binary_expr(BinaryOperator::InclusiveOr, &left, &right)
    }
    logical_and_expression ::= inclusive_or_expression;
    logical_and_expression ::= logical_and_expression(left) AndOp inclusive_or_expression(right) {
        extra.binary_expr(BinaryOperator::LogicalAnd, &left, &right)
    }
    logical_xor_expression ::= logical_and_expression;
    logical_xor_expression ::= logical_xor_expression(left) XorOp logical_and_expression(right) {
        let exp1 = extra.binary_expr(BinaryOperator::LogicalOr, &left, &right);
        let exp2 = {
            let tmp = extra.binary_expr(BinaryOperator::LogicalAnd, &left, &right).expression;
            ExpressionRule::from_expression(extra.context.expressions.append(Expression::Unary { op: UnaryOperator::Not, expr: tmp }))
        };
        extra.binary_expr(BinaryOperator::LogicalAnd, &exp1, &exp2)
    }
    logical_or_expression ::= logical_xor_expression;
    logical_or_expression ::= logical_or_expression(left) OrOp logical_xor_expression(right) {
        extra.binary_expr(BinaryOperator::LogicalOr, &left, &right)
    }

    conditional_expression ::= logical_or_expression;
    conditional_expression ::= logical_or_expression Question expression Colon assignment_expression(ae) {
        //TODO: how to do ternary here in naga?
        return Err(ErrorKind::NotImplemented("ternary exp"))
    }

    assignment_expression ::= conditional_expression;
    assignment_expression ::= unary_expression(mut pointer) assignment_operator(op) assignment_expression(value) {
        pointer.statements.extend(value.statements);
        match op {
            BinaryOperator::Equal => {
                pointer.statements.push(Statement::Store{
                    pointer: pointer.expression,
                    value: value.expression
                });
                pointer
            },
            _ => {
                let h = extra.context.expressions.append(
                    Expression::Binary{
                        op,
                        left: pointer.expression,
                        right: value.expression,
                    }
                );
                pointer.statements.push(Statement::Store{
                    pointer: pointer.expression,
                    value: h,
                });
                pointer
            }
        }
    }

    assignment_operator ::= Equal {
        BinaryOperator::Equal
    }
    assignment_operator ::= MulAssign {
        BinaryOperator::Multiply
    }
    assignment_operator ::= DivAssign {
        BinaryOperator::Divide
    }
    assignment_operator ::= ModAssign {
        BinaryOperator::Modulo
    }
    assignment_operator ::= AddAssign {
        BinaryOperator::Add
    }
    assignment_operator ::= SubAssign {
        BinaryOperator::Subtract
    }
    assignment_operator ::= LeftAssign {
        BinaryOperator::ShiftLeft
    }
    assignment_operator ::= RightAssign {
        BinaryOperator::ShiftRight
    }
    assignment_operator ::= AndAssign {
        BinaryOperator::And
    }
    assignment_operator ::= XorAssign {
        BinaryOperator::ExclusiveOr
    }
    assignment_operator ::= OrAssign {
        BinaryOperator::InclusiveOr
    }

    expression ::= assignment_expression;
    expression ::= expression(e) Comma assignment_expression(mut ae) {
        ae.statements.extend(e.statements);
        ExpressionRule {
            expression: e.expression,
            statements: ae.statements,
            sampler: None,
        }
    }

    //TODO: properly handle constant expressions
    // constant_expression ::= conditional_expression(e) {
    //     if let Expression::Constant(h) = extra.context.expressions[e] {
    //         h
    //     } else {
    //         return Err(ErrorKind::ExpectedConstant)
    //     }
    // }

    // declaration
    declaration ::= init_declarator_list(idl) Semicolon {
        Some(idl)
    }

    declaration ::= type_qualifier(t) Identifier(i) LeftBrace
        struct_declaration_list(sdl) RightBrace Semicolon {
        if i.1 == "gl_PerVertex" {
            None
        } else {
            let block = !t.is_empty();
            Some(VarDeclaration {
                type_qualifiers: t,
                ids_initializers: vec![(None, None)],
                ty: extra.module.types.fetch_or_append(Type{
                    name: Some(i.1),
                    inner: TypeInner::Struct {
                        block,
                        members: sdl,
                    },
                }),
            })
        }
    }

    declaration ::= type_qualifier(t) Identifier(i1) LeftBrace
        struct_declaration_list(sdl) RightBrace Identifier(i2) Semicolon {
        let block = !t.is_empty();
        Some(VarDeclaration {
            type_qualifiers: t,
            ids_initializers: vec![(Some(i2.1), None)],
            ty: extra.module.types.fetch_or_append(Type{
                name: Some(i1.1),
                inner: TypeInner::Struct {
                    block,
                    members: sdl,
                },
            }),
        })
    }

    // declaration ::= type_qualifier(t) Identifier(i1) LeftBrace
    //     struct_declaration_list RightBrace Identifier(i2) array_specifier Semicolon;

    init_declarator_list ::= single_declaration;
    init_declarator_list ::= init_declarator_list(mut idl) Comma Identifier(i) {
        idl.ids_initializers.push((Some(i.1), None));
        idl
    }
    // init_declarator_list ::= init_declarator_list Comma Identifier array_specifier;
    // init_declarator_list ::= init_declarator_list Comma Identifier array_specifier Equal initializer;
    init_declarator_list ::= init_declarator_list(mut idl) Comma Identifier(i) Equal initializer(init) {
        idl.ids_initializers.push((Some(i.1), Some(init)));
        idl
    }

    single_declaration ::= fully_specified_type(t) {
        let ty = t.1.ok_or_else(||ErrorKind::SemanticError("Empty type for declaration".into()))?;

        VarDeclaration {
            type_qualifiers: t.0,
            ids_initializers: vec![],
            ty,
        }
    }
    single_declaration ::= fully_specified_type(t) Identifier(i) {
        let ty = t.1.ok_or_else(|| ErrorKind::SemanticError("Empty type for declaration".into()))?;

        VarDeclaration {
            type_qualifiers: t.0,
            ids_initializers: vec![(Some(i.1), None)],
            ty,
        }
    }
    // single_declaration ::= fully_specified_type Identifier array_specifier;
    // single_declaration ::= fully_specified_type Identifier array_specifier Equal initializer;
    single_declaration ::= fully_specified_type(t) Identifier(i) Equal initializer(init) {
        let ty = t.1.ok_or_else(|| ErrorKind::SemanticError("Empty type for declaration".into()))?;

        VarDeclaration {
            type_qualifiers: t.0,
            ids_initializers: vec![(Some(i.1), Some(init))],
            ty,
        }
    }

    fully_specified_type ::= type_specifier(t) {
        (vec![], t)
    }
    fully_specified_type ::= type_qualifier(q) type_specifier(t) {
        (q,t)
    }

    interpolation_qualifier ::= Interpolation((_, i)) {
        i
    }

    layout_qualifier ::= Layout LeftParen layout_qualifier_id_list(l) RightParen {
        if let Some(&(_, loc)) = l.iter().find(|&q| q.0.as_str() == "location") {
            let interpolation = None; //TODO
            StructLayout::Binding(Binding::Location(loc, interpolation))
        } else if let Some(&(_, binding)) = l.iter().find(|&q| q.0.as_str() == "binding") {
            let group = if let Some(&(_, set)) = l.iter().find(|&q| q.0.as_str() == "set") {
                set
            } else {
                0
            };
            StructLayout::Resource(ResourceBinding{ group, binding })
        } else if l.iter().any(|q| q.0.as_str() == "push_constant") {
            StructLayout::PushConstant
        } else {
            return Err(ErrorKind::NotImplemented("unsupported layout qualifier(s)"));
        }
    }
    layout_qualifier_id_list ::= layout_qualifier_id(lqi) {
        vec![lqi]
    }
    layout_qualifier_id_list ::= layout_qualifier_id_list(mut l) Comma layout_qualifier_id(lqi) {
        l.push(lqi);
        l
    }
    layout_qualifier_id ::= Identifier(i) {
        (i.1, 0)
    }
    //TODO: handle full constant_expression instead of IntConstant
    layout_qualifier_id ::= Identifier(i) Equal IntConstant(ic) {
        (i.1, ic.1 as u32)
    }
    // layout_qualifier_id ::= Shared;

    // precise_qualifier ::= Precise;

    type_qualifier ::= single_type_qualifier(t) {
        vec![t]
    }
    type_qualifier ::= type_qualifier(mut l) single_type_qualifier(t) {
        l.push(t);
        l
    }

    single_type_qualifier ::= storage_qualifier(s) {
        TypeQualifier::StorageQualifier(s)
    }
    single_type_qualifier ::= layout_qualifier(l) {
        match l {
            StructLayout::Binding(b) => TypeQualifier::Binding(b),
            StructLayout::Resource(b) => TypeQualifier::ResourceBinding(b),
            StructLayout::PushConstant => TypeQualifier::StorageQualifier(StorageQualifier::StorageClass(StorageClass::PushConstant)),
        }
    }
    // single_type_qualifier ::= precision_qualifier;
    single_type_qualifier ::= interpolation_qualifier(i) {
        TypeQualifier::Interpolation(i)
    }
    // single_type_qualifier ::= invariant_qualifier;
    // single_type_qualifier ::= precise_qualifier;

    storage_qualifier ::= Const {
        StorageQualifier::Const
    }
    // storage_qualifier ::= InOut;
    storage_qualifier ::= In {
        StorageQualifier::Input
    }
    storage_qualifier ::= Out {
        StorageQualifier::Output
    }
    // storage_qualifier ::= Centroid;
    // storage_qualifier ::= Patch;
    // storage_qualifier ::= Sample;
    storage_qualifier ::= Uniform {
        StorageQualifier::StorageClass(StorageClass::Uniform)
    }
    //TODO: other storage qualifiers

    type_specifier ::= type_specifier_nonarray(t) {
        t.map(|t| {
            let name = t.name.clone();
            let handle = extra.module.types.fetch_or_append(t);
            if let Some(name) = name {
                extra.lookup_type.insert(name, handle);
            }
            handle
        })
    }
    //TODO: array

    type_specifier_nonarray ::= Void {
        None
    }
    type_specifier_nonarray ::= TypeName(t) {
        Some(t.1)
    };
    type_specifier_nonarray ::= struct_specifier(s) {
        Some(s)
    }

    // struct
    struct_specifier ::= Struct Identifier(i) LeftBrace  struct_declaration_list RightBrace {
        Type{
            name: Some(i.1),
            inner: TypeInner::Struct {
                block: false,
                members: vec![],
            }
        }
    }
    //struct_specifier ::= Struct LeftBrace  struct_declaration_list RightBrace;

    struct_declaration_list ::= struct_declaration(sd) {
        sd
    }
    struct_declaration_list ::= struct_declaration_list(mut sdl) struct_declaration(sd) {
        sdl.extend(sd);
        sdl
    }

    struct_declaration ::= type_specifier(t) struct_declarator_list(sdl) Semicolon {
        if let Some(ty) = t {
            sdl.iter().map(|name| StructMember {
                name: Some(name.clone()),
                ty,
                binding: None, //TODO
                //TODO: if the struct is a uniform struct, these values have to reflect
                // std140 layout. Otherwise, std430.
                size: None,
                align: None,
            }).collect()
        } else {
            return Err(ErrorKind::SemanticError("Struct member can't be void".into()))
        }
    }
    //struct_declaration ::= type_qualifier type_specifier struct_declarator_list Semicolon;

    struct_declarator_list ::= struct_declarator(sd) {
        vec![sd]
    }
    struct_declarator_list ::= struct_declarator_list(mut sdl) Comma struct_declarator(sd) {
        sdl.push(sd);
        sdl
    }

    struct_declarator ::= Identifier(i) {
        i.1
    }
    //struct_declarator ::= Identifier array_specifier;


    initializer ::= assignment_expression;
    // initializer ::= LeftBrace initializer_list RightBrace;
    // initializer ::= LeftBrace initializer_list Comma RightBrace;

    // initializer_list ::= initializer;
    // initializer_list ::= initializer_list Comma initializer;

    declaration_statement ::= declaration(d) {
        let mut statements = Vec::<Statement>::new();
        // local variables
        if let Some(d) = d {
            for (id, initializer) in d.ids_initializers {
                let id = id.ok_or_else(|| ErrorKind::SemanticError("Local var must be named".into()))?;
                // check if already declared in current scope
                #[cfg(feature = "glsl-validate")]
                {
                    if extra.context.lookup_local_var_current_scope(&id).is_some() {
                        return Err(ErrorKind::VariableAlreadyDeclared(id))
                    }
                }
                let mut init_exp: Option<Handle<Expression>> = None;
                let localVar = extra.context.local_variables.append(
                    LocalVariable {
                        name: Some(id.clone()),
                        ty: d.ty,
                        init: initializer.map(|i| {
                            statements.extend(i.statements);
                            if let Expression::Constant(constant) = extra.context.expressions[i.expression] {
                                Some(constant)
                            } else {
                                init_exp = Some(i.expression);
                                None
                            }
                        }).flatten(),
                    }
                );
                let exp = extra.context.expressions.append(Expression::LocalVariable(localVar));
                extra.context.add_local_var(id, exp);

                if let Some(value) = init_exp {
                    statements.push(
                        Statement::Store {
                            pointer: exp,
                            value,
                        }
                    );
                }
            }
        };
        match statements.len() {
            1 => statements.remove(0),
            _ => Statement::Block(statements),
        }
    }

    // statement
    statement ::= compound_statement(cs) {
        Statement::Block(cs)
    }
    statement ::= simple_statement;

    simple_statement ::= declaration_statement;
    simple_statement ::= expression_statement;
    simple_statement ::= selection_statement;
    simple_statement ::= jump_statement;
    simple_statement ::= iteration_statement;


    selection_statement ::= If LeftParen expression(e) RightParen statement(s1) Else statement(s2) {
        Statement::If {
            condition: e.expression,
            accept: vec![s1],
            reject: vec![s2],
        }
    }

    selection_statement ::= If LeftParen expression(e) RightParen statement(s) [Else] {
        Statement::If {
            condition: e.expression,
            accept: vec![s],
            reject: vec![],
        }
    }

    selection_statement ::= Switch LeftParen expression(e) RightParen LeftBrace switch_statement_list(ls) RightBrace {
        let mut default = Vec::new();
        let mut cases = Vec::new();
        for (v, body, fall_through) in ls {
            if let Some(value) = v {
                cases.push(SwitchCase {
                    value,
                    body,
                    fall_through,
                });
            } else {
                default.extend_from_slice(&body);
            }
        }
        Statement::Switch {
            selector: e.expression,
            cases,
            default,
        }
    }

    switch_statement_list ::= {
        vec![]
    }
    switch_statement_list ::= switch_statement_list(mut ssl) switch_statement((v, sl, ft)) {
        ssl.push((v, sl, ft));
        ssl
    }
    switch_statement ::= Case IntConstant(v) Colon statement_list(sl) {
        let fall_through = match sl.last() {
            Some(&Statement::Break) => false,
            _ => true,
        };
        (Some(v.1 as i32), sl, fall_through)
    }
    switch_statement ::= Default Colon statement_list(sl) {
        let fall_through = match sl.last() {
            Some(&Statement::Break) => true,
            _ => false,
        };
        (None, sl, fall_through)
    }

    iteration_statement ::= While LeftParen expression(e) RightParen compound_statement_no_new_scope(sl) {
        let mut body = Vec::with_capacity(sl.len() + 1);
        body.push(
            Statement::If {
                condition: e.expression,
                accept: vec![Statement::Break],
                reject: vec![],
            }
        );
        body.extend_from_slice(&sl);
        Statement::Loop {
            body,
            continuing: vec![],
        }
    }

    iteration_statement ::= Do compound_statement(sl) While LeftParen expression(e) RightParen  {
        let mut body = sl;
        body.push(
            Statement::If {
                condition: e.expression,
                accept: vec![Statement::Break],
                reject: vec![],
            }
        );
        Statement::Loop {
            body,
            continuing: vec![],
        }
    }

    iteration_statement ::= For LeftParen for_init_statement(s_init) for_rest_statement((cond_e, loop_e)) RightParen compound_statement_no_new_scope(sl) {
        let mut body = Vec::with_capacity(sl.len() + 2);
        if let Some(cond_e) = cond_e {
            body.push(
                Statement::If {
                    condition: cond_e.expression,
                    accept: vec![Statement::Break],
                    reject: vec![],
                }
            );
        }
        body.extend_from_slice(&sl);
        if let Some(loop_e) = loop_e {
            body.extend_from_slice(&loop_e.statements);
        }
        Statement::Block(vec![
            s_init,
            Statement::Loop {
                body,
                continuing: vec![],
            }
        ])
    }

    for_init_statement ::= expression_statement;
    for_init_statement ::= declaration_statement;
    for_rest_statement ::= condition_opt(c) Semicolon {
        (c, None)
    }
    for_rest_statement ::= condition_opt(c) Semicolon expression(e) {
        (c, Some(e))
    }

    condition_opt ::= {
        None
    }
    condition_opt ::= conditional_expression(c) {
        Some(c)
    }

    compound_statement ::= LeftBrace RightBrace {
        vec![]
    }
    compound_statement ::= left_brace_scope statement_list(sl) RightBrace {
        extra.context.remove_current_scope();
        sl
    }

    // extra rule to add scope before statement_list
    left_brace_scope ::= LeftBrace {
        extra.context.push_scope();
    }


    compound_statement_no_new_scope ::= LeftBrace RightBrace {
        vec![]
    }
    compound_statement_no_new_scope ::= LeftBrace statement_list(sl) RightBrace {
        sl
    }

    statement_list ::= statement(s) {
        //TODO: catch this earlier and don't populate the statements
        match s {
            Statement::Block(ref block) if block.is_empty() => vec![],
            _ => vec![s],
        }
    }
    statement_list ::= statement_list(mut ss) statement(s) { ss.push(s); ss }

    expression_statement ::= Semicolon  {
        Statement::Block(Vec::new())
    }
    expression_statement ::= expression(mut e) Semicolon {
        match e.statements.len() {
            1 => e.statements.remove(0),
            _ => Statement::Block(e.statements),
        }
    }



    // function
    function_prototype ::= function_declarator(f) RightParen {
        extra.add_function_prelude();
        f
    }
    function_declarator ::= function_header;
    function_declarator ::= function_header_with_parameters((f, args)) {
        for (pos, arg) in args.into_iter().enumerate() {
            if let Some(name) = arg.name.clone() {
                let exp = extra.context.expressions.append(Expression::FunctionArgument(pos as u32));
                extra.context.add_local_var(name, exp);
            }
            extra.context.arguments.push(arg);
        }
        f
    }
    function_header ::= fully_specified_type(t) Identifier(n) LeftParen {
        Function {
            name: Some(n.1),
            arguments: vec![],
            result: t.1.map(|ty| FunctionResult { ty, binding: None }),
            local_variables: Arena::<LocalVariable>::new(),
            expressions: Arena::<Expression>::new(),
            body: vec![],
        }
    }
    function_header_with_parameters ::= function_header(h) parameter_declaration(p) {
        (h, vec![p])
    }
    function_header_with_parameters ::= function_header_with_parameters((h, mut args)) Comma parameter_declaration(p) {
        args.push(p);
        (h, args)
    }
    parameter_declarator ::= parameter_type_specifier(ty) Identifier(n) {
        FunctionArgument { name: Some(n.1), ty, binding: None }
    }
    // parameter_declarator ::= type_specifier(ty) Identifier(ident) array_specifier;
    parameter_declaration ::= parameter_declarator;
    parameter_declaration ::= parameter_type_specifier(ty) {
        FunctionArgument { name: None, ty, binding: None }
    }

    parameter_type_specifier ::= type_specifier(t) {
        if let Some(ty) = t {
            ty
        } else {
            return Err(ErrorKind::SemanticError("Function parameter can't be void".into()))
        }
    }

    jump_statement ::= Continue Semicolon {
        Statement::Continue
    }
    jump_statement ::= Break Semicolon {
        Statement::Break
    }
    jump_statement ::= Return Semicolon {
        Statement::Return { value: None }
    }
    jump_statement ::= Return expression(mut e) Semicolon {
        let ret = Statement::Return{ value: Some(e.expression) };
        if !e.statements.is_empty() {
            e.statements.push(ret);
            Statement::Block(e.statements)
        } else {
            ret
        }
    }
    jump_statement ::= Discard Semicolon  {
        Statement::Kill
    } // Fragment shader only

    // Grammar Note: No 'goto'. Gotos are not supported.

    // misc
    translation_unit ::= external_declaration;
    translation_unit ::= translation_unit external_declaration;

    external_declaration ::= function_definition(f) {
        extra.declare_function(f)?
    }
    external_declaration ::= declaration(d) {
        if let Some(d) = d {
            // TODO: handle multiple storage qualifiers
            let storage = d.type_qualifiers.iter().find_map(|tq| {
                if let TypeQualifier::StorageQualifier(sc) = *tq { Some(sc) } else { None }
            }).unwrap_or(StorageQualifier::StorageClass(StorageClass::Private));

            match storage {
                StorageQualifier::StorageClass(storage_class) => {
                    // TODO: Check that the storage qualifiers allow for the bindings
                    let binding = d.type_qualifiers.iter().find_map(|tq| {
                        if let TypeQualifier::ResourceBinding(ref b) = *tq { Some(b.clone()) } else { None }
                    });
                    for (id, initializer) in d.ids_initializers {
                        let init = initializer.map(|init| extra.solve_constant(init.expression)).transpose()?;

                        // use StorageClass::Handle for texture and sampler uniforms
                        let class = if storage_class == StorageClass::Uniform {
                            match extra.module.types[d.ty].inner {
                                TypeInner::Image{..} | TypeInner::Sampler{..} => StorageClass::Handle,
                                _ => storage_class,
                            }
                        } else {
                            storage_class
                        };

                        let h = extra.module.global_variables.fetch_or_append(
                            GlobalVariable {
                                name: id.clone(),
                                class,
                                binding: binding.clone(),
                                ty: d.ty,
                                init,
                                storage_access: StorageAccess::empty(), //TODO
                            },
                        );
                        if let Some(id) = id {
                            extra.lookup_global_variables.insert(id, h);
                        }
                    }
                }
                StorageQualifier::Input => {
                    let mut binding = d.type_qualifiers.iter().find_map(|tq| {
                        if let TypeQualifier::Binding(ref b) = *tq { Some(b.clone()) } else { None }
                    });
                    let interpolation = d.type_qualifiers.iter().find_map(|tq| {
                        if let TypeQualifier::Interpolation(interp) = *tq { Some(interp) } else { None }
                    });
                    if let Some(Binding::Location(_, ref mut interp)) = binding {
                        *interp = interpolation;
                    }

                    for (id, _initializer) in d.ids_initializers {
                        if let Some(id) = id {
                            //TODO!
                            let expr = extra.context.expressions.append(Expression::FunctionArgument(0));
                            extra.context.lookup_global_var_exps.insert(id, expr);
                        }
                    }
                }
                StorageQualifier::Output => {
                    let _binding = d.type_qualifiers.iter().find_map(|tq| {
                        if let TypeQualifier::Binding(ref b) = *tq { Some(b.clone()) } else { None }
                    });
                    for (id, _initializer) in d.ids_initializers {
                        if let Some(id) = id {
                            //TODO!
                            let expr = extra.context.expressions.append(Expression::FunctionArgument(0));
                            extra.context.lookup_global_var_exps.insert(id, expr);
                        }
                    }
                }
                StorageQualifier::Const => {
                    for (id, initializer) in d.ids_initializers {
                        if let Some(init) = initializer {
                            let constant = extra.solve_constant(init.expression)?;
                            let inner = extra.module.constants[constant].inner.clone();

                            let h = extra.module.constants.fetch_or_append(
                                Constant {
                                    name: id.clone(),
                                    specialization: None, // TODO
                                    inner
                                },
                            );
                            if let Some(id) = id {
                                extra.lookup_constants.insert(id.clone(), h);
                                let expr = extra.context.expressions.append(Expression::Constant(h));
                                extra.context.lookup_constant_exps.insert(id, expr);
                            }
                        } else {
                            return Err(ErrorKind::SemanticError("Constants must have an initializer".into()))
                        }
                    }
                }
            }
        }
    }

    function_definition ::= function_prototype(f) compound_statement_no_new_scope(cs) {
        extra.function_definition(f, cs)
    };
}

pub use parser::*;
