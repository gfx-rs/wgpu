#![allow(unused_braces)]
#![allow(clippy::panic)]
use pomelo::pomelo;

pomelo! {
    //%verbose;
    %include {
        use super::super::{error::ErrorKind, token::*, ast::*};
        use crate::{Arena, BinaryOperator, Binding, Block, BuiltIn, Constant, ConstantInner, Expression,
            Function, GlobalVariable, Handle, LocalVariable, ScalarKind,
            ShaderStage, Statement, StorageClass, Type, TypeInner, VectorSize};
    }
    %token #[derive(Debug)] pub enum Token {};
    %parser pub struct Parser<'a> {};
    %extra_argument &'a mut Program;
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

    %type Unknown char;
    %type CommentStart ();
    %type CommentEnd ();

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
    %type function_definition Function;
    // statements
    %type compound_statement Block;
    %type compound_statement_no_new_scope Block;
    %type statement_list Block;
    %type statement Statement;
    %type simple_statement Statement;
    %type expression_statement Statement;

    // expressions
    %type unary_expression Handle<Expression>;
    %type postfix_expression Handle<Expression>;
    %type primary_expression Handle<Expression>;
    %type variable_identifier Handle<Expression>;

    %type function_call Handle<Expression>;
    %type function_call_generic Handle<Expression>;
    %type function_call_header_no_parameters Handle<Expression>;
    %type function_call_header_with_parameters Handle<Expression>;
    %type function_call_header Handle<Expression>;
    %type function_identifier Handle<Expression>;

    %type multiplicative_expression Handle<Expression>;
    %type additive_expression Handle<Expression>;
    %type shift_expression Handle<Expression>;
    %type relational_expression Handle<Expression>;
    %type equality_expression Handle<Expression>;
    %type and_expression Handle<Expression>;
    %type exclusive_or_expression Handle<Expression>;
    %type inclusive_or_expression Handle<Expression>;
    %type logical_and_expression Handle<Expression>;
    %type logical_xor_expression Handle<Expression>;
    %type logical_or_expression Handle<Expression>;
    %type conditional_expression Handle<Expression>;

    %type assignment_expression (Statement, Handle<Expression>);
    %type assignment_operator BinaryOperator;
    %type expression Statement;

    // types
    %type fully_specified_type Option<Handle<Type>>;
    %type type_specifier Option<Handle<Type>>;
    %type type_specifier_nonarray Option<Type>;


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
        if v.1.as_str() == "gl_Position" &&
            (extra.shader_stage == ShaderStage::Vertex ||
            extra.shader_stage == ShaderStage::Fragment) {
            let h = extra.global_variables.fetch_or_append(
                GlobalVariable {
                    name: Some(v.1),
                    class: match extra.shader_stage {
                        ShaderStage::Vertex => StorageClass::Output,
                        ShaderStage::Fragment => StorageClass::Input,
                        _ => StorageClass::Input,
                    },
                    binding: Some(Binding::BuiltIn(BuiltIn::Position)),
                    ty: extra.types.fetch_or_append(Type {
                        name: None,
                        inner: TypeInner::Vector {
                            size: VectorSize::Quad,
                            kind: ScalarKind::Float,
                            width: 4,
                        },
                    }),
                },
            );
            extra.context.expressions.append(Expression::GlobalVariable(h))
        } else {
            return Err(ErrorKind::NotImplemented("var"))
        }
    }

    primary_expression ::= variable_identifier;
    primary_expression ::= IntConstant(i) {
        let ty = extra.types.fetch_or_append(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Sint,
                width: 4,
            }
        });
        let ch = extra.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            ty,
            inner: ConstantInner::Sint(i.1)
        });
        extra.context.expressions.append(Expression::Constant(ch))
    }
    // primary_expression ::= UintConstant;
    // primary_expression ::= FloatConstant;
    // primary_expression ::= BoolConstant;
    // primary_expression ::= DoubleConstant;
    // primary_expression ::= LeftParen expression RightParen;

    postfix_expression ::= primary_expression;
    postfix_expression ::= postfix_expression LeftBracket integer_expression RightBracket {
        // TODO
        return Err(ErrorKind::NotImplemented("[]"))
    }
    postfix_expression ::= function_call;
    postfix_expression ::= postfix_expression Dot FieldSelection {
        // TODO
        return Err(ErrorKind::NotImplemented(".field"))
    }
    postfix_expression ::= postfix_expression(pe) IncOp {
        // TODO
        return Err(ErrorKind::NotImplemented("post++"))
    }
    postfix_expression ::= postfix_expression(pe) DecOp {
        // TODO
        return Err(ErrorKind::NotImplemented("post--"))
    }

    integer_expression ::= expression;

    function_call ::= function_call_generic;
    function_call_generic ::= function_call_header_with_parameters(h) RightParen {h}
    function_call_generic ::= function_call_header_no_parameters(h) RightParen {h}
    function_call_header_no_parameters ::= function_call_header(h) Void {h}
    function_call_header_no_parameters ::= function_call_header;
    function_call_header_with_parameters ::= function_call_header(h) assignment_expression(ae) {
        if let Expression::Compose{ty, components} = extra.context.expressions.get_mut(h) {
            components.push(ae.1);
        }
        //TODO: Call
        h
    }
    function_call_header_with_parameters ::= function_call_header_with_parameters(h) Comma assignment_expression(ae) {
        if let Expression::Compose{ty, components} = extra.context.expressions.get_mut(h) {
            components.push(ae.1);
        }
        //TODO: Call
        h
    }
    function_call_header ::= function_identifier(i) LeftParen {i}

    // Grammar Note: Constructors look like functions, but lexical analysis recognized most of them as
    // keywords. They are now recognized through “type_specifier”.
    // Methods (.length), subroutine array calls, and identifiers are recognized through postfix_expression.
    function_identifier ::= type_specifier(t) {
        if let Some(ty) = t {
            extra.context.expressions.append(Expression::Compose{
                ty,
                components: vec![],
            })
        } else {
            return Err(ErrorKind::NotImplemented("bad type ctor"))
        }
    }
    function_identifier ::= postfix_expression;

    unary_expression ::= postfix_expression;
    unary_expression ::= IncOp unary_expression {
        // TODO
        return Err(ErrorKind::NotImplemented("++pre"))
    }
    unary_expression ::= DecOp unary_expression {
        // TODO
        return Err(ErrorKind::NotImplemented("--pre"))
    }
    unary_expression ::= unary_operator unary_expression {
        // TODO
        return Err(ErrorKind::NotImplemented("unary_op"))
    }

    unary_operator ::= Plus;
    unary_operator ::= Dash;
    unary_operator ::= Bang;
    unary_operator ::= Tilde;
    multiplicative_expression ::= unary_expression;
    multiplicative_expression ::= multiplicative_expression(left) Star unary_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Multiply, left, right})
    }
    multiplicative_expression ::= multiplicative_expression(left) Slash unary_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Divide, left, right})
    }
    multiplicative_expression ::= multiplicative_expression(left) Percent unary_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Modulo, left, right})
    }
    additive_expression ::= multiplicative_expression;
    additive_expression ::= additive_expression(left) Plus multiplicative_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Add, left, right})
    }
    additive_expression ::= additive_expression(left) Dash multiplicative_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Subtract, left, right})
    }
    shift_expression ::= additive_expression;
    shift_expression ::= shift_expression(left) LeftOp additive_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::ShiftLeftLogical, left, right})
    }
    shift_expression ::= shift_expression(left) RightOp additive_expression(right) {
        //TODO: when to use ShiftRightArithmetic
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::ShiftRightLogical, left, right})
    }
    relational_expression ::= shift_expression;
    relational_expression ::= relational_expression(left) LeftAngle shift_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Less, left, right})
    }
    relational_expression ::= relational_expression(left) RightAngle shift_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Greater, left, right})
    }
    relational_expression ::= relational_expression(left) LeOp shift_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::LessEqual, left, right})
    }
    relational_expression ::= relational_expression(left) GeOp shift_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::GreaterEqual, left, right})
    }
    equality_expression ::= relational_expression;
    equality_expression ::= equality_expression(left) EqOp relational_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::Equal, left, right})
    }
    equality_expression ::= equality_expression(left) NeOp relational_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::NotEqual, left, right})
    }
    and_expression ::= equality_expression;
    and_expression ::= and_expression(left) Ampersand equality_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::And, left, right})
    }
    exclusive_or_expression ::= and_expression;
    exclusive_or_expression ::= exclusive_or_expression(left) Caret and_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::ExclusiveOr, left, right})
    }
    inclusive_or_expression ::= exclusive_or_expression;
    inclusive_or_expression ::= inclusive_or_expression(left) VerticalBar exclusive_or_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::InclusiveOr, left, right})
    }
    logical_and_expression ::= inclusive_or_expression;
    logical_and_expression ::= logical_and_expression(left) AndOp inclusive_or_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::LogicalAnd, left, right})
    }
    logical_xor_expression ::= logical_and_expression;
    logical_xor_expression ::= logical_xor_expression(left) XorOp logical_and_expression(right) {
        return Err(ErrorKind::NotImplemented("logical xor"))
        //TODO: naga doesn't have BinaryOperator::LogicalXor
        // extra.context.expressions.append(Expression::Binary{op: BinaryOperator::LogicalXor, left, right})
    }
    logical_or_expression ::= logical_xor_expression;
    logical_or_expression ::= logical_or_expression(left) OrOp logical_xor_expression(right) {
        extra.context.expressions.append(Expression::Binary{op: BinaryOperator::LogicalOr, left, right})
    }

    conditional_expression ::= logical_or_expression;
    conditional_expression ::= logical_or_expression Question expression Colon assignment_expression(ae) {
        //TODO: how to do ternary here in naga?
        return Err(ErrorKind::NotImplemented("ternary exp"))
    }

    assignment_expression ::= conditional_expression(ce) {
        (Statement::Empty, ce)
    }
    assignment_expression ::= unary_expression(pointer) assignment_operator(op) assignment_expression(value) {
        match op {
            BinaryOperator::Equal => (Statement::Store{pointer, value: value.1}, pointer),
            //TODO: op != Equal
        _ => {return Err(ErrorKind::NotImplemented("assign op"))}
        }
    }

    assignment_operator ::= Equal {BinaryOperator::Equal}
    assignment_operator ::= MulAssign {BinaryOperator::Multiply}
    assignment_operator ::= DivAssign {BinaryOperator::Divide}
    assignment_operator ::= ModAssign {BinaryOperator::Modulo}
    assignment_operator ::= AddAssign {BinaryOperator::Add}
    assignment_operator ::= SubAssign {BinaryOperator::Subtract}
    assignment_operator ::= LeftAssign {BinaryOperator::ShiftLeftLogical}
    assignment_operator ::= RightAssign {BinaryOperator::ShiftRightLogical}
    assignment_operator ::= AndAssign {BinaryOperator::And}
    assignment_operator ::= XorAssign {BinaryOperator::ExclusiveOr}
    assignment_operator ::= OrAssign {BinaryOperator::InclusiveOr}

    expression ::= assignment_expression(ae) {ae.0}
    expression ::= expression(e) Comma assignment_expression(ae) {
        match e {
            Statement::Block(mut b) => { b.push(ae.0); Statement::Block(b) },
            e => Statement::Block(vec![e, ae.0]),
        }
    }

    // statement
    statement ::= compound_statement(cs) {Statement::Block(cs)}
    statement ::= simple_statement;

    // Grammar Note: labeled statements for SWITCH only; 'goto' is not supported.
    //simple_statement ::= declaration_statement;
    simple_statement ::= expression_statement;
    //simple_statement ::= selection_statement;
    //simple_statement ::= switch_statement;
    //simple_statement ::= case_label;
    //simple_statement ::= iteration_statement;
    //simple_statement ::= jump_statement;

    compound_statement ::= LeftBrace RightBrace {vec![]}
    compound_statement ::= LeftBrace statement_list(sl) RightBrace {sl}

    compound_statement_no_new_scope ::= LeftBrace RightBrace  {vec![]}
    compound_statement_no_new_scope ::= LeftBrace statement_list(sl) RightBrace {sl}

    statement_list ::= statement(s) { vec![s] }
    statement_list ::= statement_list(mut ss) statement(s) { ss.push(s); ss }

    expression_statement ::= Semicolon  {Statement::Empty}
    expression_statement ::= expression(e) Semicolon  {e}



    // function
    function_prototype ::= function_declarator(f) RightParen {f}
    function_declarator ::= function_header;
    function_header ::= fully_specified_type(t) Identifier(n) LeftParen {
        Function {
            name: Some(n.1),
            parameter_types: vec![],
            return_type: t,
            global_usage: vec![],
            local_variables: Arena::<LocalVariable>::new(),
            expressions: Arena::<Expression>::new(),
            body: vec![],
        }
    }

    // type
    fully_specified_type ::= type_specifier;
    type_specifier ::= type_specifier_nonarray(t) {
        t.map(|t| {
            let name = t.name.clone();
            let handle = extra.types.fetch_or_append(t);
            if let Some(name) = name {
                extra.lookup_type.insert(name, handle);
            }
            handle
        })
    }

    type_specifier_nonarray ::= Void { None }
    type_specifier_nonarray ::= Vec4 {
        Some(Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Quad,
                kind: ScalarKind::Float,
                width: 4,
            }
        })
    }
    //TODO: remaining types

    // misc
    translation_unit ::= external_declaration;
    translation_unit ::= translation_unit external_declaration;

    external_declaration ::= function_definition(f) {
        let name = f.name.clone();
        let handle = extra.functions.append(f);
        if let Some(name) = name {
            extra.lookup_function.insert(name, handle);
        }
    }

    function_definition ::= function_prototype(mut f) compound_statement_no_new_scope(cs) {
        std::mem::swap(&mut f.expressions, &mut extra.context.expressions);
        std::mem::swap(&mut f.local_variables, &mut extra.context.local_variables);
        f.body = cs;
        f
    };
}

pub use parser::*;
