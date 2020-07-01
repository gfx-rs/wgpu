#![allow(unused_braces)]
use pomelo::pomelo;

pomelo! {
    //%verbose;
    %include {
        use super::super::{error::ErrorKind, token::*};
        use crate::{Arena, Expression, Function, LocalVariable, Module};
    }
    %token #[derive(Debug)] pub enum Token {};
    %extra_argument Module;
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

    %type Identifier String;
    %type IntConstant i64;
    %type UintConstant u64;
    %type FloatConstant f32;
    %type BoolConstant bool;
    %type DoubleConstant f64;
    %type String String;
    %type arg_list Vec<String>;
    %type function_definition Function;

    root ::= version_pragma translation_unit;
    version_pragma ::= Version IntConstant Identifier?;

    // expression
    variable_identifier ::= Identifier;

    primary_expression ::= variable_identifier;
    primary_expression ::= IntConstant;
    primary_expression ::= UintConstant;
    primary_expression ::= FloatConstant;
    primary_expression ::= BoolConstant;
    primary_expression ::= DoubleConstant;
    primary_expression ::= LeftParen expression RightParen;

    postfix_expression ::= primary_expression;
    postfix_expression ::= postfix_expression LeftBracket integer_expression RightBracket;
    postfix_expression ::= function_call;
    postfix_expression ::= postfix_expression Dot FieldSelection;
    postfix_expression ::= postfix_expression IncOp;
    postfix_expression ::= postfix_expression DecOp;

    integer_expression ::= expression;

    function_call ::= function_call_generic;
    function_call_generic ::= function_call_header_with_parameters RightParen;
    function_call_generic ::= function_call_header_no_parameters RightParen;
    function_call_header_no_parameters ::= function_call_header Void;
    function_call_header_no_parameters ::= function_call_header;
    function_call_header_with_parameters ::= function_call_header assignment_expression;
    function_call_header_with_parameters ::= function_call_header_with_parameters Comma assignment_expression;
    function_call_header ::= function_identifier LeftParen;

    // Grammar Note: Constructors look like functions, but lexical analysis recognized most of them as
    // keywords. They are now recognized through “type_specifier”.
    // Methods (.length), subroutine array calls, and identifiers are recognized through postfix_expression.
    function_identifier ::= type_specifier;
    function_identifier ::= postfix_expression;

    unary_expression ::= postfix_expression;
    unary_expression ::= IncOp unary_expression;
    unary_expression ::= DecOp unary_expression;
    unary_expression ::= unary_operator unary_expression;
    unary_operator ::= Plus;
    unary_operator ::= Dash;
    unary_operator ::= Bang;
    unary_operator ::= Tilde;
    multiplicative_expression ::= unary_expression;
    multiplicative_expression ::= multiplicative_expression Star unary_expression;
    multiplicative_expression ::= multiplicative_expression Slash unary_expression;
    multiplicative_expression ::= multiplicative_expression Percent unary_expression;
    additive_expression ::= multiplicative_expression;
    additive_expression ::= additive_expression Plus multiplicative_expression;
    additive_expression ::= additive_expression Dash multiplicative_expression;
    shift_expression ::= additive_expression;
    shift_expression ::= shift_expression LeftOp additive_expression;
    shift_expression ::= shift_expression RightOp additive_expression;
    relational_expression ::= shift_expression;
    relational_expression ::= relational_expression LeftAngle shift_expression;
    relational_expression ::= relational_expression RightAngle shift_expression;
    relational_expression ::= relational_expression LeOp shift_expression;
    relational_expression ::= relational_expression GeOp shift_expression;
    equality_expression ::= relational_expression;
    equality_expression ::= equality_expression EqOp relational_expression;
    equality_expression ::= equality_expression NeOp relational_expression;
    and_expression ::= equality_expression;
    and_expression ::= and_expression Ampersand equality_expression;
    exclusive_or_expression ::= and_expression;
    exclusive_or_expression ::= exclusive_or_expression Caret and_expression;
    inclusive_or_expression ::= exclusive_or_expression;
    inclusive_or_expression ::= inclusive_or_expression VerticalBar exclusive_or_expression;
    logical_and_expression ::= inclusive_or_expression;
    logical_and_expression ::= logical_and_expression AndOp inclusive_or_expression;
    logical_xor_expression ::= logical_and_expression;
    logical_xor_expression ::= logical_xor_expression XorOp logical_and_expression;
    logical_or_expression ::= logical_xor_expression;
    logical_or_expression ::= logical_or_expression OrOp logical_xor_expression;

    conditional_expression ::= logical_or_expression;
    conditional_expression ::= logical_or_expression Question expression Colon assignment_expression;

    assignment_expression ::= conditional_expression;
    assignment_expression ::= unary_expression assignment_operator assignment_expression;

    assignment_operator ::= Equal;
    assignment_operator ::= MulAssign;
    assignment_operator ::= DivAssign;
    assignment_operator ::= ModAssign;
    assignment_operator ::= AddAssign;
    assignment_operator ::= SubAssign;
    assignment_operator ::= LeftAssign;
    assignment_operator ::= RightAssign;
    assignment_operator ::= AndAssign;
    assignment_operator ::= XorAssign;
    assignment_operator ::= OrAssign;

    expression ::= assignment_expression;
    expression ::= expression Comma assignment_expression;

    // statement
    statement ::= compound_statement;
    statement ::= simple_statement;

    // Grammar Note: labeled statements for SWITCH only; 'goto' is not supported.
    //simple_statement ::= declaration_statement;
    simple_statement ::= expression_statement;

    compound_statement ::= LeftBrace RightBrace;
    compound_statement ::= LeftBrace statement_list RightBrace;

    compound_statement_no_new_scope ::= LeftBrace RightBrace;
    compound_statement_no_new_scope ::= LeftBrace statement_list RightBrace;

    statement_list ::= statement(s) { /*vec![s]*/ }
    statement_list ::= statement_list/*(mut ss)*/ statement(s) { /*ss.push(s); ss*/ }

    expression_statement ::= Semicolon;
    expression_statement ::= expression Semicolon;



    // function
    function_prototype ::= function_declarator RightParen;
    function_declarator ::= function_header;
    function_header ::= fully_specified_type Identifier LeftParen;

    // type
    fully_specified_type ::= type_specifier;
    type_specifier ::= type_specifier_nonarray;

    type_specifier_nonarray ::= Void;
    type_specifier_nonarray ::= Vec4;
    //TODO: remaining types

    // misc
    translation_unit ::= external_declaration;
    translation_unit ::= translation_unit external_declaration;

    external_declaration ::= function_definition(f) { extra.functions.append(f); }

    function_definition ::= function_prototype compound_statement_no_new_scope {
        Function {
            name: Some(String::from("main")),
            parameter_types: vec![],
            return_type: None,
            global_usage: vec![],
            local_variables: Arena::<LocalVariable>::new(),
            expressions: Arena::<Expression>::new(),
            body: vec![],
        }
    };
}

pub use parser::*;
