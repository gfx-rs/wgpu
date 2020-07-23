#![allow(unused_braces)]
#![allow(clippy::panic)]
use pomelo::pomelo;

pomelo! {
    //%verbose;
    %include {
        use super::super::{error::ErrorKind, token::*, ast::*};
        use crate::{Arena, Binding, BuiltIn, Constant, ConstantInner, Expression,
            Function, GlobalVariable, Handle, LocalVariable, ScalarKind,
            ShaderStage, StorageClass, Type, TypeInner, VectorSize};
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
    %type IntConstant i64;
    %type UintConstant u64;
    %type FloatConstant f32;
    %type BoolConstant bool;
    %type DoubleConstant f64;
    %type String String;
    %type arg_list Vec<String>;

    %type function_prototype Function;
    %type function_declarator Function;
    %type function_header Function;

    %type function_definition Function;

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
            extra.global_variables.fetch_or_append(
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
    }
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

    function_definition ::= function_prototype(f) compound_statement_no_new_scope {
        f
    };
}

pub use parser::*;
