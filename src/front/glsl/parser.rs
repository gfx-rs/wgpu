#![allow(clippy::panic)]
use pomelo::pomelo;

pomelo! {
    //%verbose;
    %include {
        use super::super::{error::ErrorKind, token::*, ast::*};
        use crate::{proc::Typifier, Arena, BinaryOperator, Binding, Block, Constant,
            ConstantInner, EntryPoint, Expression, Function, GlobalVariable, Handle, Interpolation,
            LocalVariable, MemberOrigin, ScalarKind, Statement, StorageAccess,
            StorageClass, StructMember, Type, TypeInner};
    }
    %token #[derive(Debug)] #[cfg_attr(test, derive(PartialEq))] pub enum Token {};
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

    %type Unknown String;
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
    %type declaration_statement Statement;
    %type jump_statement Statement;

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

    // decalartions
    %type declaration VarDeclaration;
    %type init_declarator_list VarDeclaration;
    %type single_declaration VarDeclaration;
    %type layout_qualifier Binding;
    %type layout_qualifier_id_list Vec<(String, u32)>;
    %type layout_qualifier_id (String, u32);
    %type type_qualifier Vec<TypeQualifier>;
    %type single_type_qualifier TypeQualifier;
    %type storage_qualifier StorageClass;
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
                ExpressionRule::from_expression(
                    expression
                )
            },
            None => {
                return Err(ErrorKind::UnknownVariable(v.0, v.1));
            }
        }
    }

    primary_expression ::= variable_identifier;
    primary_expression ::= IntConstant(i) {
        let ty = extra.module.types.fetch_or_append(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Sint,
                width: 4,
            }
        });
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            ty,
            inner: ConstantInner::Sint(i.1)
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    // primary_expression ::= UintConstant;
    primary_expression ::= FloatConstant(f) {
        let ty = extra.module.types.fetch_or_append(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Float,
                width: 4,
            }
        });
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            ty,
            inner: ConstantInner::Float(f.1 as f64)
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    primary_expression ::= BoolConstant(b) {
        let ty = extra.module.types.fetch_or_append(Type {
            name: None,
            inner: TypeInner::Scalar {
                kind: ScalarKind::Bool,
                width: 4,
            }
        });
        let ch = extra.module.constants.fetch_or_append(Constant {
            name: None,
            specialization: None,
            ty,
            inner: ConstantInner::Bool(b.1)
        });
        ExpressionRule::from_expression(
            extra.context.expressions.append(Expression::Constant(ch))
        )
    }
    // primary_expression ::= DoubleConstant;
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
        ExpressionRule { expression, statements: e.statements }
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
        if let FunctionCallKind::TypeConstructor(ty) = fc.kind {
            let h = extra.context.expressions.append(Expression::Compose{
                ty,
                components: fc.args,
            });
            ExpressionRule{
                expression: h,
                statements: fc.statements,
            }
        } else {
            return Err(ErrorKind::NotImplemented("Function call"));
        }
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
        h.args.push(ae.expression);
        h.statements.extend(ae.statements);
        h
    }
    function_call_header_with_parameters ::= function_call_header_with_parameters(mut h) Comma assignment_expression(ae) {
        h.args.push(ae.expression);
        h.statements.extend(ae.statements);
        h
    }
    function_call_header ::= function_identifier(i) LeftParen {
        FunctionCall {
            kind: i,
            args: vec![],
            statements: vec![],
        }
    }

    // Grammar Note: Constructors look like functions, but lexical analysis recognized most of them as
    // keywords. They are now recognized through “type_specifier”.
    // Methods (.length), subroutine array calls, and identifiers are recognized through postfix_expression.
    function_identifier ::= type_specifier(t) {
        if let Some(ty) = t {
            FunctionCallKind::TypeConstructor(ty)
        } else {
            return Err(ErrorKind::NotImplemented("bad type ctor"))
        }
    }
    function_identifier ::= postfix_expression(e) {
        FunctionCallKind::Function(e.expression)
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
    unary_expression ::= unary_operator unary_expression {
        //TODO
        return Err(ErrorKind::NotImplemented("unary_op"))
    }

    unary_operator ::= Plus;
    unary_operator ::= Dash;
    unary_operator ::= Bang;
    unary_operator ::= Tilde;
    multiplicative_expression ::= unary_expression;
    multiplicative_expression ::= multiplicative_expression(left) Star unary_expression(right) {
        extra.binary_expr(BinaryOperator::Multiply, left, right)
    }
    multiplicative_expression ::= multiplicative_expression(left) Slash unary_expression(right) {
        extra.binary_expr(BinaryOperator::Divide, left, right)
    }
    multiplicative_expression ::= multiplicative_expression(left) Percent unary_expression(right) {
        extra.binary_expr(BinaryOperator::Modulo, left, right)
    }
    additive_expression ::= multiplicative_expression;
    additive_expression ::= additive_expression(left) Plus multiplicative_expression(right) {
        extra.binary_expr(BinaryOperator::Add, left, right)
    }
    additive_expression ::= additive_expression(left) Dash multiplicative_expression(right) {
        extra.binary_expr(BinaryOperator::Subtract, left, right)
    }
    shift_expression ::= additive_expression;
    shift_expression ::= shift_expression(left) LeftOp additive_expression(right) {
        extra.binary_expr(BinaryOperator::ShiftLeftLogical, left, right)
    }
    shift_expression ::= shift_expression(left) RightOp additive_expression(right) {
        //TODO: when to use ShiftRightArithmetic
        extra.binary_expr(BinaryOperator::ShiftRightLogical, left, right)
    }
    relational_expression ::= shift_expression;
    relational_expression ::= relational_expression(left) LeftAngle shift_expression(right) {
        extra.binary_expr(BinaryOperator::Less, left, right)
    }
    relational_expression ::= relational_expression(left) RightAngle shift_expression(right) {
        extra.binary_expr(BinaryOperator::Greater, left, right)
    }
    relational_expression ::= relational_expression(left) LeOp shift_expression(right) {
        extra.binary_expr(BinaryOperator::LessEqual, left, right)
    }
    relational_expression ::= relational_expression(left) GeOp shift_expression(right) {
        extra.binary_expr(BinaryOperator::GreaterEqual, left, right)
    }
    equality_expression ::= relational_expression;
    equality_expression ::= equality_expression(left) EqOp relational_expression(right) {
        extra.binary_expr(BinaryOperator::Equal, left, right)
    }
    equality_expression ::= equality_expression(left) NeOp relational_expression(right) {
        extra.binary_expr(BinaryOperator::NotEqual, left, right)
    }
    and_expression ::= equality_expression;
    and_expression ::= and_expression(left) Ampersand equality_expression(right) {
        extra.binary_expr(BinaryOperator::And, left, right)
    }
    exclusive_or_expression ::= and_expression;
    exclusive_or_expression ::= exclusive_or_expression(left) Caret and_expression(right) {
        extra.binary_expr(BinaryOperator::ExclusiveOr, left, right)
    }
    inclusive_or_expression ::= exclusive_or_expression;
    inclusive_or_expression ::= inclusive_or_expression(left) VerticalBar exclusive_or_expression(right) {
        extra.binary_expr(BinaryOperator::InclusiveOr, left, right)
    }
    logical_and_expression ::= inclusive_or_expression;
    logical_and_expression ::= logical_and_expression(left) AndOp inclusive_or_expression(right) {
        extra.binary_expr(BinaryOperator::LogicalAnd, left, right)
    }
    logical_xor_expression ::= logical_and_expression;
    logical_xor_expression ::= logical_xor_expression(left) XorOp logical_and_expression(right) {
        return Err(ErrorKind::NotImplemented("logical xor"))
        //TODO: naga doesn't have BinaryOperator::LogicalXor
        // extra.context.expressions.append(Expression::Binary{op: BinaryOperator::LogicalXor, left, right})
    }
    logical_or_expression ::= logical_xor_expression;
    logical_or_expression ::= logical_or_expression(left) OrOp logical_xor_expression(right) {
        extra.binary_expr(BinaryOperator::LogicalOr, left, right)
    }

    conditional_expression ::= logical_or_expression;
    conditional_expression ::= logical_or_expression Question expression Colon assignment_expression(ae) {
        //TODO: how to do ternary here in naga?
        return Err(ErrorKind::NotImplemented("ternary exp"))
    }

    assignment_expression ::= conditional_expression;
    assignment_expression ::= unary_expression(mut pointer) assignment_operator(op) assignment_expression(value) {
        match op {
            BinaryOperator::Equal => {
                pointer.statements.extend(value.statements);
                pointer.statements.push(Statement::Store{
                    pointer: pointer.expression,
                    value: value.expression
                });
                pointer
            },
            //TODO: op != Equal
            _ => {return Err(ErrorKind::NotImplemented("assign op"))}
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
        BinaryOperator::ShiftLeftLogical
    }
    assignment_operator ::= RightAssign {
        BinaryOperator::ShiftRightLogical
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
        ExpressionRule{
            expression: e.expression,
            statements: ae.statements,
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
        idl
    }

    declaration ::= type_qualifier(t) Identifier(i) LeftBrace
        struct_declaration_list(sdl) RightBrace Semicolon {
        VarDeclaration{
            type_qualifiers: t,
            ids_initializers: vec![(None, None)],
            ty: extra.module.types.fetch_or_append(Type{
                name: Some(i.1),
                inner: TypeInner::Struct {
                    members: sdl
                }
            }),
        }
    }

    declaration ::= type_qualifier(t) Identifier(i1) LeftBrace
        struct_declaration_list(sdl) RightBrace Identifier(i2) Semicolon {
        VarDeclaration{
            type_qualifiers: t,
            ids_initializers: vec![(Some(i2.1), None)],
            ty: extra.module.types.fetch_or_append(Type{
                name: Some(i1.1),
                inner: TypeInner::Struct {
                    members: sdl
                }
            }),
        }
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
        let ty = t.1.ok_or(ErrorKind::SemanticError("Empty type for declaration"))?;

        VarDeclaration{
            type_qualifiers: t.0,
            ids_initializers: vec![],
            ty,
        }
    }
    single_declaration ::= fully_specified_type(t) Identifier(i) {
        let ty = t.1.ok_or(ErrorKind::SemanticError("Empty type for declaration"))?;

        VarDeclaration{
            type_qualifiers: t.0,
            ids_initializers: vec![(Some(i.1), None)],
            ty,
        }
    }
    // single_declaration ::= fully_specified_type Identifier array_specifier;
    // single_declaration ::= fully_specified_type Identifier array_specifier Equal initializer;
    single_declaration ::= fully_specified_type(t) Identifier(i) Equal initializer(init) {
        let ty = t.1.ok_or(ErrorKind::SemanticError("Empty type for declaration"))?;

        VarDeclaration{
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
            Binding::Location(loc)
        } else if let Some(&(_, binding)) = l.iter().find(|&q| q.0.as_str() == "binding") {
            let group = if let Some(&(_, set)) = l.iter().find(|&q| q.0.as_str() == "set") {
                set
            } else {
                0
            };
            Binding::Resource{ group, binding }
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
        TypeQualifier::StorageClass(s)
    }
    single_type_qualifier ::= layout_qualifier(l) {
        TypeQualifier::Binding(l)
    }
    // single_type_qualifier ::= precision_qualifier;
    single_type_qualifier ::= interpolation_qualifier(i) {
        TypeQualifier::Interpolation(i)
    }
    // single_type_qualifier ::= invariant_qualifier;
    // single_type_qualifier ::= precise_qualifier;

    storage_qualifier ::= Const {
        StorageClass::Constant
    }
    // storage_qualifier ::= InOut;
    storage_qualifier ::= In {
        StorageClass::Input
    }
    storage_qualifier ::= Out {
        StorageClass::Output
    }
    // storage_qualifier ::= Centroid;
    // storage_qualifier ::= Patch;
    // storage_qualifier ::= Sample;
    storage_qualifier ::= Uniform {
        StorageClass::Uniform
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
                members: vec![]
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
            sdl.iter().map(|name| StructMember{
                name: Some(name.clone()),
                origin: MemberOrigin::Empty,
                ty,
            }).collect()
        } else {
            return Err(ErrorKind::SemanticError("Struct member can't be void"))
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
        for (id, initializer) in d.ids_initializers {
            let id = id.ok_or(ErrorKind::SemanticError("local var must be named"))?;
            // check if already declared in current scope
            #[cfg(feature = "glsl-validate")]
            {
                if extra.context.lookup_local_var_current_scope(&id).is_some() {
                    return Err(ErrorKind::VariableAlreadyDeclared(id))
                }
            }
            let localVar = extra.context.local_variables.append(
                LocalVariable {
                    name: Some(id.clone()),
                    ty: d.ty,
                    init: initializer.map(|i| {
                        statements.extend(i.statements);
                        i.expression
                    }),
                }
            );
            let exp = extra.context.expressions.append(Expression::LocalVariable(localVar));
            extra.context.add_local_var(id, exp);
        }
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

    // Grammar Note: labeled statements for SWITCH only; 'goto' is not supported.
    simple_statement ::= declaration_statement;
    simple_statement ::= expression_statement;
    //simple_statement ::= selection_statement;
    //simple_statement ::= switch_statement;
    //simple_statement ::= case_label;
    //simple_statement ::= iteration_statement;
    simple_statement ::= jump_statement;

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
        vec![s]
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
        // prelude, add global var expressions
        for (var_handle, var) in extra.module.global_variables.iter() {
            if let Some(name) = var.name.as_ref() {
                let exp = extra.context.expressions.append(
                    Expression::GlobalVariable(var_handle)
                );
                extra.context.lookup_global_var_exps.insert(name.clone(), exp);
            } else {
                let ty = &extra.module.types[var.ty];
                // anonymous structs
                if let TypeInner::Struct { members } = &ty.inner {
                    let base = extra.context.expressions.append(
                        Expression::GlobalVariable(var_handle)
                    );
                    for (idx, member) in members.iter().enumerate() {
                        if let Some(name) = member.name.as_ref() {
                            let exp = extra.context.expressions.append(
                                Expression::AccessIndex{
                                    base,
                                    index: idx as u32,
                                }
                            );
                            extra.context.lookup_global_var_exps.insert(name.clone(), exp);
                        }
                    }
                }
            }
        }
        f
    }
    function_declarator ::= function_header;
    function_header ::= fully_specified_type(t) Identifier(n) LeftParen {
        Function {
            name: Some(n.1),
            parameter_types: vec![],
            return_type: t.1,
            global_usage: vec![],
            local_variables: Arena::<LocalVariable>::new(),
            expressions: Arena::<Expression>::new(),
            body: vec![],
        }
    }

    jump_statement ::= Continue Semicolon {
        Statement::Continue
    }
    jump_statement ::= Break Semicolon {
        Statement::Break
    }
    jump_statement ::= Return Semicolon {
        Statement::Return{ value: None }
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
        if f.name == extra.entry {
            let name = extra.entry.take().unwrap();
            extra.module.entry_points.insert(
                (extra.shader_stage, name),
                EntryPoint {
                    early_depth_test: None,
                    workgroup_size: [0; 3], //TODO
                    function: f,
                },
            );
        } else {
            let name = f.name.clone().unwrap();
            let handle = extra.module.functions.append(f);
            extra.lookup_function.insert(name, handle);
        }
    }
    external_declaration ::= declaration(d) {
        let class = d.type_qualifiers.iter().find_map(|tq| {
            if let TypeQualifier::StorageClass(sc) = tq { Some(*sc) } else { None }
        }).ok_or(ErrorKind::SemanticError("Missing storage class for global var"))?;

        let binding = d.type_qualifiers.iter().find_map(|tq| {
            if let TypeQualifier::Binding(b) = tq { Some(b.clone()) } else { None }
        });

        let interpolation = d.type_qualifiers.iter().find_map(|tq| {
            if let TypeQualifier::Interpolation(i) = tq { Some(*i) } else { None }
        });

        for (id, initializer) in d.ids_initializers {
            let h = extra.module.global_variables.fetch_or_append(
                GlobalVariable {
                    name: id.clone(),
                    class,
                    binding: binding.clone(),
                    ty: d.ty,
                    interpolation,
                    storage_access: StorageAccess::empty(), //TODO
                },
            );
            if let Some(id) = id {
                extra.lookup_global_variables.insert(id, h);
            }
        }
    }

    function_definition ::= function_prototype(mut f) compound_statement_no_new_scope(cs) {
        std::mem::swap(&mut f.expressions, &mut extra.context.expressions);
        std::mem::swap(&mut f.local_variables, &mut extra.context.local_variables);
        extra.context.clear_scopes();
        extra.context.lookup_global_var_exps.clear();
        extra.context.typifier = Typifier::new();
        f.body = cs;
        f.fill_global_use(&extra.module.global_variables);
        f
    };
}

pub use parser::*;
