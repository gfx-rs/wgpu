use naga::{valid, Expression, Function, Module, Scalar, Type, TypeInner};

#[test]
fn emit_atomic_result() {
    // We want to ensure that the *only* problem with the code is the
    // use of an `Emit` statement instead of an `Atomic` statement. So
    // validate two versions of the module varying only in that
    // aspect.
    //
    // Looking at uses of the `atomic` makes it easy to identify the
    // differences between the two variants.
    fn variant(
        atomic: bool,
    ) -> Result<naga::valid::ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_u32 = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            span,
        );
        let ty_atomic_u32 = module.types.insert(
            Type {
                name: Some("atomic<u32>".into()),
                inner: TypeInner::Atomic(Scalar::U32),
            },
            span,
        );
        let var_atomic = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("atomic_global".into()),
                space: naga::AddressSpace::WorkGroup,
                binding: None,
                ty: ty_atomic_u32,
                init: None,
            },
            span,
        );

        let mut fun = Function::default();
        let ex_global = fun
            .expressions
            .append(Expression::GlobalVariable(var_atomic), span);
        let ex_42 = fun
            .expressions
            .append(Expression::Literal(naga::Literal::U32(42)), span);
        let ex_result = fun.expressions.append(
            Expression::AtomicResult {
                ty: ty_u32,
                comparison: false,
            },
            span,
        );

        if atomic {
            fun.body.push(
                naga::Statement::Atomic {
                    pointer: ex_global,
                    fun: naga::AtomicFunction::Add,
                    value: ex_42,
                    result: ex_result,
                },
                span,
            );
        } else {
            fun.body.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(ex_result, ex_result)),
                span,
            );
        }

        module.functions.append(fun, span);

        valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module)
    }

    variant(true).expect("module should validate");
    assert!(variant(false).is_err());
}

#[test]
fn emit_call_result() {
    // We want to ensure that the *only* problem with the code is the
    // use of an `Emit` statement instead of a `Call` statement. So
    // validate two versions of the module varying only in that
    // aspect.
    //
    // Looking at uses of the `call` makes it easy to identify the
    // differences between the two variants.
    fn variant(
        call: bool,
    ) -> Result<naga::valid::ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_u32 = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            span,
        );

        let mut fun_callee = Function {
            result: Some(naga::FunctionResult {
                ty: ty_u32,
                binding: None,
            }),
            ..Function::default()
        };
        let ex_42 = fun_callee
            .expressions
            .append(Expression::Literal(naga::Literal::U32(42)), span);
        fun_callee
            .body
            .push(naga::Statement::Return { value: Some(ex_42) }, span);
        let fun_callee = module.functions.append(fun_callee, span);

        let mut fun_caller = Function::default();
        let ex_result = fun_caller
            .expressions
            .append(Expression::CallResult(fun_callee), span);

        if call {
            fun_caller.body.push(
                naga::Statement::Call {
                    function: fun_callee,
                    arguments: vec![],
                    result: Some(ex_result),
                },
                span,
            );
        } else {
            fun_caller.body.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(ex_result, ex_result)),
                span,
            );
        }

        module.functions.append(fun_caller, span);

        valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module)
    }

    variant(true).expect("should validate");
    assert!(variant(false).is_err());
}

#[test]
fn emit_workgroup_uniform_load_result() {
    // We want to ensure that the *only* problem with the code is the
    // use of an `Emit` statement instead of an `Atomic` statement. So
    // validate two versions of the module varying only in that
    // aspect.
    //
    // Looking at uses of the `wg_load` makes it easy to identify the
    // differences between the two variants.
    fn variant(
        wg_load: bool,
    ) -> Result<naga::valid::ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_u32 = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            span,
        );
        let var_workgroup = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("workgroup_global".into()),
                space: naga::AddressSpace::WorkGroup,
                binding: None,
                ty: ty_u32,
                init: None,
            },
            span,
        );

        let mut fun = Function::default();
        let ex_global = fun
            .expressions
            .append(Expression::GlobalVariable(var_workgroup), span);
        let ex_result = fun
            .expressions
            .append(Expression::WorkGroupUniformLoadResult { ty: ty_u32 }, span);

        if wg_load {
            fun.body.push(
                naga::Statement::WorkGroupUniformLoad {
                    pointer: ex_global,
                    result: ex_result,
                },
                span,
            );
        } else {
            fun.body.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(ex_result, ex_result)),
                span,
            );
        }

        module.functions.append(fun, span);

        valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module)
    }

    variant(true).expect("module should validate");
    assert!(variant(false).is_err());
}

/// Validation should reject expressions that refer to un-emitted
/// subexpressions.
#[test]
fn emit_subexpressions() {
    fn variant(
        emit: bool,
    ) -> Result<naga::valid::ModuleInfo, naga::WithSpan<naga::valid::ValidationError>> {
        let span = naga::Span::default();
        let mut module = Module::default();
        let ty_u32 = module.types.insert(
            Type {
                name: Some("u32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            span,
        );
        let var_private = module.global_variables.append(
            naga::GlobalVariable {
                name: Some("private".into()),
                space: naga::AddressSpace::Private,
                binding: None,
                ty: ty_u32,
                init: None,
            },
            span,
        );

        let mut fun = Function::default();

        // These expressions are pre-emit, so they don't need to be
        // covered by any `Emit` statement.
        let ex_var = fun
            .expressions
            .append(Expression::GlobalVariable(var_private), span);

        // This expression is neither pre-emit nor used directly by a
        // statement. We want to test whether validation notices when
        // it is not covered by an `Emit` statement.
        let ex_add = fun
            .expressions
            .append(Expression::Load { pointer: ex_var }, span);

        // This expression is used directly by the statement, so if
        // it's not covered by an `Emit`, then validation will catch
        // that.
        let ex_mul = fun.expressions.append(
            Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                left: ex_add,
                right: ex_add,
            },
            span,
        );

        if emit {
            // This `Emit` covers all expressions properly.
            fun.body.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(ex_add, ex_mul)),
                span,
            );
        } else {
            // This `Emit` covers `ex_mul` but not its subexpression `ex_add`.
            fun.body.push(
                naga::Statement::Emit(naga::Range::new_from_bounds(ex_mul, ex_mul)),
                span,
            );
        }
        fun.body.push(
            naga::Statement::Store {
                pointer: ex_var,
                value: ex_mul,
            },
            span,
        );

        module.functions.append(fun, span);

        let result = valid::Validator::new(
            valid::ValidationFlags::default(),
            valid::Capabilities::all(),
        )
        .validate(&module);

        if let Ok(ref info) = result {
            let (source, _translation_info) =
                naga::back::msl::write_string(&module, info, &<_>::default(), &<_>::default())
                    .expect("generating MSL failed");
            eprintln!("MSL output:\n{source}");
        }

        result
    }

    variant(true).expect("module should validate");
    variant(false).expect_err("validation should notice un-emitted subexpression");
}
