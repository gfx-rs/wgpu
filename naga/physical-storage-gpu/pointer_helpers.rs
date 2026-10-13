use naga::{Expression as E, Span, Statement as S, TypeInner as T};

pub fn helper(
    m: &mut naga::Module,
    pointer: naga::Handle<naga::Type>,
) -> naga::Handle<naga::Function> {
    let uint64 = m.types.insert(
        naga::Type {
            name: None,
            inner: T::Scalar(naga::Scalar::U64),
        },
        Span::UNDEFINED,
    );
    let mut f = naga::Function {
        name: Some("pointer_helper".into()),
        ..Default::default()
    };
    f.arguments.push(naga::FunctionArgument {
        name: None,
        ty: pointer,
        binding: None,
        immutable_pointee: false,
    });
    f.result = Some(naga::FunctionResult {
        ty: pointer,
        binding: None,
    });
    let p = f
        .expressions
        .append(E::FunctionArgument(0), Span::UNDEFINED);
    let one = f
        .expressions
        .append(E::Literal(naga::Literal::U64(1)), Span::UNDEFINED);
    let minus_one = f
        .expressions
        .append(E::Literal(naga::Literal::I64(-1)), Span::UNDEFINED);
    let mask = f
        .expressions
        .append(E::Literal(naga::Literal::U64(8)), Span::UNDEFINED);
    let zero = f
        .expressions
        .append(E::Literal(naga::Literal::U64(0)), Span::UNDEFINED);
    let local = f.local_variables.append(
        naga::LocalVariable {
            name: None,
            ty: pointer,
            init: None,
        },
        Span::UNDEFINED,
    );
    let place = f
        .expressions
        .append(E::LocalVariable(local), Span::UNDEFINED);
    let address = f.expressions.append(
        E::PointerCast {
            expr: p,
            ty: uint64,
        },
        Span::UNDEFINED,
    );
    let bits = f.expressions.append(
        E::Binary {
            op: naga::BinaryOperator::And,
            left: address,
            right: mask,
        },
        Span::UNDEFINED,
    );
    let condition = f.expressions.append(
        E::Binary {
            op: naga::BinaryOperator::NotEqual,
            left: bits,
            right: zero,
        },
        Span::UNDEFINED,
    );
    let restored = f.expressions.append(
        E::PointerCast {
            expr: address,
            ty: pointer,
        },
        Span::UNDEFINED,
    );
    let next = f.expressions.append(
        E::PointerOffset {
            pointer: restored,
            offset: one,
        },
        Span::UNDEFINED,
    );
    let back = f.expressions.append(
        E::PointerOffset {
            pointer: next,
            offset: minus_one,
        },
        Span::UNDEFINED,
    );
    let select = f.expressions.append(
        E::Select {
            condition,
            accept: p,
            reject: back,
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(address, select)),
        Span::UNDEFINED,
    );
    let branch = |value| {
        let mut b = naga::Block::new();
        b.push(
            S::Store {
                pointer: place,
                value,
            },
            Span::UNDEFINED,
        );
        b
    };
    f.body.push(
        S::If {
            condition,
            accept: branch(back),
            reject: branch(select),
        },
        Span::UNDEFINED,
    );
    let result = f
        .expressions
        .append(E::Load { pointer: place }, Span::UNDEFINED);
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(result, result)),
        Span::UNDEFINED,
    );
    f.body.push(
        S::Return {
            value: Some(result),
        },
        Span::UNDEFINED,
    );
    m.functions.append(f, Span::UNDEFINED)
}

fn emit(f: &mut naga::Function, expr: E) -> naga::Handle<E> {
    let pre = expr.needs_pre_emit();
    let h = f.expressions.append(expr, Span::UNDEFINED);
    if !pre {
        f.body
            .push(S::Emit(naga::Range::new_from_bounds(h, h)), Span::UNDEFINED);
    }
    h
}

/// Runtime span bounds and address-overflow checks, followed by aliased helper writes.
pub fn span_module(scalar_layout: bool) -> naga::Module {
    span_module_with_atomics(scalar_layout, false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanMode {
    Plain,
    Atomic,
    Contended,
    Feedback,
    Matrix,
    RuntimeArray,
    MatrixLayout { row_major: bool, padded: bool },
}

pub fn span_module_with_atomics(scalar_layout: bool, atomic: bool) -> naga::Module {
    span_module_with_mode(
        scalar_layout,
        if atomic {
            SpanMode::Atomic
        } else {
            SpanMode::Plain
        },
    )
}

pub fn span_module_with_mode(scalar_layout: bool, mode: SpanMode) -> naga::Module {
    let atomic = matches!(
        mode,
        SpanMode::Atomic | SpanMode::Contended | SpanMode::Feedback
    );
    let matrix_mode = mode == SpanMode::Matrix;
    let layout_mode = matches!(mode, SpanMode::MatrixLayout { .. });
    let mut m = naga::Module::default();
    let ty = |m: &mut naga::Module, inner| {
        m.types
            .insert(naga::Type { name: None, inner }, Span::UNDEFINED)
    };
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let u64_ty = ty(&mut m, T::Scalar(naga::Scalar::U64));
    let vector = ty(
        &mut m,
        T::Vector {
            size: naga::VectorSize::Tri,
            scalar: naga::Scalar::U32,
        },
    );
    let element_type = if atomic {
        ty(&mut m, T::Atomic(naga::Scalar::U32))
    } else {
        uint
    };
    let base = if layout_mode {
        ty(
            &mut m,
            T::Array {
                base: uint,
                size: naga::ArraySize::Constant(core::num::NonZeroU32::new(8).unwrap()),
                stride: 4,
            },
        )
    } else if matrix_mode {
        ty(
            &mut m,
            T::Matrix {
                columns: naga::VectorSize::Bi,
                rows: naga::VectorSize::Bi,
                scalar: naga::Scalar::F32,
            },
        )
    } else if scalar_layout {
        ty(
            &mut m,
            T::Struct {
                members: vec![
                    naga::StructMember {
                        access: None,
                        name: None,
                        ty: element_type,
                        binding: None,
                        offset: 0,
                    },
                    naga::StructMember {
                        access: None,
                        name: None,
                        ty: vector,
                        binding: None,
                        offset: 4,
                    },
                ],
                span: 16,
            },
        )
    } else {
        element_type
    };
    let ptr = ty(
        &mut m,
        T::Pointer {
            base,
            space: naga::AddressSpace::PhysicalStorage,
        },
    );
    let identity = helper(&mut m, ptr);
    let scalar_ptr = ty(
        &mut m,
        T::Pointer {
            base: element_type,
            space: naga::AddressSpace::PhysicalStorage,
        },
    );
    let leaf_identity = helper(&mut m, scalar_ptr);
    let mut update = naga::Function {
        name: Some("aliased_update".into()),
        ..Default::default()
    };
    for ty in [ptr, ptr, uint] {
        update.arguments.push(naga::FunctionArgument {
            name: None,
            ty,
            binding: None,
            immutable_pointee: false,
        });
    }
    let p = emit(&mut update, E::FunctionArgument(0));
    let q = emit(&mut update, E::FunctionArgument(1));
    let (p, q) = if matrix_mode {
        let value = emit(&mut update, E::Load { pointer: p });
        let column = emit(
            &mut update,
            E::AccessIndex {
                base: value,
                index: 0,
            },
        );
        let cell = emit(
            &mut update,
            E::AccessIndex {
                base: column,
                index: 0,
            },
        );
        let bits = emit(
            &mut update,
            E::As {
                expr: cell,
                kind: naga::ScalarKind::Uint,
                convert: None,
            },
        );
        let added = emit(&mut update, E::FunctionArgument(2));
        let changed = emit(
            &mut update,
            E::Binary {
                op: naga::BinaryOperator::Add,
                left: bits,
                right: added,
            },
        );
        let as_float = emit(
            &mut update,
            E::As {
                expr: changed,
                kind: naga::ScalarKind::Float,
                convert: None,
            },
        );
        let mut cols = Vec::new();
        let col_ty = ty(
            &mut m,
            T::Vector {
                size: naga::VectorSize::Bi,
                scalar: naga::Scalar::F32,
            },
        );
        let other = emit(
            &mut update,
            E::AccessIndex {
                base: column,
                index: 1,
            },
        );
        cols.push(emit(
            &mut update,
            E::Compose {
                ty: col_ty,
                components: vec![as_float, other],
            },
        ));
        cols.push(emit(
            &mut update,
            E::AccessIndex {
                base: value,
                index: 1,
            },
        ));
        let composed = emit(
            &mut update,
            E::Compose {
                ty: base,
                components: cols,
            },
        );
        update.body.push(
            S::Store {
                pointer: q,
                value: composed,
            },
            Span::UNDEFINED,
        );
        let pa = emit(
            &mut update,
            E::PointerCast {
                expr: p,
                ty: u64_ty,
            },
        );
        let qa = emit(
            &mut update,
            E::PointerCast {
                expr: q,
                ty: u64_ty,
            },
        );
        (
            emit(
                &mut update,
                E::PointerCast {
                    expr: pa,
                    ty: scalar_ptr,
                },
            ),
            emit(
                &mut update,
                E::PointerCast {
                    expr: qa,
                    ty: scalar_ptr,
                },
            ),
        )
    } else {
        (p, q)
    };

    let p = if scalar_layout && !matrix_mode {
        emit(&mut update, E::AccessIndex { base: p, index: 0 })
    } else {
        p
    };
    let q = if scalar_layout && !matrix_mode {
        emit(&mut update, E::AccessIndex { base: q, index: 0 })
    } else {
        q
    };
    let returned = update
        .expressions
        .append(E::CallResult(leaf_identity), Span::UNDEFINED);
    update.body.push(
        S::Call {
            function: leaf_identity,
            arguments: vec![q],
            result: Some(returned),
        },
        Span::UNDEFINED,
    );
    let q = emit(
        &mut update,
        E::PointerAlignment {
            pointer: returned,
            alignment: 4,
        },
    );
    let p = emit(
        &mut update,
        E::PointerAlignment {
            pointer: p,
            alignment: 4,
        },
    );
    let p = if atomic {
        emit(
            &mut update,
            E::AtomicPointer {
                pointer: p,
                order: naga::AtomicMemoryOrder::Release,
                failure_order: None,
            },
        )
    } else {
        p
    };
    let q = if atomic {
        emit(
            &mut update,
            E::AtomicPointer {
                pointer: q,
                order: naga::AtomicMemoryOrder::Acquire,
                failure_order: None,
            },
        )
    } else {
        q
    };

    let addend = emit(&mut update, E::FunctionArgument(2));
    let null_local = update.local_variables.append(
        naga::LocalVariable {
            name: None,
            ty: scalar_ptr,
            init: None,
        },
        Span::UNDEFINED,
    );
    let null_place = emit(&mut update, E::LocalVariable(null_local));
    let null = emit(
        &mut update,
        E::Load {
            pointer: null_place,
        },
    );
    let null_address = emit(
        &mut update,
        E::PointerCast {
            expr: null,
            ty: u64_ty,
        },
    );
    let explicit_null = emit(&mut update, E::ZeroValue(scalar_ptr));
    let explicit_address = emit(
        &mut update,
        E::PointerCast {
            expr: explicit_null,
            ty: u64_ty,
        },
    );
    let zero_address = emit(&mut update, E::Literal(naga::Literal::U64(0)));
    let default_ok = emit(
        &mut update,
        E::Binary {
            op: naga::BinaryOperator::Equal,
            left: null_address,
            right: zero_address,
        },
    );
    let explicit_ok = emit(
        &mut update,
        E::Binary {
            op: naga::BinaryOperator::Equal,
            left: explicit_address,
            right: zero_address,
        },
    );
    let nulls_ok = emit(
        &mut update,
        E::Binary {
            op: naga::BinaryOperator::LogicalAnd,
            left: default_ok,
            right: explicit_ok,
        },
    );
    let wrong = emit(&mut update, E::Literal(naga::Literal::U32(0)));
    let addend = emit(
        &mut update,
        E::Select {
            condition: nulls_ok,
            accept: addend,
            reject: wrong,
        },
    );

    let three = emit(&mut update, E::Literal(naga::Literal::U32(3)));
    let one = emit(&mut update, E::Literal(naga::Literal::U32(1)));
    let old = emit(&mut update, E::Load { pointer: q });
    let product = emit(
        &mut update,
        E::Binary {
            op: naga::BinaryOperator::Multiply,
            left: old,
            right: three,
        },
    );
    let result = emit(
        &mut update,
        E::Binary {
            op: naga::BinaryOperator::Add,
            left: product,
            right: addend,
        },
    );
    update.body.push(
        S::Store {
            pointer: p,
            value: result,
        },
        Span::UNDEFINED,
    );
    let reload = emit(&mut update, E::Load { pointer: q });
    let increment = emit(
        &mut update,
        E::Binary {
            op: naga::BinaryOperator::Add,
            left: reload,
            right: one,
        },
    );
    update.body.push(
        S::Store {
            pointer: p,
            value: increment,
        },
        Span::UNDEFINED,
    );
    let again = emit(&mut update, E::Load { pointer: q });
    let decrement = emit(
        &mut update,
        E::Binary {
            op: naga::BinaryOperator::Subtract,
            left: again,
            right: one,
        },
    );
    update.body.push(
        S::Store {
            pointer: p,
            value: decrement,
        },
        Span::UNDEFINED,
    );
    if atomic {
        let q = emit(
            &mut update,
            E::AtomicPointer {
                pointer: q,
                order: naga::AtomicMemoryOrder::AcquireRelease,
                failure_order: None,
            },
        );
        for fun in [
            naga::AtomicFunction::Add,
            naga::AtomicFunction::Subtract,
            naga::AtomicFunction::InclusiveOr,
            naga::AtomicFunction::ExclusiveOr,
            naga::AtomicFunction::ExclusiveOr,
            naga::AtomicFunction::And,
            naga::AtomicFunction::Min,
            naga::AtomicFunction::Max,
        ] {
            let operand = match fun {
                naga::AtomicFunction::And | naga::AtomicFunction::Min => {
                    emit(&mut update, E::Literal(naga::Literal::U32(u32::MAX)))
                }
                naga::AtomicFunction::InclusiveOr | naga::AtomicFunction::Max => {
                    emit(&mut update, E::Literal(naga::Literal::U32(0)))
                }
                _ => one,
            };
            update.body.push(
                S::Atomic {
                    pointer: q,
                    fun,
                    value: operand,
                    result: None,
                },
                Span::UNDEFINED,
            );
        }
        let exchanged = update.expressions.append(
            E::AtomicResult {
                ty: uint,
                comparison: false,
            },
            Span::UNDEFINED,
        );
        update.body.push(
            S::Atomic {
                pointer: q,
                fun: naga::AtomicFunction::Exchange { compare: None },
                value: one,
                result: Some(exchanged),
            },
            Span::UNDEFINED,
        );
        update.body.push(
            S::Store {
                pointer: p,
                value: exchanged,
            },
            Span::UNDEFINED,
        );
        let cas_type = m.generate_predeclared_type(
            naga::PredeclaredType::AtomicCompareExchangeWeakResult(naga::Scalar::U32),
        );
        let cas = update.expressions.append(
            E::AtomicResult {
                ty: cas_type,
                comparison: true,
            },
            Span::UNDEFINED,
        );
        update.body.push(
            S::Atomic {
                pointer: q,
                fun: naga::AtomicFunction::Exchange {
                    compare: Some(exchanged),
                },
                value: one,
                result: Some(cas),
            },
            Span::UNDEFINED,
        );
        let old_value = emit(
            &mut update,
            E::AccessIndex {
                base: cas,
                index: 0,
            },
        );
        let succeeded = emit(
            &mut update,
            E::AccessIndex {
                base: cas,
                index: 1,
            },
        );
        let restore = emit(
            &mut update,
            E::Select {
                condition: succeeded,
                accept: old_value,
                reject: one,
            },
        );
        update.body.push(
            S::Store {
                pointer: p,
                value: restore,
            },
            Span::UNDEFINED,
        );
        let impossible = emit(&mut update, E::Literal(naga::Literal::U32(u32::MAX)));
        let failed = update.expressions.append(
            E::AtomicResult {
                ty: cas_type,
                comparison: true,
            },
            Span::UNDEFINED,
        );
        update.body.push(
            S::Atomic {
                pointer: q,
                fun: naga::AtomicFunction::Exchange {
                    compare: Some(impossible),
                },
                value: one,
                result: Some(failed),
            },
            Span::UNDEFINED,
        );
        let old = emit(
            &mut update,
            E::AccessIndex {
                base: failed,
                index: 0,
            },
        );
        let succeeded = emit(
            &mut update,
            E::AccessIndex {
                base: failed,
                index: 1,
            },
        );
        let acquire = emit(
            &mut update,
            E::AtomicPointer {
                pointer: q,
                order: naga::AtomicMemoryOrder::Acquire,
                failure_order: None,
            },
        );
        let observed = emit(&mut update, E::Load { pointer: acquire });
        let unchanged = emit(
            &mut update,
            E::Binary {
                op: naga::BinaryOperator::Equal,
                left: observed,
                right: old,
            },
        );
        let valid = emit(
            &mut update,
            E::Unary {
                op: naga::UnaryOperator::LogicalNot,
                expr: succeeded,
            },
        );
        let valid = emit(
            &mut update,
            E::Binary {
                op: naga::BinaryOperator::LogicalAnd,
                left: valid,
                right: unchanged,
            },
        );
        let checked = emit(
            &mut update,
            E::Select {
                condition: valid,
                accept: old,
                reject: one,
            },
        );
        update.body.push(
            S::Store {
                pointer: p,
                value: checked,
            },
            Span::UNDEFINED,
        );
    }
    update.body.push(S::Return { value: None }, Span::UNDEFINED);
    if matches!(mode, SpanMode::Contended | SpanMode::Feedback) {
        update.body = naga::Block::new();
        update.expressions = naga::Arena::new();
        let pointer = emit(&mut update, E::FunctionArgument(0));
        let pointer = if scalar_layout {
            emit(
                &mut update,
                E::AccessIndex {
                    base: pointer,
                    index: 0,
                },
            )
        } else {
            pointer
        };
        let value = emit(&mut update, E::FunctionArgument(2));
        update.body.push(
            S::Atomic {
                pointer,
                fun: if mode == SpanMode::Feedback {
                    naga::AtomicFunction::InclusiveOr
                } else {
                    naga::AtomicFunction::Add
                },
                value,
                result: None,
            },
            Span::UNDEFINED,
        );
        update.body.push(S::Return { value: None }, Span::UNDEFINED);
    }
    if let SpanMode::MatrixLayout { row_major, padded } = mode {
        update.body = naga::Block::new();
        update.expressions = naga::Arena::new();
        let float = ty(&mut m, T::Scalar(naga::Scalar::F32));
        let float_ptr = ty(
            &mut m,
            T::Pointer {
                base: float,
                space: naga::AddressSpace::PhysicalStorage,
            },
        );
        let pointer = emit(&mut update, E::FunctionArgument(0));
        let address = emit(
            &mut update,
            E::PointerCast {
                expr: pointer,
                ty: u64_ty,
            },
        );
        let pointer = emit(
            &mut update,
            E::PointerCast {
                expr: address,
                ty: float_ptr,
            },
        );
        let stride = emit(
            &mut update,
            E::Literal(naga::Literal::U32(
                if row_major { 2 } else { 3 } + u32::from(padded),
            )),
        );
        let data = naga::CooperativeData {
            pointer,
            stride,
            row_major,
        };
        let matrix = emit(
            &mut update,
            E::MatrixLoad {
                columns: naga::VectorSize::Bi,
                rows: naga::VectorSize::Tri,
                data,
            },
        );
        let stride = emit(
            &mut update,
            E::Literal(naga::Literal::U32(
                if row_major { 3 } else { 2 } + u32::from(padded),
            )),
        );
        update.body.push(
            S::MatrixStore {
                target: matrix,
                data: naga::CooperativeData {
                    pointer,
                    stride,
                    row_major: !row_major,
                },
            },
            Span::UNDEFINED,
        );
        update.body.push(S::Return { value: None }, Span::UNDEFINED);
    }
    let update = m.functions.append(update, Span::UNDEFINED);
    let root = ty(
        &mut m,
        T::Struct {
            members: vec![
                naga::StructMember {
                    access: None,
                    name: Some("address".into()),
                    ty: u64_ty,
                    binding: None,
                    offset: 0,
                },
                naga::StructMember {
                    access: None,
                    name: Some("addend".into()),
                    ty: uint,
                    binding: None,
                    offset: 8,
                },
                naga::StructMember {
                    access: None,
                    name: Some("count".into()),
                    ty: uint,
                    binding: None,
                    offset: 12,
                },
            ],
            span: 16,
        },
    );
    let global = m.global_variables.append(
        naga::GlobalVariable {
            name: None,
            space: naga::AddressSpace::Immediate,
            binding: None,
            ty: root,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let mut f = naga::Function::default();
    f.arguments.push(naga::FunctionArgument {
        name: None,
        ty: vector,
        binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
        immutable_pointee: false,
    });
    let id = emit(&mut f, E::FunctionArgument(0));
    let index = emit(&mut f, E::AccessIndex { base: id, index: 0 });
    let roots = emit(&mut f, E::GlobalVariable(global));
    let count_field = emit(
        &mut f,
        E::AccessIndex {
            base: roots,
            index: 2,
        },
    );
    let count = emit(
        &mut f,
        E::Load {
            pointer: count_field,
        },
    );
    let in_bounds = emit(
        &mut f,
        E::Binary {
            op: naga::BinaryOperator::Less,
            left: index,
            right: count,
        },
    );
    let address_field = emit(
        &mut f,
        E::AccessIndex {
            base: roots,
            index: 0,
        },
    );
    let address = emit(
        &mut f,
        E::Load {
            pointer: address_field,
        },
    );
    let offset = emit(
        &mut f,
        E::As {
            expr: index,
            kind: naga::ScalarKind::Uint,
            convert: Some(8),
        },
    );
    let four = emit(
        &mut f,
        E::Literal(naga::Literal::U64(if layout_mode {
            32
        } else if scalar_layout || matrix_mode {
            16
        } else {
            4
        })),
    );
    let max = emit(
        &mut f,
        E::Literal(naga::Literal::U64(
            u64::MAX
                - if layout_mode {
                    31
                } else if scalar_layout || matrix_mode {
                    15
                } else {
                    3
                },
        )),
    );
    let bytes = emit(
        &mut f,
        E::Binary {
            op: naga::BinaryOperator::Multiply,
            left: offset,
            right: four,
        },
    );
    let limit = emit(
        &mut f,
        E::Binary {
            op: naga::BinaryOperator::Subtract,
            left: max,
            right: bytes,
        },
    );
    let no_overflow = emit(
        &mut f,
        E::Binary {
            op: naga::BinaryOperator::LessEqual,
            left: address,
            right: limit,
        },
    );
    let allowed = emit(
        &mut f,
        E::Binary {
            op: naga::BinaryOperator::LogicalAnd,
            left: in_bounds,
            right: no_overflow,
        },
    );
    let prelude = core::mem::take(&mut f.body);
    let p = emit(
        &mut f,
        E::PointerCast {
            expr: address,
            ty: ptr,
        },
    );
    let offset = if matches!(mode, SpanMode::Contended | SpanMode::Feedback) {
        emit(&mut f, E::Literal(naga::Literal::U64(0)))
    } else {
        offset
    };
    let element = if mode == SpanMode::RuntimeArray {
        let array = ty(
            &mut m,
            T::Array {
                base,
                size: naga::ArraySize::Dynamic,
                stride: if scalar_layout { 16 } else { 4 },
            },
        );
        let array_pointer = ty(
            &mut m,
            T::Pointer {
                base: array,
                space: naga::AddressSpace::PhysicalStorage,
            },
        );
        let p = emit(
            &mut f,
            E::PointerCast {
                expr: address,
                ty: array_pointer,
            },
        );
        emit(&mut f, E::Access { base: p, index })
    } else {
        emit(&mut f, E::PointerOffset { pointer: p, offset })
    };
    let result = f
        .expressions
        .append(E::CallResult(identity), Span::UNDEFINED);
    f.body.push(
        S::Call {
            function: identity,
            arguments: vec![element],
            result: Some(result),
        },
        Span::UNDEFINED,
    );
    let addend_field = emit(
        &mut f,
        E::AccessIndex {
            base: roots,
            index: 1,
        },
    );
    let addend = emit(
        &mut f,
        E::Load {
            pointer: addend_field,
        },
    );
    let addend = if mode == SpanMode::Feedback {
        let mask = emit(&mut f, E::Literal(naga::Literal::U32(31)));
        let bit = emit(
            &mut f,
            E::Binary {
                op: naga::BinaryOperator::And,
                left: index,
                right: mask,
            },
        );
        let one = emit(&mut f, E::Literal(naga::Literal::U32(1)));
        emit(
            &mut f,
            E::Binary {
                op: naga::BinaryOperator::ShiftLeft,
                left: one,
                right: bit,
            },
        )
    } else {
        addend
    };
    f.body.push(
        S::Call {
            function: update,
            arguments: vec![element, result, addend],
            result: None,
        },
        Span::UNDEFINED,
    );
    let accept = core::mem::replace(&mut f.body, prelude);
    f.body.push(
        S::If {
            condition: allowed,
            accept,
            reject: naga::Block::new(),
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.entry_points.push(naga::EntryPoint {
        name: "main".into(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function: f,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });
    m
}

pub fn use_checked_atomics(m: &mut naga::Module) {
    let uint = m.types.insert(
        naga::Type {
            name: None,
            inner: T::Scalar(naga::Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let f = &mut m.entry_points[0].function;
    let (result, pointer) = f
        .expressions
        .iter()
        .find_map(|(h, e)| match *e {
            E::Load { pointer } if matches!(f.expressions[pointer], E::Access { .. }) => {
                Some((h, pointer))
            }
            _ => None,
        })
        .unwrap();
    f.expressions[result] = E::AtomicResult {
        ty: uint,
        comparison: false,
    };
    let one = f
        .expressions
        .append(E::Literal(naga::Literal::U32(1)), Span::UNDEFINED);
    let ordered = f.expressions.append(
        E::AtomicPointer {
            pointer,
            order: naga::AtomicMemoryOrder::AcquireRelease,
            failure_order: None,
        },
        Span::UNDEFINED,
    );
    let aligned = f.expressions.append(
        E::PointerAlignment {
            pointer: ordered,
            alignment: 4,
        },
        Span::UNDEFINED,
    );
    let old = core::mem::take(&mut f.body);
    for (statement, span) in old.span_iter() {
        if let S::Emit(range) = statement {
            let handles = range.clone().collect::<Vec<_>>();
            if let Some(index) = handles.iter().position(|&h| h == result) {
                if index > 0 {
                    f.body.push(
                        S::Emit(naga::Range::new_from_bounds(handles[0], handles[index - 1])),
                        *span,
                    );
                }
                f.body.push(
                    S::Emit(naga::Range::new_from_bounds(ordered, aligned)),
                    Span::UNDEFINED,
                );
                f.body.push(
                    S::Atomic {
                        pointer: aligned,
                        fun: naga::AtomicFunction::Add,
                        value: one,
                        result: Some(result),
                    },
                    Span::UNDEFINED,
                );
                if index + 1 < handles.len() {
                    f.body.push(
                        S::Emit(naga::Range::new_from_bounds(
                            handles[index + 1],
                            *handles.last().unwrap(),
                        )),
                        *span,
                    );
                }
                continue;
            }
        }
        f.body.push(statement.clone(), *span);
    }
}

pub fn byte_module(signed: bool) -> naga::Module {
    fn ty(m: &mut naga::Module, inner: T) -> naga::Handle<naga::Type> {
        m.types
            .insert(naga::Type { name: None, inner }, Span::UNDEFINED)
    }
    fn emit(f: &mut naga::Function, expression: E) -> naga::Handle<E> {
        let pre = expression.needs_pre_emit();
        let h = f.expressions.append(expression, Span::UNDEFINED);
        if !pre {
            f.body
                .push(S::Emit(naga::Range::new_from_bounds(h, h)), Span::UNDEFINED);
        }
        h
    }
    let mut m = naga::Module::default();
    let address_ty = ty(&mut m, T::Scalar(naga::Scalar::U64));
    let byte_scalar = if signed {
        naga::Scalar::I8
    } else {
        naga::Scalar::U8
    };
    let byte = ty(&mut m, T::Scalar(byte_scalar));
    let wide_scalar = if signed {
        naga::Scalar::I32
    } else {
        naga::Scalar::U32
    };
    let wide = ty(&mut m, T::Scalar(wide_scalar));
    let bytes = ty(
        &mut m,
        T::Array {
            base: byte,
            size: naga::ArraySize::Constant(core::num::NonZeroU32::new(8).unwrap()),
            stride: 1,
        },
    );
    let record = ty(
        &mut m,
        T::Struct {
            members: vec![
                naga::StructMember {
                    access: None,
                    name: None,
                    ty: byte,
                    binding: None,
                    offset: 0,
                },
                naga::StructMember {
                    access: None,
                    name: None,
                    ty: bytes,
                    binding: None,
                    offset: 1,
                },
            ],
            span: 9,
        },
    );
    let record_ptr = ty(
        &mut m,
        T::Pointer {
            base: record,
            space: naga::AddressSpace::PhysicalStorage,
        },
    );
    let wide_ptr = ty(
        &mut m,
        T::Pointer {
            base: wide,
            space: naga::AddressSpace::PhysicalStorage,
        },
    );
    let args = ty(
        &mut m,
        T::Struct {
            members: vec![naga::StructMember {
                access: None,
                name: None,
                ty: address_ty,
                binding: None,
                offset: 0,
            }],
            span: 8,
        },
    );
    let global = m.global_variables.append(
        naga::GlobalVariable {
            name: None,
            space: naga::AddressSpace::Immediate,
            binding: None,
            ty: args,
            init: None,
            memory_decorations: Default::default(),
        },
        Span::UNDEFINED,
    );
    let mut f = naga::Function::default();
    let g = emit(&mut f, E::GlobalVariable(global));
    let g = emit(&mut f, E::AccessIndex { base: g, index: 0 });
    let address = emit(&mut f, E::Load { pointer: g });
    let offset = emit(&mut f, E::Literal(naga::Literal::U64(17)));
    let record_address = emit(
        &mut f,
        E::Binary {
            op: naga::BinaryOperator::Add,
            left: address,
            right: offset,
        },
    );
    let record = emit(
        &mut f,
        E::PointerCast {
            expr: record_address,
            ty: record_ptr,
        },
    );
    let array = emit(
        &mut f,
        E::AccessIndex {
            base: record,
            index: 1,
        },
    );
    for index in 0..8 {
        let byte_ptr = emit(&mut f, E::AccessIndex { base: array, index });
        let value = emit(&mut f, E::Load { pointer: byte_ptr });
        let extended = emit(
            &mut f,
            E::As {
                expr: value,
                kind: wide_scalar.kind,
                convert: Some(4),
            },
        );
        let offset = emit(
            &mut f,
            E::Literal(naga::Literal::U64(64 + u64::from(index) * 4)),
        );
        let out_address = emit(
            &mut f,
            E::Binary {
                op: naga::BinaryOperator::Add,
                left: address,
                right: offset,
            },
        );
        let out = emit(
            &mut f,
            E::PointerCast {
                expr: out_address,
                ty: wide_ptr,
            },
        );
        f.body.push(
            S::Store {
                pointer: out,
                value: extended,
            },
            Span::UNDEFINED,
        );
        let delta = emit(
            &mut f,
            E::Literal(if signed {
                naga::Literal::I8(1)
            } else {
                naga::Literal::U8(1)
            }),
        );
        let incremented = emit(
            &mut f,
            E::Binary {
                op: naga::BinaryOperator::Add,
                left: value,
                right: delta,
            },
        );
        f.body.push(
            S::Store {
                pointer: byte_ptr,
                value: incremented,
            },
            Span::UNDEFINED,
        );
    }
    let out = emit(
        &mut f,
        E::AccessIndex {
            base: record,
            index: 0,
        },
    );
    let offset = emit(&mut f, E::Literal(naga::Literal::U64(32)));
    let source_address = emit(
        &mut f,
        E::Binary {
            op: naga::BinaryOperator::Add,
            left: address,
            right: offset,
        },
    );
    let source = emit(
        &mut f,
        E::PointerCast {
            expr: source_address,
            ty: wide_ptr,
        },
    );
    let value = emit(&mut f, E::Load { pointer: source });
    let value = emit(
        &mut f,
        E::As {
            expr: value,
            kind: byte_scalar.kind,
            convert: Some(1),
        },
    );
    f.body.push(
        S::Store {
            pointer: out,
            value,
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.entry_points.push(naga::EntryPoint {
        name: "main".into(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function: f,
        incoming_ray_payload: None,
        mesh_info: None,
        task_payload: None,
    });
    m
}
