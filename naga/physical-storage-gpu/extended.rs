use super::*;

fn ty(m: &mut naga::Module, inner: T) -> naga::Handle<Type> {
    m.types.insert(Type { name: None, inner }, Span::UNDEFINED)
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

/// Extended modules compute addresses with 64-bit integers.
fn capabilities() -> naga::valid::Capabilities {
    native_capabilities() | naga::valid::Capabilities::SHADER_INT64
}

fn module() -> (naga::Module, naga::Function, naga::Handle<E>) {
    let mut m = naga::Module::default();
    let u64_ty = ty(&mut m, T::Scalar(naga::Scalar::U64));
    let args = ty(
        &mut m,
        T::Struct {
            members: vec![naga::StructMember {
                access: None,
                name: None,
                ty: u64_ty,
                binding: None,
                offset: 0,
            }],
            span: 8,
        },
    );
    let global = m.global_variables.append(
        naga::GlobalVariable {
            name: None,
            space: AddressSpace::Immediate,
            binding: None,
            ty: args,
            init: None,
            memory_decorations: Default::default(),
        },
        Span::UNDEFINED,
    );
    let mut f = naga::Function::default();
    let args = emit(&mut f, E::GlobalVariable(global));
    let address = emit(
        &mut f,
        E::AccessIndex {
            base: args,
            index: 0,
        },
    );
    let address = emit(&mut f, E::Load { pointer: address });
    (m, f, address)
}

fn pointer(
    m: &mut naga::Module,
    f: &mut naga::Function,
    address: naga::Handle<E>,
    offset: u64,
    base: naga::Handle<Type>,
) -> naga::Handle<E> {
    let offset = emit(f, E::Literal(naga::Literal::U64(offset)));
    let address = emit(
        f,
        E::Binary {
            op: naga::BinaryOperator::Add,
            left: address,
            right: offset,
        },
    );
    let pointer = ty(
        m,
        T::Pointer {
            base,
            space: AddressSpace::PhysicalStorage,
        },
    );
    emit(
        f,
        E::PointerCast {
            expr: address,
            ty: pointer,
        },
    )
}

fn finish(mut m: naga::Module, mut f: naga::Function, workgroup: u32) -> naga::Module {
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.entry_points.push(naga::EntryPoint {
        name: "main".into(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [workgroup, 1, 1],
        workgroup_size_overrides: None,
        function: f,
        incoming_ray_payload: None,
        mesh_info: None,
        task_payload: None,
    });
    m
}

/// Executes `m` directly and, for a family's representative case (`import`), again after an
/// spv-in round trip. Returns the read-back buffer of each execution.
fn dispatch(
    gpu: &Gpu,
    m: naga::Module,
    capabilities: naga::valid::Capabilities,
    input: &[u32],
    import: bool,
    family: &str,
    label: &str,
) -> Vec<Vec<u32>> {
    let words = write_spirv(&m, capabilities, Default::default());
    // The compiler suite does not build these modules. Compaction must leave their SPIR-V
    // unchanged, so a compacted copy would repeat an identical dispatch.
    let mut compacted = m.clone();
    naga::compact::compact(&mut compacted, naga::compact::KeepUnused::No);
    assert_eq!(
        write_spirv(&compacted, capabilities, Default::default()),
        words,
        "compaction changed the SPIR-V of {label}"
    );
    spirv_val(&words, false, label);
    let workgroup = m.entry_points[0].workgroup_size;
    gpu.assert_safe_api_rejects(family, &m);
    let imported = import.then(|| {
        let (m, words) = round_trip_shader(&words, capabilities, false);
        gpu.assert_safe_api_rejects(&format!("imported {family}"), &m);
        ("imported", words)
    });
    std::iter::once(("direct", words))
        .chain(imported)
        .map(|(path, words)| {
            gpu.run(
                &format!("{path} {label}"),
                words,
                workgroup,
                1,
                &[input.to_vec()],
                |_, address| {
                    assert_eq!(address % 16, 0);
                    let mut args = [0; 16];
                    args[..8].copy_from_slice(&address.to_le_bytes());
                    args
                },
            )
            .remove(0)
        })
        .collect()
}

fn atomic_module(scalar: naga::Scalar) -> naga::Module {
    let (mut m, mut f, address) = module();
    let scalar_ty = ty(&mut m, T::Scalar(scalar));
    let atomic_ty = ty(&mut m, T::Atomic(scalar));
    let p = pointer(&mut m, &mut f, address, 16, atomic_ty);
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    f.arguments.push(naga::FunctionArgument {
        name: None,
        ty: uint,
        binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::LocalInvocationIndex)),
        immutable_pointee: false,
    });
    let id = emit(&mut f, E::FunctionArgument(0));
    let id = emit(
        &mut f,
        E::As {
            expr: id,
            kind: naga::ScalarKind::Uint,
            convert: Some(8),
        },
    );
    let out = pointer(&mut m, &mut f, address, 32, scalar_ty);
    let out = emit(
        &mut f,
        E::PointerOffset {
            pointer: out,
            offset: id,
        },
    );
    let one = emit(
        &mut f,
        E::Literal(match scalar.kind {
            naga::ScalarKind::Float => naga::Literal::F32(1.0),
            naga::ScalarKind::Sint => naga::Literal::I64(1),
            naga::ScalarKind::Uint => naga::Literal::U64(1),
            _ => unreachable!(),
        }),
    );
    let result = f.expressions.append(
        E::AtomicResult {
            ty: scalar_ty,
            comparison: false,
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::Atomic {
            pointer: p,
            fun: naga::AtomicFunction::Add,
            value: one,
            result: Some(result),
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::Store {
            pointer: out,
            value: result,
        },
        Span::UNDEFINED,
    );
    finish(m, f, 64)
}

fn atomics(gpu: &Gpu) {
    for scalar in [naga::Scalar::U64, naga::Scalar::I64, naga::Scalar::F32] {
        let width = usize::from(scalar.width) / 4;
        let start = match scalar.kind {
            naga::ScalarKind::Uint => 1u64 << 40,
            naga::ScalarKind::Sint => (-(1i64 << 40)) as u64,
            naga::ScalarKind::Float => u64::from(10f32.to_bits()),
            _ => unreachable!(),
        };
        let required = capabilities()
            | if scalar == naga::Scalar::F32 {
                naga::valid::Capabilities::SHADER_FLOAT32_ATOMIC
            } else {
                naga::valid::Capabilities::SHADER_INT64_ATOMIC_ALL_OPS
            };
        let mut input = vec![GUARD; 8 + 64 * width + 4];
        input[4] = start as u32;
        if width == 2 {
            input[5] = (start >> 32) as u32;
        }
        for actual in dispatch(
            gpu,
            atomic_module(scalar),
            required,
            &input,
            scalar == naga::Scalar::I64,
            "contended atomic add",
            &format!("contended {scalar:?} atomic add"),
        ) {
            let mut returned: Vec<u64> = actual[8..8 + 64 * width]
                .chunks_exact(width)
                .map(|w| u64::from(w[0]) | if width == 2 { u64::from(w[1]) << 32 } else { 0 })
                .collect();
            returned.sort_unstable();
            let mut expected: Vec<u64> = (0..64)
                .map(|i| {
                    if scalar == naga::Scalar::F32 {
                        u64::from((10.0 + i as f32).to_bits())
                    } else {
                        start.wrapping_add(i)
                    }
                })
                .collect();
            expected.sort_unstable();
            assert_eq!(
                returned, expected,
                "every atomic returns a unique pre-add value"
            );
            let final_value = if scalar == naga::Scalar::F32 {
                u64::from(74f32.to_bits())
            } else {
                start.wrapping_add(64)
            };
            let mut expected = input.clone();
            expected[4] = final_value as u32;
            if width == 2 {
                expected[5] = (final_value >> 32) as u32;
            }
            expected[8..8 + 64 * width].copy_from_slice(&actual[8..8 + 64 * width]);
            assert_eq!(actual, expected, "final atomic value and untouched guards");
        }
    }
}

fn integer_atomics(gpu: &Gpu) {
    for signed in [false, true] {
        let scalar = if signed {
            naga::Scalar::I64
        } else {
            naga::Scalar::U64
        };
        let literal = |value: i64| {
            E::Literal(if signed {
                naga::Literal::I64(value)
            } else {
                naga::Literal::U64(value as u64)
            })
        };
        let (mut m, mut f, address) = module();
        let scalar_ty = ty(&mut m, T::Scalar(scalar));
        let atomic_ty = ty(&mut m, T::Atomic(scalar));
        let p = pointer(&mut m, &mut f, address, 16, atomic_ty);
        let zero = emit(&mut f, literal(0));
        let one = emit(&mut f, literal(1));
        let compare = emit(&mut f, literal(9));
        let operations = [
            (naga::AtomicFunction::Min, if signed { -5 } else { 5 }),
            (naga::AtomicFunction::Max, 7),
            (naga::AtomicFunction::InclusiveOr, 16),
            (naga::AtomicFunction::ExclusiveOr, 3),
            (naga::AtomicFunction::And, 15),
            (naga::AtomicFunction::Subtract, 2),
            (naga::AtomicFunction::Exchange { compare: None }, 9),
            (
                naga::AtomicFunction::Exchange {
                    compare: Some(compare),
                },
                11,
            ),
            (
                naga::AtomicFunction::Exchange {
                    compare: Some(compare),
                },
                13,
            ),
        ];
        for (i, (fun, value)) in operations.into_iter().enumerate() {
            let comparison = i >= 7;
            let result_ty = if comparison {
                m.generate_predeclared_type(naga::PredeclaredType::AtomicCompareExchangeWeakResult(
                    scalar,
                ))
            } else {
                scalar_ty
            };
            let value = emit(&mut f, literal(value));
            let result = f.expressions.append(
                E::AtomicResult {
                    ty: result_ty,
                    comparison,
                },
                Span::UNDEFINED,
            );
            let ordered = if comparison {
                emit(
                    &mut f,
                    E::AtomicPointer {
                        pointer: p,
                        order: naga::AtomicMemoryOrder::AcquireRelease,
                        failure_order: Some(if i == 7 {
                            naga::AtomicMemoryOrder::Relaxed
                        } else {
                            naga::AtomicMemoryOrder::Acquire
                        }),
                    },
                )
            } else {
                p
            };
            f.body.push(
                S::Atomic {
                    pointer: ordered,
                    fun,
                    value,
                    result: Some(result),
                },
                Span::UNDEFINED,
            );
            let (old, exchanged) = if comparison {
                let old = emit(
                    &mut f,
                    E::AccessIndex {
                        base: result,
                        index: 0,
                    },
                );
                let success = emit(
                    &mut f,
                    E::AccessIndex {
                        base: result,
                        index: 1,
                    },
                );
                (
                    old,
                    emit(
                        &mut f,
                        E::Select {
                            condition: success,
                            accept: one,
                            reject: zero,
                        },
                    ),
                )
            } else {
                (result, zero)
            };
            for (j, value) in [old, exchanged].into_iter().enumerate() {
                let out = pointer(
                    &mut m,
                    &mut f,
                    address,
                    32 + ((i * 2 + j) * 8) as u64,
                    scalar_ty,
                );
                f.body.push(
                    S::Store {
                        pointer: out,
                        value,
                    },
                    Span::UNDEFINED,
                );
            }
        }
        let value = emit(&mut f, E::Load { pointer: p });
        let out = pointer(&mut m, &mut f, address, 176, scalar_ty);
        f.body.push(
            S::Store {
                pointer: out,
                value,
            },
            Span::UNDEFINED,
        );
        let value = emit(&mut f, literal(22));
        f.body.push(S::Store { pointer: p, value }, Span::UNDEFINED);
        let mut input = vec![GUARD; 50];
        let set = |words: &mut [u32], index: usize, value: i64| {
            let bits = value as u64;
            words[index] = bits as u32;
            words[index + 1] = (bits >> 32) as u32;
        };
        set(&mut input, 4, 1 << 40);
        let mut expected = input.clone();
        set(&mut expected, 4, 22);
        for (i, old) in [1 << 40, if signed { -5 } else { 5 }, 7, 23, 20, 4, 2, 9, 11]
            .into_iter()
            .enumerate()
        {
            set(&mut expected, 8 + i * 4, old);
            set(&mut expected, 10 + i * 4, i64::from(i == 7));
        }
        set(&mut expected, 44, 11);
        for actual in dispatch(
            gpu,
            finish(m, f, 1),
            capabilities() | naga::valid::Capabilities::SHADER_INT64_ATOMIC_ALL_OPS,
            &input,
            signed,
            "64-bit atomic operations",
            &format!("{scalar:?} atomic load/store, min/max, bitwise, subtract, exchange and CAS"),
        ) {
            assert_eq!(actual, expected);
        }
    }
}

fn size(n: u32) -> naga::CooperativeSize {
    match n {
        8 => naga::CooperativeSize::Eight,
        16 => naga::CooperativeSize::Sixteen,
        _ => panic!("unsupported cooperative dimension {n}"),
    }
}

fn cooperative(gpu: &Gpu, properties: &[wgpu::CooperativeMatrixProperties]) {
    let p = properties
        .iter()
        .find(|p| {
            p.ab_type == wgpu::CooperativeScalarType::F16
                && p.cr_type == wgpu::CooperativeScalarType::F32
                && [p.m_size, p.n_size, p.k_size]
                    .into_iter()
                    .all(|n| n == 8 || n == 16)
                && !p.saturating_accumulation
        })
        .expect("an f16/f32 cooperative matrix configuration representable in Naga is required");
    // SAFETY: the HAL guard keeps the instance and physical device alive.
    let subgroup_size = unsafe {
        let hal = gpu.device.as_hal::<Vulkan>().unwrap();
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
        hal.shared_instance()
            .raw_instance()
            .get_physical_device_properties2(hal.raw_physical_device(), &mut properties);
        subgroup.subgroup_size
    };
    println!("Executing cooperative configuration {p:?}");
    for row_major in [false, true] {
        for padded in [false, true] {
            let (mut m, mut f, address) = module();
            let half_ty = ty(&mut m, T::Scalar(naga::Scalar::F16));
            let float_ty = ty(&mut m, T::Scalar(naga::Scalar::F32));
            let mut input = vec![GUARD; 4];
            let mut matrices = Vec::new();
            let mut output = (0usize, 0usize);
            for (index, (rows, columns)) in [
                (p.m_size, p.k_size),
                (p.k_size, p.n_size),
                (p.m_size, p.n_size),
                (p.m_size, p.n_size),
            ]
            .into_iter()
            .enumerate()
            {
                let is_half = index < 2;
                let layout = if index == 3 { !row_major } else { row_major };
                let stride = if layout { columns } else { rows } + if padded { 8 } else { 0 };
                let count = (if layout { rows } else { columns } * stride) as usize;
                let start = input.len();
                input.resize(start + count / if is_half { 2 } else { 1 }, GUARD);
                if index < 3 {
                    for r in 0..rows {
                        for c in 0..columns {
                            let value = match index {
                                0 => ((r + c) % 3) as f32 - 1.0,
                                1 => ((r * 2 + c) % 4) as f32 - 1.0,
                                _ => (r + c) as f32,
                            };
                            let slot = (if layout {
                                r * stride + c
                            } else {
                                c * stride + r
                            }) as usize;
                            if is_half {
                                let bits = u32::from(half::f16::from_f32(value).to_bits());
                                let shift = (slot % 2) * 16;
                                input[start + slot / 2] = (input[start + slot / 2]
                                    & !(0xffff << shift))
                                    | (bits << shift);
                            } else {
                                input[start + slot] = value.to_bits();
                            }
                        }
                    }
                }
                let pointer = pointer(
                    &mut m,
                    &mut f,
                    address,
                    (start * 4) as u64,
                    if is_half { half_ty } else { float_ty },
                );
                let pointer = if padded {
                    emit(
                        &mut f,
                        E::CoherentPointer {
                            pointer,
                            scope: naga::MemoryScope::Device,
                        },
                    )
                } else {
                    pointer
                };
                let stride_expr = emit(&mut f, E::Literal(naga::Literal::U32(stride)));
                let data = naga::CooperativeData {
                    pointer,
                    stride: stride_expr,
                    row_major: layout,
                };
                if index < 3 {
                    let result = emit(
                        &mut f,
                        E::CooperativeLoad {
                            columns: size(columns),
                            rows: size(rows),
                            role: [
                                naga::CooperativeRole::A,
                                naga::CooperativeRole::B,
                                naga::CooperativeRole::C,
                            ][index],
                            data,
                        },
                    );
                    matrices.push(result);
                } else {
                    let result = emit(
                        &mut f,
                        E::CooperativeMultiplyAdd {
                            a: matrices[0],
                            b: matrices[1],
                            c: matrices[2],
                        },
                    );
                    f.body.push(
                        S::CooperativeStore {
                            target: result,
                            data,
                        },
                        Span::UNDEFINED,
                    );
                    output = (start, stride as usize);
                }
                input.extend_from_slice(&[GUARD; 4]);
            }
            let mut expected = input.clone();
            for r in 0..p.m_size {
                for c in 0..p.n_size {
                    let value = (r + c) as f32
                        + (0..p.k_size)
                            .map(|k| {
                                (((r + k) % 3) as f32 - 1.0) * (((k * 2 + c) % 4) as f32 - 1.0)
                            })
                            .sum::<f32>();
                    let slot = if !row_major {
                        r as usize * output.1 + c as usize
                    } else {
                        c as usize * output.1 + r as usize
                    };
                    expected[output.0 + slot] = value.to_bits();
                }
            }
            let required = capabilities()
                | naga::valid::Capabilities::SHADER_FLOAT16
                | naga::valid::Capabilities::COOPERATIVE_MATRIX
                | if padded {
                    naga::valid::Capabilities::COHERENT_PHYSICAL_MEMORY
                } else {
                    naga::valid::Capabilities::empty()
                };
            for actual in dispatch(
                gpu,
                finish(m, f, subgroup_size),
                required,
                &input,
                row_major && padded,
                "cooperative multiply-add",
                &format!("cooperative multiply-add row_major={row_major} padded={padded}"),
            ) {
                assert_eq!(
                    actual, expected,
                    "cooperative product, inputs, padding and guards"
                );
            }
        }
    }
}

fn coherent_accesses(gpu: &Gpu) {
    for scope in [
        naga::MemoryScope::Device,
        naga::MemoryScope::QueueFamily,
        naga::MemoryScope::Workgroup,
    ] {
        let (mut m, mut f, address) = module();
        let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
        let array = ty(
            &mut m,
            T::Array {
                base: scalar,
                size: naga::ArraySize::Constant(core::num::NonZeroU32::new(64).unwrap()),
                stride: 4,
            },
        );
        f.arguments.push(naga::FunctionArgument {
            name: None,
            ty: scalar,
            binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::LocalInvocationIndex)),
            immutable_pointee: false,
        });
        let lane = emit(&mut f, E::FunctionArgument(0));
        let one = emit(&mut f, E::Literal(naga::Literal::U32(1)));
        let count = emit(&mut f, E::Literal(naga::Literal::U32(64)));
        let value = emit(
            &mut f,
            E::Binary {
                op: naga::BinaryOperator::Add,
                left: lane,
                right: one,
            },
        );
        let data = pointer(&mut m, &mut f, address, 16, array);
        let own = emit(
            &mut f,
            E::Access {
                base: data,
                index: lane,
            },
        );
        let own = emit(
            &mut f,
            E::CoherentPointer {
                pointer: own,
                scope,
            },
        );
        f.body.push(
            S::Store {
                pointer: own,
                value,
            },
            Span::UNDEFINED,
        );
        f.body
            .push(S::ControlBarrier(naga::Barrier::STORAGE), Span::UNDEFINED);
        let neighbor = emit(
            &mut f,
            E::Binary {
                op: naga::BinaryOperator::Modulo,
                left: value,
                right: count,
            },
        );
        let source = emit(
            &mut f,
            E::Access {
                base: data,
                index: neighbor,
            },
        );
        let source = emit(
            &mut f,
            E::CoherentPointer {
                pointer: source,
                scope,
            },
        );
        let value = emit(&mut f, E::Load { pointer: source });
        let output = pointer(&mut m, &mut f, address, 288, array);
        let output = emit(
            &mut f,
            E::Access {
                base: output,
                index: lane,
            },
        );
        f.body.push(
            S::Store {
                pointer: output,
                value,
            },
            Span::UNDEFINED,
        );
        let input = vec![0xdeadbeef; 140];
        let mut expected = input.clone();
        for lane in 0..64 {
            expected[4 + lane] = lane as u32 + 1;
            expected[72 + lane] = ((lane + 1) % 64) as u32 + 1;
        }
        for actual in dispatch(
            gpu,
            finish(m, f, 64),
            capabilities() | naga::valid::Capabilities::COHERENT_PHYSICAL_MEMORY,
            &input,
            scope == naga::MemoryScope::QueueFamily,
            "coherent neighbor exchange",
            &format!("coherent neighbor exchange {scope:?}"),
        ) {
            assert_eq!(actual, expected, "neighbor values and untouched guards");
        }
    }
}

fn byte_accesses(gpu: &Gpu) {
    for signed in [false, true] {
        let mut bytes = [0xcd; 128];
        let values = [0u8, 1, 127, 128, 254, 255, 42, 200];
        bytes[18..26].copy_from_slice(&values);
        bytes[32..36].copy_from_slice(&0x123456abu32.to_le_bytes());
        let input: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        bytes[17] = 0xab;
        for (i, value) in values.into_iter().enumerate() {
            let extended = if signed {
                i32::from(value as i8) as u32
            } else {
                u32::from(value)
            };
            bytes[64 + i * 4..68 + i * 4].copy_from_slice(&extended.to_le_bytes());
            bytes[18 + i] = value.wrapping_add(1);
        }
        let expected: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        for actual in dispatch(
            gpu,
            pointer_helpers::byte_module(signed),
            capabilities() | naga::valid::Capabilities::SHADER_INT8,
            &input,
            signed,
            "packed bytes",
            &format!("packed bytes signed={signed}"),
        ) {
            assert_eq!(
                actual, expected,
                "byte extension, arithmetic, truncation, and untouched neighbors"
            );
        }
    }
}

pub fn run() {
    let (device, queue, properties) = device_with_features(
        wgpu::Features::SHADER_INT64_ATOMIC_ALL_OPS
            | wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::SHADER_F16
            | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
    );
    let gpu = Gpu::new((device, queue));
    atomics(&gpu);
    integer_atomics(&gpu);
    coherent_accesses(&gpu);
    byte_accesses(&gpu);
    cooperative(&gpu, &properties);
    drop(gpu);
    let errors = wgpu_hal::VALIDATION_CANARY.get_and_reset();
    assert!(errors.is_empty(), "Vulkan validation errors: {errors:#?}");
}
