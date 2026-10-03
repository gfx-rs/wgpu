#![cfg(spv_out)]

use naga::valid::{Capabilities as C, TypeError, ValidationError, ValidationFlags, Validator};
use naga::{AddressSpace, Expression as E, Span, Statement as S, Type, TypeInner as T};
use rspirv::binary::Disassemble;

fn ty(m: &mut naga::Module, inner: T) -> naga::Handle<Type> {
    m.types.insert(Type { name: None, inner }, Span::UNDEFINED)
}

fn named_ty(m: &mut naga::Module, name: &str, inner: T) -> naga::Handle<Type> {
    m.types.insert(
        Type {
            name: Some(name.into()),
            inner,
        },
        Span::UNDEFINED,
    )
}

/// A `PhysicalStorage` pointer to `base`.
fn ptr_to(m: &mut naga::Module, base: naga::Handle<Type>) -> naga::Handle<Type> {
    ty(
        m,
        T::Pointer {
            base,
            space: AddressSpace::PhysicalStorage,
        },
    )
}

/// The first `PhysicalStorage` pointer type in `m`.
fn physical_pointer(m: &naga::Module) -> naga::Handle<Type> {
    m.types
        .iter()
        .find(|(_, t)| t.inner.pointer_space() == Some(AddressSpace::PhysicalStorage))
        .unwrap()
        .0
}

fn member(name: Option<&str>, ty: naga::Handle<Type>, offset: u32) -> naga::StructMember {
    naga::StructMember {
        access: None,
        name: name.map(Into::into),
        ty,
        binding: None,
        offset,
    }
}

fn argument(ty: naga::Handle<Type>) -> naga::FunctionArgument {
    naga::FunctionArgument {
        name: None,
        ty,
        binding: None,
        immutable_pointee: false,
    }
}

fn returns(ty: naga::Handle<Type>) -> naga::FunctionResult {
    naga::FunctionResult { ty, binding: None }
}

/// A function whose only argument has type `ty`, with that argument emitted.
fn pointer_function(ty: naga::Handle<Type>) -> (naga::Function, naga::Handle<E>) {
    let mut f = naga::Function::default();
    f.arguments.push(argument(ty));
    let argument = append_emit(&mut f, E::FunctionArgument(0));
    (f, argument)
}

fn local(
    locals: &mut naga::Arena<naga::LocalVariable>,
    ty: naga::Handle<Type>,
) -> naga::Handle<naga::LocalVariable> {
    locals.append(
        naga::LocalVariable {
            name: None,
            ty,
            init: None,
        },
        Span::UNDEFINED,
    )
}

fn fixed_size(count: u32) -> naga::ArraySize {
    naga::ArraySize::Constant(core::num::NonZeroU32::new(count).unwrap())
}

fn compute_entry(function: naga::Function) -> naga::EntryPoint {
    naga::EntryPoint {
        name: "main".into(),
        stage: naga::ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    }
}

fn module() -> naga::Module {
    module_with_array(false)
}

fn module_with_array(dynamic: bool) -> naga::Module {
    module_with_options(dynamic, false)
}

fn module_with_options(dynamic: bool, pointer_array: bool) -> naga::Module {
    let mut m = naga::Module::default();
    let scalar = ty(&mut m, T::Scalar(naga::Scalar::F32));
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let member_type = if dynamic {
        ty(
            &mut m,
            T::Array {
                base: scalar,
                size: fixed_size(4),
                stride: 4,
            },
        )
    } else {
        scalar
    };
    let data = named_ty(
        &mut m,
        "Data",
        T::Struct {
            members: vec![member(Some("value"), member_type, 0)],
            span: if dynamic { 16 } else { 4 },
        },
    );
    let pointer = ptr_to(&mut m, data);
    let root_member = if pointer_array {
        ty(
            &mut m,
            T::Array {
                base: pointer,
                size: fixed_size(4),
                stride: 8,
            },
        )
    } else {
        pointer
    };
    let root = named_ty(
        &mut m,
        "Root",
        T::Struct {
            members: vec![
                member(Some("pointer"), root_member, 0),
                member(Some("index"), uint, if pointer_array { 32 } else { 8 }),
            ],
            span: if pointer_array { 40 } else { 16 },
        },
    );
    let global = m.global_variables.append(
        naga::GlobalVariable {
            name: Some("root".into()),
            space: if pointer_array {
                AddressSpace::Storage {
                    access: naga::StorageAccess::LOAD,
                }
            } else {
                AddressSpace::Immediate
            },
            binding: pointer_array.then_some(naga::ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: root,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let mut f = naga::Function::default();
    let g = f
        .expressions
        .append(E::GlobalVariable(global), Span::UNDEFINED);
    let field = f
        .expressions
        .append(E::AccessIndex { base: g, index: 0 }, Span::UNDEFINED);
    let pointer_field = if pointer_array {
        let index_ptr = f
            .expressions
            .append(E::AccessIndex { base: g, index: 1 }, Span::UNDEFINED);
        let index = f
            .expressions
            .append(E::Load { pointer: index_ptr }, Span::UNDEFINED);
        f.expressions
            .append(E::Access { base: field, index }, Span::UNDEFINED)
    } else {
        field
    };
    let ptr = f.expressions.append(
        E::Load {
            pointer: pointer_field,
        },
        Span::UNDEFINED,
    );
    let value_ptr = f.expressions.append(
        E::AccessIndex {
            base: ptr,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let value_ptr = if dynamic {
        let index_ptr = f
            .expressions
            .append(E::AccessIndex { base: g, index: 1 }, Span::UNDEFINED);
        let index = f
            .expressions
            .append(E::Load { pointer: index_ptr }, Span::UNDEFINED);
        f.expressions.append(
            E::Access {
                base: value_ptr,
                index,
            },
            Span::UNDEFINED,
        )
    } else {
        value_ptr
    };
    let value = f
        .expressions
        .append(E::Load { pointer: value_ptr }, Span::UNDEFINED);
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(field, value)),
        Span::UNDEFINED,
    );
    f.body.push(
        S::Store {
            pointer: value_ptr,
            value,
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.entry_points.push(compute_entry(f));
    m
}

/// `module_with_array(true)` with its `f32` element replaced by `atomic<u32>`.
fn atomic_array_module() -> naga::Module {
    let mut m = module_with_array(true);
    let scalar = m
        .types
        .iter()
        .find(|(_, t)| t.inner == T::Scalar(naga::Scalar::F32))
        .unwrap()
        .0;
    m.types.replace(
        scalar,
        Type {
            name: None,
            inner: T::Atomic(naga::Scalar::U32),
        },
    );
    m
}

/// Validates `m` and emits it with `ReadZeroSkipWrite` index checks (and
/// buffer checks when `buffer` is set), then validates the emitted words.
fn checked_words(m: &naga::Module, name: &str, buffer: bool) -> String {
    let info = Validator::new(ValidationFlags::all(), C::all())
        .validate(m)
        .unwrap();
    let policy = naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite;
    let words = naga::back::spv::write_vec(
        m,
        &info,
        &naga::back::spv::Options {
            bounds_check_policies: naga::proc::BoundsCheckPolicies {
                index: policy,
                buffer: if buffer { policy } else { Default::default() },
                ..Default::default()
            },
            ..Default::default()
        },
        None,
    )
    .unwrap();
    validate_emitted_words(&words, name, false)
}

#[test]
fn physical_storage_requires_capability() {
    let err = Validator::new(ValidationFlags::all(), C::default())
        .validate(&module())
        .unwrap_err();
    assert!(format!("{err:?}").contains("PHYSICAL_STORAGE_BUFFER_ADDRESSES"));
}

#[test]
fn physical_storage_load_store_spirv() {
    use naga::proc::BoundsCheckPolicy as P;
    for (m, name, policy) in [
        (module(), "direct", P::Unchecked),
        (module_with_array(true), "checked", P::ReadZeroSkipWrite),
        (module_with_array(true), "restrict", P::Restrict),
    ] {
        check_spirv(m, name, policy);
    }
}

fn check_spirv(m: naga::Module, name: &str, policy: naga::proc::BoundsCheckPolicy) {
    check_spirv_aligned(
        m,
        name,
        policy,
        4,
        C::default() | C::IMMEDIATES | C::PHYSICAL_STORAGE_BUFFER_ADDRESSES,
    );
}

fn check_spirv_aligned(
    m: naga::Module,
    name: &str,
    policy: naga::proc::BoundsCheckPolicy,
    alignment: u32,
    capabilities: C,
) {
    let info = Validator::new(ValidationFlags::all(), capabilities)
        .validate(&m)
        .unwrap();
    let options = naga::back::spv::Options {
        lang_version: (1, 3),
        bounds_check_policies: naga::proc::BoundsCheckPolicies {
            index: policy,
            buffer: policy,
            ..Default::default()
        },
        ..Default::default()
    };
    let last = m.entry_points[0]
        .function
        .expressions
        .iter()
        .last()
        .unwrap()
        .0;
    assert!(info.get_entry_point(0)[last]
        .uniformity
        .non_uniform_result
        .is_some());
    let words = naga::back::spv::write_vec(&m, &info, &options, None).unwrap();
    let text = validate_spirv_words(&words, name, false);
    assert!(
        text.contains("OpMemoryModel PhysicalStorageBuffer64"),
        "{text}"
    );
    assert!(
        text.contains("OpCapability PhysicalStorageBufferAddresses"),
        "{text}"
    );
    assert!(text.contains(&format!("Aligned {alignment}")), "{text}");
}

#[test]
fn physical_storage_is_not_a_global_address_space() {
    let mut m = module();
    m.global_variables.iter_mut().next().unwrap().1.space = AddressSpace::PhysicalStorage;
    let error = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap_err();
    assert!(format!("{error:?}").contains("InvalidUsage(PhysicalStorage)"));
}

#[cfg(any(
    feature = "wgsl-out",
    feature = "msl-out",
    feature = "hlsl-out",
    feature = "glsl-out"
))]
#[test]
fn physical_storage_text_backends_rejected() {
    let m = module();
    let info = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap();
    #[cfg(feature = "wgsl-out")]
    assert!(matches!(
        naga::back::wgsl::write_string(&m, &info, naga::back::wgsl::WriterFlags::empty()),
        Err(naga::back::wgsl::Error::PhysicalStorageUnsupported)
    ));
    #[cfg(feature = "msl-out")]
    assert!(matches!(
        naga::back::msl::write_string(&m, &info, &Default::default(), &Default::default()),
        Err(naga::back::msl::Error::PhysicalStorageUnsupported)
    ));
    #[cfg(feature = "hlsl-out")]
    {
        let options = naga::back::hlsl::Options::default();
        let pipeline = naga::back::hlsl::PipelineOptions::default();
        let mut writer = naga::back::hlsl::Writer::new(String::new(), &options, &pipeline);
        assert!(matches!(
            writer.write(&m, &info, None),
            Err(naga::back::hlsl::Error::PhysicalStorageUnsupported)
        ));
    }
    #[cfg(feature = "glsl-out")]
    {
        let options = naga::back::glsl::Options::default();
        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: "main".into(),
            multiview: None,
        };
        assert!(matches!(
            naga::back::glsl::Writer::new(
                String::new(),
                &m,
                &info,
                &options,
                &pipeline,
                Default::default()
            ),
            Err(naga::back::glsl::Error::PhysicalStorageUnsupported)
        ));
    }
}

#[test]
fn physical_storage_invalid_pointees() {
    for case in [
        "bool",
        "cooperative",
        "nested-cooperative",
        "misaligned-member",
        "zero-stride",
        "overlapping-stride",
        "runtime-array",
    ] {
        let mut m = naga::Module::default();
        let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
        let array = |m: &mut naga::Module, size, stride| {
            ty(
                m,
                T::Array {
                    base: uint,
                    size,
                    stride,
                },
            )
        };
        let cooperative = |m: &mut naga::Module| {
            ty(
                m,
                T::CooperativeMatrix {
                    columns: naga::CooperativeSize::Eight,
                    rows: naga::CooperativeSize::Eight,
                    scalar: naga::Scalar::F32,
                    role: naga::CooperativeRole::A,
                },
            )
        };
        let base = match case {
            "bool" => ty(&mut m, T::Scalar(naga::Scalar::BOOL)),
            "cooperative" => cooperative(&mut m),
            "nested-cooperative" => {
                let matrix = cooperative(&mut m);
                ty(
                    &mut m,
                    T::Struct {
                        members: vec![member(None, matrix, 0)],
                        span: 256,
                    },
                )
            }
            "misaligned-member" => ty(
                &mut m,
                T::Struct {
                    members: vec![member(None, uint, 2)],
                    span: 8,
                },
            ),
            "zero-stride" => array(&mut m, fixed_size(4), 0),
            "overlapping-stride" => array(&mut m, fixed_size(4), 2),
            "runtime-array" => array(&mut m, naga::ArraySize::Dynamic, 4),
            _ => unreachable!(),
        };
        let pointer = ptr_to(&mut m, base);
        let result = Validator::new(ValidationFlags::all(), C::all())
            .validate(&m)
            .map_err(|error| error.into_inner());
        match case {
            "runtime-array" => {
                result.unwrap();
            }
            _ => assert!(
                matches!(
                    result,
                    Err(ValidationError::Type {
                        handle,
                        source: TypeError::InvalidPointerBase(invalid),
                        ..
                    }) if handle == pointer && invalid == base
                ),
                "{case}: {result:?}"
            ),
        }
    }
}

#[test]
fn physical_storage_pointer_null_bounds_check() {
    let m = module_with_options(false, true);
    let text = checked_words(&m, "null-checked-pointer", true);
    assert!(
        text.contains("OpPhi") && text.contains("OpBitcast"),
        "{text}"
    );
    check_spirv(m, "pointer-array", naga::proc::BoundsCheckPolicy::Restrict);
}

fn roundtrip(mut m: naga::Module, base: naga::Handle<Type>, chase: bool) -> naga::Module {
    let pointer = ptr_to(&mut m, base);
    let root = ty(
        &mut m,
        T::Struct {
            members: vec![member(None, pointer, 0)],
            span: 8,
        },
    );
    let global = m.global_variables.append(
        naga::GlobalVariable {
            name: None,
            space: AddressSpace::Immediate,
            binding: None,
            ty: root,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let mut f = naga::Function::default();
    let g = f
        .expressions
        .append(E::GlobalVariable(global), Span::UNDEFINED);
    let field = f
        .expressions
        .append(E::AccessIndex { base: g, index: 0 }, Span::UNDEFINED);
    let ptr = f
        .expressions
        .append(E::Load { pointer: field }, Span::UNDEFINED);
    let ptr = if chase {
        f.expressions
            .append(E::Load { pointer: ptr }, Span::UNDEFINED)
    } else {
        ptr
    };
    let value = f
        .expressions
        .append(E::Load { pointer: ptr }, Span::UNDEFINED);
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(field, value)),
        Span::UNDEFINED,
    );
    f.body.push(
        S::Store {
            pointer: ptr,
            value,
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.entry_points.push(compute_entry(f));
    m
}

#[test]
fn physical_storage_aggregate_layout_and_spirv() {
    let mut m = naga::Module::default();
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let wide = ty(
        &mut m,
        T::Scalar(naga::Scalar {
            kind: naga::ScalarKind::Uint,
            width: 8,
        }),
    );
    let vector = ty(
        &mut m,
        T::Vector {
            size: naga::VectorSize::Tri,
            scalar: naga::Scalar::F32,
        },
    );
    let matrix = ty(
        &mut m,
        T::Matrix {
            columns: naga::VectorSize::Tri,
            rows: naga::VectorSize::Bi,
            scalar: naga::Scalar::F32,
        },
    );
    let inner = ty(
        &mut m,
        T::Struct {
            members: vec![
                member(None, vector, 0),
                member(None, matrix, 16),
                member(None, wide, 40),
            ],
            span: 48,
        },
    );
    let array = ty(
        &mut m,
        T::Array {
            base: inner,
            size: fixed_size(2),
            stride: 48,
        },
    );
    let outer = ty(
        &mut m,
        T::Struct {
            members: vec![member(None, uint, 0), member(None, array, 16)],
            span: 112,
        },
    );
    let mut layout = naga::proc::Layouter::default();
    layout.update(m.to_ctx()).unwrap();
    assert_eq!(layout[inner].size, 48);
    assert_eq!(layout[inner].alignment, naga::proc::Alignment::SIXTEEN);
    assert_eq!(layout[outer].size, 112);
    check_spirv_aligned(
        roundtrip(m, outer, false),
        "nested-aggregate",
        naga::proc::BoundsCheckPolicy::Unchecked,
        8,
        C::all(),
    );
}

#[test]
fn physical_storage_pointer_chasing_spirv() {
    let mut m = naga::Module::default();
    let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let pointer = ptr_to(&mut m, scalar);
    let m = roundtrip(m, pointer, true);
    check_spirv(
        m,
        "pointer-chasing",
        naga::proc::BoundsCheckPolicy::Unchecked,
    );
}

#[test]
fn physical_storage_function_uses_require_native_capability() {
    for usage in ["local", "argument", "return"] {
        let mut m = module();
        let pointer = physical_pointer(&m);
        let mut f = naga::Function::default();
        match usage {
            "local" => {
                local(&mut f.local_variables, pointer);
                f.body.push(S::Return { value: None }, Span::UNDEFINED);
            }
            "argument" => {
                f.arguments.push(argument(pointer));
                f.body.push(S::Return { value: None }, Span::UNDEFINED);
            }
            "return" => {
                f.result = Some(returns(pointer));
                let global = m.global_variables.iter().next().unwrap().0;
                let root = f
                    .expressions
                    .append(E::GlobalVariable(global), Span::UNDEFINED);
                let field = f.expressions.append(
                    E::AccessIndex {
                        base: root,
                        index: 0,
                    },
                    Span::UNDEFINED,
                );
                let value = f
                    .expressions
                    .append(E::Load { pointer: field }, Span::UNDEFINED);
                f.body.push(
                    S::Emit(naga::Range::new_from_bounds(field, value)),
                    Span::UNDEFINED,
                );
                f.body
                    .push(S::Return { value: Some(value) }, Span::UNDEFINED);
            }
            _ => unreachable!(),
        }
        m.functions.append(f, Span::UNDEFINED);
        let caps = C::default() | C::IMMEDIATES;
        let error = Validator::new(ValidationFlags::all(), caps)
            .validate(&m)
            .unwrap_err();
        assert!(
            format!("{error:?}").contains("PHYSICAL_STORAGE_BUFFER_ADDRESSES"),
            "{usage}: {error:?}"
        );
        validate_native_words(
            &m,
            caps | C::PHYSICAL_STORAGE_BUFFER_ADDRESSES,
            &format!("native-function-{usage}"),
            false,
        );
    }
}

#[test]
fn physical_storage_cooperative_memory() {
    let mut m = module();
    let f = &mut m.entry_points[0].function;
    f.body.cull(f.body.len() - 1..);
    let pointer = f
        .expressions
        .iter()
        .find(|(_, e)| matches!(e, E::Load { .. }))
        .unwrap()
        .0;
    let element = f
        .expressions
        .iter()
        .find(|(_, e)| matches!(e, E::AccessIndex { base, .. } if *base == pointer))
        .unwrap()
        .0;
    let stride = f
        .expressions
        .append(E::Literal(naga::Literal::U32(8)), Span::UNDEFINED);
    let result = f.expressions.append(
        E::CooperativeLoad {
            columns: naga::CooperativeSize::Eight,
            rows: naga::CooperativeSize::Eight,
            role: naga::CooperativeRole::A,
            data: naga::CooperativeData {
                pointer: element,
                stride,
                row_major: true,
            },
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(result, result)),
        Span::UNDEFINED,
    );
    f.body.push(
        S::CooperativeStore {
            target: result,
            data: naga::CooperativeData {
                pointer: element,
                stride,
                row_major: true,
            },
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    let error = Validator::new(ValidationFlags::all(), C::all() - C::COOPERATIVE_MATRIX)
        .validate(&m)
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("COOPERATIVE_MATRIX"),
        "{error:?}"
    );
    let text = validate_native_words(&m, C::all(), "cooperative-memory", false);
    assert!(text.contains("OpCooperativeMatrixLoadKHR"), "{text}");
    assert!(text.contains("OpCooperativeMatrixStoreKHR"), "{text}");
    assert!(
        text.lines()
            .filter(|line| line.contains("OpCooperativeMatrixLoadKHR")
                || line.contains("OpCooperativeMatrixStoreKHR"))
            .all(|line| line.contains("Aligned 4")),
        "{text}"
    );
    let atomic = ty(&mut m, T::Atomic(naga::Scalar::U32));
    let pointer = ptr_to(&mut m, atomic);
    let (mut f, pointer) = pointer_function(pointer);
    let value = append_emit(&mut f, E::Literal(naga::Literal::U32(1)));
    f.body.push(
        S::Atomic {
            pointer,
            value,
            fun: naga::AtomicFunction::Add,
            result: None,
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.functions.append(f, Span::UNDEFINED);
    let text = validate_native_words(&m, C::all(), "cooperative-device-atomics", false);
    assert!(
        text.contains("OpCapability VulkanMemoryModelDeviceScope"),
        "{text}"
    );
}

#[path = "../../physical-storage-gpu/pointer_helpers.rs"]
mod pointer_helpers;

#[test]
fn physical_storage_function_pointer_operations() {
    let mut m = module();
    let pointer = physical_pointer(&m);
    pointer_helpers::helper(&mut m, pointer);
    for compact in [false, true] {
        if compact {
            naga::compact::compact(&mut m, naga::compact::KeepUnused::Yes);
        }
        check_spirv_aligned(
            m.clone(),
            if compact { "helper-compact" } else { "helper" },
            naga::proc::BoundsCheckPolicy::Restrict,
            4,
            C::default() | C::IMMEDIATES | C::SHADER_INT64 | C::PHYSICAL_STORAGE_BUFFER_ADDRESSES,
        );
    }
}

#[test]
fn physical_storage_scalar_layout_is_explicit() {
    let mut m = module();
    let data = m
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("Data"))
        .unwrap()
        .0;
    let scalar = if let T::Struct { ref members, .. } = m.types[data].inner {
        members[0].ty
    } else {
        unreachable!()
    };
    // Rebuild type order because member types must precede their structures.
    let mut types = naga::UniqueArena::new();
    for (h, ty) in m.types.iter() {
        if h == data {
            break;
        }
        types.insert(ty.clone(), Span::UNDEFINED);
    }
    let vector = types.insert(
        Type {
            name: None,
            inner: T::Vector {
                size: naga::VectorSize::Tri,
                scalar: naga::Scalar::F32,
            },
        },
        Span::UNDEFINED,
    );
    let scalar_data = types.insert(
        Type {
            name: Some("ScalarData".into()),
            inner: T::Struct {
                members: vec![member(None, scalar, 0), member(None, vector, 4)],
                span: 16,
            },
        },
        Span::UNDEFINED,
    );
    let pointer = types.insert(
        Type {
            name: None,
            inner: T::Pointer {
                base: scalar_data,
                space: AddressSpace::PhysicalStorage,
            },
        },
        Span::UNDEFINED,
    );
    m = naga::Module {
        types,
        ..Default::default()
    };
    let caps = C::default() | C::PHYSICAL_STORAGE_BUFFER_ADDRESSES;
    assert!(Validator::new(ValidationFlags::all(), caps)
        .validate(&m)
        .is_err());
    Validator::new(
        ValidationFlags::all(),
        caps | C::PHYSICAL_STORAGE_SCALAR_LAYOUT,
    )
    .validate(&m)
    .unwrap();
    let mut fun = naga::Function::default();
    fun.arguments.push(argument(pointer));
    let arg = fun
        .expressions
        .append(E::FunctionArgument(0), Span::UNDEFINED);
    let field = fun.expressions.append(
        E::AccessIndex {
            base: arg,
            index: 1,
        },
        Span::UNDEFINED,
    );
    let value = fun
        .expressions
        .append(E::Load { pointer: field }, Span::UNDEFINED);
    fun.body.push(
        S::Emit(naga::Range::new_from_bounds(field, value)),
        Span::UNDEFINED,
    );
    fun.body.push(
        S::Store {
            pointer: field,
            value,
        },
        Span::UNDEFINED,
    );
    fun.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.functions.append(fun, Span::UNDEFINED);
    m.entry_points
        .push(compute_entry(naga::Function::default()));
    validate_native_words(
        &m,
        caps | C::PHYSICAL_STORAGE_SCALAR_LAYOUT,
        "scalar-layout",
        true,
    );
}

fn native_words(m: &naga::Module, caps: C) -> Vec<u32> {
    let info = Validator::new(ValidationFlags::all(), caps)
        .validate(m)
        .unwrap();
    naga::back::spv::write_vec(
        m,
        &info,
        &naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        },
        None,
    )
    .unwrap()
}

fn validate_native_words(m: &naga::Module, caps: C, name: &str, scalar: bool) -> String {
    validate_emitted_words(&native_words(m, caps), name, scalar)
}

#[cfg(feature = "spv-in")]
fn import(words: &[u32], name: &str) -> naga::Module {
    naga::front::spv::Frontend::new(
        words.iter().copied(),
        &naga::front::spv::Options {
            adjust_coordinate_space: false,
            ..Default::default()
        },
    )
    .parse()
    .unwrap_or_else(|error| panic!("import {name}: {error:?}"))
}

/// Runs `spirv-val` on `words` and, with `spv-in`, checks that importing,
/// compacting and re-exporting them stays valid and preserves memory operands.
fn validate_emitted_words(words: &[u32], name: &str, scalar: bool) -> String {
    let text = validate_spirv_words(words, name, scalar);
    #[cfg(feature = "spv-in")]
    {
        let mut parsed = import(words, name);
        for compact in [false, true] {
            if compact {
                naga::compact::compact(&mut parsed, naga::compact::KeepUnused::No);
            }
            let info = Validator::new(ValidationFlags::all(), C::all())
                .validate(&parsed)
                .unwrap_or_else(|error| {
                    panic!("validate import {name}, compact={compact}: {error:?}")
                });
            let output = naga::back::spv::write_vec(
                &parsed,
                &info,
                &naga::back::spv::Options {
                    lang_version: (1, 3),
                    ..Default::default()
                },
                None,
            )
            .unwrap_or_else(|error| panic!("export {name}: {error:?}"));
            validate_spirv_words(&output, &format!("{name}-import-{compact}"), scalar);
            if !compact {
                assert_eq!(
                    native_memory_operands(words),
                    native_memory_operands(&output),
                    "memory semantics changed while importing {name}"
                );
            }
        }
    }
    text
}

fn validate_spirv_words(words: &[u32], name: &str, scalar: bool) -> String {
    static NEXT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
    let text = rspirv::dr::load_words(words).unwrap().disassemble();
    let file_name: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let file = std::env::temp_dir().join(format!(
        "naga-physical-storage-{file_name}-{}-{}.spv",
        std::process::id(),
        NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::write(
        &file,
        words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut command = std::process::Command::new(
        std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into()),
    );
    command.args(["--target-env", "vulkan1.1"]);
    if scalar {
        command.arg("--scalar-block-layout");
    }
    let result = command
        .arg(&file)
        .output()
        .expect("spirv-val is required; install the Vulkan SDK or set SPIRV_VAL");
    std::fs::remove_file(file).unwrap();
    assert!(
        result.status.success(),
        "{name}: {}
{text}",
        String::from_utf8_lossy(&result.stderr)
    );
    text
}

#[test]
fn physical_storage_aggregate_arguments_returns_and_locals() {
    let mut m = module();
    let root = m.global_variables.iter().next().unwrap().1.ty;
    let mut f = naga::Function::default();
    f.arguments.push(argument(root));
    f.result = Some(returns(root));
    let arg = f
        .expressions
        .append(E::FunctionArgument(0), Span::UNDEFINED);
    let local = local(&mut f.local_variables, root);
    let place = f
        .expressions
        .append(E::LocalVariable(local), Span::UNDEFINED);
    f.body.push(
        S::Store {
            pointer: place,
            value: arg,
        },
        Span::UNDEFINED,
    );
    let value = f
        .expressions
        .append(E::Load { pointer: place }, Span::UNDEFINED);
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(value, value)),
        Span::UNDEFINED,
    );
    f.body
        .push(S::Return { value: Some(value) }, Span::UNDEFINED);
    m.functions.append(f, Span::UNDEFINED);
    let text = validate_native_words(&m, C::all(), "aggregate-helper", false);
    assert!(text.contains("OpReturnValue"));
}

#[test]
fn physical_storage_casts_and_offsets_reject_invalid_types() {
    for source in [
        naga::Literal::U32(0),
        naga::Literal::F64(0.0),
        naga::Literal::I64(0),
    ] {
        let mut m = module();
        let ptr = physical_pointer(&m);
        let f = &mut m.entry_points[0].function;
        let value = f.expressions.append(E::Literal(source), Span::UNDEFINED);
        f.expressions.append(
            E::PointerCast {
                expr: value,
                ty: ptr,
            },
            Span::UNDEFINED,
        );
        let error = Validator::new(ValidationFlags::all(), C::all())
            .validate(&m)
            .unwrap_err();
        assert!(
            format!("{error:?}").contains("InvalidCastArgument"),
            "{error:?}"
        );
    }
    let mut m = module();
    let f = &mut m.entry_points[0].function;
    let value = f
        .expressions
        .append(E::Literal(naga::Literal::U64(1)), Span::UNDEFINED);
    f.expressions.append(
        E::PointerOffset {
            pointer: value,
            offset: value,
        },
        Span::UNDEFINED,
    );
    assert!(Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .is_err());
}

#[test]
fn physical_storage_bounded_span_and_aliasing_helpers() {
    let basic =
        C::default() | C::IMMEDIATES | C::SHADER_INT64 | C::PHYSICAL_STORAGE_BUFFER_ADDRESSES;
    for scalar in [false, true] {
        let mut m = pointer_helpers::span_module(scalar);
        let caps = if scalar {
            assert!(Validator::new(ValidationFlags::all(), basic)
                .validate(&m)
                .is_err());
            basic | C::PHYSICAL_STORAGE_SCALAR_LAYOUT
        } else {
            basic
        };
        for compact in [false, true] {
            if compact {
                naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
            }
            let text = validate_native_words(&m, caps, &format!("span-{scalar}-{compact}"), scalar);
            for instruction in [
                "OpConvertUToPtr",
                "OpConvertPtrToU",
                "OpFunctionCall",
                "AliasedPointer",
                "OpSelectionMerge",
            ] {
                assert!(text.contains(instruction), "{instruction}: {text}");
            }
        }
    }
}

#[test]
fn physical_storage_pointer_holder_parameter_is_aliased() {
    let mut m = module();
    let pointer = physical_pointer(&m);
    let holder = ty(
        &mut m,
        T::Pointer {
            base: pointer,
            space: AddressSpace::Function,
        },
    );
    let mut f = naga::Function::default();
    f.arguments.push(argument(holder));
    f.result = Some(returns(pointer));
    let p = f
        .expressions
        .append(E::FunctionArgument(0), Span::UNDEFINED);
    let v = f
        .expressions
        .append(E::Load { pointer: p }, Span::UNDEFINED);
    f.body
        .push(S::Emit(naga::Range::new_from_bounds(v, v)), Span::UNDEFINED);
    f.body.push(S::Return { value: Some(v) }, Span::UNDEFINED);
    m.functions.append(f, Span::UNDEFINED);
    let text = validate_native_words(&m, C::all(), "holder-parameter", false);
    assert!(text.contains("AliasedPointer"));
}

#[test]
fn physical_storage_pointer_zero_values_and_constants() {
    for (holder, init_kind) in [
        ("pointer", None),
        ("pointer", Some("zero")),
        ("aggregate", None),
        ("aggregate", Some("zero")),
        ("array", Some("compose")),
    ] {
        let mut m = module();
        let pointer = physical_pointer(&m);
        let local_ty = match holder {
            "pointer" => pointer,
            "aggregate" => m.global_variables.iter().next().unwrap().1.ty,
            _ => ty(
                &mut m,
                T::Array {
                    base: pointer,
                    size: fixed_size(2),
                    stride: 8,
                },
            ),
        };
        let f = &mut m.entry_points[0].function;
        f.body.cull(f.body.len() - 1..);
        let init = init_kind.map(|kind| {
            if kind == "compose" {
                let zero = f.expressions.append(E::ZeroValue(pointer), Span::UNDEFINED);
                f.expressions.append(
                    E::Compose {
                        ty: local_ty,
                        components: vec![zero, zero],
                    },
                    Span::UNDEFINED,
                )
            } else {
                f.expressions
                    .append(E::ZeroValue(local_ty), Span::UNDEFINED)
            }
        });
        let local = f.local_variables.append(
            naga::LocalVariable {
                name: None,
                ty: local_ty,
                init,
            },
            Span::UNDEFINED,
        );
        let p = append_emit(f, E::LocalVariable(local));
        let value = append_emit(f, E::Load { pointer: p });
        f.body.push(S::Store { pointer: p, value }, Span::UNDEFINED);
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        let text =
            validate_native_words(&m, C::all(), &format!("null-{holder}-{init_kind:?}"), false);
        assert!(text.contains("OpBitcast"), "{text}");
    }
    for cast in [false, true] {
        let mut m = module();
        let pointer = physical_pointer(&m);
        let expression = if cast {
            let zero = m
                .global_expressions
                .append(E::Literal(naga::Literal::U64(0)), Span::UNDEFINED);
            E::PointerCast {
                expr: zero,
                ty: pointer,
            }
        } else {
            E::ZeroValue(pointer)
        };
        m.global_expressions.append(expression, Span::UNDEFINED);
        assert!(Validator::new(ValidationFlags::all(), C::all())
            .validate(&m)
            .is_err());
    }
}

#[test]
fn physical_storage_span_survives_override_processing() {
    let mut m = pointer_helpers::span_module(false);
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let init = m
        .global_expressions
        .append(E::Literal(naga::Literal::U32(1)), Span::UNDEFINED);
    let count = m.overrides.append(
        naga::Override {
            name: Some("workgroup".into()),
            id: None,
            ty: uint,
            init: Some(init),
        },
        Span::UNDEFINED,
    );
    let expr = m
        .global_expressions
        .append(E::Override(count), Span::UNDEFINED);
    m.entry_points[0].workgroup_size_overrides = Some([Some(expr), None, None]);
    let info = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap();
    let mut constants = naga::back::PipelineConstants::default();
    constants.insert("workgroup".into(), 4.0);
    let (resolved, _) = naga::back::pipeline_constants::process_overrides(
        &m,
        &info,
        Some((naga::ShaderStage::Compute, "main")),
        &constants,
    )
    .unwrap();
    assert_eq!(resolved.entry_points[0].workgroup_size[0], 4);
    validate_native_words(&resolved, C::all(), "span-overrides", false);
}

#[test]
fn physical_storage_span_root_layout() {
    let mut m = naga::Module::default();
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let pointer = ptr_to(&mut m, uint);
    let span = ty(
        &mut m,
        T::Struct {
            members: vec![
                member(None, pointer, 0),
                member(None, uint, 8),
                member(None, uint, 12),
            ],
            span: 16,
        },
    );
    let mut members: Vec<_> = (0..20).map(|i| member(None, span, i * 16)).collect();
    members.extend((0..8).map(|i| member(None, uint, 320 + i * 4)));
    let root = ty(&mut m, T::Struct { members, span: 352 });
    let root_pointer = ptr_to(&mut m, root);
    let span_pointer = ptr_to(&mut m, span);
    let mut f = naga::Function::default();
    f.arguments.push(argument(root_pointer));
    f.result = Some(returns(span_pointer));
    let arg = f
        .expressions
        .append(E::FunctionArgument(0), Span::UNDEFINED);
    let member = f.expressions.append(
        E::AccessIndex {
            base: arg,
            index: 19,
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(member, member)),
        Span::UNDEFINED,
    );
    f.body.push(
        S::Return {
            value: Some(member),
        },
        Span::UNDEFINED,
    );
    m.functions.append(f, Span::UNDEFINED);
    m.entry_points
        .push(compute_entry(naga::Function::default()));
    let mut layout = naga::proc::Layouter::default();
    layout.update(m.to_ctx()).unwrap();
    assert_eq!(layout[pointer].size, 8);
    assert_eq!(layout[pointer].alignment, naga::proc::Alignment::EIGHT);
    assert_eq!(layout[span].size, 16);
    assert_eq!(layout[root].size, 352);
    let text = validate_native_words(&m, C::all(), "span-root", false);
    assert!(text.contains("Offset 304"));
    assert!(text.contains("Offset 348"));
}

#[test]
fn physical_storage_projected_pointer_values() {
    for operation in ["cast", "offset", "select", "compose", "store", "call"] {
        let mut m = module();
        let pointer = physical_pointer(&m);
        let scalar = m
            .types
            .iter()
            .find(|(_, t)| t.inner == T::Scalar(naga::Scalar::F32))
            .unwrap()
            .0;
        let scalar_ptr = ptr_to(&mut m, scalar);
        let uint64 = ty(&mut m, T::Scalar(naga::Scalar::U64));
        let holder = ty(
            &mut m,
            T::Struct {
                members: vec![member(None, scalar_ptr, 0)],
                span: 8,
            },
        );
        let helper = pointer_helpers::helper(&mut m, scalar_ptr);
        let mut f = naga::Function::default();
        f.arguments.push(argument(pointer));
        f.result = Some(returns(scalar_ptr));
        let arg = f
            .expressions
            .append(E::FunctionArgument(0), Span::UNDEFINED);
        let zero = f
            .expressions
            .append(E::Literal(naga::Literal::U64(0)), Span::UNDEFINED);
        let condition = f
            .expressions
            .append(E::Literal(naga::Literal::Bool(true)), Span::UNDEFINED);
        let local = local(&mut f.local_variables, scalar_ptr);
        let place = f
            .expressions
            .append(E::LocalVariable(local), Span::UNDEFINED);
        let field = f.expressions.append(
            E::AccessIndex {
                base: arg,
                index: 0,
            },
            Span::UNDEFINED,
        );
        f.body.push(
            S::Emit(naga::Range::new_from_bounds(field, field)),
            Span::UNDEFINED,
        );
        let result = match operation {
            "cast" => {
                let address = f.expressions.append(
                    E::PointerCast {
                        expr: field,
                        ty: uint64,
                    },
                    Span::UNDEFINED,
                );
                let result = f.expressions.append(
                    E::PointerCast {
                        expr: address,
                        ty: scalar_ptr,
                    },
                    Span::UNDEFINED,
                );
                f.body.push(
                    S::Emit(naga::Range::new_from_bounds(address, result)),
                    Span::UNDEFINED,
                );
                result
            }
            "offset" | "select" => {
                let expression = if operation == "offset" {
                    E::PointerOffset {
                        pointer: field,
                        offset: zero,
                    }
                } else {
                    E::Select {
                        condition,
                        accept: field,
                        reject: field,
                    }
                };
                let result = f.expressions.append(expression, Span::UNDEFINED);
                f.body.push(
                    S::Emit(naga::Range::new_from_bounds(result, result)),
                    Span::UNDEFINED,
                );
                result
            }
            "compose" => {
                let record = f.expressions.append(
                    E::Compose {
                        ty: holder,
                        components: vec![field],
                    },
                    Span::UNDEFINED,
                );
                let result = f.expressions.append(
                    E::AccessIndex {
                        base: record,
                        index: 0,
                    },
                    Span::UNDEFINED,
                );
                f.body.push(
                    S::Emit(naga::Range::new_from_bounds(record, result)),
                    Span::UNDEFINED,
                );
                result
            }
            "store" => {
                f.body.push(
                    S::Store {
                        pointer: place,
                        value: field,
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
                result
            }
            "call" => {
                let result = f.expressions.append(E::CallResult(helper), Span::UNDEFINED);
                f.body.push(
                    S::Call {
                        function: helper,
                        arguments: vec![field],
                        result: Some(result),
                    },
                    Span::UNDEFINED,
                );
                result
            }
            _ => unreachable!(),
        };
        f.body.push(
            S::Return {
                value: Some(result),
            },
            Span::UNDEFINED,
        );
        m.functions.append(f, Span::UNDEFINED);
        validate_native_words(&m, C::all(), operation, false);
    }
}

#[test]
fn physical_storage_atomic_operations() {
    let caps =
        C::default() | C::IMMEDIATES | C::SHADER_INT64 | C::PHYSICAL_STORAGE_BUFFER_ADDRESSES;
    for scalar in [false, true] {
        let mut m = pointer_helpers::span_module_with_atomics(scalar, true);
        assert!(Validator::new(
            ValidationFlags::all(),
            caps - C::PHYSICAL_STORAGE_BUFFER_ADDRESSES
        )
        .validate(&m)
        .is_err());
        for compact in [false, true] {
            if compact {
                naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
            }
            let text = validate_native_words(
                &m,
                caps | C::PHYSICAL_STORAGE_SCALAR_LAYOUT,
                &format!("atomics-{scalar}-{compact}"),
                scalar,
            );
            for op in [
                "OpAtomicLoad",
                "OpAtomicStore",
                "OpAtomicIAdd",
                "OpAtomicISub",
                "OpAtomicOr",
                "OpAtomicXor",
                "OpAtomicAnd",
                "OpAtomicUMin",
                "OpAtomicUMax",
                "OpAtomicExchange",
                "OpAtomicCompareExchange",
            ] {
                assert!(text.contains(op), "missing {op}: {text}");
            }
        }
    }
}

#[test]
fn physical_storage_standalone_matrix_pointer() {
    for (array, projected) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut m = module();
        let matrix = ty(
            &mut m,
            T::Matrix {
                columns: naga::VectorSize::Tri,
                rows: naga::VectorSize::Tri,
                scalar: naga::Scalar::F32,
            },
        );
        let base = if array {
            ty(
                &mut m,
                T::Array {
                    base: matrix,
                    size: fixed_size(2),
                    stride: 48,
                },
            )
        } else {
            matrix
        };
        let ptr = ptr_to(&mut m, base);
        let identity = pointer_helpers::helper(&mut m, ptr);
        let argument_ty = if projected {
            let record = ty(
                &mut m,
                T::Struct {
                    members: vec![member(None, base, 0)],
                    span: if array { 96 } else { 48 },
                },
            );
            ptr_to(&mut m, record)
        } else {
            ptr
        };

        let (mut f, p) = pointer_function(argument_ty);
        let p = if projected {
            append_emit(&mut f, E::AccessIndex { base: p, index: 0 })
        } else {
            p
        };
        let result = f
            .expressions
            .append(E::CallResult(identity), Span::UNDEFINED);
        f.body.push(
            S::Call {
                function: identity,
                arguments: vec![p],
                result: Some(result),
            },
            Span::UNDEFINED,
        );
        let value = append_emit(&mut f, E::Load { pointer: result });
        f.body.push(
            S::Store {
                pointer: result,
                value,
            },
            Span::UNDEFINED,
        );
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        m.functions.append(f, Span::UNDEFINED);
        let text = validate_native_words(
            &m,
            C::all(),
            &format!("matrix-pointer-{array}-{projected}"),
            false,
        );
        assert!(text.contains("MatrixStride 16"));
    }
}

fn append_emit(f: &mut naga::Function, expression: E) -> naga::Handle<E> {
    let pre = expression.needs_pre_emit();
    let h = f.expressions.append(expression, Span::UNDEFINED);
    if !pre {
        f.body
            .push(S::Emit(naga::Range::new_from_bounds(h, h)), Span::UNDEFINED);
    }
    h
}

#[test]
fn physical_storage_explicit_alignment() {
    for alignment in [0, 2, 3, 4, 16, 256] {
        let mut m = module();
        let scalar = ty(&mut m, T::Scalar(naga::Scalar::F32));
        let pointer = ptr_to(&mut m, scalar);
        let (mut f, p) = pointer_function(pointer);
        let aligned = append_emit(
            &mut f,
            E::PointerAlignment {
                pointer: p,
                alignment,
            },
        );
        let v = append_emit(&mut f, E::Load { pointer: aligned });
        f.body.push(
            S::Store {
                pointer: aligned,
                value: v,
            },
            Span::UNDEFINED,
        );
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        let helper = m.functions.append(f, Span::UNDEFINED);
        let entry = &mut m.entry_points[0].function;
        let argument = entry
            .body
            .iter()
            .find_map(|s| match *s {
                S::Store { pointer, .. } => Some(pointer),
                _ => None,
            })
            .unwrap();
        entry.body.cull(entry.body.len() - 1..);
        entry.body.push(
            S::Call {
                function: helper,
                arguments: vec![argument],
                result: None,
            },
            Span::UNDEFINED,
        );
        entry.body.push(S::Return { value: None }, Span::UNDEFINED);
        if !alignment.is_power_of_two() || alignment < 4 {
            let err = Validator::new(ValidationFlags::all(), C::all())
                .validate(&m)
                .unwrap_err();
            assert!(format!("{err:?}").contains("InvalidPointerAlignment"));
        } else {
            for compact in [false, true] {
                if compact {
                    naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
                }
                let text = validate_native_words(
                    &m,
                    C::all(),
                    &format!("aligned-{alignment}-{compact}"),
                    false,
                );
                assert!(
                    text.matches(&format!("Aligned {alignment}")).count() >= 2,
                    "{text}"
                );
            }
        }
    }
}

#[test]
fn physical_storage_atomic_ordering_restrictions() {
    for order in [
        naga::AtomicMemoryOrder::Relaxed,
        naga::AtomicMemoryOrder::Acquire,
        naga::AtomicMemoryOrder::Release,
        naga::AtomicMemoryOrder::AcquireRelease,
    ] {
        for store in [false, true] {
            let mut m = module();
            let atomic = ty(&mut m, T::Atomic(naga::Scalar::U32));
            let pointer = ptr_to(&mut m, atomic);
            let (mut f, p) = pointer_function(pointer);
            let p = append_emit(
                &mut f,
                E::AtomicPointer {
                    pointer: p,
                    order,
                    failure_order: None,
                },
            );
            if store {
                let value = append_emit(&mut f, E::Literal(naga::Literal::U32(7)));
                f.body.push(S::Store { pointer: p, value }, Span::UNDEFINED);
            } else {
                append_emit(&mut f, E::Load { pointer: p });
            }
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            m.functions.append(f, Span::UNDEFINED);
            let legal = order == naga::AtomicMemoryOrder::Relaxed
                || order
                    == if store {
                        naga::AtomicMemoryOrder::Release
                    } else {
                        naga::AtomicMemoryOrder::Acquire
                    };
            if legal {
                validate_native_words(&m, C::all(), &format!("order-{order:?}-{store}"), false);
            } else {
                assert!(Validator::new(ValidationFlags::all(), C::all())
                    .validate(&m)
                    .is_err());
            }
        }
    }
}

#[test]
fn physical_storage_checked_atomic_load_store() {
    let text = checked_words(&atomic_array_module(), "checked-atomic", false);
    assert!(
        text.contains("OpAtomicLoad") && text.contains("OpAtomicStore"),
        "{text}"
    );
}

#[test]
fn physical_storage_atomic_scalar_capabilities() {
    for scalar in [
        naga::Scalar::U32,
        naga::Scalar::I32,
        naga::Scalar::U64,
        naga::Scalar::I64,
        naga::Scalar::F32,
    ] {
        let mut m = module();
        let atom = ty(&mut m, T::Atomic(scalar));
        let value_type = ty(&mut m, T::Scalar(scalar));
        let pointer = ptr_to(&mut m, atom);
        let mut f = naga::Function::default();
        for ty in [pointer, value_type] {
            f.arguments.push(argument(ty));
        }
        let p = append_emit(&mut f, E::FunctionArgument(0));
        let value = append_emit(&mut f, E::FunctionArgument(1));
        f.body.push(
            S::Atomic {
                pointer: p,
                fun: naga::AtomicFunction::Add,
                value,
                result: None,
            },
            Span::UNDEFINED,
        );
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        m.functions.append(f, Span::UNDEFINED);
        let basic = C::default() | C::IMMEDIATES | C::PHYSICAL_STORAGE_BUFFER_ADDRESSES;
        if scalar.width == 8 || scalar == naga::Scalar::F32 {
            assert!(Validator::new(ValidationFlags::all(), basic)
                .validate(&m)
                .is_err());
        }
        validate_native_words(
            &m,
            C::all(),
            &format!("atomic-{:?}-{}", scalar.kind, scalar.width),
            false,
        );
    }
}

#[test]
fn physical_storage_extended_spans_and_compaction() {
    use pointer_helpers::SpanMode;
    for mode in [SpanMode::Matrix, SpanMode::Contended, SpanMode::Feedback] {
        let scalar = mode == SpanMode::Matrix;
        let mut m = pointer_helpers::span_module_with_mode(scalar, mode);
        for compact in [false, true] {
            if compact {
                naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
            }
            let text =
                validate_native_words(&m, C::all(), &format!("span-{mode:?}-{compact}"), scalar);
            assert!(text.contains(match mode {
                SpanMode::Matrix => "MatrixStride 8",
                SpanMode::Contended => "OpAtomicIAdd",
                SpanMode::Feedback => "OpAtomicOr",
                _ => unreachable!(),
            }));
            assert!(text.contains("OpConvertUToPtr"));
        }
    }
}

#[test]
fn physical_storage_atomic_aggregate_load_rejected() {
    let mut m = module();
    let atomic = ty(&mut m, T::Atomic(naga::Scalar::U32));
    let record = ty(
        &mut m,
        T::Struct {
            members: vec![member(None, atomic, 0)],
            span: 4,
        },
    );
    let pointer = ptr_to(&mut m, record);
    let (mut f, p) = pointer_function(pointer);
    append_emit(&mut f, E::Load { pointer: p });
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.functions.append(f, Span::UNDEFINED);
    let err = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap_err();
    assert!(format!("{err:?}").contains("InvalidPointerType"));
}

#[test]
fn physical_storage_checked_atomic_updates() {
    for annotated in [false, true] {
        for returned in [false, true] {
            let mut m = atomic_array_module();
            let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
            let f = &mut m.entry_points[0].function;
            let pointer = f
                .expressions
                .iter()
                .find_map(|(_, e)| match *e {
                    E::Load { pointer } if matches!(f.expressions[pointer], E::Access { .. }) => {
                        Some(pointer)
                    }
                    _ => None,
                })
                .unwrap();
            f.body.cull(f.body.len() - 1..);
            let pointer = if annotated {
                let pointer = append_emit(
                    f,
                    E::AtomicPointer {
                        pointer,
                        order: naga::AtomicMemoryOrder::AcquireRelease,
                        failure_order: None,
                    },
                );
                append_emit(
                    f,
                    E::PointerAlignment {
                        pointer,
                        alignment: 4,
                    },
                )
            } else {
                pointer
            };
            let value = append_emit(f, E::Literal(naga::Literal::U32(7)));
            let result = returned.then(|| {
                f.expressions.append(
                    E::AtomicResult {
                        ty: uint,
                        comparison: false,
                    },
                    Span::UNDEFINED,
                )
            });
            f.body.push(
                S::Atomic {
                    pointer,
                    fun: naga::AtomicFunction::Add,
                    value,
                    result,
                },
                Span::UNDEFINED,
            );
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            let text = checked_words(&m, &format!("checked-update-{annotated}-{returned}"), false);
            assert_eq!(text.matches("OpAtomicIAdd").count(), 1, "{text}");
            assert!(
                text.contains("OpPhi") && text.contains("OpBranchConditional"),
                "{text}"
            );
        }
    }
    // The GPU workload's variant: the checked load becomes the atomic's result.
    let mut m = atomic_array_module();
    pointer_helpers::use_checked_atomics(&mut m);
    let text = checked_words(&m, "checked-update-workload", false);
    assert_eq!(text.matches("OpAtomicIAdd").count(), 1, "{text}");
    assert!(
        text.contains("OpPhi") && text.contains("OpBranchConditional"),
        "{text}"
    );
}

#[test]
fn physical_storage_runtime_array_access() {
    for wrapped in [false, true] {
        let mut m = module_with_array(true);
        let array = m
            .types
            .iter()
            .find(|(_, t)| matches!(t.inner, T::Array { .. }))
            .unwrap()
            .0;
        let T::Array { base, stride, .. } = m.types[array].inner else {
            unreachable!()
        };
        m.types.replace(
            array,
            Type {
                name: None,
                inner: T::Array {
                    base,
                    stride,
                    size: naga::ArraySize::Dynamic,
                },
            },
        );
        if !wrapped {
            let pointer = physical_pointer(&m);
            m.types.replace(
                pointer,
                Type {
                    name: None,
                    inner: T::Pointer {
                        base: array,
                        space: AddressSpace::PhysicalStorage,
                    },
                },
            );
            let f = &mut m.entry_points[0].function;
            let (projected, pointer) = f
                .expressions
                .iter()
                .find_map(|(h, e)| match *e {
                    E::AccessIndex { base, index: 0 }
                        if matches!(f.expressions[base], E::Load { .. }) =>
                    {
                        Some((h, base))
                    }
                    _ => None,
                })
                .unwrap();
            f.expressions[projected] = E::PointerAlignment {
                pointer,
                alignment: 4,
            };
        }
        let info = Validator::new(ValidationFlags::all(), C::all())
            .validate(&m)
            .unwrap();
        for policy in [
            naga::proc::BoundsCheckPolicy::Unchecked,
            naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            naga::proc::BoundsCheckPolicy::Restrict,
        ] {
            let words = naga::back::spv::write_vec(
                &m,
                &info,
                &naga::back::spv::Options {
                    bounds_check_policies: naga::proc::BoundsCheckPolicies {
                        index: policy,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            );
            if policy == naga::proc::BoundsCheckPolicy::Unchecked {
                let text =
                    validate_emitted_words(&words.unwrap(), &format!("runtime-{wrapped}"), false);
                assert!(text.contains("OpTypeRuntimeArray"), "{text}");
                assert!(!text.contains("OpArrayLength"), "{text}");
            } else {
                assert!(format!("{:?}", words.unwrap_err())
                    .contains("device pointers do not carry runtime array lengths"));
            }
        }
        let f = &mut m.entry_points[0].function;
        let base = f
            .expressions
            .iter()
            .find_map(|(_, e)| match *e {
                E::Access { base, .. } => Some(base),
                _ => None,
            })
            .unwrap();
        f.body.cull(f.body.len() - 1..);
        append_emit(f, E::ArrayLength(base));
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        assert!(Validator::new(ValidationFlags::all(), C::all())
            .validate(&m)
            .is_err());
    }
}

#[test]
fn physical_storage_matrix_memory_layouts() {
    for row_major in [false, true] {
        for stride in [3, 4, 8] {
            let mut m = module();
            let f = &mut m.entry_points[0].function;
            f.body.cull(f.body.len() - 1..);
            let pointer = f.expressions.iter().find_map(|(_, e)| match *e {
                E::Load { pointer } if matches!(f.expressions[pointer], E::AccessIndex { base, .. } if matches!(f.expressions[base], E::Load { .. })) => Some(pointer),
                _ => None,
            }).unwrap();
            let stride = append_emit(f, E::Literal(naga::Literal::U32(stride)));
            let data = naga::CooperativeData {
                pointer,
                stride,
                row_major,
            };
            let value = append_emit(
                f,
                E::MatrixLoad {
                    columns: naga::VectorSize::Bi,
                    rows: naga::VectorSize::Tri,
                    data,
                },
            );
            f.body.push(
                S::MatrixStore {
                    target: value,
                    data,
                },
                Span::UNDEFINED,
            );
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            for compact in [false, true] {
                if compact {
                    naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
                }
                let text = validate_native_words(
                    &m,
                    C::all(),
                    &format!("matrix-layout-{row_major}-{stride:?}-{compact}"),
                    false,
                );
                assert!(text.contains("OpTypeMatrix"), "{text}");
                assert_eq!(
                    text.lines()
                        .filter(|l| l.contains("OpLoad") && l.contains("Aligned 4"))
                        .count(),
                    7,
                    "{text}"
                );
            }
            assert!(
                Validator::new(ValidationFlags::all(), C::all() - C::SHADER_INT64)
                    .validate(&m)
                    .is_err()
            );
        }
    }
}

#[test]
fn physical_storage_immutable_arguments() {
    let mut m = module();
    let scalar = ty(&mut m, T::Scalar(naga::Scalar::F32));
    let pointer = ptr_to(&mut m, scalar);
    let mut f = naga::Function::default();
    f.arguments.push(argument(pointer));
    f.arguments[0].immutable_pointee = true;
    f.result = Some(returns(scalar));
    let p = append_emit(&mut f, E::FunctionArgument(0));
    let value = append_emit(&mut f, E::Load { pointer: p });
    f.body
        .push(S::Return { value: Some(value) }, Span::UNDEFINED);
    let function = m.functions.append(f, Span::UNDEFINED);
    let text = validate_native_words(&m, C::all(), "immutable-argument", false);
    assert!(
        text.lines()
            .any(|line| line.contains("OpDecorate") && line.ends_with(" Restrict")),
        "{text}"
    );
    naga::compact::compact(&mut m, naga::compact::KeepUnused::Yes);
    validate_native_words(&m, C::all(), "immutable-compact", false);
    // Stores through an immutable argument are covered by
    // `physical_storage_immutable_access_paths` ("direct").
    m.functions[function].arguments[0].immutable_pointee = false;
    let function_pointer = ty(
        &mut m,
        T::Pointer {
            base: scalar,
            space: naga::AddressSpace::Function,
        },
    );
    for non_physical in [scalar, function_pointer] {
        let mut m = m.clone();
        let mut arg = argument(non_physical);
        arg.immutable_pointee = true;
        m.functions[function].arguments.push(arg);
        let error = Validator::new(ValidationFlags::all(), C::all())
            .validate(&m)
            .unwrap_err();
        assert!(
            format!("{error:?}").contains("InvalidImmutablePointerArgument(1)"),
            "{error:?}"
        );
    }
}

#[test]
fn physical_storage_native_matrix_invalid_inputs() {
    for scalar in [naga::Scalar::U32, naga::Scalar::F32] {
        for stride_float in [false, true] {
            let mut m = module();
            let base = ty(&mut m, T::Scalar(scalar));
            let pointer = ptr_to(&mut m, base);
            let (mut f, pointer) = pointer_function(pointer);
            let stride = append_emit(
                &mut f,
                E::Literal(if stride_float {
                    naga::Literal::F32(3.0)
                } else {
                    naga::Literal::U32(3)
                }),
            );
            append_emit(
                &mut f,
                E::MatrixLoad {
                    columns: naga::VectorSize::Bi,
                    rows: naga::VectorSize::Tri,
                    data: naga::CooperativeData {
                        pointer,
                        stride,
                        row_major: false,
                    },
                },
            );
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            m.functions.append(f, Span::UNDEFINED);
            let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
            assert_eq!(
                result.is_ok(),
                scalar == naga::Scalar::F32 && !stride_float,
                "{result:?}"
            );
        }
    }
}

#[test]
fn physical_storage_runtime_array_offset_rejected() {
    let mut m = module();
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let array = ty(
        &mut m,
        T::Array {
            base: uint,
            size: naga::ArraySize::Dynamic,
            stride: 4,
        },
    );
    let pointer = ptr_to(&mut m, array);
    let (mut f, pointer) = pointer_function(pointer);
    let offset = append_emit(&mut f, E::Literal(naga::Literal::U64(1)));
    append_emit(&mut f, E::PointerOffset { pointer, offset });
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    m.functions.append(f, Span::UNDEFINED);
    let error = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("InvalidPointerType"),
        "{error:?}"
    );
}

#[test]
fn physical_storage_checked_projected_pointer() {
    let mut m = module_with_array(true);
    let float = ty(&mut m, T::Scalar(naga::Scalar::F32));
    let pointer_type = ptr_to(&mut m, float);
    let f = &mut m.entry_points[0].function;
    let pointer = f
        .expressions
        .iter()
        .find_map(|(h, e)| matches!(e, E::Access { .. }).then_some(h))
        .unwrap();
    let local = local(&mut f.local_variables, pointer_type);
    f.body.cull(f.body.len() - 1..);
    let place = append_emit(f, E::LocalVariable(local));
    f.body.push(
        S::Store {
            pointer: place,
            value: pointer,
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    let text = checked_words(&m, "checked-pointer-projection", false);
    assert!(
        text.contains("OpPhi") && text.contains("OpBitcast"),
        "{text}"
    );
}

#[test]
fn physical_storage_checked_compare_exchange() {
    let mut m = atomic_array_module();
    let uint = ty(&mut m, T::Scalar(naga::Scalar::U32));
    let boolean = ty(&mut m, T::Scalar(naga::Scalar::BOOL));
    let result_type = ty(
        &mut m,
        T::Struct {
            members: vec![
                member(Some("old_value"), uint, 0),
                member(Some("exchanged"), boolean, 4),
            ],
            span: 8,
        },
    );
    let f = &mut m.entry_points[0].function;
    f.body.cull(f.body.len() - 1..);
    let pointer = f
        .expressions
        .iter()
        .find_map(|(h, e)| matches!(e, E::Access { .. }).then_some(h))
        .unwrap();
    let pointer = append_emit(
        f,
        E::AtomicPointer {
            pointer,
            order: naga::AtomicMemoryOrder::AcquireRelease,
            failure_order: None,
        },
    );
    let one = append_emit(f, E::Literal(naga::Literal::U32(1)));
    let result = f.expressions.append(
        E::AtomicResult {
            ty: result_type,
            comparison: true,
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::Atomic {
            pointer,
            fun: naga::AtomicFunction::Exchange { compare: Some(one) },
            value: one,
            result: Some(result),
        },
        Span::UNDEFINED,
    );
    let observed = append_emit(
        f,
        E::AccessIndex {
            base: result,
            index: 0,
        },
    );
    let destination = f
        .expressions
        .iter()
        .find_map(|(h, e)| matches!(e, E::Access { .. }).then_some(h))
        .unwrap();
    f.body.push(
        S::Store {
            pointer: destination,
            value: observed,
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    let text = checked_words(&m, "checked-compare-exchange", false);
    assert!(
        text.contains("OpAtomicCompareExchange") && text.contains("OpPhi"),
        "{text}"
    );
}

#[test]
fn physical_storage_checked_uniform_matrix_record() {
    let mut m = module();
    let pointer = physical_pointer(&m);
    let matrix = ty(
        &mut m,
        T::Matrix {
            columns: naga::VectorSize::Bi,
            rows: naga::VectorSize::Bi,
            scalar: naga::Scalar::F32,
        },
    );
    let record = ty(
        &mut m,
        T::Struct {
            members: vec![member(None, pointer, 0), member(None, matrix, 16)],
            span: 32,
        },
    );
    let array = ty(
        &mut m,
        T::Array {
            base: record,
            size: fixed_size(2),
            stride: 32,
        },
    );
    let uniform = m.global_variables.append(
        naga::GlobalVariable {
            name: None,
            space: AddressSpace::Uniform,
            binding: Some(naga::ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: array,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let root = m.global_variables.iter().next().unwrap().0;
    let f = &mut m.entry_points[0].function;
    f.body.cull(f.body.len() - 1..);
    let g = append_emit(f, E::GlobalVariable(root));
    let index_ptr = append_emit(f, E::AccessIndex { base: g, index: 1 });
    let index = append_emit(f, E::Load { pointer: index_ptr });
    let g = append_emit(f, E::GlobalVariable(uniform));
    let element = append_emit(f, E::Access { base: g, index });
    let value = append_emit(f, E::Load { pointer: element });
    let local = local(&mut f.local_variables, record);
    let destination = append_emit(f, E::LocalVariable(local));
    f.body.push(
        S::Store {
            pointer: destination,
            value,
        },
        Span::UNDEFINED,
    );
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    let text = checked_words(&m, "checked-uniform-pointer-matrix", true);
    assert!(
        text.contains("OpPhi") && text.contains("OpFunctionCall"),
        "{text}"
    );
}
#[test]
fn physical_storage_coherent_checked_accesses() {
    for scope in [
        naga::MemoryScope::Device,
        naga::MemoryScope::QueueFamily,
        naga::MemoryScope::Workgroup,
        naga::MemoryScope::Subgroup,
        naga::MemoryScope::Invocation,
    ] {
        let mut m = module();
        let scalar = ty(&mut m, T::Scalar(naga::Scalar::F32));
        let integer = ty(&mut m, T::Scalar(naga::Scalar::U32));
        let address = ty(&mut m, T::Scalar(naga::Scalar::U64));
        let ptr = ptr_to(&mut m, scalar);
        let array = ty(
            &mut m,
            T::Array {
                base: scalar,
                size: fixed_size(4),
                stride: 4,
            },
        );
        let array_ptr = ptr_to(&mut m, array);
        let mut f = naga::Function::default();
        for ty in [ptr, integer] {
            f.arguments.push(argument(ty));
        }
        let p = append_emit(&mut f, E::FunctionArgument(0));
        let index = append_emit(&mut f, E::FunctionArgument(1));
        let p = append_emit(
            &mut f,
            E::PointerCast {
                expr: p,
                ty: address,
            },
        );
        let p = append_emit(
            &mut f,
            E::PointerCast {
                expr: p,
                ty: array_ptr,
            },
        );
        let p = append_emit(&mut f, E::Access { base: p, index });
        let p = append_emit(&mut f, E::CoherentPointer { pointer: p, scope });
        let p = append_emit(
            &mut f,
            E::PointerAlignment {
                pointer: p,
                alignment: 4,
            },
        );
        let value = append_emit(&mut f, E::Load { pointer: p });
        f.body.push(S::Store { pointer: p, value }, Span::UNDEFINED);
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        let function = m.functions.append(f, Span::UNDEFINED);
        let entry = &mut m.entry_points[0].function;
        let pointer = entry
            .body
            .iter()
            .find_map(|s| {
                if let S::Store { pointer, .. } = *s {
                    Some(pointer)
                } else {
                    None
                }
            })
            .unwrap();
        entry.body.cull(entry.body.len() - 1..);
        let index = append_emit(entry, E::Literal(naga::Literal::U32(4)));
        entry.body.push(
            S::Call {
                function,
                arguments: vec![pointer, index],
                result: None,
            },
            Span::UNDEFINED,
        );
        entry.body.push(S::Return { value: None }, Span::UNDEFINED);
        let error = Validator::new(
            ValidationFlags::all(),
            C::all() - C::COHERENT_PHYSICAL_MEMORY,
        )
        .validate(&m)
        .unwrap_err();
        assert!(format!("{error:?}").contains("COHERENT_PHYSICAL_MEMORY"));
        // Every scope is emitted (and its scope constant round-tripped through
        // `validate_emitted_words`); the compaction x bounds-check matrix only
        // needs to run for a couple of representative scopes.
        let full = matches!(
            scope,
            naga::MemoryScope::Device | naga::MemoryScope::Workgroup
        );
        for compact in [false, true] {
            if compact {
                if !full {
                    break;
                }
                naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
            }
            for policy in [
                naga::proc::BoundsCheckPolicy::Unchecked,
                naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
            ] {
                if !full && policy == naga::proc::BoundsCheckPolicy::Unchecked {
                    continue;
                }
                let info = Validator::new(ValidationFlags::all(), C::all())
                    .validate(&m)
                    .unwrap();
                let words = naga::back::spv::write_vec(
                    &m,
                    &info,
                    &naga::back::spv::Options {
                        lang_version: (1, 3),
                        bounds_check_policies: naga::proc::BoundsCheckPolicies {
                            index: policy,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
                let text = validate_emitted_words(
                    &words,
                    &format!("coherent-{scope:?}-{policy:?}-{compact}"),
                    false,
                );
                for flag in [
                    "MakePointerVisible",
                    "MakePointerAvailable",
                    "NonPrivatePointer",
                    "VulkanMemoryModel",
                ] {
                    assert!(text.contains(flag), "{text}");
                }
                if policy == naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite {
                    assert!(text.contains("OpBranchConditional"));
                }
            }
        }
    }
}

#[test]
fn physical_storage_independent_cas_ordering() {
    use naga::AtomicMemoryOrder as O;
    for order in [O::Relaxed, O::Acquire, O::Release, O::AcquireRelease] {
        for failure_order in [O::Relaxed, O::Acquire, O::Release, O::AcquireRelease] {
            let mut m = module();
            let atomic = ty(&mut m, T::Atomic(naga::Scalar::U32));
            let ptr = ptr_to(&mut m, atomic);
            let result_ty = m.generate_predeclared_type(
                naga::PredeclaredType::AtomicCompareExchangeWeakResult(naga::Scalar::U32),
            );
            let (mut f, p) = pointer_function(ptr);
            let p = append_emit(
                &mut f,
                E::AtomicPointer {
                    pointer: p,
                    order,
                    failure_order: Some(failure_order),
                },
            );
            let one = append_emit(&mut f, E::Literal(naga::Literal::U32(1)));
            let result = f.expressions.append(
                E::AtomicResult {
                    ty: result_ty,
                    comparison: true,
                },
                Span::UNDEFINED,
            );
            f.body.push(
                S::Atomic {
                    pointer: p,
                    fun: naga::AtomicFunction::Exchange { compare: Some(one) },
                    value: one,
                    result: Some(result),
                },
                Span::UNDEFINED,
            );
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            m.functions.append(f, Span::UNDEFINED);
            let legal = failure_order == O::Relaxed
                || (failure_order == O::Acquire && matches!(order, O::Acquire | O::AcquireRelease));
            if legal {
                let text = validate_native_words(
                    &m,
                    C::all(),
                    &format!("cas-{order:?}-{failure_order:?}"),
                    false,
                );
                let line = text
                    .lines()
                    .find(|line| line.contains("OpAtomicCompareExchange"))
                    .unwrap();
                let operands: Vec<_> = line
                    .split("OpAtomicCompareExchange")
                    .nth(1)
                    .unwrap()
                    .split_whitespace()
                    .collect();
                let constant = |id: &str| -> u32 {
                    text.lines()
                        .find_map(|line| {
                            let tokens: Vec<_> = line.split_whitespace().collect();
                            (tokens.len() == 5 && tokens[0] == id && tokens[2] == "OpConstant")
                                .then(|| tokens[4].parse().unwrap())
                        })
                        .unwrap()
                };
                let semantics = |order| match order {
                    O::Relaxed => 0,
                    O::Acquire => 0x42,
                    O::Release => 0x44,
                    O::AcquireRelease => 0x48,
                };
                assert_eq!(constant(operands[3]), semantics(order), "success ordering");
                assert_eq!(
                    constant(operands[4]),
                    semantics(failure_order),
                    "failure ordering"
                );
            } else {
                let error = Validator::new(ValidationFlags::all(), C::all())
                    .validate(&m)
                    .unwrap_err();
                assert!(format!("{error:?}").contains("InvalidAtomicFailureOrder"));
            }
        }
    }
}
#[test]
fn physical_storage_byte_pointees() {
    for signed in [false, true] {
        let mut m = pointer_helpers::byte_module(signed);
        let error = Validator::new(ValidationFlags::all(), C::all() - C::SHADER_INT8)
            .validate(&m)
            .unwrap_err();
        assert!(format!("{error:?}").contains("SHADER_INT8"));
        for compact in [false, true] {
            if compact {
                naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
            }
            let text =
                validate_native_words(&m, C::all(), &format!("bytes-{signed}-{compact}"), false);
            for instruction in [
                "OpTypeInt 8",
                "ArrayStride 1",
                "Aligned 1",
                "OpIAdd",
                "BitAccess",
            ] {
                assert!(text.contains(instruction), "missing {instruction}: {text}");
            }
            assert!(
                text.contains(if signed { "OpSConvert" } else { "OpUConvert" }),
                "{text}"
            );
        }
        let mut atomic = naga::Module::default();
        ty(
            &mut atomic,
            T::Atomic(if signed {
                naga::Scalar::I8
            } else {
                naga::Scalar::U8
            }),
        );
        assert!(Validator::new(ValidationFlags::all(), C::all())
            .validate(&atomic)
            .is_err());
    }
}

#[test]
fn physical_storage_byte_backend_boundaries() {
    for mode in 0..4 {
        let mut m = naga::Module::default();
        match mode {
            0 => {
                ty(&mut m, T::Scalar(naga::Scalar::U8));
            }
            1 => {
                m.global_expressions
                    .append(E::Literal(naga::Literal::I8(-1)), Span::UNDEFINED);
            }
            _ => {
                let mut f = naga::Function::default();
                let value = append_emit(&mut f, E::Literal(naga::Literal::U32(255)));
                append_emit(
                    &mut f,
                    E::As {
                        expr: value,
                        kind: naga::ScalarKind::Uint,
                        convert: Some(1),
                    },
                );
                f.body.push(S::Return { value: None }, Span::UNDEFINED);
                if mode == 2 {
                    m.functions.append(f, Span::UNDEFINED);
                } else {
                    m.entry_points.push(compute_entry(f));
                }
            }
        }
        let info = Validator::new(ValidationFlags::all(), C::all())
            .validate(&m)
            .unwrap();
        #[cfg(feature = "wgsl-out")]
        assert!(matches!(
            naga::back::wgsl::write_string(&m, &info, naga::back::wgsl::WriterFlags::empty()),
            Err(naga::back::wgsl::Error::Int8Unsupported)
        ));
        #[cfg(feature = "msl-out")]
        assert!(matches!(
            naga::back::msl::write_string(&m, &info, &Default::default(), &Default::default()),
            Err(naga::back::msl::Error::Int8Unsupported)
        ));
        #[cfg(feature = "hlsl-out")]
        {
            let options = naga::back::hlsl::Options::default();
            let pipeline = naga::back::hlsl::PipelineOptions::default();
            let mut writer = naga::back::hlsl::Writer::new(String::new(), &options, &pipeline);
            assert!(matches!(
                writer.write(&m, &info, None),
                Err(naga::back::hlsl::Error::Int8Unsupported)
            ));
        }
        #[cfg(feature = "glsl-out")]
        {
            let options = naga::back::glsl::Options::default();
            let pipeline = naga::back::glsl::PipelineOptions {
                shader_stage: naga::ShaderStage::Compute,
                entry_point: "main".into(),
                multiview: None,
            };
            assert!(matches!(
                naga::back::glsl::Writer::new(
                    String::new(),
                    &m,
                    &info,
                    &options,
                    &pipeline,
                    Default::default()
                ),
                Err(naga::back::glsl::Error::Int8Unsupported)
            ));
        }
    }
    let mut m = naga::Module::default();
    let byte = ty(&mut m, T::Scalar(naga::Scalar::U8));
    m.global_variables.append(
        naga::GlobalVariable {
            name: None,
            space: AddressSpace::Immediate,
            binding: None,
            ty: byte,
            init: None,
            memory_decorations: Default::default(),
        },
        Span::UNDEFINED,
    );
    assert!(Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .is_err());
}
#[test]
fn physical_storage_coherent_matrix_accesses() {
    for row_major in [false, true] {
        let mut m = module();
        let scalar = ty(&mut m, T::Scalar(naga::Scalar::F32));
        let pointer = ptr_to(&mut m, scalar);
        let (mut f, pointer) = pointer_function(pointer);
        let pointer = append_emit(
            &mut f,
            E::CoherentPointer {
                pointer,
                scope: naga::MemoryScope::Device,
            },
        );
        let stride = append_emit(&mut f, E::Literal(naga::Literal::U32(4)));
        let data = naga::CooperativeData {
            pointer,
            stride,
            row_major,
        };
        let matrix = append_emit(
            &mut f,
            E::MatrixLoad {
                columns: naga::VectorSize::Bi,
                rows: naga::VectorSize::Tri,
                data,
            },
        );
        f.body.push(
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
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        m.functions.append(f, Span::UNDEFINED);
        let text =
            validate_native_words(&m, C::all(), &format!("coherent-matrix-{row_major}"), false);
        assert_eq!(text.matches("MakePointerVisible").count(), 6, "{text}");
        assert_eq!(text.matches("MakePointerAvailable").count(), 6, "{text}");
    }
}

#[test]
fn physical_storage_bytes_are_not_stage_io() {
    for kind in [naga::ScalarKind::Sint, naga::ScalarKind::Uint] {
        for width in [1, 4] {
            for vector in [false, true] {
                for output in [false, true] {
                    let mut m = naga::Module::default();
                    let scalar = naga::Scalar { kind, width };
                    let value_ty = ty(
                        &mut m,
                        if vector {
                            T::Vector {
                                size: naga::VectorSize::Quad,
                                scalar,
                            }
                        } else {
                            T::Scalar(scalar)
                        },
                    );
                    let binding = naga::Binding::Location {
                        location: 0,
                        interpolation: Some(naga::Interpolation::Flat),
                        sampling: None,
                        blend_src: None,
                        per_primitive: false,
                    };
                    let mut f = naga::Function::default();
                    let value = if output {
                        f.result = Some(naga::FunctionResult {
                            ty: value_ty,
                            binding: Some(binding),
                        });
                        Some(append_emit(&mut f, E::ZeroValue(value_ty)))
                    } else {
                        f.arguments.push(naga::FunctionArgument {
                            name: None,
                            ty: value_ty,
                            binding: Some(binding),
                            immutable_pointee: false,
                        });
                        None
                    };
                    f.body.push(S::Return { value }, Span::UNDEFINED);
                    m.entry_points.push(naga::EntryPoint {
                        stage: naga::ShaderStage::Fragment,
                        workgroup_size: [0, 0, 0],
                        ..compute_entry(f)
                    });
                    let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
                    if width == 1 {
                        let error = result.unwrap_err();
                        assert!(format!("{error:?}").contains("NotIOShareable"), "{error:?}");
                    } else {
                        result.unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn physical_storage_workloads_round_trip() {
    use pointer_helpers::SpanMode;
    let mut modules = vec![
        ("module".to_string(), module(), false),
        (
            "bytes".to_string(),
            pointer_helpers::byte_module(false),
            false,
        ),
        (
            "signed-bytes".to_string(),
            pointer_helpers::byte_module(true),
            false,
        ),
    ];
    for scalar in [false, true] {
        for mode in [
            SpanMode::Plain,
            SpanMode::Atomic,
            SpanMode::Contended,
            SpanMode::Feedback,
            SpanMode::Matrix,
            SpanMode::RuntimeArray,
            SpanMode::MatrixLayout {
                row_major: false,
                padded: false,
            },
            SpanMode::MatrixLayout {
                row_major: false,
                padded: true,
            },
            SpanMode::MatrixLayout {
                row_major: true,
                padded: false,
            },
            SpanMode::MatrixLayout {
                row_major: true,
                padded: true,
            },
        ] {
            modules.push((
                format!("{mode:?}-{scalar}"),
                pointer_helpers::span_module_with_mode(scalar, mode),
                scalar,
            ));
        }
    }
    for (name, mut m, scalar) in modules {
        for compact in [false, true] {
            if compact {
                naga::compact::compact(&mut m, naga::compact::KeepUnused::No);
            }
            let name = format!("workload-{name}-{compact}");
            let words = native_words(&m, C::all());
            validate_emitted_words(&words, &name, scalar);
            // The importer must reconstruct physical pointers, not lower them
            // to something that validates without the capability.
            #[cfg(feature = "spv-in")]
            {
                let result = Validator::new(
                    ValidationFlags::all(),
                    C::all() - C::PHYSICAL_STORAGE_BUFFER_ADDRESSES,
                )
                .validate(&import(&words, &name))
                .map_err(|error| error.into_inner());
                assert!(
                    matches!(
                        result,
                        Err(ValidationError::Type {
                            source: TypeError::MissingCapability(
                                C::PHYSICAL_STORAGE_BUFFER_ADDRESSES
                            ),
                            ..
                        })
                    ),
                    "{name}: {result:?}"
                );
            }
        }
    }
}

#[cfg(feature = "spv-in")]
fn native_memory_operands(words: &[u32]) -> Vec<Vec<u32>> {
    let instructions: Vec<_> = spirv_instructions(words)
        .map(|i| &words[i..i + (words[i] >> 16) as usize])
        .collect();
    let constants: naga::FastHashMap<_, _> = instructions
        .iter()
        .filter(|w| w[0] as u16 == spirv::Op::Constant as u16 && w.len() == 4)
        .map(|w| (w[2], w[3]))
        .collect();
    let mut signatures = Vec::new();
    for w in instructions {
        let op = spirv::Op::from_u32(w[0] & 0xffff).unwrap();
        let memory = match op {
            spirv::Op::Load => Some(4),
            spirv::Op::Store => Some(3),
            spirv::Op::CooperativeMatrixLoadKHR => Some(6),
            spirv::Op::CooperativeMatrixStoreKHR => Some(5),
            _ => None,
        };
        if let Some(start) = memory {
            if w.len() <= start {
                continue;
            }
            let flags = spirv::MemoryAccess::from_bits(w[start]).unwrap();
            let mut values = vec![op as u32, w[start]];
            let mut i = start + 1;
            if flags.contains(spirv::MemoryAccess::ALIGNED) {
                values.push(w[i]);
                i += 1;
            }
            for flag in [
                spirv::MemoryAccess::MAKE_POINTER_AVAILABLE,
                spirv::MemoryAccess::MAKE_POINTER_VISIBLE,
            ] {
                if flags.contains(flag) {
                    values.push(constants[&w[i]]);
                    i += 1;
                }
            }
            signatures.push(values);
        } else {
            let start = match op {
                spirv::Op::AtomicStore => 2,
                spirv::Op::AtomicLoad
                | spirv::Op::AtomicExchange
                | spirv::Op::AtomicCompareExchange
                | spirv::Op::AtomicIIncrement
                | spirv::Op::AtomicIDecrement
                | spirv::Op::AtomicIAdd
                | spirv::Op::AtomicISub
                | spirv::Op::AtomicSMin
                | spirv::Op::AtomicUMin
                | spirv::Op::AtomicSMax
                | spirv::Op::AtomicUMax
                | spirv::Op::AtomicAnd
                | spirv::Op::AtomicOr
                | spirv::Op::AtomicXor
                | spirv::Op::AtomicFAddEXT => 4,
                _ => continue,
            };
            let mut values = vec![op as u32, constants[&w[start]], constants[&w[start + 1]]];
            if op == spirv::Op::AtomicCompareExchange {
                values.push(constants[&w[start + 2]]);
            }
            signatures.push(values);
        }
    }
    signatures.sort();
    signatures
}

#[cfg(feature = "spv-in")]
fn spirv_instructions(words: &[u32]) -> impl Iterator<Item = usize> + '_ {
    core::iter::successors((words.len() > 5).then_some(5), |&i| {
        let next = i + (words[i] >> 16) as usize;
        (next < words.len()).then_some(next)
    })
}

#[cfg(feature = "spv-in")]
#[test]
fn physical_storage_import_rejects_unrepresentable_memory_semantics() {
    let mut m = module();
    let f = &mut m.entry_points[0].function;
    let pointer = f
        .body
        .iter()
        .find_map(|s| match *s {
            S::Store { pointer, .. } => Some(pointer),
            _ => None,
        })
        .unwrap();
    f.body.cull(f.body.len() - 1..);
    let pointer = append_emit(
        f,
        E::CoherentPointer {
            pointer,
            scope: naga::MemoryScope::Device,
        },
    );
    let value = append_emit(f, E::Load { pointer });
    f.body.push(S::Store { pointer, value }, Span::UNDEFINED);
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    let info = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap();
    let words = naga::back::spv::write_vec(
        &m,
        &info,
        &naga::back::spv::Options {
            lang_version: (1, 3),
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let parse =
        |w: &[u32]| naga::front::spv::Frontend::new(w.iter().copied(), &Default::default()).parse();
    parse(&words).unwrap();
    let locate = |op: spirv::Op| {
        spirv_instructions(&words)
            .find(|&i| words[i] as u16 == op as u16)
            .unwrap()
    };
    let memory_model = locate(spirv::Op::MemoryModel);
    let load = spirv_instructions(&words)
        .find(|&i| words[i] as u16 == spirv::Op::Load as u16 && words[i] >> 16 == 7)
        .unwrap();
    let physical_cap = spirv_instructions(&words)
        .find(|&i| {
            words[i] as u16 == spirv::Op::Capability as u16
                && words[i + 1] == spirv::Capability::PhysicalStorageBufferAddresses as u32
        })
        .unwrap();
    for (name, offset, value) in [
        (
            "logical addressing",
            memory_model + 1,
            spirv::AddressingModel::Logical as u32,
        ),
        (
            "missing capability",
            physical_cap + 1,
            spirv::Capability::Shader as u32,
        ),
        (
            "coherence without Vulkan memory model",
            memory_model + 2,
            spirv::MemoryModel::GLSL450 as u32,
        ),
        ("non-power-of-two alignment", load + 5, 3),
        (
            "volatile physical access",
            load + 4,
            words[load + 4] | spirv::MemoryAccess::VOLATILE.bits(),
        ),
        (
            "nontemporal physical access",
            load + 4,
            words[load + 4] | spirv::MemoryAccess::NONTEMPORAL.bits(),
        ),
        (
            "private coherent access",
            load + 4,
            words[load + 4] & !spirv::MemoryAccess::NON_PRIVATE_POINTER.bits(),
        ),
        (
            "available load",
            load + 4,
            (words[load + 4] & !spirv::MemoryAccess::MAKE_POINTER_VISIBLE.bits())
                | spirv::MemoryAccess::MAKE_POINTER_AVAILABLE.bits(),
        ),
        (
            "missing alignment",
            load + 4,
            words[load + 4] & !spirv::MemoryAccess::ALIGNED.bits(),
        ),
        (
            "unknown memory operand",
            load + 4,
            words[load + 4] | 0x8000_0000,
        ),
    ] {
        let mut broken = words.clone();
        broken[offset] = value;
        assert!(parse(&broken).is_err(), "accepted {name}");
    }
    let scope_id = words[load + 6];
    let scope = spirv_instructions(&words)
        .find(|&i| words[i] as u16 == spirv::Op::Constant as u16 && words[i + 2] == scope_id)
        .unwrap();
    let mut broken = words.clone();
    broken[scope + 3] = spirv::Scope::CrossDevice as u32;
    assert!(parse(&broken).is_err());

    let m = pointer_helpers::span_module_with_atomics(false, true);
    let info = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap();
    let words = naga::back::spv::write_vec(&m, &info, &Default::default(), None).unwrap();
    parse(&words).unwrap();
    let cast = spirv_instructions(&words)
        .find(|&i| words[i] as u16 == spirv::Op::ConvertUToPtr as u16)
        .unwrap();
    let mut broken = words.clone();
    broken[cast] = (broken[cast] & 0xffff_0000) | spirv::Op::ConvertPtrToU as u32;
    assert!(parse(&broken).is_err());
    let atomic = spirv_instructions(&words)
        .find(|&i| words[i] as u16 == spirv::Op::AtomicLoad as u16)
        .unwrap();
    let mut broken = words.clone();
    broken[atomic + 5] = words[atomic + 4];
    assert!(
        parse(&broken).is_err(),
        "unsupported atomic order must not become Relaxed"
    );
}

#[test]
fn physical_storage_immutable_access_paths() {
    for path in [
        "direct",
        "stored",
        "reassigned",
        "returned",
        "offset",
        "alignment",
        "coherent",
        "cast",
        "select",
        "aggregate",
        "mutable-member",
        "nested-aggregate",
        "nested-mutable-member",
        "helper-write",
        "helper-read",
        "nested-helper-write",
        "matrix",
        "atomic",
        "cooperative",
    ] {
        for immutable in [false, true] {
            let mut m = module();
            let scalar = ty(&mut m, T::Scalar(naga::Scalar::F32));
            let ptr = ptr_to(&mut m, scalar);
            let (mut helper, arg) = pointer_function(ptr);
            let value = append_emit(&mut helper, E::Load { pointer: arg });
            if path.ends_with("helper-write") {
                helper.body.push(
                    S::Store {
                        pointer: arg,
                        value,
                    },
                    Span::UNDEFINED,
                );
            }
            helper.body.push(S::Return { value: None }, Span::UNDEFINED);
            let mut callee = m.functions.append(helper, Span::UNDEFINED);
            if path == "nested-helper-write" {
                let (mut helper, arg) = pointer_function(ptr);
                let condition = append_emit(&mut helper, E::Literal(naga::Literal::Bool(true)));
                let mut accept = naga::Block::new();
                accept.push(
                    S::Call {
                        function: callee,
                        arguments: vec![arg],
                        result: None,
                    },
                    Span::UNDEFINED,
                );
                helper.body.push(
                    S::If {
                        condition,
                        accept,
                        reject: naga::Block::new(),
                    },
                    Span::UNDEFINED,
                );
                helper.body.push(S::Return { value: None }, Span::UNDEFINED);
                callee = m.functions.append(helper, Span::UNDEFINED);
            }
            let mut f = naga::Function::default();
            for _ in 0..2 {
                f.arguments.push(argument(ptr));
            }
            if immutable {
                f.arguments[0].immutable_pointee = true;
            }
            let p = append_emit(&mut f, E::FunctionArgument(0));
            let q = append_emit(&mut f, E::FunctionArgument(1));
            let p = match path {
                "stored" | "reassigned" => {
                    let local = local(&mut f.local_variables, ptr);
                    let holder = append_emit(&mut f, E::LocalVariable(local));
                    f.body.push(
                        S::Store {
                            pointer: holder,
                            value: p,
                        },
                        Span::UNDEFINED,
                    );
                    if path == "reassigned" {
                        f.body.push(
                            S::Store {
                                pointer: holder,
                                value: q,
                            },
                            Span::UNDEFINED,
                        );
                    }
                    append_emit(&mut f, E::Load { pointer: holder })
                }
                "returned" => {
                    let mut identity = naga::Function::default();
                    identity.arguments.push(argument(ptr));
                    identity.result = Some(returns(ptr));
                    let arg = append_emit(&mut identity, E::FunctionArgument(0));
                    identity
                        .body
                        .push(S::Return { value: Some(arg) }, Span::UNDEFINED);
                    let identity = m.functions.append(identity, Span::UNDEFINED);
                    let result = f
                        .expressions
                        .append(E::CallResult(identity), Span::UNDEFINED);
                    f.body.push(
                        S::Call {
                            function: identity,
                            arguments: vec![p],
                            result: Some(result),
                        },
                        Span::UNDEFINED,
                    );
                    result
                }
                "offset" => {
                    let offset = append_emit(&mut f, E::Literal(naga::Literal::I64(1)));
                    append_emit(&mut f, E::PointerOffset { pointer: p, offset })
                }
                "alignment" => append_emit(
                    &mut f,
                    E::PointerAlignment {
                        pointer: p,
                        alignment: 4,
                    },
                ),
                "coherent" => append_emit(
                    &mut f,
                    E::CoherentPointer {
                        pointer: p,
                        scope: naga::MemoryScope::Device,
                    },
                ),
                "cast" => {
                    let address = ty(&mut m, T::Scalar(naga::Scalar::U64));
                    let value = append_emit(
                        &mut f,
                        E::PointerCast {
                            expr: p,
                            ty: address,
                        },
                    );
                    append_emit(
                        &mut f,
                        E::PointerCast {
                            expr: value,
                            ty: ptr,
                        },
                    )
                }
                "select" => {
                    let condition = append_emit(&mut f, E::Literal(naga::Literal::Bool(true)));
                    append_emit(
                        &mut f,
                        E::Select {
                            condition,
                            accept: p,
                            reject: q,
                        },
                    )
                }
                "aggregate" | "mutable-member" | "nested-aggregate" | "nested-mutable-member" => {
                    let record = ty(
                        &mut m,
                        T::Struct {
                            members: [p, q]
                                .iter()
                                .enumerate()
                                .map(|(i, _)| member(None, ptr, i as u32 * 8))
                                .collect(),
                            span: 16,
                        },
                    );
                    let record_ty = record;
                    let record = append_emit(
                        &mut f,
                        E::Compose {
                            ty: record,
                            components: vec![p, q],
                        },
                    );
                    let record = if path.starts_with("nested-") {
                        let outer = ty(
                            &mut m,
                            T::Struct {
                                members: vec![member(None, record_ty, 0)],
                                span: 16,
                            },
                        );
                        let outer = append_emit(
                            &mut f,
                            E::Compose {
                                ty: outer,
                                components: vec![record],
                            },
                        );
                        append_emit(
                            &mut f,
                            E::AccessIndex {
                                base: outer,
                                index: 0,
                            },
                        )
                    } else {
                        record
                    };
                    append_emit(
                        &mut f,
                        E::AccessIndex {
                            base: record,
                            index: u32::from(path.ends_with("mutable-member")),
                        },
                    )
                }
                _ => p,
            };
            if path.contains("helper-") {
                f.body.push(
                    S::Call {
                        function: callee,
                        arguments: vec![p],
                        result: None,
                    },
                    Span::UNDEFINED,
                );
            } else if path == "atomic" {
                let address = ty(&mut m, T::Scalar(naga::Scalar::U64));
                let atomic = ty(&mut m, T::Atomic(naga::Scalar::U32));
                let atomic_ptr = ptr_to(&mut m, atomic);
                let address = append_emit(
                    &mut f,
                    E::PointerCast {
                        expr: p,
                        ty: address,
                    },
                );
                let pointer = append_emit(
                    &mut f,
                    E::PointerCast {
                        expr: address,
                        ty: atomic_ptr,
                    },
                );
                let pointer = append_emit(
                    &mut f,
                    E::AtomicPointer {
                        pointer,
                        order: naga::AtomicMemoryOrder::Relaxed,
                        failure_order: None,
                    },
                );
                let value = append_emit(&mut f, E::Literal(naga::Literal::U32(1)));
                f.body.push(
                    S::Atomic {
                        pointer,
                        fun: naga::AtomicFunction::Add,
                        value,
                        result: None,
                    },
                    Span::UNDEFINED,
                );
            } else if path == "cooperative" {
                let stride = append_emit(&mut f, E::Literal(naga::Literal::U32(8)));
                let data = naga::CooperativeData {
                    pointer: p,
                    stride,
                    row_major: false,
                };
                let target = append_emit(
                    &mut f,
                    E::CooperativeLoad {
                        columns: naga::CooperativeSize::Eight,
                        rows: naga::CooperativeSize::Eight,
                        role: naga::CooperativeRole::C,
                        data,
                    },
                );
                f.body
                    .push(S::CooperativeStore { target, data }, Span::UNDEFINED);
            } else if path == "matrix" {
                let matrix = ty(
                    &mut m,
                    T::Matrix {
                        columns: naga::VectorSize::Bi,
                        rows: naga::VectorSize::Bi,
                        scalar: naga::Scalar::F32,
                    },
                );
                let target = append_emit(&mut f, E::ZeroValue(matrix));
                let stride = append_emit(&mut f, E::Literal(naga::Literal::U32(2)));
                f.body.push(
                    S::MatrixStore {
                        target,
                        data: naga::CooperativeData {
                            pointer: p,
                            stride,
                            row_major: false,
                        },
                    },
                    Span::UNDEFINED,
                );
            } else {
                let value = append_emit(&mut f, E::Literal(naga::Literal::F32(1.0)));
                f.body.push(S::Store { pointer: p, value }, Span::UNDEFINED);
            }
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            m.functions.append(f, Span::UNDEFINED);
            let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
            if immutable
                && !(path.ends_with("mutable-member")
                    || path == "helper-read"
                    || path == "reassigned")
            {
                let error = result.expect_err(&format!("accepted immutable write through {path}"));
                assert!(
                    format!("{error:?}").contains("InvalidStorePointer")
                        || format!("{error:?}").contains("PointerAccess"),
                    "{path}: {error:?}"
                );
            } else {
                result.unwrap_or_else(|error| {
                    panic!("rejected writable/read-only control {path}: {error:?}")
                });
                if path.contains("aggregate") || path.ends_with("mutable-member") {
                    validate_native_words(
                        &m,
                        C::all(),
                        &format!("access-audit-{path}-{immutable}"),
                        false,
                    );
                }
            }
        }
    }
}

#[test]
fn physical_storage_read_only_holder_does_not_make_pointee_read_only() {
    let mut m = module_with_options(false, true);
    Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap();
    let f = &mut m.entry_points[0].function;
    let (value, pointer) = f
        .expressions
        .iter()
        .find_map(|(value, expr)| match *expr {
            E::Load { pointer } if matches!(f.expressions[pointer], E::Access { .. }) => {
                Some((value, pointer))
            }
            _ => None,
        })
        .unwrap();
    f.body.cull(f.body.len() - 1..);
    f.body.push(S::Store { pointer, value }, Span::UNDEFINED);
    f.body.push(S::Return { value: None }, Span::UNDEFINED);
    let error = Validator::new(ValidationFlags::all(), C::all())
        .validate(&m)
        .unwrap_err();
    assert!(
        format!("{error:?}").contains("InvalidStorePointer"),
        "{error:?}"
    );
}

#[test]
fn physical_storage_alias_dataflow() {
    // A pointer stored to and reloaded from a local is covered by
    // `physical_storage_immutable_access_paths` ("stored").
    for path in [
        "array-dynamic",
        "array-dynamic-store",
        "array-writable",
        "global",
        "physical-holder",
        "reassign",
        "snapshot",
        "branch",
        "branch-overwrite",
        "loop",
        "loop-overwrite",
        "struct",
        "struct-sibling",
        "return",
        "return-sibling",
        "aggregate-call",
        "out-parameter",
        "integer",
    ] {
        for immutable in [false, true] {
            let mut m = module();
            let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
            let ptr = ptr_to(&mut m, scalar);
            let pair = ty(
                &mut m,
                T::Struct {
                    members: (0..2).map(|i| member(None, ptr, i * 8)).collect(),
                    span: 16,
                },
            );
            let mut f = naga::Function::default();
            for _ in 0..2 {
                f.arguments.push(argument(ptr));
            }
            let bool_ty = ty(&mut m, T::Scalar(naga::Scalar::BOOL));
            f.arguments.push(argument(bool_ty));
            if immutable {
                f.arguments[0].immutable_pointee = true;
            }
            let p = append_emit(&mut f, E::FunctionArgument(0));
            let q = append_emit(&mut f, E::FunctionArgument(1));
            let condition = append_emit(&mut f, E::FunctionArgument(2));
            let one = append_emit(&mut f, E::Literal(naga::Literal::U32(1)));
            let holder = if path == "global" {
                let global = m.global_variables.append(
                    naga::GlobalVariable {
                        name: None,
                        space: AddressSpace::Storage {
                            access: naga::StorageAccess::LOAD | naga::StorageAccess::STORE,
                        },
                        binding: Some(naga::ResourceBinding {
                            group: 0,
                            binding: 1,
                        }),
                        ty: pair,
                        init: None,
                        memory_decorations: naga::MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                let root = append_emit(&mut f, E::GlobalVariable(global));
                append_emit(
                    &mut f,
                    E::AccessIndex {
                        base: root,
                        index: 0,
                    },
                )
            } else if path == "physical-holder" {
                let holder_ty = ptr_to(&mut m, ptr);
                f.arguments.push(argument(holder_ty));
                append_emit(&mut f, E::FunctionArgument(3))
            } else {
                let local = local(&mut f.local_variables, ptr);
                append_emit(&mut f, E::LocalVariable(local))
            };
            let pointer = match path {
                "array-dynamic" | "array-dynamic-store" | "array-writable" => {
                    let array = ty(
                        &mut m,
                        T::Array {
                            base: ptr,
                            size: fixed_size(2),
                            stride: 8,
                        },
                    );
                    let value = append_emit(
                        &mut f,
                        E::Compose {
                            ty: array,
                            components: vec![if path == "array-dynamic" { p } else { q }, q],
                        },
                    );
                    let local = local(&mut f.local_variables, array);
                    let place = append_emit(&mut f, E::LocalVariable(local));
                    f.body.push(
                        S::Store {
                            pointer: place,
                            value,
                        },
                        Span::UNDEFINED,
                    );
                    let zero = append_emit(&mut f, E::Literal(naga::Literal::U32(0)));
                    let index = append_emit(
                        &mut f,
                        E::Select {
                            condition,
                            accept: zero,
                            reject: one,
                        },
                    );
                    let element = append_emit(&mut f, E::Access { base: place, index });
                    if path == "array-dynamic-store" {
                        f.body.push(
                            S::Store {
                                pointer: element,
                                value: p,
                            },
                            Span::UNDEFINED,
                        );
                        let element = append_emit(
                            &mut f,
                            E::AccessIndex {
                                base: place,
                                index: 1,
                            },
                        );
                        append_emit(&mut f, E::Load { pointer: element })
                    } else {
                        append_emit(&mut f, E::Load { pointer: element })
                    }
                }
                "struct" | "struct-sibling" => {
                    let value = append_emit(
                        &mut f,
                        E::Compose {
                            ty: pair,
                            components: vec![p, q],
                        },
                    );
                    let local = local(&mut f.local_variables, pair);
                    let place = append_emit(&mut f, E::LocalVariable(local));
                    f.body.push(
                        S::Store {
                            pointer: place,
                            value,
                        },
                        Span::UNDEFINED,
                    );
                    let field = append_emit(
                        &mut f,
                        E::AccessIndex {
                            base: place,
                            index: u32::from(path.ends_with("sibling")),
                        },
                    );
                    append_emit(&mut f, E::Load { pointer: field })
                }
                "return" | "return-sibling" | "aggregate-call" => {
                    let mut helper = naga::Function::default();
                    helper.arguments.push(argument(pair));
                    helper.result = Some(returns(pair));
                    let arg = append_emit(&mut helper, E::FunctionArgument(0));
                    if path == "aggregate-call" {
                        let target = append_emit(
                            &mut helper,
                            E::AccessIndex {
                                base: arg,
                                index: 0,
                            },
                        );
                        let value = append_emit(&mut helper, E::Literal(naga::Literal::U32(1)));
                        helper.body.push(
                            S::Store {
                                pointer: target,
                                value,
                            },
                            Span::UNDEFINED,
                        );
                    }
                    helper
                        .body
                        .push(S::Return { value: Some(arg) }, Span::UNDEFINED);
                    let helper = m.functions.append(helper, Span::UNDEFINED);
                    let arg = append_emit(
                        &mut f,
                        E::Compose {
                            ty: pair,
                            components: vec![p, q],
                        },
                    );
                    let result = f.expressions.append(E::CallResult(helper), Span::UNDEFINED);
                    f.body.push(
                        S::Call {
                            function: helper,
                            arguments: vec![arg],
                            result: Some(result),
                        },
                        Span::UNDEFINED,
                    );
                    append_emit(
                        &mut f,
                        E::AccessIndex {
                            base: result,
                            index: u32::from(path.ends_with("sibling")),
                        },
                    )
                }
                "out-parameter" => {
                    let holder_ty = ty(
                        &mut m,
                        T::Pointer {
                            base: ptr,
                            space: AddressSpace::Function,
                        },
                    );
                    let mut helper = naga::Function::default();
                    for ty in [ptr, holder_ty] {
                        helper.arguments.push(argument(ty));
                    }
                    let value = append_emit(&mut helper, E::FunctionArgument(0));
                    let target = append_emit(&mut helper, E::FunctionArgument(1));
                    helper.body.push(
                        S::Store {
                            pointer: target,
                            value,
                        },
                        Span::UNDEFINED,
                    );
                    helper.body.push(S::Return { value: None }, Span::UNDEFINED);
                    let helper = m.functions.append(helper, Span::UNDEFINED);
                    f.body.push(
                        S::Call {
                            function: helper,
                            arguments: vec![p, holder],
                            result: None,
                        },
                        Span::UNDEFINED,
                    );
                    append_emit(&mut f, E::Load { pointer: holder })
                }
                "integer" => {
                    let address = ty(&mut m, T::Scalar(naga::Scalar::U64));
                    let address = append_emit(
                        &mut f,
                        E::PointerCast {
                            expr: p,
                            ty: address,
                        },
                    );
                    let offset = append_emit(&mut f, E::Literal(naga::Literal::U64(4)));
                    let address = append_emit(
                        &mut f,
                        E::Binary {
                            op: naga::BinaryOperator::Add,
                            left: address,
                            right: offset,
                        },
                    );
                    append_emit(
                        &mut f,
                        E::PointerCast {
                            expr: address,
                            ty: ptr,
                        },
                    )
                }
                "branch" | "branch-overwrite" => {
                    f.body.push(
                        S::Store {
                            pointer: holder,
                            value: q,
                        },
                        Span::UNDEFINED,
                    );
                    let mut accept = naga::Block::new();
                    accept.push(
                        S::Store {
                            pointer: holder,
                            value: p,
                        },
                        Span::UNDEFINED,
                    );
                    f.body.push(
                        S::If {
                            condition,
                            accept,
                            reject: naga::Block::new(),
                        },
                        Span::UNDEFINED,
                    );
                    if path.ends_with("overwrite") {
                        f.body.push(
                            S::Store {
                                pointer: holder,
                                value: q,
                            },
                            Span::UNDEFINED,
                        );
                    }
                    append_emit(&mut f, E::Load { pointer: holder })
                }
                "loop" | "loop-overwrite" => {
                    f.body.push(
                        S::Store {
                            pointer: holder,
                            value: q,
                        },
                        Span::UNDEFINED,
                    );
                    let outer = core::mem::take(&mut f.body);
                    if path.ends_with("overwrite") {
                        f.body.push(
                            S::Store {
                                pointer: holder,
                                value: q,
                            },
                            Span::UNDEFINED,
                        );
                    }
                    let alias = append_emit(&mut f, E::Load { pointer: holder });
                    f.body.push(
                        S::Store {
                            pointer: alias,
                            value: one,
                        },
                        Span::UNDEFINED,
                    );
                    f.body.push(
                        S::Store {
                            pointer: holder,
                            value: p,
                        },
                        Span::UNDEFINED,
                    );
                    let body = core::mem::replace(&mut f.body, outer);
                    f.body.push(
                        S::Loop {
                            body,
                            continuing: naga::Block::new(),
                            break_if: Some(condition),
                        },
                        Span::UNDEFINED,
                    );
                    q
                }
                _ => {
                    f.body.push(
                        S::Store {
                            pointer: holder,
                            value: p,
                        },
                        Span::UNDEFINED,
                    );
                    let snapshot = append_emit(&mut f, E::Load { pointer: holder });
                    if matches!(path, "reassign" | "snapshot") {
                        f.body.push(
                            S::Store {
                                pointer: holder,
                                value: q,
                            },
                            Span::UNDEFINED,
                        );
                    }
                    if path == "snapshot" {
                        snapshot
                    } else {
                        append_emit(&mut f, E::Load { pointer: holder })
                    }
                }
            };
            f.body.push(
                S::Store {
                    pointer,
                    value: one,
                },
                Span::UNDEFINED,
            );
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            m.functions.append(f, Span::UNDEFINED);
            let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
            let writable = matches!(
                path,
                "reassign"
                    | "array-writable"
                    | "branch-overwrite"
                    | "loop-overwrite"
                    | "struct-sibling"
                    | "return-sibling"
            );
            if immutable && !writable {
                let error = format!("{:?}", result.expect_err(path));
                assert!(
                    error.contains("InvalidStorePointer") || error.contains("PointerAccess"),
                    "{path}: {error}"
                );
            } else {
                result.unwrap_or_else(|error| panic!("{path}/{immutable}: {error:?}"));
            }
        }
    }
}

#[test]
fn physical_storage_member_access() {
    use naga::StorageAccess as A;
    for access in [A::LOAD, A::STORE, A::LOAD | A::STORE, A::empty()] {
        for index in [0, 1] {
            for operation in [
                "load",
                "store",
                "whole-load",
                "whole-store",
                "cast-store",
                "helper-store",
                "stored-store",
                "returned-store",
            ] {
                if index == 1 && operation.starts_with("whole-") {
                    // Whole-record operations ignore the member index.
                    continue;
                }
                let mut m = module();
                let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
                let record = ty(
                    &mut m,
                    T::Struct {
                        members: (0..2)
                            .map(|i| naga::StructMember {
                                access: (i == 0).then_some(access),
                                name: Some(format!("field{i}")),
                                ty: scalar,
                                binding: None,
                                offset: i * 4,
                            })
                            .collect(),
                        span: 8,
                    },
                );
                let ptr = ptr_to(&mut m, record);
                let (mut f, p) = pointer_function(ptr);
                let mut target = if operation.starts_with("whole-") {
                    p
                } else {
                    append_emit(&mut f, E::AccessIndex { base: p, index })
                };
                if operation == "cast-store" {
                    let address = ty(&mut m, T::Scalar(naga::Scalar::U64));
                    let pointer = ptr_to(&mut m, scalar);
                    let value = append_emit(
                        &mut f,
                        E::PointerCast {
                            expr: target,
                            ty: address,
                        },
                    );
                    target = append_emit(
                        &mut f,
                        E::PointerCast {
                            expr: value,
                            ty: pointer,
                        },
                    );
                }
                if matches!(operation, "stored-store" | "returned-store") {
                    let pointer = ptr_to(&mut m, scalar);
                    if operation == "stored-store" {
                        let local = local(&mut f.local_variables, pointer);
                        let holder = append_emit(&mut f, E::LocalVariable(local));
                        f.body.push(
                            S::Store {
                                pointer: holder,
                                value: target,
                            },
                            Span::UNDEFINED,
                        );
                        target = append_emit(&mut f, E::Load { pointer: holder });
                    } else {
                        let mut helper = naga::Function::default();
                        helper.arguments.push(argument(pointer));
                        helper.result = Some(returns(pointer));
                        let arg = append_emit(&mut helper, E::FunctionArgument(0));
                        helper
                            .body
                            .push(S::Return { value: Some(arg) }, Span::UNDEFINED);
                        let helper = m.functions.append(helper, Span::UNDEFINED);
                        let result = f.expressions.append(E::CallResult(helper), Span::UNDEFINED);
                        f.body.push(
                            S::Call {
                                function: helper,
                                arguments: vec![target],
                                result: Some(result),
                            },
                            Span::UNDEFINED,
                        );
                        target = result;
                    }
                }
                if operation.ends_with("load") {
                    append_emit(&mut f, E::Load { pointer: target });
                } else if operation == "helper-store" {
                    let pointer = ptr_to(&mut m, scalar);
                    let (mut helper, pointer) = pointer_function(pointer);
                    let value = append_emit(&mut helper, E::Literal(naga::Literal::U32(1)));
                    helper
                        .body
                        .push(S::Store { pointer, value }, Span::UNDEFINED);
                    helper.body.push(S::Return { value: None }, Span::UNDEFINED);
                    let helper = m.functions.append(helper, Span::UNDEFINED);
                    f.body.push(
                        S::Call {
                            function: helper,
                            arguments: vec![target],
                            result: None,
                        },
                        Span::UNDEFINED,
                    );
                } else {
                    let value = append_emit(
                        &mut f,
                        E::ZeroValue(if operation == "whole-store" {
                            record
                        } else {
                            scalar
                        }),
                    );
                    f.body.push(
                        S::Store {
                            pointer: target,
                            value,
                        },
                        Span::UNDEFINED,
                    );
                }
                f.body.push(S::Return { value: None }, Span::UNDEFINED);
                m.functions.append(f, Span::UNDEFINED);
                let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
                let required = if operation.ends_with("load") {
                    A::LOAD
                } else {
                    A::STORE
                };
                if (index == 0 || operation.starts_with("whole-")) && !access.contains(required) {
                    let error = format!("{:?}", result.expect_err(operation));
                    assert!(
                        error.contains("PointerAccess"),
                        "{operation}/{access:?}: {error}"
                    );
                } else {
                    result.unwrap_or_else(|error| panic!("{operation}/{access:?}: {error:?}"));
                    // Emit every operation for an unrestricted member, and a
                    // plain load/store for each restricted member layout.
                    if access == A::LOAD | A::STORE || matches!(operation, "load" | "store") {
                        validate_native_words(
                            &m,
                            C::all(),
                            &format!("member-{operation}-{index}-{}", access.bits()),
                            false,
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[cfg(feature = "spv-in")]
fn physical_storage_import_member_restrictions() {
    use rspirv::{
        binary::Assemble,
        dr::{Instruction, Operand},
        spirv::{Decoration, Op},
    };
    for holder in [false, true] {
        for decoration in [Decoration::NonWritable, Decoration::NonReadable] {
            let m = module();
            let info = Validator::new(ValidationFlags::all(), C::all())
                .validate(&m)
                .unwrap();
            let words = naga::back::spv::write_vec(&m, &info, &Default::default(), None).unwrap();
            let mut spv = rspirv::dr::load_words(&words).unwrap();
            let target = spv
                .types_global_values
                .iter()
                .find(|inst| {
                    inst.class.opcode == Op::TypeStruct
                        && inst.operands.len() == if holder { 2 } else { 1 }
                })
                .unwrap()
                .result_id
                .unwrap();
            spv.annotations.push(Instruction::new(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(target),
                    Operand::LiteralBit32(0),
                    Operand::Decoration(decoration),
                ],
            ));
            let words = spv.assemble();
            let mut parsed =
                naga::front::spv::Frontend::new(words.iter().copied(), &Default::default())
                    .parse()
                    .unwrap();
            for compact in [false, true] {
                if compact {
                    naga::compact::compact(&mut parsed, naga::compact::KeepUnused::No);
                }
                assert!(parsed.types.iter().any(|(_, ty)| matches!(&ty.inner,
                    T::Struct { members, .. } if members.iter().any(|member| member.access == Some(
                        if decoration == Decoration::NonWritable { naga::StorageAccess::LOAD } else { naga::StorageAccess::STORE }
                    )))));
                let result = Validator::new(ValidationFlags::all(), C::all()).validate(&parsed);
                if holder && decoration == Decoration::NonWritable {
                    let info =
                        result.expect("read-only pointer holder must not restrict its pointee");
                    let output =
                        naga::back::spv::write_vec(&parsed, &info, &Default::default(), None)
                            .unwrap();
                    let text =
                        validate_spirv_words(&output, &format!("member-holder-{compact}"), false);
                    assert!(text.contains("NonWritable"));
                } else {
                    let error = format!(
                        "{:?}",
                        result.expect_err("imported member access restriction was lost")
                    );
                    assert!(error.contains("InvalidPointerAccess"), "{error}");
                }
            }
        }
    }
}

#[test]
fn physical_storage_immutable_callee_contract() {
    for before in [false, true] {
        for same_pointer in [false, true] {
            let mut m = module();
            let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
            let ptr = ptr_to(&mut m, scalar);
            let (mut helper, arg) = pointer_function(ptr);
            helper.arguments[0].immutable_pointee = true;
            append_emit(&mut helper, E::Load { pointer: arg });
            helper.body.push(S::Return { value: None }, Span::UNDEFINED);
            let helper = m.functions.append(helper, Span::UNDEFINED);
            let mut f = naga::Function::default();
            for _ in 0..2 {
                f.arguments.push(argument(ptr));
            }
            let p = append_emit(&mut f, E::FunctionArgument(0));
            let q = append_emit(&mut f, E::FunctionArgument(1));
            let value = append_emit(&mut f, E::Literal(naga::Literal::U32(1)));
            let store = S::Store {
                pointer: if same_pointer { p } else { q },
                value,
            };
            if before {
                f.body.push(store.clone(), Span::UNDEFINED);
            }
            f.body.push(
                S::Call {
                    function: helper,
                    arguments: vec![p],
                    result: None,
                },
                Span::UNDEFINED,
            );
            if !before {
                f.body.push(store, Span::UNDEFINED);
            }
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            m.functions.append(f, Span::UNDEFINED);
            let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
            if same_pointer {
                let error = format!(
                    "{:?}",
                    result.expect_err("caller violates immutable pointee promise")
                );
                assert!(error.contains("InvalidStorePointer"), "{error}");
            } else {
                result.unwrap();
            }
        }
    }
}

#[test]
fn physical_storage_access_analysis_complexity() {
    for depth in [4, 20] {
        let mut m = module();
        let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
        let pointer = ptr_to(&mut m, scalar);
        let mut previous = None;
        for _ in 0..depth {
            let (mut f, arg) = pointer_function(pointer);
            f.arguments[0].immutable_pointee = true;
            if let Some(function) = previous {
                for _ in 0..2 {
                    f.body.push(
                        S::Call {
                            function,
                            arguments: vec![arg],
                            result: None,
                        },
                        Span::UNDEFINED,
                    );
                }
            }
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            previous = Some(m.functions.append(f, Span::UNDEFINED));
        }
        let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
        if depth == 4 {
            result.unwrap();
        } else {
            let error = format!(
                "{:?}",
                result.expect_err("exponential call expansion must be bounded")
            );
            assert!(error.contains("PointerAccessAnalysisLimit"), "{error}");
        }
    }
}

#[test]
fn physical_storage_access_analysis_nested_arrays() {
    for depth in [4, 130] {
        let mut m = module();
        let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
        let pointer = ptr_to(&mut m, scalar);
        let mut marker = naga::Function::default();
        marker.arguments.push(argument(pointer));
        marker.arguments[0].immutable_pointee = true;
        marker.body.push(S::Return { value: None }, Span::UNDEFINED);
        m.functions.append(marker, Span::UNDEFINED);
        let mut array = pointer;
        for _ in 0..depth {
            array = ty(
                &mut m,
                T::Array {
                    base: array,
                    size: fixed_size(1),
                    stride: 8,
                },
            );
        }
        let mut f = naga::Function::default();
        f.arguments.push(argument(array));
        f.body.push(S::Return { value: None }, Span::UNDEFINED);
        m.functions.append(f, Span::UNDEFINED);
        let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
        if depth == 4 {
            result.unwrap();
        } else {
            let error = format!(
                "{:?}",
                result.expect_err("array provenance nesting must be bounded")
            );
            assert!(error.contains("PointerAccessAnalysisLimit"), "{error}");
        }
    }
}

#[test]
fn physical_storage_atomic_address_aliases() {
    for path in [
        "exchange-store",
        "exchange-result",
        "exchange-overwrite",
        "add",
        "cas-store",
        "cas-result",
    ] {
        for immutable in [false, true] {
            let mut m = module();
            let scalar = ty(&mut m, T::Scalar(naga::Scalar::U32));
            let pointer = ptr_to(&mut m, scalar);
            let address = ty(&mut m, T::Scalar(naga::Scalar::U64));
            let atomic = ty(&mut m, T::Atomic(naga::Scalar::U64));
            let holder = ptr_to(&mut m, atomic);
            let mut f = naga::Function::default();
            for ty in [pointer, pointer, holder] {
                f.arguments.push(argument(ty));
            }
            if immutable {
                f.arguments[0].immutable_pointee = true;
            }
            let p = append_emit(&mut f, E::FunctionArgument(0));
            let q = append_emit(&mut f, E::FunctionArgument(1));
            let holder = append_emit(&mut f, E::FunctionArgument(2));
            let p = append_emit(
                &mut f,
                E::PointerCast {
                    expr: p,
                    ty: address,
                },
            );
            let q = append_emit(
                &mut f,
                E::PointerCast {
                    expr: q,
                    ty: address,
                },
            );
            f.body.push(
                S::Store {
                    pointer: holder,
                    value: p,
                },
                Span::UNDEFINED,
            );
            let comparison = path.starts_with("cas-");
            let result_ty = if comparison {
                m.generate_predeclared_type(naga::PredeclaredType::AtomicCompareExchangeWeakResult(
                    naga::Scalar::U64,
                ))
            } else {
                address
            };
            let old = f.expressions.append(
                E::AtomicResult {
                    ty: result_ty,
                    comparison,
                },
                Span::UNDEFINED,
            );
            let zero = append_emit(&mut f, E::Literal(naga::Literal::U64(0)));
            f.body.push(
                S::Atomic {
                    pointer: holder,
                    fun: if path == "add" {
                        naga::AtomicFunction::Add
                    } else {
                        naga::AtomicFunction::Exchange {
                            compare: comparison.then_some(p),
                        }
                    },
                    value: if path == "add" {
                        zero
                    } else if path == "exchange-store" {
                        p
                    } else {
                        q
                    },
                    result: Some(old),
                },
                Span::UNDEFINED,
            );
            let value = if path == "cas-result" {
                append_emit(
                    &mut f,
                    E::AccessIndex {
                        base: old,
                        index: 0,
                    },
                )
            } else if path == "exchange-result" {
                old
            } else {
                append_emit(&mut f, E::Load { pointer: holder })
            };
            let pointer = append_emit(
                &mut f,
                E::PointerCast {
                    expr: value,
                    ty: pointer,
                },
            );
            let value = append_emit(&mut f, E::Literal(naga::Literal::U32(1)));
            f.body.push(S::Store { pointer, value }, Span::UNDEFINED);
            f.body.push(S::Return { value: None }, Span::UNDEFINED);
            m.functions.append(f, Span::UNDEFINED);
            let result = Validator::new(ValidationFlags::all(), C::all()).validate(&m);
            if immutable && path != "exchange-overwrite" {
                let error = format!(
                    "{:?}",
                    result.expect_err("atomic address operation lost immutable provenance")
                );
                assert!(error.contains("InvalidStorePointer"), "{path}: {error}");
            } else {
                result.unwrap_or_else(|error| panic!("{path}: {error:?}"));
            }
        }
    }
}
