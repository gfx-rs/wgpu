use std::{
    borrow::Cow,
    cell::RefCell,
    num::NonZeroU32,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::Duration,
};

use ash::vk;
use naga::{AddressSpace, Expression as E, Span, Statement as S, Type, TypeInner as T};

#[cfg(feature = "extended-native-tests")]
mod extended;
#[cfg_attr(
    not(feature = "extended-native-tests"),
    expect(dead_code, reason = "byte records only execute in the extended cases")
)]
mod pointer_helpers;
use pointer_helpers::SpanMode;

type Vulkan = wgpu::hal::api::Vulkan;
const COUNT: u32 = 64;
const GUARD: u32 = 0xfeed_beef;

/// Span case: element count, scalar layout, and span mode.
type SpanCase = (u32, bool, SpanMode);

/// Validator capabilities required by every physical pointer module in this package. Each case
/// adds only what it needs, so a capability gap is not masked by an over-broad set.
fn native_capabilities() -> naga::valid::Capabilities {
    naga::valid::Capabilities::IMMEDIATES
        | naga::valid::Capabilities::PHYSICAL_STORAGE_BUFFER_ADDRESSES
}

/// Validates `module` with exactly `capabilities` and writes SPIR-V 1.3.
fn write_spirv(
    module: &naga::Module,
    capabilities: naga::valid::Capabilities,
    bounds_check_policies: naga::proc::BoundsCheckPolicies,
) -> Vec<u32> {
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(module)
        .unwrap_or_else(|error| panic!("validate with {capabilities:?}: {error:?}"));
    naga::back::spv::write_vec(
        module,
        &info,
        &naga::back::spv::Options {
            lang_version: (1, 3),
            bounds_check_policies,
            use_storage_input_output_16: false,
            ..Default::default()
        },
        None,
    )
    .unwrap()
}

/// Runs `spirv-val` (from `SPIRV_VAL` or PATH) on `words`, removing its temporary file.
fn spirv_val(words: &[u32], scalar_layout: bool, context: &str) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "naga-physical-storage-gpu-{}-{}.spv",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(
        &path,
        words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let validation = std::process::Command::new(
        std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into()),
    )
    .args(["--target-env", "vulkan1.1"])
    .args(scalar_layout.then_some("--scalar-block-layout"))
    .arg(&path)
    .output();
    std::fs::remove_file(&path).unwrap();
    let validation = validation.expect("install spirv-val or set SPIRV_VAL");
    assert!(
        validation.status.success(),
        "spirv-val rejected {context}: {}\n{}",
        String::from_utf8_lossy(&validation.stderr),
        rspirv::binary::Disassemble::disassemble(&rspirv::dr::load_words(words).unwrap())
    );
    println!("Validated {} SPIR-V words: {context}", words.len());
}

fn shader_capabilities(helper: bool, span: Option<SpanCase>) -> naga::valid::Capabilities {
    let mut capabilities = native_capabilities();
    if helper || span.is_some() {
        // The pointer helper and span modules compute addresses with 64-bit integers.
        capabilities |= naga::valid::Capabilities::SHADER_INT64;
    }
    if let Some((_, scalar, mode)) = span {
        // The scalar-layout matrix span has no under-aligned pointee, so it validates
        // without the scalar layout capability.
        if scalar && mode != SpanMode::Matrix {
            capabilities |= naga::valid::Capabilities::PHYSICAL_STORAGE_SCALAR_LAYOUT;
        }
    }
    capabilities
}

fn shader(
    policy: naga::proc::BoundsCheckPolicy,
    extended: bool,
    span: Option<SpanCase>,
    atomic: bool,
) -> (naga::Module, Vec<u32>) {
    let mut m = naga::Module::default();
    let uint = m.types.insert(
        Type {
            name: None,
            inner: T::Scalar(naga::Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let vector = m.types.insert(
        Type {
            name: None,
            inner: T::Vector {
                size: naga::VectorSize::Tri,
                scalar: naga::Scalar::U32,
            },
        },
        Span::UNDEFINED,
    );
    let element_type = if atomic {
        m.types.insert(
            Type {
                name: None,
                inner: T::Atomic(naga::Scalar::U32),
            },
            Span::UNDEFINED,
        )
    } else {
        uint
    };
    let array = m.types.insert(
        Type {
            name: None,
            inner: T::Array {
                base: element_type,
                size: naga::ArraySize::Constant(NonZeroU32::new(COUNT).unwrap()),
                stride: 4,
            },
        },
        Span::UNDEFINED,
    );
    let pointer = m.types.insert(
        Type {
            name: None,
            inner: T::Pointer {
                base: array,
                space: AddressSpace::PhysicalStorage,
            },
        },
        Span::UNDEFINED,
    );
    let root = m.types.insert(
        Type {
            name: Some("Arguments".into()),
            inner: T::Struct {
                members: vec![
                    naga::StructMember {
                        access: None,
                        name: Some("address".into()),
                        ty: pointer,
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
                        name: Some("index_bias".into()),
                        ty: uint,
                        binding: None,
                        offset: 12,
                    },
                ],
                span: 16,
            },
        },
        Span::UNDEFINED,
    );
    let args = m.global_variables.append(
        naga::GlobalVariable {
            name: Some("args".into()),
            space: AddressSpace::Immediate,
            binding: None,
            ty: root,
            init: None,
            memory_decorations: naga::MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let helper = extended.then(|| pointer_helpers::helper(&mut m, pointer));
    let mut f = naga::Function::default();
    f.arguments.push(naga::FunctionArgument {
        name: Some("id".into()),
        ty: vector,
        binding: Some(naga::Binding::BuiltIn(naga::BuiltIn::GlobalInvocationId)),
        immutable_pointee: false,
    });
    let id = f
        .expressions
        .append(E::FunctionArgument(0), Span::UNDEFINED);
    let global = f
        .expressions
        .append(E::GlobalVariable(args), Span::UNDEFINED);
    let three = f
        .expressions
        .append(E::Literal(naga::Literal::U32(3)), Span::UNDEFINED);
    let zero = f
        .expressions
        .append(E::Literal(naga::Literal::U32(0)), Span::UNDEFINED);
    let id_index = f
        .expressions
        .append(E::AccessIndex { base: id, index: 0 }, Span::UNDEFINED);
    let bias_field = f.expressions.append(
        E::AccessIndex {
            base: global,
            index: 2,
        },
        Span::UNDEFINED,
    );
    let bias = f.expressions.append(
        E::Load {
            pointer: bias_field,
        },
        Span::UNDEFINED,
    );
    let index = f.expressions.append(
        E::Binary {
            op: naga::BinaryOperator::Add,
            left: id_index,
            right: bias,
        },
        Span::UNDEFINED,
    );
    let address_field = f.expressions.append(
        E::AccessIndex {
            base: global,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let address = f.expressions.append(
        E::Load {
            pointer: address_field,
        },
        Span::UNDEFINED,
    );
    let address = if let Some(helper) = helper {
        f.body.push(
            S::Emit(naga::Range::new_from_bounds(id_index, address)),
            Span::UNDEFINED,
        );
        let result = f.expressions.append(E::CallResult(helper), Span::UNDEFINED);
        f.body.push(
            S::Call {
                function: helper,
                arguments: vec![address],
                result: Some(result),
            },
            Span::UNDEFINED,
        );
        result
    } else {
        address
    };
    let element = f.expressions.append(
        E::Access {
            base: address,
            index,
        },
        Span::UNDEFINED,
    );
    let old = f
        .expressions
        .append(E::Load { pointer: element }, Span::UNDEFINED);
    let product = f.expressions.append(
        E::Binary {
            op: naga::BinaryOperator::Multiply,
            left: old,
            right: three,
        },
        Span::UNDEFINED,
    );
    let addend_field = f.expressions.append(
        E::AccessIndex {
            base: global,
            index: 1,
        },
        Span::UNDEFINED,
    );
    let addend = f.expressions.append(
        E::Load {
            pointer: addend_field,
        },
        Span::UNDEFINED,
    );
    let result = f.expressions.append(
        E::Binary {
            op: naga::BinaryOperator::Add,
            left: product,
            right: addend,
        },
        Span::UNDEFINED,
    );
    let observation = f.expressions.append(
        E::AccessIndex {
            base: address,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let observe = f.expressions.append(
        E::Binary {
            op: naga::BinaryOperator::NotEqual,
            left: bias,
            right: zero,
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::Emit(naga::Range::new_from_bounds(
            if extended { element } else { id_index },
            observe,
        )),
        Span::UNDEFINED,
    );
    f.body.push(
        S::Store {
            pointer: element,
            value: result,
        },
        Span::UNDEFINED,
    );
    let mut observed = naga::Block::new();
    observed.push(
        S::Store {
            pointer: observation,
            value: result,
        },
        Span::UNDEFINED,
    );
    f.body.push(
        S::If {
            condition: observe,
            accept: observed,
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
    if let Some((_, scalar, mode)) = span {
        m = match mode {
            SpanMode::Plain => pointer_helpers::span_module(scalar),
            SpanMode::Atomic => pointer_helpers::span_module_with_atomics(scalar, true),
            _ => pointer_helpers::span_module_with_mode(scalar, mode),
        };
    }
    if atomic {
        pointer_helpers::use_checked_atomics(&mut m);
    }
    assert!(m
        .global_variables
        .iter()
        .all(|(_, var)| var.binding.is_none()));
    let words = write_spirv(
        &m,
        shader_capabilities(extended, span),
        naga::proc::BoundsCheckPolicies {
            index: policy,
            buffer: policy,
            ..Default::default()
        },
    );
    let disassembly =
        rspirv::binary::Disassemble::disassemble(&rspirv::dr::load_words(&words).unwrap());
    assert!(disassembly.contains("OpMemoryModel PhysicalStorageBuffer64"));
    assert!(disassembly.contains("OpCapability PhysicalStorageBufferAddresses"));
    if span.is_some_and(|(_, _, mode)| {
        matches!(
            mode,
            SpanMode::Atomic | SpanMode::Contended | SpanMode::Feedback
        )
    }) {
        assert!(
            disassembly.contains(if span.unwrap().2 == SpanMode::Feedback {
                "OpAtomicOr"
            } else {
                "OpAtomicIAdd"
            })
        );
        if span.unwrap().2 == SpanMode::Atomic {
            for op in [
                "OpAtomicLoad",
                "OpAtomicStore",
                "OpAtomicOr",
                "OpAtomicCompareExchange",
            ] {
                assert!(disassembly.contains(op));
            }
        }
    } else if atomic {
        for op in [
            "OpAtomicIAdd",
            "OpAtomicStore",
            "OpPhi",
            "OpBranchConditional",
        ] {
            assert!(disassembly.contains(op), "missing {op}: {disassembly}");
        }
    } else {
        assert!(disassembly.contains("Aligned 4"));
    }
    spirv_val(
        &words,
        span.is_some_and(|(_, scalar, _)| scalar),
        "Naga IR physical pointer shader",
    );
    (m, words)
}

fn device() -> (wgpu::Device, wgpu::Queue) {
    let (device, queue, _) = device_with_features(wgpu::Features::empty());
    (device, queue)
}

fn device_with_features(
    extra: wgpu::Features,
) -> (
    wgpu::Device,
    wgpu::Queue,
    Vec<wgpu::CooperativeMatrixProperties>,
) {
    // Fail if validation is unavailable, rather than silently running without it.
    let entry = unsafe { ash::Entry::load() }.unwrap();
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap();
    assert!(
        layers
            .iter()
            .any(|layer| layer.layer_name_as_c_str().unwrap() == c"VK_LAYER_KHRONOS_validation"),
        "install Vulkan validation layers or set VK_ADD_LAYER_PATH"
    );
    assert!(wgpu_hal::VALIDATION_CANARY.get_and_reset().is_empty());
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::VALIDATION | wgpu::InstanceFlags::DEBUG,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
    let features = wgpu::Features::IMMEDIATES
        | wgpu::Features::PASSTHROUGH_SHADERS
        | wgpu::Features::SHADER_INT64
        | extra;
    let adapter = adapters.into_iter().find(|adapter| {
        let info = adapter.get_info();
        println!("Vulkan adapter: {} ({:?}), driver {} {}", info.name, info.device_type, info.driver, info.driver_info);
        if !(matches!(info.device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu)
                || (std::env::var_os("NAGA_ALLOW_SOFTWARE_ADAPTER").as_deref() == Some(std::ffi::OsStr::new("1")) && info.device_type == wgpu::DeviceType::Cpu))
            || !adapter.features().contains(features) { return false; }
        // SAFETY: the adapter keeps the instance and physical device alive.
        unsafe {
            let hal = adapter.as_hal::<Vulkan>().unwrap();
            let mut bda = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
            let mut scalar = vk::PhysicalDeviceScalarBlockLayoutFeatures::default();
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut bda).push_next(&mut scalar);
            hal.shared_instance().raw_instance().get_physical_device_features2(hal.raw_physical_device(), &mut query);
            bda.buffer_device_address == vk::TRUE && scalar.scalar_block_layout == vk::TRUE
        }
    }).expect("a Vulkan adapter supporting bufferDeviceAddress and passthrough is required; CPU adapters require NAGA_ALLOW_SOFTWARE_ADAPTER=1");
    let properties = adapter.cooperative_matrix_properties();
    println!("Cooperative matrix configurations: {properties:?}");
    let limits = wgpu::Limits {
        max_immediate_size: 16,
        ..Default::default()
    };
    let desc = wgpu::DeviceDescriptor {
        label: Some("physical pointer GPU test"),
        // SAFETY: these tests validate the native shader and its resource contracts.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        required_features: features,
        required_limits: limits,
        ..Default::default()
    };
    // SAFETY: all enabled features were queried above; `opened` belongs to this adapter.
    unsafe {
        let hal = adapter.as_hal::<Vulkan>().unwrap();
        let mut bda =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut scalar =
            vk::PhysicalDeviceScalarBlockLayoutFeatures::default().scalar_block_layout(true);
        let mut storage8 = vk::PhysicalDevice8BitStorageFeatures::default();
        if cfg!(feature = "extended-native-tests") {
            let mut int8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
            let mut query = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut storage8)
                .push_next(&mut int8);
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_features2(hal.raw_physical_device(), &mut query);
            assert_eq!(storage8.storage_buffer8_bit_access, vk::TRUE);
            assert_eq!(storage8.uniform_and_storage_buffer8_bit_access, vk::TRUE);
            assert_eq!(int8.shader_int8, vk::TRUE);
            storage8 = vk::PhysicalDevice8BitStorageFeatures::default()
                .storage_buffer8_bit_access(true)
                .uniform_and_storage_buffer8_bit_access(true);
        }
        let opened = hal
            .open_with_callback(
                features,
                &desc.required_limits,
                &desc.memory_hints,
                Some(Box::new(|args| {
                    args.extensions.push(ash::khr::buffer_device_address::NAME);
                    args.extensions.push(ash::ext::scalar_block_layout::NAME);
                    *args.create_info = args.create_info.push_next(&mut bda).push_next(&mut scalar);
                    if cfg!(feature = "extended-native-tests") {
                        args.extensions.push(ash::khr::_8bit_storage::NAME);
                        *args.create_info = args.create_info.push_next(&mut storage8);
                    }
                })),
            )
            .unwrap();
        drop(hal);
        let (device, queue) = adapter
            .create_device_from_hal::<Vulkan>(opened, &desc)
            .unwrap();
        (device, queue, properties)
    }
}

fn addressed_buffer(device: &wgpu::Device, values: &[u32]) -> (wgpu::Buffer, u64) {
    let size = std::mem::size_of_val(values) as u64;
    // SAFETY: memory is compatible, bound and coherent; the import callback owns cleanup.
    unsafe {
        let hal = device.as_hal::<Vulkan>().unwrap();
        let raw = hal.raw_device().clone();
        let instance = hal.shared_instance().raw_instance();
        let buffer = raw
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::STORAGE_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let requirements = raw.get_buffer_memory_requirements(buffer);
        let properties = instance.get_physical_device_memory_properties(hal.raw_physical_device());
        let memory_type = (0..properties.memory_type_count)
            .find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[i as usize].property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .expect("host-visible coherent memory required for this test");
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = raw
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type)
                    .push_next(&mut flags),
                None,
            )
            .unwrap();
        raw.bind_buffer_memory(buffer, memory, 0).unwrap();
        let mapped = raw
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .unwrap();
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            mapped.cast::<u8>(),
            size as usize,
        );
        raw.unmap_memory(memory);
        let bda = ash::khr::buffer_device_address::Device::new(instance, &raw);
        let address =
            bda.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
        assert_ne!(address, 0);
        let keep_device_alive = device.clone();
        let imported = wgpu::hal::vulkan::Buffer::from_raw_externally_owned(
            buffer,
            Box::new(move || {
                raw.destroy_buffer(buffer, None);
                raw.free_memory(memory, None);
                drop(keep_device_alive);
            }),
        );
        drop(hal);
        let buffer = device.create_buffer_from_hal::<Vulkan>(
            imported,
            &wgpu::BufferDescriptor {
                label: Some("addressable data with guards"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        );
        (buffer, address)
    }
}
/// A test device that checks safe-API rejection once per module family.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    rejected: RefCell<naga::FastHashSet<String>>,
}

impl Gpu {
    fn new((device, queue): (wgpu::Device, wgpu::Queue)) -> Self {
        Self {
            device,
            queue,
            rejected: RefCell::default(),
        }
    }

    /// Safe wgpu must reject physical pointer IR. Each module family is checked once per device;
    /// repeating the check for every execution of an identical module adds no evidence.
    fn assert_safe_api_rejects(&self, family: &str, module: &naga::Module) {
        if !self.rejected.borrow_mut().insert(family.to_owned()) {
            return;
        }
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _rejected = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("physical pointers must not pass safe validation"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module.clone())),
            });
        let error =
            pollster::block_on(scope.pop()).expect("safe wgpu must reject physical pointer IR");
        assert!(
            matches!(error, wgpu::Error::Validation { .. }),
            "{family}: {error:?}"
        );
        println!("Safe shader API rejected {family}");
    }

    /// Loads `words` through passthrough, dispatches once per input buffer, and reads every
    /// buffer back. `immediates` receives the buffer index and its device address.
    fn run(
        &self,
        label: &str,
        words: Vec<u32>,
        workgroup_size: [u32; 3],
        workgroups: u32,
        inputs: &[Vec<u32>],
        immediates: impl Fn(usize, u64) -> [u8; 16],
    ) -> Vec<Vec<u32>> {
        let device = &self.device;
        // SAFETY: callers validate the SPIR-V and its entry-point metadata. Addressed buffers
        // remain live and aligned until execution completes.
        let shader = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(label),
                spirv: Some(Cow::Owned(words)),
                entry_points: Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: Cow::Borrowed("main"),
                    workgroup_size: (workgroup_size[0], workgroup_size[1], workgroup_size[2]),
                }]),
                ..Default::default()
            })
        };
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 16,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers: Vec<_> = inputs
            .iter()
            .map(|values| addressed_buffer(device, values))
            .collect();
        for (i, (_, address)) in buffers.iter().enumerate() {
            assert!(buffers[..i].iter().all(|(_, other)| other != address));
        }
        let sizes: Vec<u64> = inputs
            .iter()
            .map(|values| std::mem::size_of_val(values.as_slice()) as u64)
            .collect();
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: sizes.iter().sum(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut offset = 0;
        for (i, ((buffer, address), &size)) in buffers.iter().zip(&sizes).enumerate() {
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.transition_resources(
                    std::iter::once(wgpu::BufferTransition {
                        buffer,
                        state: wgpu::BufferUses::STORAGE_READ_WRITE,
                    }),
                    std::iter::empty(),
                );
                pass.set_immediates(0, &immediates(i, *address));
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(buffer, 0, &readback, offset, size);
            offset += size;
        }
        self.queue.submit([encoder.finish()]);
        let (send, receive) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).unwrap()
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
        receive
            .recv_timeout(Duration::from_secs(30))
            .unwrap()
            .unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let words: Vec<u32> = mapped
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        drop(mapped);
        readback.unmap();
        let mut rest = words.as_slice();
        let outputs = sizes
            .iter()
            .map(|&size| {
                let (output, tail) = rest.split_at(size as usize / 4);
                rest = tail;
                output.to_vec()
            })
            .collect();
        println!("Executed {label}");
        outputs
    }
}

/// Parses `words` with spv-in, re-emits them, and validates the result with `capabilities`.
fn round_trip_shader(
    words: &[u32],
    capabilities: naga::valid::Capabilities,
    scalar_layout: bool,
) -> (naga::Module, Vec<u32>) {
    let module = naga::front::spv::Frontend::new(
        words.iter().copied(),
        &naga::front::spv::Options {
            adjust_coordinate_space: false,
            ..Default::default()
        },
    )
    .parse()
    .unwrap();
    let words = write_spirv(&module, capabilities, Default::default());
    spirv_val(&words, scalar_layout, "imported and re-emitted SPIR-V");
    (module, words)
}

#[test]
fn naga_ir_physical_pointers_execute_through_wgpu_passthrough() {
    use naga::proc::BoundsCheckPolicy as Policy;
    const {
        assert!(
            cfg!(debug_assertions),
            "validation canary requires a debug test build"
        );
    }
    let gpu = Gpu::new(device());
    // Every case runs directly; `import` marks the representative of each case family that
    // also runs through spv-in. The compiler suite covers compaction of these modules.
    for policy in [
        Policy::Unchecked,
        Policy::ReadZeroSkipWrite,
        Policy::Restrict,
    ] {
        for helper in [false, true] {
            // Unchecked has no out-of-range case, so its in-range helper case is imported.
            let import = helper && policy == Policy::Unchecked;
            execute(&gpu, policy, false, helper, None, false, import);
            if policy != Policy::Unchecked {
                execute(&gpu, policy, true, helper, None, false, helper);
            }
        }
    }
    for count in [0, 17, COUNT] {
        let span = (count, false, SpanMode::Plain);
        execute(
            &gpu,
            Policy::Restrict,
            false,
            true,
            Some(span),
            false,
            count == 17,
        );
    }
    let span = (16, true, SpanMode::Plain);
    execute(&gpu, Policy::Restrict, false, true, Some(span), false, true);
    for scalar in [false, true] {
        let span = (if scalar { 16 } else { COUNT }, scalar, SpanMode::Atomic);
        execute(
            &gpu,
            Policy::Restrict,
            false,
            true,
            Some(span),
            false,
            !scalar,
        );
    }
    let layout = |row_major, padded| (8, false, SpanMode::MatrixLayout { row_major, padded });
    for (span, import) in [
        ((COUNT, false, SpanMode::Contended), true),
        ((COUNT, false, SpanMode::Feedback), true),
        ((16, true, SpanMode::Matrix), true),
        (layout(false, false), false),
        (layout(true, false), false),
        (layout(false, true), false),
        (layout(true, true), true),
    ] {
        execute(
            &gpu,
            Policy::Restrict,
            false,
            true,
            Some(span),
            false,
            import,
        );
    }
    let span = (COUNT, false, SpanMode::RuntimeArray);
    execute(
        &gpu,
        Policy::Unchecked,
        false,
        true,
        Some(span),
        false,
        true,
    );
    for out_of_bounds in [false, true] {
        let policy = Policy::ReadZeroSkipWrite;
        execute(
            &gpu,
            policy,
            out_of_bounds,
            false,
            None,
            true,
            out_of_bounds,
        );
    }
    drop(gpu);
    let errors = wgpu_hal::VALIDATION_CANARY.get_and_reset();
    assert!(errors.is_empty(), "Vulkan validation errors: {errors:#?}");
    #[cfg(feature = "extended-native-tests")]
    extended::run();
}

/// Executes one case directly and, when `import` is set, again after an spv-in round trip.
fn execute(
    gpu: &Gpu,
    policy: naga::proc::BoundsCheckPolicy,
    out_of_bounds: bool,
    extended: bool,
    span: Option<SpanCase>,
    atomic: bool,
    import: bool,
) {
    println!("Case: {policy:?}, out_of_bounds={out_of_bounds}, extended={extended}, span={span:?}, atomic={atomic}, import={import}");
    let (module, words) = shader(policy, extended, span, atomic);
    let family = match span {
        None => format!("extended={extended}, atomic={atomic}"),
        Some((_, _, SpanMode::MatrixLayout { .. })) => "MatrixLayout".to_owned(),
        Some((_, _, mode)) => format!("{mode:?}"),
    };
    gpu.assert_safe_api_rejects(&family, &module);
    let imported = import.then(|| {
        let mut capabilities = shader_capabilities(extended, span);
        if atomic {
            // spv-in introduces a u64 type absent from the source when importing the checked u32
            // atomic module, so the re-emitted SPIR-V declares Int64 although the original does not.
            capabilities |= naga::valid::Capabilities::SHADER_INT64;
        }
        let (module, words) = round_trip_shader(
            &words,
            capabilities,
            span.is_some_and(|(_, scalar, _)| scalar),
        );
        gpu.assert_safe_api_rejects(&format!("imported {family}"), &module);
        ("imported", words)
    });
    let inputs: Vec<Vec<u32>> = [17, 1000]
        .into_iter()
        .map(|seed| {
            std::iter::once(GUARD)
                .chain((0..COUNT).map(|i| seed + i * 5))
                .chain(std::iter::once(GUARD))
                .collect()
        })
        .collect();
    let bias = span.map_or(if out_of_bounds { COUNT } else { 0 }, |(count, _, _)| count);
    let workgroups = if span.is_some() {
        COUNT * 2
    } else if out_of_bounds {
        1
    } else {
        COUNT
    };
    for (path, words) in std::iter::once(("direct", words)).chain(imported) {
        let outputs = gpu.run(
            &format!("{path} Naga IR physical pointer shader"),
            words,
            [1, 1, 1],
            workgroups,
            &inputs,
            |i, address| {
                let addend = [7u32, 11][i];
                println!(
                    "Dispatching buffer {i} at device address {:#x}, addend={addend}",
                    address + 4
                );
                let mut args = [0u8; 16];
                args[..8].copy_from_slice(&(address + 4).to_le_bytes());
                args[8..12].copy_from_slice(&addend.to_le_bytes());
                args[12..16].copy_from_slice(&bias.to_le_bytes());
                args
            },
        );
        for (i, (input, actual)) in inputs.iter().zip(&outputs).enumerate() {
            verify(policy, out_of_bounds, span, i, input, actual);
        }
    }
}

fn verify(
    policy: naga::proc::BoundsCheckPolicy,
    out_of_bounds: bool,
    span: Option<SpanCase>,
    i: usize,
    input: &[u32],
    actual: &[u32],
) {
    assert_eq!(actual[0], GUARD, "leading guard, buffer {i}");
    assert_eq!(
        actual[COUNT as usize + 1],
        GUARD,
        "trailing guard, buffer {i}"
    );
    for j in 0..COUNT as usize {
        let changed = span.map_or(
            !out_of_bounds
                || (policy == naga::proc::BoundsCheckPolicy::Restrict && j == COUNT as usize - 1),
            |(count, scalar, _)| {
                if scalar {
                    j.is_multiple_of(4) && j / 4 < count as usize
                } else {
                    j < count as usize
                }
            },
        );
        let expected = if let Some((_, _, SpanMode::MatrixLayout { row_major, padded })) = span {
            let destination_stride = if row_major { 3 } else { 2 } + usize::from(padded);
            let source_stride = if row_major { 2 } else { 3 } + usize::from(padded);
            let slot = j % 8;
            let (column, row) = if row_major {
                (slot / destination_stride, slot % destination_stride)
            } else {
                (slot % destination_stride, slot / destination_stride)
            };
            if column < 2 && row < 3 {
                let source = if row_major {
                    row * source_stride + column
                } else {
                    column * source_stride + row
                };
                input[j / 8 * 8 + source + 1]
            } else {
                input[j + 1]
            }
        } else if span.is_some_and(|(_, _, mode)| mode == SpanMode::Feedback) {
            if j == 0 {
                u32::MAX
            } else {
                input[j + 1]
            }
        } else if let Some((count, _, SpanMode::Contended)) = span {
            input[j + 1] + if j == 0 { count * [7, 11][i] } else { 0 }
        } else if span.is_some_and(|(_, _, mode)| mode == SpanMode::Matrix) && changed {
            (input[j + 1] + [7, 11][i]) * 3 + [7, 11][i]
        } else if out_of_bounds && j == 0 {
            if policy == naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite {
                [7, 11][i]
            } else {
                input[COUNT as usize] * 3 + [7, 11][i]
            }
        } else if changed {
            input[j + 1] * 3 + [7, 11][i]
        } else {
            input[j + 1]
        };
        assert_eq!(actual[j + 1], expected, "buffer {i}, element {j}");
    }
    println!(
        "Verified buffer {i}: all {COUNT} expected values and both guards; first={}, last={}",
        actual[1], actual[COUNT as usize]
    );
}
