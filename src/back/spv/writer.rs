/*! Standard Portable Intermediate Representation (SPIR-V) backend !*/
use super::{helpers, Instruction, LogicalLayout, PhysicalLayout, WriterFlags};
use crate::{Bytes, FastHashMap, FastHashSet, ImageFlags, VectorSize};
use spirv::{Op, Word};
use std::collections::hash_map::Entry;

const BITS_PER_BYTE: Bytes = 8;

enum Signedness {
    Unsigned = 0,
    Signed = 1,
}

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LocalType {
    Scalar {
        kind: crate::ScalarKind,
        width: Bytes,
    },
    Vector {
        size: crate::VectorSize,
        kind: crate::ScalarKind,
        width: Bytes,
    },
    Pointer {
        base: crate::Handle<crate::Type>,
        class: spirv::StorageClass,
    },
}

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LookupType {
    Handle(crate::Handle<crate::Type>),
    Local(LocalType),
}

fn map_dim(dim: crate::ImageDimension) -> spirv::Dim {
    match dim {
        crate::ImageDimension::D1 => spirv::Dim::Dim1D,
        crate::ImageDimension::D2 => spirv::Dim::Dim2D,
        crate::ImageDimension::D3 => spirv::Dim::Dim2D,
        crate::ImageDimension::Cube => spirv::Dim::DimCube,
    }
}

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
struct LookupFunctionType {
    parameter_type_ids: Vec<Word>,
    return_type_id: Word,
}

pub struct Writer {
    physical_layout: PhysicalLayout,
    logical_layout: LogicalLayout,
    id_count: u32,
    capabilities: FastHashSet<spirv::Capability>,
    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    writer_flags: WriterFlags,
    void_type: Option<u32>,
    lookup_type: FastHashMap<LookupType, Word>,
    lookup_function: FastHashMap<crate::Handle<crate::Function>, Word>,
    lookup_function_type: FastHashMap<LookupFunctionType, Word>,
    lookup_constant: FastHashMap<crate::Handle<crate::Constant>, Word>,
    lookup_global_variable: FastHashMap<crate::Handle<crate::GlobalVariable>, Word>,
}

impl Writer {
    pub fn new(header: &crate::Header, writer_flags: WriterFlags) -> Self {
        Writer {
            physical_layout: PhysicalLayout::new(header),
            logical_layout: LogicalLayout::default(),
            id_count: 0,
            capabilities: FastHashSet::default(),
            debugs: vec![],
            annotations: vec![],
            writer_flags,
            void_type: None,
            lookup_type: FastHashMap::default(),
            lookup_function: FastHashMap::default(),
            lookup_function_type: FastHashMap::default(),
            lookup_constant: FastHashMap::default(),
            lookup_global_variable: FastHashMap::default(),
        }
    }

    fn generate_id(&mut self) -> Word {
        self.id_count += 1;
        self.id_count
    }

    fn try_add_capabilities(&mut self, capabilities: &[spirv::Capability]) {
        for capability in capabilities.iter() {
            self.capabilities.insert(*capability);
        }
    }

    fn get_type_id(&mut self, arena: &crate::Arena<crate::Type>, lookup_ty: LookupType) -> Word {
        if let Entry::Occupied(e) = self.lookup_type.entry(lookup_ty) {
            *e.get()
        } else {
            match lookup_ty {
                LookupType::Handle(handle) => self.write_type_declaration_arena(arena, handle),
                LookupType::Local(local_ty) => self.write_type_declaration_local(arena, local_ty),
            }
        }
    }

    fn get_constant_id(
        &mut self,
        handle: crate::Handle<crate::Constant>,
        ir_module: &crate::Module,
    ) -> Word {
        match self.lookup_constant.entry(handle) {
            Entry::Occupied(e) => *e.get(),
            _ => {
                let (instruction, id) = self.write_constant_type(handle, ir_module);
                instruction.to_words(&mut self.logical_layout.declarations);
                id
            }
        }
    }

    fn get_global_variable_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        global_arena: &crate::Arena<crate::GlobalVariable>,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> Word {
        match self.lookup_global_variable.entry(handle) {
            Entry::Occupied(e) => *e.get(),
            _ => {
                let global_variable = &global_arena[handle];
                let (instruction, id) = self.write_global_variable(arena, global_variable, handle);
                instruction.to_words(&mut self.logical_layout.declarations);
                id
            }
        }
    }

    fn get_function_type(
        &mut self,
        ty: Option<crate::Handle<crate::Type>>,
        arena: &crate::Arena<crate::Type>,
    ) -> Word {
        match ty {
            Some(handle) => self.get_type_id(arena, LookupType::Handle(handle)),
            None => match self.void_type {
                Some(id) => id,
                None => {
                    let id = self.generate_id();
                    self.void_type = Some(id);
                    self.instruction_type_void(id)
                        .to_words(&mut self.logical_layout.declarations);
                    id
                }
            },
        }
    }

    fn get_pointer_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
        class: spirv::StorageClass,
    ) -> Word {
        let ty = &arena[handle];
        let ty_id = self.get_type_id(arena, LookupType::Handle(handle));
        match ty.inner {
            crate::TypeInner::Pointer { .. } => ty_id,
            _ => {
                match self
                    .lookup_type
                    .entry(LookupType::Local(LocalType::Pointer {
                        base: handle,
                        class,
                    })) {
                    Entry::Occupied(e) => *e.get(),
                    _ => {
                        let pointer_id = self.generate_id();
                        let instruction = self.instruction_type_pointer(pointer_id, class, ty_id);
                        instruction.to_words(&mut self.logical_layout.declarations);
                        self.lookup_type.insert(
                            LookupType::Local(LocalType::Pointer {
                                base: handle,
                                class,
                            }),
                            pointer_id,
                        );
                        pointer_id
                    }
                }
            }
        }
    }

    ///
    /// Debug Instructions
    ///

    fn instruction_source(
        &self,
        source_language: spirv::SourceLanguage,
        version: u32,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::Source);
        instruction.add_operand(source_language as u32);
        instruction.add_operands(helpers::bytes_to_words(&version.to_le_bytes()));
        instruction
    }

    fn instruction_name(&self, target_id: Word, name: &str) -> Instruction {
        let mut instruction = Instruction::new(Op::Name);
        instruction.set_result(target_id);
        instruction.add_operands(helpers::string_to_words(name));
        instruction
    }

    ///
    /// Annotation Instructions
    ///

    fn instruction_decorate(
        &self,
        target_id: Word,
        decoration: spirv::Decoration,
        operands: &[Word],
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::Decorate);
        instruction.add_operand(target_id);
        instruction.add_operand(decoration as u32);
        instruction.add_operands(Vec::from(operands));
        instruction
    }

    ///
    /// Extension Instructions
    ///

    fn instruction_ext_inst_import(&mut self, name: &str) -> Instruction {
        let mut instruction = Instruction::new(Op::ExtInstImport);
        let id = self.generate_id();
        instruction.set_result(id);
        instruction.add_operands(helpers::string_to_words(name));
        instruction
    }

    ///
    /// Mode-Setting Instructions
    ///

    fn instruction_memory_model(&mut self) -> Instruction {
        let mut instruction = Instruction::new(Op::MemoryModel);
        let addressing_model = spirv::AddressingModel::Logical;
        let memory_model = spirv::MemoryModel::GLSL450;
        self.try_add_capabilities(addressing_model.required_capabilities());
        self.try_add_capabilities(memory_model.required_capabilities());

        instruction.add_operand(addressing_model as u32);
        instruction.add_operand(memory_model as u32);
        instruction
    }

    fn instruction_entry_point(
        &mut self,
        entry_point: &crate::EntryPoint,
        ir_module: &crate::Module,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::EntryPoint);

        let function_id = *self.lookup_function.get(&entry_point.function).unwrap();

        let exec_model = match entry_point.stage {
            crate::ShaderStage::Vertex => spirv::ExecutionModel::Vertex,
            crate::ShaderStage::Fragment => spirv::ExecutionModel::Fragment,
            crate::ShaderStage::Compute => spirv::ExecutionModel::GLCompute,
        };

        instruction.add_operand(exec_model as u32);
        instruction.add_operand(function_id);
        instruction.add_operands(helpers::string_to_words(entry_point.name.as_str()));

        let function = &ir_module.functions[entry_point.function];
        for ((handle, _), &usage) in ir_module
            .global_variables
            .iter()
            .zip(&function.global_usage)
        {
            if usage.contains(crate::GlobalUse::STORE) || usage.contains(crate::GlobalUse::LOAD) {
                let id = self.get_global_variable_id(
                    &ir_module.types,
                    &ir_module.global_variables,
                    handle,
                );
                instruction.add_operand(id);
            }
        }

        self.try_add_capabilities(exec_model.required_capabilities());
        match entry_point.stage {
            crate::ShaderStage::Vertex => {}
            crate::ShaderStage::Fragment => {
                let execution_mode = spirv::ExecutionMode::OriginUpperLeft;
                self.try_add_capabilities(execution_mode.required_capabilities());
                self.instruction_execution_mode(function_id, execution_mode)
                    .to_words(&mut self.logical_layout.execution_modes);
            }
            crate::ShaderStage::Compute => {}
        }

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            self.debugs
                .push(self.instruction_name(function_id, entry_point.name.as_str()));
        }

        instruction
    }

    fn instruction_execution_mode(
        &self,
        function_id: Word,
        execution_mode: spirv::ExecutionMode,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::ExecutionMode);
        instruction.add_operand(function_id);
        instruction.add_operand(execution_mode as u32);
        instruction
    }

    fn instruction_capability(&self, capability: spirv::Capability) -> Instruction {
        let mut instruction = Instruction::new(Op::Capability);
        instruction.add_operand(capability as u32);
        instruction
    }

    ///
    /// Type-Declaration Instructions
    ///

    fn instruction_type_void(&self, id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeVoid);
        instruction.set_result(id);
        instruction
    }

    fn instruction_type_bool(&self, id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeBool);
        instruction.set_result(id);
        instruction
    }

    fn instruction_type_int(&self, id: Word, width: Word, signedness: Signedness) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeInt);
        instruction.set_result(id);
        instruction.add_operand(width);
        instruction.add_operand(signedness as u32);
        instruction
    }

    fn instruction_type_float(&self, id: Word, width: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeFloat);
        instruction.set_result(id);
        instruction.add_operand(width);
        instruction
    }

    fn instruction_type_vector(
        &self,
        id: Word,
        component_type_id: Word,
        component_count: VectorSize,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeVector);
        instruction.set_result(id);
        instruction.add_operand(component_type_id);
        instruction.add_operand(component_count as u32);
        instruction
    }

    fn instruction_type_matrix(
        &self,
        id: Word,
        column_type_id: Word,
        column_count: VectorSize,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeMatrix);
        instruction.set_result(id);
        instruction.add_operand(column_type_id);
        instruction.add_operand(column_count as u32);
        instruction
    }

    fn instruction_type_image(
        &self,
        id: Word,
        sampled_type_id: Word,
        dim: spirv::Dim,
        flags: ImageFlags,
        comparison: bool,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeImage);
        instruction.set_result(id);
        instruction.add_operand(sampled_type_id);
        instruction.add_operand(dim as u32);

        instruction.add_operand(if comparison { 1 } else { 0 });

        instruction.add_operand(if flags.contains(crate::ImageFlags::ARRAYED) {
            1
        } else {
            0
        });

        instruction.add_operand(if flags.contains(crate::ImageFlags::MULTISAMPLED) {
            1
        } else {
            0
        });

        instruction.add_operand(if flags.contains(crate::ImageFlags::SAMPLED) {
            1
        } else {
            0
        });

        // TODO Image Format defaults to Unknown, not yet in IR
        instruction.add_operand(spirv::ImageFormat::Unknown as u32);

        // Access Qualifier
        instruction.add_operand(
            if flags.contains(crate::ImageFlags::CAN_STORE)
                && flags.contains(crate::ImageFlags::CAN_LOAD)
            {
                2
            } else if flags.contains(crate::ImageFlags::CAN_STORE) {
                1
            } else {
                0
            },
        );
        instruction
    }

    fn instruction_type_sampler(&self, id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeSampler);
        instruction.set_result(id);
        instruction
    }

    fn instruction_type_array(
        &self,
        id: Word,
        element_type_id: Word,
        length_id: Word,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeArray);
        instruction.set_result(id);
        instruction.add_operand(element_type_id);
        instruction.add_operand(length_id);
        instruction
    }

    fn instruction_type_runtime_array(&self, id: Word, element_type_id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeRuntimeArray);
        instruction.set_result(id);
        instruction.add_operand(element_type_id);
        instruction
    }

    fn instruction_type_struct(&self, id: Word, member_ids: Vec<Word>) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeStruct);
        instruction.set_result(id);
        instruction.add_operands(member_ids);
        instruction
    }

    fn instruction_type_pointer(
        &self,
        id: Word,
        storage_class: spirv::StorageClass,
        type_id: Word,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::TypePointer);
        instruction.set_result(id);
        instruction.add_operand(storage_class as u32);
        instruction.add_operand(type_id);
        instruction
    }

    fn instruction_type_function(
        &self,
        id: Word,
        return_type_id: Word,
        parameter_ids: Vec<Word>,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeFunction);
        instruction.set_result(id);
        instruction.add_operand(return_type_id);
        instruction.add_operands(parameter_ids);
        instruction
    }

    ///
    /// Constant-Creation Instructions
    ///

    fn instruction_constant_true(&self, scalar_constant_id: Word, id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::ConstantTrue);
        instruction.set_type(scalar_constant_id);
        instruction.set_result(id);
        instruction
    }

    fn instruction_constant_false(&self, scalar_constant_id: Word, id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::ConstantFalse);
        instruction.set_type(scalar_constant_id);
        instruction.set_result(id);
        instruction
    }

    fn instruction_constant(
        &self,
        scalar_constant_id: Word,
        id: Word,
        values: &[Word],
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::Constant);
        instruction.set_type(scalar_constant_id);
        instruction.set_result(id);
        for value in values {
            instruction.add_operand(*value);
        }
        instruction
    }

    fn instruction_constant_composite(
        &self,
        composite_type_id: Word,
        id: Word,
        constituent_ids: Vec<Word>,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::ConstantComposite);
        instruction.set_type(composite_type_id);
        instruction.set_result(id);
        instruction.add_operands(constituent_ids);
        instruction
    }

    ///
    /// Memory Instructions
    ///

    fn instruction_variable(
        &self,
        pointer_type_id: Word,
        id: Word,
        storage_class: spirv::StorageClass,
        initializer_id: Option<Word>,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::Variable);
        instruction.set_type(pointer_type_id);
        instruction.set_result(id);
        instruction.add_operand(storage_class as u32);

        if let Some(initializer_id) = initializer_id {
            instruction.add_operand(initializer_id);
        }

        instruction
    }

    fn instruction_load(
        &self,
        type_id: Word,
        id: Word,
        pointer_type_id: Word,
        memory_access: Option<spirv::MemoryAccess>,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::Load);
        instruction.set_type(type_id);
        instruction.set_result(id);
        instruction.add_operand(pointer_type_id);

        instruction.add_operand(if let Some(memory_access) = memory_access {
            memory_access.bits()
        } else {
            spirv::MemoryAccess::NONE.bits()
        });

        instruction
    }

    fn instruction_store(&self, pointer_type_id: Word, object_id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::Store);
        instruction.set_type(pointer_type_id);
        instruction.add_operand(object_id);
        instruction
    }

    ///
    /// Function Instructions
    ///

    fn instruction_function(
        &self,
        return_type_id: Word,
        id: Word,
        function_control: spirv::FunctionControl,
        function_type_id: Word,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::Function);
        instruction.set_type(return_type_id);
        instruction.set_result(id);
        instruction.add_operand(function_control.bits());
        instruction.add_operand(function_type_id);
        instruction
    }

    fn instruction_function_end(&self) -> Instruction {
        Instruction::new(Op::FunctionEnd)
    }

    ///
    /// Image Instructions
    ///

    ///
    /// Conversion Instructions
    ///

    ///
    /// Composite Instructions
    ///

    fn instruction_composite_construct(
        &self,
        composite_type_id: Word,
        id: Word,
        constituent_ids: Vec<Word>,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::CompositeConstruct);
        instruction.set_type(composite_type_id);
        instruction.set_result(id);
        instruction.add_operands(constituent_ids);
        instruction
    }

    ///
    /// Arithmetic Instructions
    ///

    fn instruction_vector_times_scalar(
        &self,
        float_type_id: Word,
        id: Word,
        vector_type_id: Word,
        scalar_type_id: Word,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::VectorTimesScalar);
        instruction.set_type(float_type_id);
        instruction.set_result(id);
        instruction.add_operand(vector_type_id);
        instruction.add_operand(scalar_type_id);
        instruction
    }

    ///
    /// Bit Instructions
    ///

    ///
    /// Relational and Logical Instructions
    ///

    ///
    /// Derivative Instructions
    ///

    ///
    /// Control-Flow Instructions
    ///

    fn instruction_label(&self, id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::Label);
        instruction.set_result(id);
        instruction
    }

    fn instruction_return(&self) -> Instruction {
        Instruction::new(Op::Return)
    }

    fn instruction_return_value(&self, value_id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::ReturnValue);
        instruction.add_operand(value_id);
        instruction
    }

    ///
    /// Atomic Instructions
    ///

    ///
    /// Primitive Instructions
    ///

    fn write_scalar(&self, id: Word, kind: crate::ScalarKind, width: Bytes) -> Instruction {
        let bits = (width * BITS_PER_BYTE) as u32;
        match kind {
            crate::ScalarKind::Sint => self.instruction_type_int(id, bits, Signedness::Signed),
            crate::ScalarKind::Uint => self.instruction_type_int(id, bits, Signedness::Unsigned),
            crate::ScalarKind::Float => self.instruction_type_float(id, bits),
            crate::ScalarKind::Bool => self.instruction_type_bool(id),
        }
    }

    fn parse_to_spirv_storage_class(&self, class: crate::StorageClass) -> spirv::StorageClass {
        match class {
            crate::StorageClass::Constant => spirv::StorageClass::UniformConstant,
            crate::StorageClass::Function => spirv::StorageClass::Function,
            crate::StorageClass::Input => spirv::StorageClass::Input,
            crate::StorageClass::Output => spirv::StorageClass::Output,
            crate::StorageClass::Private => spirv::StorageClass::Private,
            crate::StorageClass::StorageBuffer => spirv::StorageClass::StorageBuffer,
            crate::StorageClass::Uniform => spirv::StorageClass::Uniform,
            crate::StorageClass::WorkGroup => spirv::StorageClass::Workgroup,
        }
    }

    fn write_type_declaration_local(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        local_ty: LocalType,
    ) -> Word {
        let id = self.generate_id();
        let instruction = match local_ty {
            LocalType::Scalar { kind, width } => self.write_scalar(id, kind, width),
            LocalType::Vector { size, .. } => {
                let scalar_id = self.get_type_id(arena, LookupType::Local(local_ty));
                self.instruction_type_vector(id, scalar_id, size)
            }
            LocalType::Pointer { .. } => unimplemented!(),
        };

        self.lookup_type.insert(LookupType::Local(local_ty), id);
        instruction.to_words(&mut self.logical_layout.declarations);
        id
    }

    fn write_type_declaration_arena(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
    ) -> Word {
        let ty = &arena[handle];
        let id = self.generate_id();

        let instruction = match ty.inner {
            crate::TypeInner::Scalar { kind, width } => {
                self.lookup_type
                    .insert(LookupType::Local(LocalType::Scalar { kind, width }), id);
                self.write_scalar(id, kind, width)
            }
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_id =
                    self.get_type_id(arena, LookupType::Local(LocalType::Scalar { kind, width }));
                self.lookup_type.insert(
                    LookupType::Local(LocalType::Vector { size, kind, width }),
                    id,
                );
                self.instruction_type_vector(id, scalar_id, size)
            }
            crate::TypeInner::Matrix {
                columns,
                rows: _,
                kind,
                width,
            } => {
                let vector_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Vector {
                        size: columns,
                        kind,
                        width,
                    }),
                );
                self.instruction_type_matrix(id, vector_id, columns)
            }
            crate::TypeInner::Image { base, dim, flags } => {
                let type_id = self.get_type_id(arena, LookupType::Handle(base));
                let dim = map_dim(dim);
                self.try_add_capabilities(dim.required_capabilities());
                self.instruction_type_image(id, type_id, dim, flags, false)
            }
            crate::TypeInner::DepthImage { dim, arrayed } => {
                let type_id = 0; //TODO!
                let dim = map_dim(dim);
                self.try_add_capabilities(dim.required_capabilities());

                let flags = if arrayed {
                    crate::ImageFlags::ARRAYED
                } else {
                    crate::ImageFlags::empty()
                };
                self.instruction_type_image(id, type_id, dim, flags, true)
            }
            crate::TypeInner::Sampler { comparison: _ } => self.instruction_type_sampler(id),
            crate::TypeInner::Array { size, stride, .. } => {
                if let Some(array_stride) = stride {
                    self.annotations.push(self.instruction_decorate(
                        id,
                        spirv::Decoration::ArrayStride,
                        &[array_stride.get()],
                    ));
                }

                let type_id = self.get_type_id(arena, LookupType::Handle(handle));
                match size {
                    crate::ArraySize::Static(length) => {
                        self.instruction_type_array(id, type_id, length)
                    }
                    crate::ArraySize::Dynamic => self.instruction_type_runtime_array(id, type_id),
                }
            }
            crate::TypeInner::Struct { ref members } => {
                let mut member_ids = Vec::with_capacity(members.len());
                for member in members {
                    let member_id = self.get_type_id(arena, LookupType::Handle(member.ty));
                    member_ids.push(member_id);
                }
                self.instruction_type_struct(id, member_ids)
            }
            crate::TypeInner::Pointer { base, class } => {
                let type_id = self.get_type_id(arena, LookupType::Handle(base));
                self.lookup_type.insert(
                    LookupType::Local(LocalType::Pointer {
                        base,
                        class: self.parse_to_spirv_storage_class(class),
                    }),
                    id,
                );
                self.instruction_type_pointer(id, self.parse_to_spirv_storage_class(class), type_id)
            }
        };

        self.lookup_type.insert(LookupType::Handle(handle), id);
        instruction.to_words(&mut self.logical_layout.declarations);
        id
    }

    fn write_constant_type(
        &mut self,
        handle: crate::Handle<crate::Constant>,
        ir_module: &crate::Module,
    ) -> (Instruction, Word) {
        let id = self.generate_id();
        self.lookup_constant.insert(handle, id);
        let constant = &ir_module.constants[handle];
        let arena = &ir_module.types;

        match constant.inner {
            crate::ConstantInner::Sint(val) => {
                let ty = &ir_module.types[constant.ty];
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        4 => self.instruction_constant(type_id, id, &[val as u32]),
                        8 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            self.instruction_constant(type_id, id, &[low, high])
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };
                (instruction, id)
            }
            crate::ConstantInner::Uint(val) => {
                let ty = &ir_module.types[constant.ty];
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        4 => self.instruction_constant(type_id, id, &[val as u32]),
                        8 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            self.instruction_constant(type_id, id, &[low, high])
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };

                (instruction, id)
            }
            crate::ConstantInner::Float(val) => {
                let ty = &ir_module.types[constant.ty];
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        4 => self.instruction_constant(type_id, id, &[(val as f32).to_bits()]),
                        8 => {
                            let bits = f64::to_bits(val);
                            let (low, high) = ((bits >> 32) as u32, bits as u32);
                            self.instruction_constant(type_id, id, &[low, high])
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };
                (instruction, id)
            }
            crate::ConstantInner::Bool(val) => {
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = if val {
                    self.instruction_constant_true(type_id, id)
                } else {
                    self.instruction_constant_false(type_id, id)
                };

                (instruction, id)
            }
            crate::ConstantInner::Composite(ref constituents) => {
                let mut constituent_ids = Vec::with_capacity(constituents.len());
                for constituent in constituents.iter() {
                    let constituent_id = self.get_constant_id(*constituent, &ir_module);
                    constituent_ids.push(constituent_id);
                }

                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));
                let instruction = self.instruction_constant_composite(type_id, id, constituent_ids);
                (instruction, id)
            }
        }
    }

    fn write_global_variable(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        global_variable: &crate::GlobalVariable,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> (Instruction, Word) {
        let id = self.generate_id();

        let class = self.parse_to_spirv_storage_class(global_variable.class);
        self.try_add_capabilities(class.required_capabilities());

        let pointer_id = self.get_pointer_id(arena, global_variable.ty, class);
        let instruction = self.instruction_variable(pointer_id, id, class, None);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            self.debugs
                .push(self.instruction_name(id, global_variable.name.as_ref().unwrap().as_str()));
        }

        match global_variable.binding.as_ref().unwrap() {
            crate::Binding::Location(location) => {
                self.annotations.push(self.instruction_decorate(
                    id,
                    spirv::Decoration::Location,
                    &[*location],
                ));
            }
            crate::Binding::Descriptor { set, binding } => {
                self.annotations.push(self.instruction_decorate(
                    id,
                    spirv::Decoration::DescriptorSet,
                    &[*set],
                ));
                self.annotations.push(self.instruction_decorate(
                    id,
                    spirv::Decoration::Binding,
                    &[*binding],
                ));
            }
            crate::Binding::BuiltIn(built_in) => {
                let built_in = match built_in {
                    crate::BuiltIn::BaseInstance => spirv::BuiltIn::BaseInstance,
                    crate::BuiltIn::BaseVertex => spirv::BuiltIn::BaseVertex,
                    crate::BuiltIn::ClipDistance => spirv::BuiltIn::ClipDistance,
                    crate::BuiltIn::InstanceIndex => spirv::BuiltIn::InstanceIndex,
                    crate::BuiltIn::Position => spirv::BuiltIn::Position,
                    crate::BuiltIn::VertexIndex => spirv::BuiltIn::VertexIndex,
                    crate::BuiltIn::PointSize => spirv::BuiltIn::PointSize,
                    crate::BuiltIn::FragCoord => spirv::BuiltIn::FragCoord,
                    crate::BuiltIn::FrontFacing => spirv::BuiltIn::FrontFacing,
                    crate::BuiltIn::SampleIndex => spirv::BuiltIn::SampleId,
                    crate::BuiltIn::FragDepth => spirv::BuiltIn::FragDepth,
                    crate::BuiltIn::GlobalInvocationId => spirv::BuiltIn::GlobalInvocationId,
                    crate::BuiltIn::LocalInvocationId => spirv::BuiltIn::LocalInvocationId,
                    crate::BuiltIn::LocalInvocationIndex => spirv::BuiltIn::LocalInvocationIndex,
                    crate::BuiltIn::WorkGroupId => spirv::BuiltIn::WorkgroupId,
                };

                self.annotations.push(self.instruction_decorate(
                    id,
                    spirv::Decoration::BuiltIn,
                    &[built_in as u32],
                ));
            }
        }

        // TODO Initializer is optional and not (yet) included in the IR

        self.lookup_global_variable.insert(handle, id);
        (instruction, id)
    }

    fn write_function_type(&mut self, lookup_function_type: LookupFunctionType) -> Word {
        match self
            .lookup_function_type
            .entry(lookup_function_type.clone())
        {
            Entry::Occupied(e) => *e.get(),
            _ => {
                let id = self.generate_id();
                let instruction = self.instruction_type_function(
                    id,
                    lookup_function_type.return_type_id,
                    lookup_function_type.parameter_type_ids.clone(),
                );
                instruction.to_words(&mut self.logical_layout.declarations);
                self.lookup_function_type.insert(lookup_function_type, id);
                id
            }
        }
    }

    fn write_function(
        &mut self,
        handle: crate::Handle<crate::Function>,
        function: &crate::Function,
        arena: &crate::Arena<crate::Type>,
    ) -> Instruction {
        let id = self.generate_id();

        let return_type_id = self.get_function_type(function.return_type, arena);

        let mut parameter_type_ids = Vec::with_capacity(function.parameter_types.len());
        for parameter_type in function.parameter_types.iter() {
            parameter_type_ids.push(self.get_type_id(arena, LookupType::Handle(*parameter_type)))
        }

        let lookup_function_type = LookupFunctionType {
            return_type_id,
            parameter_type_ids,
        };

        let type_function_id = self.write_function_type(lookup_function_type);

        let instruction = self.instruction_function(
            return_type_id,
            id,
            spirv::FunctionControl::empty(),
            type_function_id,
        );

        self.lookup_function.insert(handle, id);
        instruction
    }

    fn write_expression<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        function: &crate::Function,
        expression: &crate::Expression,
        output: &mut Vec<Instruction>,
    ) -> (Word, &'a crate::TypeInner) {
        match expression {
            crate::Expression::GlobalVariable(handle) => {
                let var = &ir_module.global_variables[*handle];
                let inner = &ir_module.types[var.ty].inner;
                let id = self.get_global_variable_id(
                    &ir_module.types,
                    &ir_module.global_variables,
                    *handle,
                );
                (id, inner)
            }
            crate::Expression::Constant(handle) => {
                let var = &ir_module.constants[*handle];
                let inner = &ir_module.types[var.ty].inner;
                let id = self.get_constant_id(*handle, ir_module);
                (id, inner)
            }
            crate::Expression::Compose { ty, components } => {
                let var = &ir_module.types[*ty];
                let inner = &var.inner;
                let id = self.generate_id();
                let type_id = self.get_type_id(&ir_module.types, LookupType::Handle(*ty));

                let mut constituent_ids = Vec::with_capacity(components.len());
                for component in components {
                    let expression = &function.expressions[*component];
                    let (component_id, _) =
                        self.write_expression(ir_module, &function, expression, output);
                    constituent_ids.push(component_id);
                }

                let instruction =
                    self.instruction_composite_construct(type_id, id, constituent_ids);
                output.push(instruction);

                (id, inner)
            }
            crate::Expression::Binary { op, left, right } => {
                match op {
                    crate::BinaryOperator::Multiply => {
                        // TODO OpVectorTimesScalar is only supported
                        let id = self.generate_id();
                        let left_expression = &function.expressions[*left];
                        let right_expression = &function.expressions[*right];
                        let (left_id, left_inner) =
                            self.write_expression(ir_module, function, left_expression, output);
                        let (right_id, right_inner) =
                            self.write_expression(ir_module, function, right_expression, output);

                        let (result_type_id, vector_id, scalar_id) = match (left_inner, right_inner)
                        {
                            (
                                crate::TypeInner::Vector { size, kind, width },
                                crate::TypeInner::Scalar { .. },
                            ) => {
                                let result_type_id = *self
                                    .lookup_type
                                    .get(&LookupType::Local(LocalType::Vector {
                                        size: *size,
                                        kind: *kind,
                                        width: *width,
                                    }))
                                    .unwrap();

                                (result_type_id, left_id, right_id)
                            }
                            (
                                crate::TypeInner::Scalar { .. },
                                crate::TypeInner::Vector { size, kind, width },
                            ) => {
                                let result_type_id = *self
                                    .lookup_type
                                    .get(&LookupType::Local(LocalType::Vector {
                                        size: *size,
                                        kind: *kind,
                                        width: *width,
                                    }))
                                    .unwrap();
                                (result_type_id, right_id, left_id)
                            }
                            _ => unreachable!("Expression requires both a scalar and vector"),
                        };

                        // TODO Quick fix
                        let load_id = self.generate_id();

                        let load_instruction =
                            self.instruction_load(result_type_id, load_id, vector_id, None);
                        output.push(load_instruction);

                        let instruction = self.instruction_vector_times_scalar(
                            result_type_id,
                            id,
                            load_id,
                            scalar_id,
                        );
                        output.push(instruction);

                        // TODO Not sure how or what to return
                        (
                            id,
                            &crate::TypeInner::Scalar {
                                kind: crate::ScalarKind::Float,
                                width: 10,
                            },
                        )
                    }
                    _ => unimplemented!("{:?}", op),
                }
            }
            crate::Expression::LocalVariable(variable) => {
                let id = self.generate_id();
                let var = &function.local_variables[*variable];
                let ty = &ir_module.types[var.ty];

                let pointer_id =
                    self.get_pointer_id(&ir_module.types, var.ty, spirv::StorageClass::Function);

                let instruction =
                    self.instruction_variable(pointer_id, id, spirv::StorageClass::Function, None);
                output.push(instruction);
                (id, &ty.inner)
            }
            _ => unimplemented!("{:?}", expression),
        }
    }

    fn write_function_block(
        &mut self,
        ir_module: &crate::Module,
        function: &crate::Function,
        statement: &crate::Statement,
        output: &mut Vec<Instruction>,
    ) -> Instruction {
        match statement {
            crate::Statement::Return { .. } => match function.return_type {
                Some(ty) => {
                    let value_id = self.get_type_id(&ir_module.types, LookupType::Handle(ty));
                    self.instruction_return_value(value_id)
                }
                None => self.instruction_return(),
            },
            crate::Statement::Store { pointer, value } => {
                let pointer_expression = &function.expressions[*pointer];
                let value_expression = &function.expressions[*value];
                let (pointer_id, _) =
                    self.write_expression(ir_module, function, pointer_expression, output);
                let (value_id, _) =
                    self.write_expression(ir_module, function, value_expression, output);

                self.instruction_store(pointer_id, value_id)
            }
            _ => unimplemented!(),
        }
    }

    fn write_physical_layout(&mut self) {
        self.physical_layout.bound = self.id_count + 1;
    }

    fn write_logical_layout(&mut self, ir_module: &crate::Module) {
        self.instruction_ext_inst_import("GLSL.std.450")
            .to_words(&mut self.logical_layout.ext_inst_imports);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            self.debugs
                .push(self.instruction_source(spirv::SourceLanguage::GLSL, 450));
        }

        // Looking through all global variable, types, constants.
        // Doing this because we also want to include not used parts of the module
        // to be included in the output
        for (handle, _) in ir_module.types.iter() {
            self.get_type_id(&ir_module.types, LookupType::Handle(handle));
        }

        for (handle, _) in ir_module.global_variables.iter() {
            self.get_global_variable_id(&ir_module.types, &ir_module.global_variables, handle);
        }

        for (handle, _) in ir_module.constants.iter() {
            self.get_constant_id(handle, &ir_module);
        }

        for annotation in self.annotations.iter() {
            annotation.to_words(&mut self.logical_layout.annotations);
        }

        for capability in self.capabilities.iter() {
            self.instruction_capability(*capability)
                .to_words(&mut self.logical_layout.capabilities);
        }

        for (handle, function) in ir_module.functions.iter() {
            let mut function_instructions: Vec<Instruction> = vec![];
            function_instructions.push(self.write_function(handle, function, &ir_module.types));

            let id = self.generate_id();
            function_instructions.push(self.instruction_label(id));

            for block in function.body.iter() {
                let mut output: Vec<Instruction> = vec![];
                let instruction =
                    self.write_function_block(ir_module, function, &block, &mut output);
                function_instructions.append(&mut output);
                function_instructions.push(instruction);
            }

            function_instructions.push(self.instruction_function_end());
            for instruction in function_instructions.iter() {
                instruction.to_words(&mut self.logical_layout.function_definitions);
            }
        }

        for entry_point in ir_module.entry_points.iter() {
            let entry_point_instruction = self.instruction_entry_point(entry_point, ir_module);
            entry_point_instruction.to_words(&mut self.logical_layout.entry_points);
        }

        self.instruction_memory_model()
            .to_words(&mut self.logical_layout.memory_model);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            for debug in self.debugs.iter() {
                debug.to_words(&mut self.logical_layout.debugs);
            }
        }
    }

    pub fn write(&mut self, ir_module: &crate::Module) -> Vec<Word> {
        let mut words: Vec<Word> = vec![];

        self.write_logical_layout(ir_module);
        self.write_physical_layout();

        self.physical_layout.in_words(&mut words);
        self.logical_layout.in_words(&mut words);
        words
    }
}

#[cfg(test)]
mod tests {
    use crate::back::spv::test_framework::*;
    use crate::back::spv::{Writer, WriterFlags};
    use crate::Header;
    use spirv::*;

    #[test]
    fn test_writer_generate_id() {
        let mut writer = create_writer();

        assert_eq!(writer.id_count, 0);
        writer.generate_id();
        assert_eq!(writer.id_count, 1);
    }

    #[test]
    fn test_try_add_capabilities() {
        let mut writer = create_writer();

        assert_eq!(writer.capabilities.len(), 0);
        writer.try_add_capabilities(&[spirv::Capability::Shader]);
        assert_eq!(writer.capabilities.len(), 1);

        writer.try_add_capabilities(&[spirv::Capability::Shader]);
        assert_eq!(writer.capabilities.len(), 1);
    }

    #[test]
    fn test_instruction_capability() {
        let writer = create_writer();
        let instruction = writer.instruction_capability(spirv::Capability::Shader);
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::Capability,
            wc: 2,
            type_id: false,
            result_id: false,
            operands: true,
        };

        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_ext_inst_import() {
        let mut writer = create_writer();
        let import_name = "GLSL.std.450";
        let instruction = writer.instruction_ext_inst_import(import_name);
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::ExtInstImport,
            wc: 2,
            type_id: false,
            result_id: true,
            operands: true,
        };

        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_memory_model() {
        let mut writer = create_writer();
        let instruction = writer.instruction_memory_model();
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::MemoryModel,
            wc: 3,
            type_id: false,
            result_id: false,
            operands: true,
        };
        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_name() {
        let writer = create_writer();
        let instruction = writer.instruction_name(1, "Test");
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::Name,
            wc: 3,
            type_id: false,
            result_id: true,
            operands: true,
        };
        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_execution_mode() {
        let writer = create_writer();
        let instruction = writer.instruction_execution_mode(1, ExecutionMode::OriginUpperLeft);
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::ExecutionMode,
            wc: 3,
            type_id: false,
            result_id: false,
            operands: true,
        };
        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_source() {
        let writer = create_writer();
        let version = 450;
        let instruction = writer.instruction_source(SourceLanguage::GLSL, version);
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::Source,
            wc: 3,
            type_id: false,
            result_id: false,
            operands: true,
        };
        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_label() {
        let writer = create_writer();
        let instruction = writer.instruction_label(1);
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::Label,
            wc: 2,
            type_id: false,
            result_id: true,
            operands: false,
        };
        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_function_end() {
        let writer = create_writer();
        let instruction = writer.instruction_function_end();
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::FunctionEnd,
            wc: 1,
            type_id: false,
            result_id: false,
            operands: false,
        };
        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_instruction_decorate() {
        let writer = create_writer();
        let instruction = writer.instruction_decorate(1, Decoration::Location, &[1]);
        let mut output = vec![];

        let requirements = SpecRequirements {
            op: Op::Decorate,
            wc: 3,
            type_id: false,
            result_id: false,
            operands: true,
        };
        validate_spec_requirements(requirements, &instruction);

        instruction.to_words(&mut output);
        validate_instruction(output.as_slice(), &instruction);
    }

    #[test]
    fn test_write_physical_layout() {
        let mut writer = create_writer();
        assert_eq!(writer.physical_layout.bound, 0);
        writer.write_physical_layout();
        assert_eq!(writer.physical_layout.bound, 1);
    }

    fn create_writer() -> Writer {
        let header = Header {
            generator: 0,
            version: (1, 0, 0),
        };
        Writer::new(&header, WriterFlags::NONE)
    }
}
