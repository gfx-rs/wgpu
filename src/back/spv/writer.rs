/*! Standard Portable Intermediate Representation (SPIR-V) backend !*/
use super::{helpers, Instruction, LogicalLayout, PhysicalLayout, WriterFlags};
use spirv::{Op, Word};
use std::collections::hash_map::Entry;

const BITS_PER_BYTE: crate::Bytes = 8;

struct Block {
    label: Option<Instruction>,
    body: Vec<Instruction>,
    termination: Option<Instruction>,
}

impl Block {
    pub fn new() -> Self {
        Block {
            label: None,
            body: vec![],
            termination: None,
        }
    }
}

struct LocalVariable {
    id: Word,
    name: Option<String>,
    instruction: Instruction,
}

struct Function {
    signature: Option<Instruction>,
    parameters: Vec<Instruction>,
    variables: Vec<LocalVariable>,
    blocks: Vec<Block>,
}

impl Function {
    pub fn new() -> Self {
        Function {
            signature: None,
            parameters: vec![],
            variables: vec![],
            blocks: vec![],
        }
    }

    fn to_words(&self, sink: &mut impl Extend<Word>) {
        self.signature.as_ref().unwrap().to_words(sink);
        for instruction in self.parameters.iter() {
            instruction.to_words(sink);
        }
        for (index, block) in self.blocks.iter().enumerate() {
            block.label.as_ref().unwrap().to_words(sink);
            if index == 0 {
                for local_var in self.variables.iter() {
                    local_var.instruction.to_words(sink);
                }
            }
            for instruction in block.body.iter() {
                instruction.to_words(sink);
            }
            block.termination.as_ref().unwrap().to_words(sink);
        }
    }
}

enum Signedness {
    Unsigned = 0,
    Signed = 1,
}

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LocalType {
    Scalar {
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    Vector {
        size: crate::VectorSize,
        kind: crate::ScalarKind,
        width: crate::Bytes,
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
    capabilities: crate::FastHashSet<spirv::Capability>,
    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    writer_flags: WriterFlags,
    void_type: Option<u32>,
    lookup_type: crate::FastHashMap<LookupType, Word>,
    lookup_function: crate::FastHashMap<crate::Handle<crate::Function>, Word>,
    lookup_function_type: crate::FastHashMap<LookupFunctionType, Word>,
    lookup_constant: crate::FastHashMap<crate::Handle<crate::Constant>, Word>,
    lookup_global_variable: crate::FastHashMap<crate::Handle<crate::GlobalVariable>, Word>,
}

impl Writer {
    pub fn new(header: &crate::Header, writer_flags: WriterFlags) -> Self {
        Writer {
            physical_layout: PhysicalLayout::new(header),
            logical_layout: LogicalLayout::default(),
            id_count: 0,
            capabilities: crate::FastHashSet::default(),
            debugs: vec![],
            annotations: vec![],
            writer_flags,
            void_type: None,
            lookup_type: crate::FastHashMap::default(),
            lookup_function: crate::FastHashMap::default(),
            lookup_function_type: crate::FastHashMap::default(),
            lookup_constant: crate::FastHashMap::default(),
            lookup_global_variable: crate::FastHashMap::default(),
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

    fn get_function_return_type(
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
            crate::ShaderStage::Fragment { .. } => spirv::ExecutionModel::Fragment,
            crate::ShaderStage::Compute { .. } => spirv::ExecutionModel::GLCompute,
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
            crate::ShaderStage::Fragment { .. } => {
                let execution_mode = spirv::ExecutionMode::OriginUpperLeft;
                self.try_add_capabilities(execution_mode.required_capabilities());
                self.instruction_execution_mode(function_id, execution_mode)
                    .to_words(&mut self.logical_layout.execution_modes);
            }
            crate::ShaderStage::Compute { .. } => {}
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
        component_count: crate::VectorSize,
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
        column_count: crate::VectorSize,
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
        arrayed: bool,
        class: crate::ImageClass,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::TypeImage);
        instruction.set_result(id);
        instruction.add_operand(sampled_type_id);
        instruction.add_operand(dim as u32);

        instruction.add_operand(match class {
            crate::ImageClass::Depth => 1,
            _ => 0,
        });
        instruction.add_operand(if arrayed { 1 } else { 0 });
        instruction.add_operand(match class {
            crate::ImageClass::Multisampled => 1,
            _ => 0,
        });
        instruction.add_operand(match class {
            crate::ImageClass::Sampled => 1,
            _ => 0,
        });

        let (format, access) = match class {
            crate::ImageClass::Storage(format, access) => {
                let spv_format = match format {
                    crate::StorageFormat::R8Unorm => spirv::ImageFormat::R8,
                    crate::StorageFormat::R8Snorm => spirv::ImageFormat::R8Snorm,
                    crate::StorageFormat::R8Uint => spirv::ImageFormat::R8ui,
                    crate::StorageFormat::R8Sint => spirv::ImageFormat::R8i,
                    crate::StorageFormat::R16Uint => spirv::ImageFormat::R16ui,
                    crate::StorageFormat::R16Sint => spirv::ImageFormat::R16i,
                    crate::StorageFormat::R16Float => spirv::ImageFormat::R16f,
                    crate::StorageFormat::Rg8Unorm => spirv::ImageFormat::Rg8,
                    crate::StorageFormat::Rg8Snorm => spirv::ImageFormat::Rg8Snorm,
                    crate::StorageFormat::Rg8Uint => spirv::ImageFormat::Rg8ui,
                    crate::StorageFormat::Rg8Sint => spirv::ImageFormat::Rg8i,
                    crate::StorageFormat::R32Uint => spirv::ImageFormat::R32ui,
                    crate::StorageFormat::R32Sint => spirv::ImageFormat::R32i,
                    crate::StorageFormat::R32Float => spirv::ImageFormat::R32f,
                    crate::StorageFormat::Rg16Uint => spirv::ImageFormat::Rg16ui,
                    crate::StorageFormat::Rg16Sint => spirv::ImageFormat::Rg16i,
                    crate::StorageFormat::Rg16Float => spirv::ImageFormat::Rg16f,
                    crate::StorageFormat::Rgba8Unorm => spirv::ImageFormat::Rgba8,
                    crate::StorageFormat::Rgba8Snorm => spirv::ImageFormat::Rgba8Snorm,
                    crate::StorageFormat::Rgba8Uint => spirv::ImageFormat::Rgba8ui,
                    crate::StorageFormat::Rgba8Sint => spirv::ImageFormat::Rgba8i,
                    crate::StorageFormat::Rgb10a2Unorm => spirv::ImageFormat::Rgb10a2ui,
                    crate::StorageFormat::Rg11b10Float => spirv::ImageFormat::R11fG11fB10f,
                    crate::StorageFormat::Rg32Uint => spirv::ImageFormat::Rg32ui,
                    crate::StorageFormat::Rg32Sint => spirv::ImageFormat::Rg32i,
                    crate::StorageFormat::Rg32Float => spirv::ImageFormat::Rg32f,
                    crate::StorageFormat::Rgba16Uint => spirv::ImageFormat::Rgba16ui,
                    crate::StorageFormat::Rgba16Sint => spirv::ImageFormat::Rgba16i,
                    crate::StorageFormat::Rgba16Float => spirv::ImageFormat::Rgba16f,
                    crate::StorageFormat::Rgba32Uint => spirv::ImageFormat::Rgba32ui,
                    crate::StorageFormat::Rgba32Sint => spirv::ImageFormat::Rgba32i,
                    crate::StorageFormat::Rgba32Float => spirv::ImageFormat::Rgba32f,
                };
                (spv_format, access)
            }
            _ => (spirv::ImageFormat::Unknown, crate::StorageAccess::empty()),
        };

        instruction.add_operand(format as u32);
        // Access Qualifier
        if !access.is_empty() {
            instruction.add_operand(if access == crate::StorageAccess::all() {
                2
            } else if access.contains(crate::StorageAccess::STORE) {
                1
            } else {
                0
            });
        }

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
        result_type_id: Word,
        result_id: Word,
        storage_class: spirv::StorageClass,
        initializer_id: Option<Word>,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::Variable);
        instruction.set_type(result_type_id);
        instruction.set_result(result_id);
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

    fn instruction_function_parameter(&self, result_type_id: Word, id: Word) -> Instruction {
        let mut instruction = Instruction::new(Op::FunctionParameter);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction
    }

    fn instruction_function_end(&self) -> Instruction {
        Instruction::new(Op::FunctionEnd)
    }

    fn instruction_function_call(
        &self,
        result_type_id: Word,
        id: Word,
        function_id: Word,
        argument_ids: Vec<Word>,
    ) -> Instruction {
        let mut instruction = Instruction::new(Op::FunctionCall);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(function_id);
        instruction.add_operands(argument_ids);
        instruction
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

    fn write_scalar(&self, id: Word, kind: crate::ScalarKind, width: crate::Bytes) -> Instruction {
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
            crate::TypeInner::Image {
                kind,
                dim,
                arrayed,
                class,
            } => {
                let type_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Vector {
                        size: crate::VectorSize::Quad,
                        kind,
                        width: 4,
                    }),
                );
                let dim = map_dim(dim);
                self.try_add_capabilities(dim.required_capabilities());
                self.instruction_type_image(id, type_id, dim, arrayed, class)
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

        if let Some(interpolation) = global_variable.interpolation {
            let decoration = match interpolation {
                crate::Interpolation::Linear => Some(spirv::Decoration::NoPerspective),
                crate::Interpolation::Flat => Some(spirv::Decoration::Flat),
                crate::Interpolation::Patch => Some(spirv::Decoration::Patch),
                crate::Interpolation::Centroid => Some(spirv::Decoration::Centroid),
                crate::Interpolation::Sample => Some(spirv::Decoration::Sample),
                crate::Interpolation::Perspective => None,
            };
            if let Some(decoration) = decoration {
                self.annotations
                    .push(self.instruction_decorate(id, decoration, &[]));
            }
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

    fn get_function_type(
        &mut self,
        lookup_function_type: LookupFunctionType,
        parameter_pointer_ids: Vec<Word>,
    ) -> Word {
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
                    parameter_pointer_ids,
                );
                instruction.to_words(&mut self.logical_layout.declarations);
                self.lookup_function_type.insert(lookup_function_type, id);
                id
            }
        }
    }

    fn write_expression<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        ir_function: &crate::Function,
        expression: &crate::Expression,
        block: &mut Block,
        function: &mut Function,
    ) -> Option<(Word, Option<crate::Handle<crate::Type>>)> {
        match expression {
            crate::Expression::GlobalVariable(handle) => {
                let var = &ir_module.global_variables[*handle];
                let id = self.get_global_variable_id(
                    &ir_module.types,
                    &ir_module.global_variables,
                    *handle,
                );
                Some((id, Some(var.ty)))
            }
            crate::Expression::Constant(handle) => {
                let var = &ir_module.constants[*handle];
                let id = self.get_constant_id(*handle, ir_module);
                Some((id, Some(var.ty)))
            }
            crate::Expression::Compose { ty, components } => {
                let id = self.generate_id();
                let type_id = self.get_type_id(&ir_module.types, LookupType::Handle(*ty));

                let mut constituent_ids = Vec::with_capacity(components.len());
                for component in components {
                    let expression = &ir_function.expressions[*component];
                    let (component_id, _) = self
                        .write_expression(ir_module, &ir_function, expression, block, function)
                        .unwrap();
                    constituent_ids.push(component_id);
                }

                let instruction =
                    self.instruction_composite_construct(type_id, id, constituent_ids);
                block.body.push(instruction);
                Some((id, Some(*ty)))
            }
            crate::Expression::Binary { op, left, right } => {
                match op {
                    crate::BinaryOperator::Multiply => {
                        // TODO OpVectorTimesScalar is only supported
                        let id = self.generate_id();
                        let left_expression = &ir_function.expressions[*left];
                        let right_expression = &ir_function.expressions[*right];
                        let (left_id, left_ty) = self
                            .write_expression(
                                ir_module,
                                ir_function,
                                left_expression,
                                block,
                                function,
                            )
                            .unwrap();
                        let (right_id, right_ty) = self
                            .write_expression(
                                ir_module,
                                ir_function,
                                right_expression,
                                block,
                                function,
                            )
                            .unwrap();

                        let left_ty_inner = &ir_module.types[left_ty.unwrap()].inner;
                        let right_ty_inner = &ir_module.types[right_ty.unwrap()].inner;

                        let (result_type_id, vector_id, scalar_id) =
                            match (left_ty_inner, right_ty_inner) {
                                (
                                    crate::TypeInner::Vector { .. },
                                    crate::TypeInner::Scalar { .. },
                                ) => (
                                    self.get_type_id(
                                        &ir_module.types,
                                        LookupType::Handle(left_ty.unwrap()),
                                    ),
                                    left_id,
                                    right_id,
                                ),
                                (
                                    crate::TypeInner::Scalar { .. },
                                    crate::TypeInner::Vector { .. },
                                ) => (
                                    self.get_type_id(
                                        &ir_module.types,
                                        LookupType::Handle(right_ty.unwrap()),
                                    ),
                                    right_id,
                                    left_id,
                                ),
                                _ => unreachable!("Expression requires both a scalar and vector"),
                            };

                        let load_id = self.generate_id();
                        let load_instruction =
                            self.instruction_load(result_type_id, load_id, vector_id, None);
                        block.body.push(load_instruction);

                        let instruction = self.instruction_vector_times_scalar(
                            result_type_id,
                            id,
                            load_id,
                            scalar_id,
                        );
                        block.body.push(instruction);
                        Some((id, None))
                    }

                    _ => unimplemented!("{:?}", op),
                }
            }
            crate::Expression::LocalVariable(variable) => {
                let var = &ir_function.local_variables[*variable];
                let id = if let Some(local_var) = function
                    .variables
                    .iter()
                    .find(|&v| v.name.as_ref().unwrap() == var.name.as_ref().unwrap())
                {
                    local_var.id
                } else {
                    panic!("Could not find: {:?}", var)
                };

                Some((id, Some(var.ty)))
            }
            crate::Expression::FunctionParameter(index) => {
                let handle = ir_function.parameter_types.get(*index as usize).unwrap();
                let type_id = self.get_type_id(&ir_module.types, LookupType::Handle(*handle));
                let load_id = self.generate_id();

                block.body.push(self.instruction_load(
                    type_id,
                    load_id,
                    function.parameters[*index as usize].result_id.unwrap(),
                    None,
                ));
                Some((load_id, Some(*handle)))
            }
            crate::Expression::Call { origin, arguments } => match origin {
                crate::FunctionOrigin::Local(local_function) => {
                    let origin_function = &ir_module.functions[*local_function];
                    let id = self.generate_id();
                    let mut argument_ids = vec![];

                    for argument in arguments {
                        let expression = &ir_function.expressions[*argument];
                        let (id, ty) = self
                            .write_expression(ir_module, ir_function, expression, block, function)
                            .unwrap();

                        // Create variable - OpVariable
                        // Store value to variable - OpStore
                        // Use id of variable

                        let pointer_id = self.get_pointer_id(
                            &ir_module.types,
                            ty.unwrap(),
                            spirv::StorageClass::Function,
                        );

                        let variable_id = self.generate_id();
                        function.variables.push(LocalVariable {
                            id: variable_id,
                            name: None,
                            instruction: self.instruction_variable(
                                pointer_id,
                                variable_id,
                                spirv::StorageClass::Function,
                                None,
                            ),
                        });
                        block.body.push(self.instruction_store(variable_id, id));
                        argument_ids.push(variable_id);
                    }

                    let return_type_id = self
                        .get_function_return_type(origin_function.return_type, &ir_module.types);

                    block.body.push(self.instruction_function_call(
                        return_type_id,
                        id,
                        *self.lookup_function.get(local_function).unwrap(),
                        argument_ids,
                    ));
                    Some((id, None))
                }
                _ => unimplemented!("{:?}", origin),
            },
            _ => unimplemented!("{:?}", expression),
        }
    }

    fn write_function_statement(
        &mut self,
        ir_module: &crate::Module,
        ir_function: &crate::Function,
        statement: &crate::Statement,
        block: &mut Block,
        function: &mut Function,
    ) {
        match statement {
            crate::Statement::Return { value } => match ir_function.return_type {
                Some(_) => {
                    let expression = &ir_function.expressions[value.unwrap()];
                    let (id, ty) = self
                        .write_expression(ir_module, ir_function, expression, block, function)
                        .unwrap();

                    let id = match expression {
                        crate::Expression::LocalVariable(_) => {
                            let load_id = self.generate_id();
                            let value_ty_id = self
                                .get_type_id(&ir_module.types, LookupType::Handle(ty.unwrap()));
                            block.body.push(self.instruction_load(
                                value_ty_id,
                                load_id,
                                id,
                                None,
                            ));
                            load_id
                        }
                        _ => id
                    };
                    block.termination = Some(self.instruction_return_value(id));
                }
                None => block.termination = Some(self.instruction_return()),
            },
            crate::Statement::Store { pointer, value } => {
                let pointer_expression = &ir_function.expressions[*pointer];
                let value_expression = &ir_function.expressions[*value];
                let (pointer_id, _) = self
                    .write_expression(ir_module, ir_function, pointer_expression, block, function)
                    .unwrap();
                let (value_id, value_ty) = self
                    .write_expression(ir_module, ir_function, value_expression, block, function)
                    .unwrap();

                let value_id = match value_expression {
                    crate::Expression::LocalVariable(_) => {
                        let load_id = self.generate_id();
                        let value_ty_id = self
                            .get_type_id(&ir_module.types, LookupType::Handle(value_ty.unwrap()));
                        block.body.push(self.instruction_load(
                            value_ty_id,
                            load_id,
                            value_id,
                            None,
                        ));
                        load_id
                    }
                    _ => value_id,
                };

                block
                    .body
                    .push(self.instruction_store(pointer_id, value_id));
            }
            crate::Statement::Empty => {}
            _ => unimplemented!("{:?}", statement),
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

        for (handle, ir_function) in ir_module.functions.iter() {
            let mut function = Function::new();

            for (_, variable) in ir_function.local_variables.iter() {
                let id = self.generate_id();

                let init_word = match variable.init {
                    Some(exp) => match &ir_function.expressions[exp] {
                        crate::Expression::Constant(handle) => {
                            Some(self.get_constant_id(*handle, ir_module))
                        }
                        _ => unreachable!(),
                    },
                    None => None,
                };

                let pointer_id = self.get_pointer_id(
                    &ir_module.types,
                    variable.ty,
                    spirv::StorageClass::Function,
                );
                function.variables.push(LocalVariable {
                    id,
                    name: variable.name.clone(),
                    instruction: self.instruction_variable(
                        pointer_id,
                        id,
                        spirv::StorageClass::Function,
                        init_word,
                    ),
                });
            }

            let return_type_id =
                self.get_function_return_type(ir_function.return_type, &ir_module.types);
            let mut parameter_type_ids = Vec::with_capacity(ir_function.parameter_types.len());

            let mut function_parameter_pointer_ids = vec![];

            for parameter_type in ir_function.parameter_types.iter() {
                let id = self.generate_id();
                let pointer_id = self.get_pointer_id(
                    &ir_module.types,
                    *parameter_type,
                    spirv::StorageClass::Function,
                );

                function_parameter_pointer_ids.push(pointer_id);
                parameter_type_ids
                    .push(self.get_type_id(&ir_module.types, LookupType::Handle(*parameter_type)));
                function
                    .parameters
                    .push(self.instruction_function_parameter(pointer_id, id));
            }

            let lookup_function_type = LookupFunctionType {
                return_type_id,
                parameter_type_ids,
            };

            let id = self.generate_id();
            let function_type =
                self.get_function_type(lookup_function_type, function_parameter_pointer_ids);
            function.signature = Some(self.instruction_function(
                return_type_id,
                id,
                spirv::FunctionControl::empty(),
                function_type,
            ));

            self.lookup_function.insert(handle, id);

            let mut block = Block::new();
            let id = self.generate_id();
            block.label = Some(self.instruction_label(id));
            for statement in ir_function.body.iter() {
                self.write_function_statement(
                    ir_module,
                    ir_function,
                    &statement,
                    &mut block,
                    &mut function,
                );
            }
            function.blocks.push(block);

            function.to_words(&mut self.logical_layout.function_definitions);
            self.instruction_function_end()
                .to_words(&mut self.logical_layout.function_definitions);
        }

        for entry_point in ir_module.entry_points.iter() {
            let entry_point_instruction = self.instruction_entry_point(entry_point, ir_module);
            entry_point_instruction.to_words(&mut self.logical_layout.entry_points);
        }

        for capability in self.capabilities.iter() {
            self.instruction_capability(*capability)
                .to_words(&mut self.logical_layout.capabilities);
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
