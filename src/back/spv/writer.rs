/*! Standard Portable Intermediate Representation (SPIR-V) backend !*/
use crate::back::spv::{helpers, Instruction, LogicalLayout, PhysicalLayout, WriterFlags};
use crate::{FastHashMap, FastHashSet};
use spirv::*;

trait LookupHelper<T> {
    type Target;
    fn lookup_id(&self, handle: crate::Handle<T>) -> Option<Word>;
    fn lookup_handle(&self, word: Word) -> Option<crate::Handle<T>>;
}

impl<T> LookupHelper<T> for FastHashMap<Word, crate::Handle<T>> {
    type Target = T;

    fn lookup_id(&self, handle: crate::Handle<T>) -> Option<Word> {
        let mut word = None;
        for (k, v) in self.iter() {
            if *v == handle {
                word = Some(*k);
                break;
            }
        }
        word
    }

    fn lookup_handle(&self, word: u32) -> Option<crate::Handle<T>> {
        let mut handle = None;
        for (k, v) in self.iter() {
            if *k == word {
                handle = Some(*v);
                break;
            }
        }
        handle
    }
}

#[derive(Debug, PartialEq)]
struct LookupFunctionType {
    parameter_type_ids: Vec<Word>,
    return_type_id: Word,
}

pub struct Writer {
    physical_layout: PhysicalLayout,
    logical_layout: LogicalLayout,
    id_count: u32,
    capabilities: FastHashSet<Capability>,
    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    writer_flags: WriterFlags,
    void_type: Option<u32>,
    lookup_type: FastHashMap<Word, crate::Handle<crate::Type>>,
    lookup_function: FastHashMap<Word, crate::Handle<crate::Function>>,
    lookup_function_type: FastHashMap<Word, LookupFunctionType>,
    lookup_constant: FastHashMap<Word, crate::Handle<crate::Constant>>,
    lookup_global_variable: FastHashMap<Word, crate::Handle<crate::GlobalVariable>>,
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

    fn try_add_capabilities(&mut self, capabilities: &[Capability]) {
        for capability in capabilities.iter() {
            self.capabilities.insert(*capability);
        }
    }

    fn instruction_capability(&self, capability: Capability) -> Instruction {
        let mut instruction = Instruction::new(Op::Capability);
        instruction.add_operand(capability as u32);
        instruction
    }

    fn instruction_ext_inst_import(&mut self, name: &str) -> Instruction {
        let mut instruction = Instruction::new(Op::ExtInstImport);
        let id = self.generate_id();
        instruction.set_result(id);
        instruction.add_operands(helpers::string_to_words(name));
        instruction
    }

    fn instruction_memory_model(&mut self) -> Instruction {
        let mut instruction = Instruction::new(Op::MemoryModel);
        let addressing_model = AddressingModel::Logical;
        let memory_model = MemoryModel::GLSL450;
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
        let function_id = self
            .lookup_function
            .lookup_id(entry_point.function)
            .unwrap();

        instruction.add_operand(entry_point.exec_model as u32);
        instruction.add_operand(function_id);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            let mut debug_instruction = Instruction::new(Op::Name);
            debug_instruction.set_result(function_id);
            debug_instruction.add_operands(helpers::string_to_words(entry_point.name.as_str()));
            self.debugs.push(debug_instruction);
        }

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

        self.try_add_capabilities(entry_point.exec_model.required_capabilities());
        match entry_point.exec_model {
            ExecutionModel::Vertex => {}
            ExecutionModel::Fragment => {
                let execution_mode = ExecutionMode::OriginUpperLeft;
                self.try_add_capabilities(execution_mode.required_capabilities());

                let mut execution_mode_instruction = Instruction::new(Op::ExecutionMode);
                execution_mode_instruction.add_operand(function_id);
                execution_mode_instruction.add_operand(execution_mode as u32);
                execution_mode_instruction.to_words(&mut self.logical_layout.execution_modes);
            }
            _ => unimplemented!(),
        }

        instruction
    }

    fn get_type_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
    ) -> Word {
        match self.lookup_type.lookup_id(handle) {
            Some(word) => word,
            None => {
                let (instruction, id) = self.instruction_type_declaration(arena, handle);
                instruction.to_words(&mut self.logical_layout.declarations);
                id
            }
        }
    }

    fn get_constant_id(
        &mut self,
        handle: crate::Handle<crate::Constant>,
        ir_module: &crate::Module,
    ) -> Word {
        match self.lookup_constant.lookup_id(handle) {
            Some(word) => word,
            None => {
                let (instruction, id) = self.instruction_constant_type(handle, ir_module);
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
        match self.lookup_global_variable.lookup_id(handle) {
            Some(word) => word,
            None => {
                let global_variable = &global_arena[handle];
                let (instruction, id) =
                    self.instruction_global_variable(arena, global_variable, handle);
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
            Some(handle) => self.get_type_id(arena, handle),
            None => match self.void_type {
                Some(id) => id,
                None => {
                    let id = self.generate_id();

                    let mut instruction = Instruction::new(Op::TypeVoid);
                    instruction.set_result(id);

                    self.void_type = Some(id);
                    instruction.to_words(&mut self.logical_layout.declarations);
                    id
                }
            },
        }
    }

    fn find_scalar_handle(
        &self,
        arena: &crate::Arena<crate::Type>,
        kind: crate::ScalarKind,
        width: u8,
    ) -> crate::Handle<crate::Type> {
        let mut scalar_handle = None;
        for (handle, ty) in arena.iter() {
            match ty.inner {
                crate::TypeInner::Scalar {
                    kind: _kind,
                    width: _width,
                } => {
                    if kind == _kind && width == _width {
                        scalar_handle = Some(handle);
                        break;
                    }
                }
                _ => continue,
            }
        }
        scalar_handle.unwrap()
    }

    fn instruction_type_declaration(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
    ) -> (Instruction, Word) {
        let ty = &arena[handle];
        let id = self.generate_id();
        let mut instruction;

        match ty.inner {
            crate::TypeInner::Scalar { kind, width } => {
                match kind {
                    crate::ScalarKind::Sint => {
                        instruction = Instruction::new(Op::TypeInt);
                        instruction.set_result(id);
                        instruction.add_operand(width as u32);
                        instruction.add_operand(0x1u32);
                    }
                    crate::ScalarKind::Uint => {
                        instruction = Instruction::new(Op::TypeInt);
                        instruction.set_result(id);
                        instruction.add_operand(width as u32);
                        instruction.add_operand(0x0u32);
                    }
                    crate::ScalarKind::Float => {
                        instruction = Instruction::new(Op::TypeFloat);
                        instruction.set_result(id);
                        instruction.add_operand(width as u32);
                    }
                    crate::ScalarKind::Bool => {
                        instruction = Instruction::new(Op::TypeBool);
                        instruction.set_result(id);
                    }
                }
                self.lookup_type.insert(id, handle);
            }
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_handle = self.find_scalar_handle(arena, kind, width);
                let scalar_id = self.get_type_id(arena, scalar_handle);

                instruction = Instruction::new(Op::TypeVector);
                instruction.set_result(id);
                instruction.add_operand(scalar_id);
                instruction.add_operand(size as u32);

                self.lookup_type.insert(id, handle);
            }
            crate::TypeInner::Matrix {
                columns,
                rows: _,
                kind,
                width,
            } => {
                let scalar_handle = self.find_scalar_handle(arena, kind, width);
                let scalar_id = self.get_type_id(arena, scalar_handle);

                instruction = Instruction::new(Op::TypeMatrix);
                instruction.set_result(id);
                instruction.add_operand(scalar_id);
                instruction.add_operand(columns as u32);
            }
            crate::TypeInner::Pointer { base, class } => {
                let type_id = self.get_type_id(arena, base);
                instruction = Instruction::new(Op::TypePointer);
                instruction.set_result(id);
                instruction.add_operand(class as u32);
                instruction.add_operand(type_id);

                self.lookup_type.insert(id, handle);
            }
            crate::TypeInner::Array { base, size, stride } => {
                if let Some(array_stride) = stride {
                    let mut instruction = Instruction::new(Op::Decorate);
                    instruction.add_operand(id);
                    instruction.add_operand(Decoration::ArrayStride as u32);
                    instruction.add_operand(array_stride.get());
                    self.annotations.push(instruction);
                }

                let type_id = self.get_type_id(arena, handle);

                instruction = Instruction::new(Op::TypeArray);
                instruction.set_result(id);
                instruction.add_operand(type_id);

                match size {
                    crate::ArraySize::Static(word) => {
                        instruction.add_operand(word);
                    }
                    _ => panic!("Array size {:?} unsupported", size),
                }

                self.lookup_type.insert(id, base);
            }
            crate::TypeInner::Struct { ref members } => {
                instruction = Instruction::new(Op::TypeStruct);
                instruction.set_result(id);

                for member in members {
                    let type_id = self.get_type_id(arena, member.ty);
                    instruction.add_operand(type_id);
                }

                self.lookup_type.insert(id, handle);
            }
            crate::TypeInner::Image { base, dim, flags } => {
                let type_id = self.get_type_id(arena, base);
                self.try_add_capabilities(dim.required_capabilities());

                instruction = Instruction::new(Op::TypeImage);
                instruction.set_result(id);
                instruction.add_operand(type_id);
                instruction.add_operand(dim as u32);

                // TODO Add Depth, but how to determine? Not yet in the WGSL spec
                instruction.add_operand(1);

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

                if let Dim::DimSubpassData = dim {
                    instruction.add_operand(2);
                    instruction.add_operand(ImageFormat::Unknown as u32);
                } else {
                    instruction.add_operand(if flags.contains(crate::ImageFlags::SAMPLED) {
                        1
                    } else {
                        0
                    });

                    // TODO Defaults to Unknown, not yet in IR
                    instruction.add_operand(ImageFormat::Unknown as u32);
                };

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

                self.lookup_type.insert(id, base);
            }
            crate::TypeInner::Sampler { comparison: _ } => {
                instruction = Instruction::new(Op::TypeSampler);
                instruction.set_result(id);
                self.lookup_type.insert(id, handle);
            }
        }

        (instruction, id)
    }

    fn instruction_constant_type(
        &mut self,
        handle: crate::Handle<crate::Constant>,
        ir_module: &crate::Module,
    ) -> (Instruction, Word) {
        let id = self.generate_id();
        self.lookup_constant.insert(id, handle);
        let constant = &ir_module.constants[handle];
        let arena = &ir_module.types;

        match constant.inner {
            crate::ConstantInner::Sint(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::Constant);
                instruction.set_type(type_id);
                instruction.set_result(id);

                let ty = &ir_module.types[constant.ty];
                match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        32 => {
                            instruction.add_operand(val as u32);
                        }
                        64 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            instruction.add_operand(low);
                            instruction.add_operand(high);
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }

                (instruction, id)
            }
            crate::ConstantInner::Uint(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::Constant);
                instruction.set_type(type_id);
                instruction.set_result(id);

                let ty = &ir_module.types[constant.ty];
                match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        32 => {
                            instruction.add_operand(val as u32);
                        }
                        64 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            instruction.add_operand(low);
                            instruction.add_operand(high);
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }

                (instruction, id)
            }
            crate::ConstantInner::Float(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::Constant);
                instruction.set_type(type_id);
                instruction.set_result(id);

                let ty = &ir_module.types[constant.ty];
                match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        32 => {
                            instruction.add_operand((val as f32).to_bits());
                        }
                        64 => {
                            let bits = f64::to_bits(val);
                            let (low, high) = ((bits >> 32) as u32, bits as u32);
                            instruction.add_operand(low);
                            instruction.add_operand(high);
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                }

                (instruction, id)
            }
            crate::ConstantInner::Bool(val) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(if val {
                    Op::ConstantTrue
                } else {
                    Op::ConstantFalse
                });

                instruction.set_type(type_id);
                instruction.set_result(id);
                (instruction, id)
            }
            crate::ConstantInner::Composite(ref constituents) => {
                let type_id = self.get_type_id(arena, constant.ty);

                let mut instruction = Instruction::new(Op::ConstantComposite);
                instruction.set_type(type_id);
                instruction.set_result(id);

                for constituent in constituents.iter() {
                    let id = self.get_constant_id(*constituent, &ir_module);
                    instruction.add_operand(id);
                }

                (instruction, id)
            }
        }
    }

    fn get_pointer_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
        class: StorageClass,
    ) -> Word {
        let ty = &arena[handle];
        let type_id = self.get_type_id(arena, handle);
        match ty.inner {
            crate::TypeInner::Pointer { .. } => type_id,
            _ => {
                let pointer_id = self.generate_id();
                let mut instruction = Instruction::new(Op::TypePointer);
                instruction.set_result(pointer_id);
                instruction.add_operand((class) as u32);
                instruction.add_operand(type_id);
                instruction.to_words(&mut self.logical_layout.declarations);

                /* TODO
                    Not able to lookup Pointer, because there is no Handle in the IR for it.
                    Idea would be to not have any handles at all in the lookups, so we aren't bound
                    to the IR. We can then insert, like here runtime values to the lookups
                */
                // self.lookup_type.insert(pointer_id, global_variable.ty);
                pointer_id
            }
        }
    }

    fn instruction_global_variable(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        global_variable: &crate::GlobalVariable,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> (Instruction, Word) {
        let mut instruction = Instruction::new(Op::Variable);
        let id = self.generate_id();

        self.try_add_capabilities(global_variable.class.required_capabilities());

        let pointer_id = self.get_pointer_id(arena, global_variable.ty, global_variable.class);

        instruction.set_type(pointer_id);
        instruction.set_result(id);
        instruction.add_operand(global_variable.class as u32);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            let mut debug_instruction = Instruction::new(Op::Name);
            debug_instruction.set_result(id);
            debug_instruction.add_operands(helpers::string_to_words(
                global_variable.name.as_ref().unwrap().as_str(),
            ));
            self.debugs.push(debug_instruction);
        }

        match global_variable.binding.as_ref().unwrap() {
            crate::Binding::Location(location) => {
                let mut instruction = Instruction::new(Op::Decorate);
                instruction.add_operand(id);
                instruction.add_operand(Decoration::Location as u32);
                instruction.add_operand(*location);
                self.annotations.push(instruction);
            }
            crate::Binding::Descriptor { set, binding } => {
                let mut set_instruction = Instruction::new(Op::Decorate);
                set_instruction.add_operand(id);
                set_instruction.add_operand(Decoration::DescriptorSet as u32);
                set_instruction.add_operand(*set);
                self.annotations.push(set_instruction);

                let mut binding_instruction = Instruction::new(Op::Decorate);
                binding_instruction.add_operand(id);
                binding_instruction.add_operand(Decoration::Binding as u32);
                binding_instruction.add_operand(*binding);
                self.annotations.push(binding_instruction);
            }
            crate::Binding::BuiltIn(built_in) => {
                let built_in_u32: u32 = unsafe { std::mem::transmute(*built_in) };

                let mut instruction = Instruction::new(Op::Decorate);
                instruction.add_operand(id);
                instruction.add_operand(Decoration::BuiltIn as u32);
                instruction.add_operand(built_in_u32);
                self.annotations.push(instruction);
            }
        }

        // TODO Initializer is optional and not (yet) included in the IR

        self.lookup_global_variable.insert(id, handle);
        (instruction, id)
    }

    fn write_physical_layout(&mut self) {
        self.physical_layout.bound = self.id_count + 1;
    }

    fn instruction_source(&self) -> Instruction {
        let version = 450u32;

        let mut instruction = Instruction::new(Op::Source);
        instruction.add_operand(SourceLanguage::GLSL as u32);
        instruction.add_operands(helpers::bytes_to_words(&version.to_le_bytes()));
        instruction
    }

    fn instruction_function_type(&mut self, lookup_function_type: LookupFunctionType) -> Word {
        let mut id = None;

        for (k, v) in self.lookup_function_type.iter() {
            if v.eq(&lookup_function_type) {
                id = Some(*k);
                break;
            }
        }

        if id.is_none() {
            let _id = self.generate_id();
            id = Some(_id);

            let mut instruction = Instruction::new(Op::TypeFunction);
            instruction.set_result(_id);
            instruction.add_operand(lookup_function_type.return_type_id);

            for parameter_type_id in lookup_function_type.parameter_type_ids.iter() {
                instruction.add_operand(*parameter_type_id);
            }

            self.lookup_function_type.insert(_id, lookup_function_type);
            instruction.to_words(&mut self.logical_layout.declarations);
        }

        id.unwrap()
    }

    fn instruction_function(
        &mut self,
        handle: crate::Handle<crate::Function>,
        function: &crate::Function,
        arena: &crate::Arena<crate::Type>,
    ) -> Instruction {
        let id = self.generate_id();

        let return_type_id = self.get_function_type(function.return_type, arena);

        let mut instruction = Instruction::new(Op::Function);
        instruction.set_type(return_type_id);
        instruction.set_result(id);

        let control_u32: Word = unsafe { std::mem::transmute(function.control) };

        instruction.add_operand(control_u32);

        let mut parameter_type_ids = Vec::with_capacity(function.parameter_types.len());
        for parameter_type in function.parameter_types.iter() {
            parameter_type_ids.push(self.get_type_id(arena, *parameter_type))
        }

        let lookup_function_type = LookupFunctionType {
            return_type_id,
            parameter_type_ids,
        };

        let type_function_id = self.instruction_function_type(lookup_function_type);

        instruction.add_operand(type_function_id);

        self.lookup_function.insert(id, handle);

        instruction
    }

    fn parse_expression<'a>(
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
                let type_id = self.get_type_id(&ir_module.types, *ty);

                let mut instruction = Instruction::new(Op::CompositeConstruct);
                instruction.set_type(type_id);
                instruction.set_result(id);

                for component in components {
                    let expression = &function.expressions[*component];
                    let (component_id, _) =
                        self.parse_expression(ir_module, &function, expression, output);
                    instruction.add_operand(component_id);
                }

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
                            self.parse_expression(ir_module, function, left_expression, output);
                        let (right_id, right_inner) =
                            self.parse_expression(ir_module, function, right_expression, output);

                        let mut result_type_id = None;
                        let mut vector_id = None;
                        let mut scalar_id = None;

                        match left_inner {
                            crate::TypeInner::Vector { size, kind, width } => {
                                vector_id = Some(left_id);
                                for (k, v) in self.lookup_type.iter() {
                                    let ty = &ir_module.types[*v];
                                    match ty.inner {
                                        crate::TypeInner::Vector {
                                            size: _size,
                                            kind: _kind,
                                            width: _width,
                                        } => {
                                            if size == &_size && kind == &_kind && width == &_width
                                            {
                                                result_type_id = Some(*k);
                                                break;
                                            }
                                        }
                                        _ => continue,
                                    }
                                }
                            }
                            _ => scalar_id = Some(left_id),
                        }

                        match right_inner {
                            crate::TypeInner::Vector { size, kind, width } => {
                                vector_id = Some(right_id);
                                for (k, v) in self.lookup_type.iter() {
                                    let ty = &ir_module.types[*v];
                                    match ty.inner {
                                        crate::TypeInner::Vector {
                                            size: _size,
                                            kind: _kind,
                                            width: _width,
                                        } => {
                                            if size == &_size && kind == &_kind && width == &_width
                                            {
                                                result_type_id = Some(*k);
                                                break;
                                            }
                                        }
                                        _ => continue,
                                    }
                                }
                            }
                            _ => scalar_id = Some(right_id),
                        }

                        // TODO Quick fix
                        let load_id = self.generate_id();
                        let mut instruction = Instruction::new(Op::Load);
                        instruction.set_type(result_type_id.unwrap());
                        instruction.set_result(load_id);
                        instruction.add_operand(vector_id.unwrap());

                        output.push(instruction);

                        let mut instruction = Instruction::new(Op::VectorTimesScalar);
                        instruction.set_type(result_type_id.unwrap());
                        instruction.set_result(id);
                        instruction.add_operand(load_id);
                        instruction.add_operand(scalar_id.unwrap());
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
                    self.get_pointer_id(&ir_module.types, var.ty, StorageClass::Function);

                let mut instruction = Instruction::new(Op::Variable);
                instruction.set_type(pointer_id);
                instruction.set_result(id);
                instruction.add_operand(StorageClass::Function as u32);
                (id, &ty.inner)
            }
            _ => unimplemented!("{:?}", expression),
        }
    }

    fn instruction_function_block(
        &mut self,
        ir_module: &crate::Module,
        function: &crate::Function,
        statement: &crate::Statement,
        output: &mut Vec<Instruction>,
    ) -> Instruction {
        match statement {
            crate::Statement::Return { value: _ } => match function.return_type {
                Some(_) => unimplemented!(),
                None => Instruction::new(Op::Return),
            },
            crate::Statement::Store { pointer, value } => {
                let mut instruction = Instruction::new(Op::Store);

                let pointer_expression = &function.expressions[*pointer];
                let value_expression = &function.expressions[*value];
                let (pointer_id, _) =
                    self.parse_expression(ir_module, function, pointer_expression, output);
                let (value_id, _) =
                    self.parse_expression(ir_module, function, value_expression, output);

                instruction.add_operand(pointer_id);
                instruction.add_operand(value_id);

                instruction
            }
            _ => unimplemented!(),
        }
    }

    fn instruction_label(&mut self) -> Instruction {
        let mut instruction = Instruction::new(Op::Label);
        instruction.set_result(self.generate_id());
        instruction
    }

    fn instruction_function_end(&self) -> Instruction {
        Instruction::new(Op::FunctionEnd)
    }

    fn write_logical_layout(&mut self, ir_module: &crate::Module) {
        self.instruction_ext_inst_import("GLSL.std.450")
            .to_words(&mut self.logical_layout.ext_inst_imports);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            self.debugs.push(self.instruction_source());
        }

        for (handle, function) in ir_module.functions.iter() {
            let mut function_instructions: Vec<Instruction> = vec![];
            function_instructions.push(self.instruction_function(
                handle,
                function,
                &ir_module.types,
            ));

            function_instructions.push(self.instruction_label());

            for block in function.body.iter() {
                let mut output: Vec<Instruction> = vec![];
                let instruction =
                    self.instruction_function_block(ir_module, function, &block, &mut output);
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

        // Looking through all global variable, types, constants.
        // Doing this because we also want to include not used parts of the module
        // to be included in the output
        for (handle, _) in ir_module.global_variables.iter() {
            self.get_global_variable_id(&ir_module.types, &ir_module.global_variables, handle);
        }

        for (handle, _) in ir_module.types.iter() {
            self.get_type_id(&ir_module.types, handle);
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
