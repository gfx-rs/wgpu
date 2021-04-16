use super::helpers;
use spirv::{Op, Word};

pub(super) enum Signedness {
    Unsigned = 0,
    Signed = 1,
}

pub(super) enum SampleLod {
    Explicit,
    Implicit,
}

pub(super) struct Case {
    pub value: Word,
    pub label_id: Word,
}

impl super::Instruction {
    //
    //  Debug Instructions
    //

    pub(super) fn source(source_language: spirv::SourceLanguage, version: u32) -> Self {
        let mut instruction = Self::new(Op::Source);
        instruction.add_operand(source_language as u32);
        instruction.add_operands(helpers::bytes_to_words(&version.to_le_bytes()));
        instruction
    }

    pub(super) fn name(target_id: Word, name: &str) -> Self {
        let mut instruction = Self::new(Op::Name);
        instruction.add_operand(target_id);
        instruction.add_operands(helpers::string_to_words(name));
        instruction
    }

    pub(super) fn member_name(target_id: Word, member: Word, name: &str) -> Self {
        let mut instruction = Self::new(Op::MemberName);
        instruction.add_operand(target_id);
        instruction.add_operand(member);
        instruction.add_operands(helpers::string_to_words(name));
        instruction
    }

    //
    //  Annotation Instructions
    //

    pub(super) fn decorate(
        target_id: Word,
        decoration: spirv::Decoration,
        operands: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::Decorate);
        instruction.add_operand(target_id);
        instruction.add_operand(decoration as u32);
        for operand in operands {
            instruction.add_operand(*operand)
        }
        instruction
    }

    pub(super) fn member_decorate(
        target_id: Word,
        member_index: Word,
        decoration: spirv::Decoration,
        operands: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::MemberDecorate);
        instruction.add_operand(target_id);
        instruction.add_operand(member_index);
        instruction.add_operand(decoration as u32);
        for operand in operands {
            instruction.add_operand(*operand)
        }
        instruction
    }

    //
    //  Extension Instructions
    //

    pub(super) fn extension(name: &str) -> Self {
        let mut instruction = Self::new(Op::Extension);
        instruction.add_operands(helpers::string_to_words(name));
        instruction
    }

    pub(super) fn ext_inst_import(id: Word, name: &str) -> Self {
        let mut instruction = Self::new(Op::ExtInstImport);
        instruction.set_result(id);
        instruction.add_operands(helpers::string_to_words(name));
        instruction
    }

    pub(super) fn ext_inst(
        set_id: Word,
        op: spirv::GLOp,
        result_type_id: Word,
        id: Word,
        operands: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::ExtInst);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(set_id);
        instruction.add_operand(op as u32);
        for operand in operands {
            instruction.add_operand(*operand)
        }
        instruction
    }

    //
    //  Mode-Setting Instructions
    //

    pub(super) fn memory_model(
        addressing_model: spirv::AddressingModel,
        memory_model: spirv::MemoryModel,
    ) -> Self {
        let mut instruction = Self::new(Op::MemoryModel);
        instruction.add_operand(addressing_model as u32);
        instruction.add_operand(memory_model as u32);
        instruction
    }

    pub(super) fn entry_point(
        execution_model: spirv::ExecutionModel,
        entry_point_id: Word,
        name: &str,
        interface_ids: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::EntryPoint);
        instruction.add_operand(execution_model as u32);
        instruction.add_operand(entry_point_id);
        instruction.add_operands(helpers::string_to_words(name));

        for interface_id in interface_ids {
            instruction.add_operand(*interface_id);
        }

        instruction
    }

    pub(super) fn execution_mode(
        entry_point_id: Word,
        execution_mode: spirv::ExecutionMode,
        args: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::ExecutionMode);
        instruction.add_operand(entry_point_id);
        instruction.add_operand(execution_mode as u32);
        for arg in args {
            instruction.add_operand(*arg);
        }
        instruction
    }

    pub(super) fn capability(capability: spirv::Capability) -> Self {
        let mut instruction = Self::new(Op::Capability);
        instruction.add_operand(capability as u32);
        instruction
    }

    //
    //  Type-Declaration Instructions
    //

    pub(super) fn type_void(id: Word) -> Self {
        let mut instruction = Self::new(Op::TypeVoid);
        instruction.set_result(id);
        instruction
    }

    pub(super) fn type_bool(id: Word) -> Self {
        let mut instruction = Self::new(Op::TypeBool);
        instruction.set_result(id);
        instruction
    }

    pub(super) fn type_int(id: Word, width: Word, signedness: Signedness) -> Self {
        let mut instruction = Self::new(Op::TypeInt);
        instruction.set_result(id);
        instruction.add_operand(width);
        instruction.add_operand(signedness as u32);
        instruction
    }

    pub(super) fn type_float(id: Word, width: Word) -> Self {
        let mut instruction = Self::new(Op::TypeFloat);
        instruction.set_result(id);
        instruction.add_operand(width);
        instruction
    }

    pub(super) fn type_vector(
        id: Word,
        component_type_id: Word,
        component_count: crate::VectorSize,
    ) -> Self {
        let mut instruction = Self::new(Op::TypeVector);
        instruction.set_result(id);
        instruction.add_operand(component_type_id);
        instruction.add_operand(component_count as u32);
        instruction
    }

    pub(super) fn type_matrix(
        id: Word,
        column_type_id: Word,
        column_count: crate::VectorSize,
    ) -> Self {
        let mut instruction = Self::new(Op::TypeMatrix);
        instruction.set_result(id);
        instruction.add_operand(column_type_id);
        instruction.add_operand(column_count as u32);
        instruction
    }

    pub(super) fn type_image(
        id: Word,
        sampled_type_id: Word,
        dim: spirv::Dim,
        arrayed: bool,
        image_class: crate::ImageClass,
    ) -> Self {
        let mut instruction = Self::new(Op::TypeImage);
        instruction.set_result(id);
        instruction.add_operand(sampled_type_id);
        instruction.add_operand(dim as u32);

        let (depth, multi, sampled) = match image_class {
            crate::ImageClass::Sampled { kind: _, multi } => (false, multi, true),
            crate::ImageClass::Depth => (true, false, true),
            crate::ImageClass::Storage(_) => (false, false, false),
        };
        instruction.add_operand(depth as u32);
        instruction.add_operand(arrayed as u32);
        instruction.add_operand(multi as u32);
        instruction.add_operand(if sampled { 1 } else { 2 });

        let format = match image_class {
            crate::ImageClass::Storage(format) => match format {
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
            },
            _ => spirv::ImageFormat::Unknown,
        };

        instruction.add_operand(format as u32);
        instruction
    }

    pub(super) fn type_sampler(id: Word) -> Self {
        let mut instruction = Self::new(Op::TypeSampler);
        instruction.set_result(id);
        instruction
    }

    pub(super) fn type_sampled_image(id: Word, image_type_id: Word) -> Self {
        let mut instruction = Self::new(Op::TypeSampledImage);
        instruction.set_result(id);
        instruction.add_operand(image_type_id);
        instruction
    }

    pub(super) fn type_array(id: Word, element_type_id: Word, length_id: Word) -> Self {
        let mut instruction = Self::new(Op::TypeArray);
        instruction.set_result(id);
        instruction.add_operand(element_type_id);
        instruction.add_operand(length_id);
        instruction
    }

    pub(super) fn type_runtime_array(id: Word, element_type_id: Word) -> Self {
        let mut instruction = Self::new(Op::TypeRuntimeArray);
        instruction.set_result(id);
        instruction.add_operand(element_type_id);
        instruction
    }

    pub(super) fn type_struct(id: Word, member_ids: &[Word]) -> Self {
        let mut instruction = Self::new(Op::TypeStruct);
        instruction.set_result(id);

        for member_id in member_ids {
            instruction.add_operand(*member_id)
        }

        instruction
    }

    pub(super) fn type_pointer(
        id: Word,
        storage_class: spirv::StorageClass,
        type_id: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::TypePointer);
        instruction.set_result(id);
        instruction.add_operand(storage_class as u32);
        instruction.add_operand(type_id);
        instruction
    }

    pub(super) fn type_function(id: Word, return_type_id: Word, parameter_ids: &[Word]) -> Self {
        let mut instruction = Self::new(Op::TypeFunction);
        instruction.set_result(id);
        instruction.add_operand(return_type_id);

        for parameter_id in parameter_ids {
            instruction.add_operand(*parameter_id);
        }

        instruction
    }

    //
    //  Constant-Creation Instructions
    //

    pub(super) fn constant_true(result_type_id: Word, id: Word) -> Self {
        let mut instruction = Self::new(Op::ConstantTrue);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction
    }

    pub(super) fn constant_false(result_type_id: Word, id: Word) -> Self {
        let mut instruction = Self::new(Op::ConstantFalse);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction
    }

    pub(super) fn constant(result_type_id: Word, id: Word, values: &[Word]) -> Self {
        let mut instruction = Self::new(Op::Constant);
        instruction.set_type(result_type_id);
        instruction.set_result(id);

        for value in values {
            instruction.add_operand(*value);
        }

        instruction
    }

    pub(super) fn constant_composite(
        result_type_id: Word,
        id: Word,
        constituent_ids: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::ConstantComposite);
        instruction.set_type(result_type_id);
        instruction.set_result(id);

        for constituent_id in constituent_ids {
            instruction.add_operand(*constituent_id);
        }

        instruction
    }

    //
    //  Memory Instructions
    //

    pub(super) fn variable(
        result_type_id: Word,
        id: Word,
        storage_class: spirv::StorageClass,
        initializer_id: Option<Word>,
    ) -> Self {
        let mut instruction = Self::new(Op::Variable);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(storage_class as u32);

        if let Some(initializer_id) = initializer_id {
            instruction.add_operand(initializer_id);
        }

        instruction
    }

    pub(super) fn load(
        result_type_id: Word,
        id: Word,
        pointer_id: Word,
        memory_access: Option<spirv::MemoryAccess>,
    ) -> Self {
        let mut instruction = Self::new(Op::Load);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(pointer_id);

        if let Some(memory_access) = memory_access {
            instruction.add_operand(memory_access.bits());
        }

        instruction
    }

    pub(super) fn store(
        pointer_id: Word,
        object_id: Word,
        memory_access: Option<spirv::MemoryAccess>,
    ) -> Self {
        let mut instruction = Self::new(Op::Store);
        instruction.add_operand(pointer_id);
        instruction.add_operand(object_id);

        if let Some(memory_access) = memory_access {
            instruction.add_operand(memory_access.bits());
        }

        instruction
    }

    pub(super) fn access_chain(
        result_type_id: Word,
        id: Word,
        base_id: Word,
        index_ids: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::AccessChain);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(base_id);

        for index_id in index_ids {
            instruction.add_operand(*index_id);
        }

        instruction
    }

    //
    //  Function Instructions
    //

    pub(super) fn function(
        return_type_id: Word,
        id: Word,
        function_control: spirv::FunctionControl,
        function_type_id: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::Function);
        instruction.set_type(return_type_id);
        instruction.set_result(id);
        instruction.add_operand(function_control.bits());
        instruction.add_operand(function_type_id);
        instruction
    }

    pub(super) fn function_parameter(result_type_id: Word, id: Word) -> Self {
        let mut instruction = Self::new(Op::FunctionParameter);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction
    }

    pub(super) fn function_end() -> Self {
        Self::new(Op::FunctionEnd)
    }

    pub(super) fn function_call(
        result_type_id: Word,
        id: Word,
        function_id: Word,
        argument_ids: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::FunctionCall);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(function_id);

        for argument_id in argument_ids {
            instruction.add_operand(*argument_id);
        }

        instruction
    }

    //
    //  Image Instructions
    //

    pub(super) fn sampled_image(
        result_type_id: Word,
        id: Word,
        image: Word,
        sampler: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::SampledImage);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(image);
        instruction.add_operand(sampler);
        instruction
    }

    pub(super) fn image_sample(
        result_type_id: Word,
        id: Word,
        lod: SampleLod,
        sampled_image: Word,
        coordinates: Word,
        depth_ref: Option<Word>,
    ) -> Self {
        let op = match (lod, depth_ref) {
            (SampleLod::Explicit, None) => Op::ImageSampleExplicitLod,
            (SampleLod::Implicit, None) => Op::ImageSampleImplicitLod,
            (SampleLod::Explicit, Some(_)) => Op::ImageSampleDrefExplicitLod,
            (SampleLod::Implicit, Some(_)) => Op::ImageSampleDrefImplicitLod,
        };

        let mut instruction = Self::new(op);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(sampled_image);
        instruction.add_operand(coordinates);
        if let Some(dref) = depth_ref {
            instruction.add_operand(dref);
        }

        instruction
    }

    pub(super) fn image_fetch(
        result_type_id: Word,
        id: Word,
        image: Word,
        coordinates: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::ImageFetch);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(image);
        instruction.add_operand(coordinates);
        instruction
    }

    pub(super) fn image_read(
        result_type_id: Word,
        id: Word,
        image: Word,
        coordinates: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::ImageRead);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(image);
        instruction.add_operand(coordinates);
        instruction
    }

    pub(super) fn image_write(image: Word, coordinates: Word, value: Word) -> Self {
        let mut instruction = Self::new(Op::ImageWrite);
        instruction.add_operand(image);
        instruction.add_operand(coordinates);
        instruction.add_operand(value);
        instruction
    }

    //
    //  Conversion Instructions
    //
    pub(super) fn unary(op: Op, result_type_id: Word, id: Word, value: Word) -> Self {
        let mut instruction = Self::new(op);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(value);
        instruction
    }

    //
    //  Composite Instructions
    //

    pub(super) fn composite_construct(
        result_type_id: Word,
        id: Word,
        constituent_ids: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::CompositeConstruct);
        instruction.set_type(result_type_id);
        instruction.set_result(id);

        for constituent_id in constituent_ids {
            instruction.add_operand(*constituent_id);
        }

        instruction
    }

    pub(super) fn composite_extract(
        result_type_id: Word,
        id: Word,
        composite_id: Word,
        indices: &[Word],
    ) -> Self {
        let mut instruction = Self::new(Op::CompositeExtract);
        instruction.set_type(result_type_id);
        instruction.set_result(id);

        instruction.add_operand(composite_id);
        for index in indices {
            instruction.add_operand(*index);
        }

        instruction
    }

    pub(super) fn vector_extract_dynamic(
        result_type_id: Word,
        id: Word,
        vector_id: Word,
        index_id: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::VectorExtractDynamic);
        instruction.set_type(result_type_id);
        instruction.set_result(id);

        instruction.add_operand(vector_id);
        instruction.add_operand(index_id);

        instruction
    }

    //
    // Arithmetic Instructions
    //
    pub(super) fn binary(
        op: Op,
        result_type_id: Word,
        id: Word,
        operand_1: Word,
        operand_2: Word,
    ) -> Self {
        let mut instruction = Self::new(op);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(operand_1);
        instruction.add_operand(operand_2);
        instruction
    }

    //
    // Bit Instructions
    //

    //
    // Relational and Logical Instructions
    //

    //
    // Derivative Instructions
    //

    pub(super) fn derive_x(result_type_id: Word, id: Word, expr_id: Word) -> Self {
        let mut instruction = Self::new(Op::DPdx);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(expr_id);
        instruction
    }

    pub(super) fn derive_y(result_type_id: Word, id: Word, expr_id: Word) -> Self {
        let mut instruction = Self::new(Op::DPdy);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(expr_id);
        instruction
    }

    pub(super) fn derive_width(result_type_id: Word, id: Word, expr_id: Word) -> Self {
        let mut instruction = Self::new(Op::Fwidth);
        instruction.set_type(result_type_id);
        instruction.set_result(id);
        instruction.add_operand(expr_id);
        instruction
    }

    //
    // Control-Flow Instructions
    //

    pub(super) fn selection_merge(
        merge_id: Word,
        selection_control: spirv::SelectionControl,
    ) -> Self {
        let mut instruction = Self::new(Op::SelectionMerge);
        instruction.add_operand(merge_id);
        instruction.add_operand(selection_control.bits());
        instruction
    }

    pub(super) fn loop_merge(
        merge_id: Word,
        continuing_id: Word,
        selection_control: spirv::SelectionControl,
    ) -> Self {
        let mut instruction = Self::new(Op::LoopMerge);
        instruction.add_operand(merge_id);
        instruction.add_operand(continuing_id);
        instruction.add_operand(selection_control.bits());
        instruction
    }

    pub(super) fn label(id: Word) -> Self {
        let mut instruction = Self::new(Op::Label);
        instruction.set_result(id);
        instruction
    }

    pub(super) fn branch(id: Word) -> Self {
        let mut instruction = Self::new(Op::Branch);
        instruction.add_operand(id);
        instruction
    }

    // TODO Branch Weights not implemented.
    pub(super) fn branch_conditional(
        condition_id: Word,
        true_label: Word,
        false_label: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::BranchConditional);
        instruction.add_operand(condition_id);
        instruction.add_operand(true_label);
        instruction.add_operand(false_label);
        instruction
    }

    pub(super) fn switch(selector_id: Word, default_id: Word, cases: &[Case]) -> Self {
        let mut instruction = Self::new(Op::Switch);
        instruction.add_operand(selector_id);
        instruction.add_operand(default_id);
        for case in cases {
            instruction.add_operand(case.value);
            instruction.add_operand(case.label_id);
        }
        instruction
    }

    pub(super) fn select(
        result_type_id: Word,
        id: Word,
        condition_id: Word,
        accept_id: Word,
        reject_id: Word,
    ) -> Self {
        let mut instruction = Self::new(Op::Select);
        instruction.add_operand(result_type_id);
        instruction.add_operand(id);
        instruction.add_operand(condition_id);
        instruction.add_operand(accept_id);
        instruction.add_operand(reject_id);
        instruction
    }

    pub(super) fn kill() -> Self {
        Self::new(Op::Kill)
    }

    pub(super) fn return_void() -> Self {
        Self::new(Op::Return)
    }

    pub(super) fn return_value(value_id: Word) -> Self {
        let mut instruction = Self::new(Op::ReturnValue);
        instruction.add_operand(value_id);
        instruction
    }

    //
    //  Atomic Instructions
    //

    //
    //  Primitive Instructions
    //
}
