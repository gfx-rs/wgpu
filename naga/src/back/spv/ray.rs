/*!
Generating SPIR-V for ray query operations.
*/

use alloc::{vec, vec::Vec};

use super::{
    Block, BlockContext, Function, FunctionArgument, Instruction, LookupFunctionType, NumericType,
    Writer,
};
use crate::{arena::Handle, back::spv::LookupRayQueryFunction};

/// helper function to check if a particular flag is set in a u32.
fn write_ray_flags_contains_flags(
    writer: &mut Writer,
    block: &mut Block,
    id: spirv::Word,
    flag: u32,
) -> spirv::Word {
    let bit_id = writer.get_constant_scalar(crate::Literal::U32(flag));
    let zero_id = writer.get_constant_scalar(crate::Literal::U32(0));
    let u32_type_id = writer.get_u32_type_id();
    let bool_ty = writer.get_bool_type_id();

    let and_id = writer.id_gen.next();
    block.body.push(Instruction::binary(
        spirv::Op::BitwiseAnd,
        u32_type_id,
        and_id,
        id,
        bit_id,
    ));

    let eq_id = writer.id_gen.next();
    block.body.push(Instruction::binary(
        spirv::Op::INotEqual,
        bool_ty,
        eq_id,
        and_id,
        zero_id,
    ));

    eq_id
}

impl Writer {
    pub(super) fn write_ray_query_get_intersection_function(
        &mut self,
        is_committed: bool,
        ir_module: &crate::Module,
    ) -> spirv::Word {
        if let Some(&word) =
            self.ray_query_functions
                .get(&LookupRayQueryFunction::GetIntersection {
                    committed: is_committed,
                })
        {
            return word;
        }
        let ray_intersection = ir_module.special_types.ray_intersection.unwrap();
        let intersection_type_id = self.get_handle_type_id(ray_intersection);
        let intersection_pointer_type_id =
            self.get_pointer_type_id(intersection_type_id, spirv::StorageClass::Function);

        let flag_type_id = self.get_u32_type_id();
        let flag_pointer_type_id =
            self.get_pointer_type_id(flag_type_id, spirv::StorageClass::Function);

        let transform_type_id = self.get_numeric_type_id(NumericType::Matrix {
            columns: crate::VectorSize::Quad,
            rows: crate::VectorSize::Tri,
            scalar: crate::Scalar::F32,
        });
        let transform_pointer_type_id =
            self.get_pointer_type_id(transform_type_id, spirv::StorageClass::Function);

        let barycentrics_type_id = self.get_numeric_type_id(NumericType::Vector {
            size: crate::VectorSize::Bi,
            scalar: crate::Scalar::F32,
        });
        let barycentrics_pointer_type_id =
            self.get_pointer_type_id(barycentrics_type_id, spirv::StorageClass::Function);

        let bool_type_id = self.get_bool_type_id();
        let bool_pointer_type_id =
            self.get_pointer_type_id(bool_type_id, spirv::StorageClass::Function);

        let scalar_type_id = self.get_f32_type_id();
        let float_pointer_type_id = self.get_f32_pointer_type_id(spirv::StorageClass::Function);

        let argument_type_id = self.get_ray_query_pointer_id();

        let func_ty = self.get_function_type(LookupFunctionType {
            parameter_type_ids: vec![argument_type_id, flag_pointer_type_id],
            return_type_id: intersection_type_id,
        });

        let mut function = Function::default();
        let func_id = self.id_gen.next();
        function.signature = Some(Instruction::function(
            intersection_type_id,
            func_id,
            spirv::FunctionControl::empty(),
            func_ty,
        ));
        let blank_intersection = self.get_constant_null(intersection_type_id);
        let query_id = self.id_gen.next();
        let instruction = Instruction::function_parameter(argument_type_id, query_id);
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 0,
        });

        let intersection_tracker_id = self.id_gen.next();
        let instruction =
            Instruction::function_parameter(flag_pointer_type_id, intersection_tracker_id);
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 1,
        });

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let blank_intersection_id = self.id_gen.next();
        // This must be before everything else in the function.
        block.body.push(Instruction::variable(
            intersection_pointer_type_id,
            blank_intersection_id,
            spirv::StorageClass::Function,
            Some(blank_intersection),
        ));

        let intersection_id = self.get_constant_scalar(crate::Literal::U32(if is_committed {
            spirv::RayQueryIntersection::RayQueryCommittedIntersectionKHR
        } else {
            spirv::RayQueryIntersection::RayQueryCandidateIntersectionKHR
        } as _));

        let loaded_ray_query_tracker_id = self.id_gen.next();
        block.body.push(Instruction::load(
            flag_type_id,
            loaded_ray_query_tracker_id,
            intersection_tracker_id,
            None,
        ));
        let proceeded_id = write_ray_flags_contains_flags(
            self,
            &mut block,
            loaded_ray_query_tracker_id,
            super::RayQueryPoint::PROCEED.bits(),
        );
        let finished_proceed_id = write_ray_flags_contains_flags(
            self,
            &mut block,
            loaded_ray_query_tracker_id,
            super::RayQueryPoint::FINISHED_TRAVERSAL.bits(),
        );
        let proceed_finished_correct_id = if is_committed {
            finished_proceed_id
        } else {
            let not_finished_id = self.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::LogicalNot,
                bool_type_id,
                not_finished_id,
                finished_proceed_id,
            ));
            not_finished_id
        };

        let is_valid_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_type_id,
            is_valid_id,
            proceed_finished_correct_id,
            proceeded_id,
        ));

        let valid_id = self.id_gen.next();
        let mut valid_block = Block::new(valid_id);

        let final_label_id = self.id_gen.next();
        let mut final_block = Block::new(final_label_id);

        block.body.push(Instruction::selection_merge(
            final_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            block,
            Instruction::branch_conditional(is_valid_id, valid_id, final_label_id),
        );

        let raw_kind_id = self.id_gen.next();
        valid_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionTypeKHR,
                flag_type_id,
                raw_kind_id,
                query_id,
                intersection_id,
            ));
        let kind_id = if is_committed {
            // Nothing to do: the IR value matches `spirv::RayQueryCommittedIntersectionType`
            raw_kind_id
        } else {
            // Remap from the candidate kind to IR
            let condition_id = self.id_gen.next();
            let committed_triangle_kind_id = self.get_constant_scalar(crate::Literal::U32(
                spirv::RayQueryCandidateIntersectionType::RayQueryCandidateIntersectionTriangleKHR
                    as _,
            ));
            valid_block.body.push(Instruction::binary(
                spirv::Op::IEqual,
                self.get_bool_type_id(),
                condition_id,
                raw_kind_id,
                committed_triangle_kind_id,
            ));
            let kind_id = self.id_gen.next();
            valid_block.body.push(Instruction::select(
                flag_type_id,
                kind_id,
                condition_id,
                self.get_constant_scalar(crate::Literal::U32(
                    crate::RayQueryIntersection::Triangle as _,
                )),
                self.get_constant_scalar(crate::Literal::U32(
                    crate::RayQueryIntersection::Aabb as _,
                )),
            ));
            kind_id
        };
        let idx_id = self.get_index_constant(0);
        let access_idx = self.id_gen.next();
        valid_block.body.push(Instruction::access_chain(
            flag_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        valid_block
            .body
            .push(Instruction::store(access_idx, kind_id, None));

        let not_none_comp_id = self.id_gen.next();
        let none_id =
            self.get_constant_scalar(crate::Literal::U32(crate::RayQueryIntersection::None as _));
        valid_block.body.push(Instruction::binary(
            spirv::Op::INotEqual,
            self.get_bool_type_id(),
            not_none_comp_id,
            kind_id,
            none_id,
        ));

        let not_none_label_id = self.id_gen.next();
        let mut not_none_block = Block::new(not_none_label_id);

        let outer_merge_label_id = self.id_gen.next();
        let outer_merge_block = Block::new(outer_merge_label_id);

        valid_block.body.push(Instruction::selection_merge(
            outer_merge_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            valid_block,
            Instruction::branch_conditional(
                not_none_comp_id,
                not_none_label_id,
                outer_merge_label_id,
            ),
        );

        let instance_custom_index_id = self.id_gen.next();
        not_none_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionInstanceCustomIndexKHR,
                flag_type_id,
                instance_custom_index_id,
                query_id,
                intersection_id,
            ));
        let instance_id = self.id_gen.next();
        not_none_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionInstanceIdKHR,
                flag_type_id,
                instance_id,
                query_id,
                intersection_id,
            ));
        let sbt_record_offset_id = self.id_gen.next();
        not_none_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR,
                flag_type_id,
                sbt_record_offset_id,
                query_id,
                intersection_id,
            ));
        let geometry_index_id = self.id_gen.next();
        not_none_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionGeometryIndexKHR,
                flag_type_id,
                geometry_index_id,
                query_id,
                intersection_id,
            ));
        let primitive_index_id = self.id_gen.next();
        not_none_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionPrimitiveIndexKHR,
                flag_type_id,
                primitive_index_id,
                query_id,
                intersection_id,
            ));

        //Note: there is also `OpRayQueryGetIntersectionCandidateAABBOpaqueKHR`,
        // but it's not a property of an intersection.

        let object_to_world_id = self.id_gen.next();
        not_none_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionObjectToWorldKHR,
                transform_type_id,
                object_to_world_id,
                query_id,
                intersection_id,
            ));
        let world_to_object_id = self.id_gen.next();
        not_none_block
            .body
            .push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionWorldToObjectKHR,
                transform_type_id,
                world_to_object_id,
                query_id,
                intersection_id,
            ));

        // instance custom index
        let idx_id = self.get_index_constant(2);
        let access_idx = self.id_gen.next();
        not_none_block.body.push(Instruction::access_chain(
            flag_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        not_none_block.body.push(Instruction::store(
            access_idx,
            instance_custom_index_id,
            None,
        ));

        // instance
        let idx_id = self.get_index_constant(3);
        let access_idx = self.id_gen.next();
        not_none_block.body.push(Instruction::access_chain(
            flag_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        not_none_block
            .body
            .push(Instruction::store(access_idx, instance_id, None));

        let idx_id = self.get_index_constant(4);
        let access_idx = self.id_gen.next();
        not_none_block.body.push(Instruction::access_chain(
            flag_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        not_none_block
            .body
            .push(Instruction::store(access_idx, sbt_record_offset_id, None));

        let idx_id = self.get_index_constant(5);
        let access_idx = self.id_gen.next();
        not_none_block.body.push(Instruction::access_chain(
            flag_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        not_none_block
            .body
            .push(Instruction::store(access_idx, geometry_index_id, None));

        let idx_id = self.get_index_constant(6);
        let access_idx = self.id_gen.next();
        not_none_block.body.push(Instruction::access_chain(
            flag_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        not_none_block
            .body
            .push(Instruction::store(access_idx, primitive_index_id, None));

        let idx_id = self.get_index_constant(9);
        let access_idx = self.id_gen.next();
        not_none_block.body.push(Instruction::access_chain(
            transform_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        not_none_block
            .body
            .push(Instruction::store(access_idx, object_to_world_id, None));

        let idx_id = self.get_index_constant(10);
        let access_idx = self.id_gen.next();
        not_none_block.body.push(Instruction::access_chain(
            transform_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        not_none_block
            .body
            .push(Instruction::store(access_idx, world_to_object_id, None));

        let tri_comp_id = self.id_gen.next();
        let tri_id = self.get_constant_scalar(crate::Literal::U32(
            crate::RayQueryIntersection::Triangle as _,
        ));
        not_none_block.body.push(Instruction::binary(
            spirv::Op::IEqual,
            self.get_bool_type_id(),
            tri_comp_id,
            kind_id,
            tri_id,
        ));

        let tri_label_id = self.id_gen.next();
        let mut tri_block = Block::new(tri_label_id);

        let merge_label_id = self.id_gen.next();
        let merge_block = Block::new(merge_label_id);
        // t
        {
            let block = if is_committed {
                &mut not_none_block
            } else {
                &mut tri_block
            };
            let t_id = self.id_gen.next();
            block.body.push(Instruction::ray_query_get_intersection(
                spirv::Op::RayQueryGetIntersectionTKHR,
                scalar_type_id,
                t_id,
                query_id,
                intersection_id,
            ));
            let idx_id = self.get_index_constant(1);
            let access_idx = self.id_gen.next();
            block.body.push(Instruction::access_chain(
                float_pointer_type_id,
                access_idx,
                blank_intersection_id,
                &[idx_id],
            ));
            block.body.push(Instruction::store(access_idx, t_id, None));
        }
        not_none_block.body.push(Instruction::selection_merge(
            merge_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            not_none_block,
            Instruction::branch_conditional(not_none_comp_id, tri_label_id, merge_label_id),
        );

        let barycentrics_id = self.id_gen.next();
        tri_block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionBarycentricsKHR,
            barycentrics_type_id,
            barycentrics_id,
            query_id,
            intersection_id,
        ));

        let front_face_id = self.id_gen.next();
        tri_block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionFrontFaceKHR,
            bool_type_id,
            front_face_id,
            query_id,
            intersection_id,
        ));

        let idx_id = self.get_index_constant(7);
        let access_idx = self.id_gen.next();
        tri_block.body.push(Instruction::access_chain(
            barycentrics_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        tri_block
            .body
            .push(Instruction::store(access_idx, barycentrics_id, None));

        let idx_id = self.get_index_constant(8);
        let access_idx = self.id_gen.next();
        tri_block.body.push(Instruction::access_chain(
            bool_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        tri_block
            .body
            .push(Instruction::store(access_idx, front_face_id, None));
        function.consume(tri_block, Instruction::branch(merge_label_id));
        function.consume(merge_block, Instruction::branch(outer_merge_label_id));
        function.consume(outer_merge_block, Instruction::branch(final_label_id));

        let loaded_blank_intersection_id = self.id_gen.next();
        final_block.body.push(Instruction::load(
            intersection_type_id,
            loaded_blank_intersection_id,
            blank_intersection_id,
            None,
        ));
        function.consume(
            final_block,
            Instruction::return_value(loaded_blank_intersection_id),
        );

        function.to_words(&mut self.logical_layout.function_definitions);
        self.ray_query_functions.insert(
            LookupRayQueryFunction::GetIntersection {
                committed: is_committed,
            },
            func_id,
        );
        func_id
    }

    fn write_ray_query_initialize(&mut self, ir_module: &crate::Module) -> spirv::Word {
        if let Some(&word) = self
            .ray_query_functions
            .get(&LookupRayQueryFunction::Initialize)
        {
            return word;
        }

        let ray_query_type_id = self.get_ray_query_pointer_id();
        let acceleration_structure_type_id =
            self.get_localtype_id(super::LocalType::AccelerationStructure);
        let ray_desc_type_id = self.get_handle_type_id(
            ir_module
                .special_types
                .ray_desc
                .expect("ray desc should be set if ray queries are being initialized"),
        );

        let u32_ty = self.get_u32_type_id();
        let u32_ptr_ty = self.get_pointer_type_id(u32_ty, spirv::StorageClass::Function);

        let bool_type_id = self.get_bool_type_id();
        let bool_vec3_type_id = self.get_vec3_bool_type_id();

        let func_ty = self.get_function_type(LookupFunctionType {
            parameter_type_ids: vec![
                ray_query_type_id,
                acceleration_structure_type_id,
                ray_desc_type_id,
                u32_ptr_ty,
            ],
            return_type_id: self.void_type,
        });

        let mut function = Function::default();
        let func_id = self.id_gen.next();
        function.signature = Some(Instruction::function(
            self.void_type,
            func_id,
            spirv::FunctionControl::empty(),
            func_ty,
        ));

        let query_id = self.id_gen.next();
        let instruction = Instruction::function_parameter(ray_query_type_id, query_id);
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 0,
        });

        let acceleration_structure_id = self.id_gen.next();
        let instruction = Instruction::function_parameter(
            acceleration_structure_type_id,
            acceleration_structure_id,
        );
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 1,
        });

        let desc_id = self.id_gen.next();
        let instruction = Instruction::function_parameter(ray_desc_type_id, desc_id);
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 2,
        });

        let init_tracker_id = self.id_gen.next();
        let instruction = Instruction::function_parameter(u32_ptr_ty, init_tracker_id);
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 3,
        });

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let flag_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::U32));

        //Note: composite extract indices and types must match `generate_ray_desc_type`
        let ray_flags_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            flag_type_id,
            ray_flags_id,
            desc_id,
            &[0],
        ));
        let cull_mask_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            flag_type_id,
            cull_mask_id,
            desc_id,
            &[1],
        ));

        let scalar_type_id = self.get_numeric_type_id(NumericType::Scalar(crate::Scalar::F32));
        let tmin_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            scalar_type_id,
            tmin_id,
            desc_id,
            &[2],
        ));
        let tmax_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            scalar_type_id,
            tmax_id,
            desc_id,
            &[3],
        ));

        let vector_type_id = self.get_numeric_type_id(NumericType::Vector {
            size: crate::VectorSize::Tri,
            scalar: crate::Scalar::F32,
        });
        let ray_origin_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            vector_type_id,
            ray_origin_id,
            desc_id,
            &[4],
        ));
        let ray_dir_id = self.id_gen.next();
        block.body.push(Instruction::composite_extract(
            vector_type_id,
            ray_dir_id,
            desc_id,
            &[5],
        ));

        let tmin_le_tmax_id = self.id_gen.next();
        // Because this checks if tmin and tmax are ordered too (i.e: not NaN), there is no need for an additional check.
        block.body.push(Instruction::binary(
            spirv::Op::FOrdLessThanEqual,
            bool_type_id,
            tmin_le_tmax_id,
            tmin_id,
            tmax_id,
        ));

        let tmin_ge_zero_id = self.id_gen.next();
        let zero_id = self.get_constant_scalar(crate::Literal::F32(0.0));
        block.body.push(Instruction::binary(
            spirv::Op::FOrdGreaterThanEqual,
            bool_type_id,
            tmin_ge_zero_id,
            tmin_id,
            zero_id,
        ));

        let ray_origin_infinite_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::IsInf,
            bool_vec3_type_id,
            ray_origin_infinite_id,
            ray_origin_id,
        ));
        let any_ray_origin_infinite_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::Any,
            bool_type_id,
            any_ray_origin_infinite_id,
            ray_origin_infinite_id,
        ));

        let ray_origin_nan_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::IsNan,
            bool_vec3_type_id,
            ray_origin_nan_id,
            ray_origin_id,
        ));
        let any_ray_origin_nan_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::Any,
            bool_type_id,
            any_ray_origin_nan_id,
            ray_origin_nan_id,
        ));

        let ray_origin_not_finite_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalOr,
            bool_type_id,
            ray_origin_not_finite_id,
            any_ray_origin_nan_id,
            any_ray_origin_infinite_id,
        ));

        let all_ray_origin_finite_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::LogicalNot,
            bool_type_id,
            all_ray_origin_finite_id,
            ray_origin_not_finite_id,
        ));

        let ray_dir_infinite_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::IsInf,
            bool_vec3_type_id,
            ray_dir_infinite_id,
            ray_dir_id,
        ));
        let any_ray_dir_infinite_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::Any,
            bool_type_id,
            any_ray_dir_infinite_id,
            ray_dir_infinite_id,
        ));

        let ray_dir_nan_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::IsNan,
            bool_vec3_type_id,
            ray_dir_nan_id,
            ray_dir_id,
        ));
        let any_ray_dir_nan_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::Any,
            bool_type_id,
            any_ray_dir_nan_id,
            ray_dir_nan_id,
        ));

        let ray_dir_not_finite_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalOr,
            bool_type_id,
            ray_dir_not_finite_id,
            any_ray_dir_nan_id,
            any_ray_dir_infinite_id,
        ));

        let all_ray_dir_finite_id = self.id_gen.next();
        block.body.push(Instruction::unary(
            spirv::Op::LogicalNot,
            bool_type_id,
            all_ray_dir_finite_id,
            ray_dir_not_finite_id,
        ));

        /// Writes spirv to check that less than two booleans are true
        ///
        /// For each boolean: removes it, `and`s it with all others (i.e for all possible combinations of two booleans in the list checks to see if both are true).
        /// Then `or`s all of these checks together. This produces whether two or more booleans are true.
        fn write_less_than_2_true(
            writer: &mut Writer,
            block: &mut Block,
            mut bools: Vec<spirv::Word>,
        ) -> spirv::Word {
            assert!(bools.len() > 1, "Must have multiple booleans!");
            let bool_ty = writer.get_bool_type_id();
            let mut each_two_true = Vec::new();
            while let Some(last_bool) = bools.pop() {
                for &bool in &bools {
                    let both_true_id = writer.id_gen.next();
                    block.body.push(Instruction::binary(
                        spirv::Op::LogicalAnd,
                        bool_ty,
                        both_true_id,
                        last_bool,
                        bool,
                    ));
                    each_two_true.push(both_true_id);
                }
            }
            let mut all_or_id = each_two_true.pop().expect("since this must have multiple booleans, there must be at least one thing in `each_two_true");
            for two_true in each_two_true {
                let new_all_or_id = writer.id_gen.next();
                block.body.push(Instruction::binary(
                    spirv::Op::LogicalOr,
                    bool_ty,
                    new_all_or_id,
                    all_or_id,
                    two_true,
                ));
                all_or_id = new_all_or_id;
            }

            let less_than_two_id = writer.id_gen.next();
            block.body.push(Instruction::unary(
                spirv::Op::LogicalNot,
                bool_ty,
                less_than_two_id,
                all_or_id,
            ));
            less_than_two_id
        }

        let contains_skip_triangles = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::SKIP_TRIANGLES.bits(),
        );
        let contains_skip_aabbs = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::SKIP_AABBS.bits(),
        );

        let not_contain_skip_triangles_aabbs = write_less_than_2_true(
            self,
            &mut block,
            vec![contains_skip_triangles, contains_skip_aabbs],
        );

        let contains_cull_back = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::CULL_BACK_FACING.bits(),
        );
        let contains_cull_front = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::CULL_FRONT_FACING.bits(),
        );

        let not_contain_skip_triangles_cull = write_less_than_2_true(
            self,
            &mut block,
            vec![
                contains_skip_triangles,
                contains_cull_back,
                contains_cull_front,
            ],
        );

        let contains_opaque = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::FORCE_OPAQUE.bits(),
        );
        let contains_no_opaque = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::FORCE_NO_OPAQUE.bits(),
        );
        let contains_cull_opaque = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::CULL_OPAQUE.bits(),
        );
        let contains_cull_no_opaque = write_ray_flags_contains_flags(
            self,
            &mut block,
            ray_flags_id,
            crate::RayFlag::CULL_NO_OPAQUE.bits(),
        );

        let not_contain_multiple_opaque = write_less_than_2_true(
            self,
            &mut block,
            vec![
                contains_opaque,
                contains_no_opaque,
                contains_cull_opaque,
                contains_cull_no_opaque,
            ],
        );

        let tmin_tmax_valid_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_type_id,
            tmin_tmax_valid_id,
            tmin_le_tmax_id,
            tmin_ge_zero_id,
        ));

        let origin_dir_valid_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_type_id,
            origin_dir_valid_id,
            all_ray_origin_finite_id,
            all_ray_dir_finite_id,
        ));

        let flags_skip_tri_aabbs_tri_cull_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_type_id,
            flags_skip_tri_aabbs_tri_cull_id,
            not_contain_skip_triangles_aabbs,
            not_contain_skip_triangles_cull,
        ));
        let flags_valid_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_type_id,
            flags_valid_id,
            flags_skip_tri_aabbs_tri_cull_id,
            not_contain_multiple_opaque,
        ));

        let tmin_tmax_origin_dir_valid_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_type_id,
            tmin_tmax_origin_dir_valid_id,
            tmin_tmax_valid_id,
            origin_dir_valid_id,
        ));

        let all_valid_id = self.id_gen.next();
        block.body.push(Instruction::binary(
            spirv::Op::LogicalAnd,
            bool_type_id,
            all_valid_id,
            tmin_tmax_origin_dir_valid_id,
            flags_valid_id,
        ));

        let merge_label_id = self.id_gen.next();
        let merge_block = Block::new(merge_label_id);

        let invalid_label_id = self.id_gen.next();
        let mut invalid_block = Block::new(invalid_label_id);

        let valid_label_id = self.id_gen.next();
        let mut valid_block = Block::new(valid_label_id);

        block.body.push(Instruction::selection_merge(
            merge_label_id,
            spirv::SelectionControl::NONE,
        ));

        function.consume(
            block,
            Instruction::branch_conditional(all_valid_id, valid_label_id, invalid_label_id),
        );

        valid_block.body.push(Instruction::ray_query_initialize(
            query_id,
            acceleration_structure_id,
            ray_flags_id,
            cull_mask_id,
            ray_origin_id,
            tmin_id,
            ray_dir_id,
            tmax_id,
        ));

        let const_initialized = self.get_constant_scalar(crate::Literal::U32(
            super::RayQueryPoint::INITIALIZED.bits(),
        ));
        valid_block
            .body
            .push(Instruction::store(init_tracker_id, const_initialized, None));

        function.consume(valid_block, Instruction::branch(merge_label_id));

        if self
            .flags
            .contains(super::WriterFlags::PRINT_ON_RAY_QUERY_INITIALIZATION_FAIL)
        {
            self.write_debug_printf(
                &mut invalid_block,
                "Naga ignored invalid arguments to rayQueryInitialize with flags: %u t_min: %f t_max: %f origin: %v4f dir: %v4f",
                &[
                    ray_flags_id,
                    tmin_id,
                    tmax_id,
                    ray_origin_id,
                    ray_dir_id,
                ],
            );
        }

        function.consume(invalid_block, Instruction::branch(merge_label_id));

        function.consume(merge_block, Instruction::return_void());

        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_query_functions
            .insert(LookupRayQueryFunction::Initialize, func_id);
        func_id
    }

    fn write_ray_query_proceed(&mut self) -> spirv::Word {
        if let Some(&word) = self
            .ray_query_functions
            .get(&LookupRayQueryFunction::Proceed)
        {
            return word;
        }

        let ray_query_type_id = self.get_ray_query_pointer_id();

        let u32_ty = self.get_u32_type_id();
        let u32_ptr_ty = self.get_pointer_type_id(u32_ty, spirv::StorageClass::Function);

        let bool_type_id = self.get_bool_type_id();
        let bool_ptr_ty = self.get_pointer_type_id(bool_type_id, spirv::StorageClass::Function);

        let func_ty = self.get_function_type(LookupFunctionType {
            parameter_type_ids: vec![ray_query_type_id, u32_ptr_ty],
            return_type_id: bool_type_id,
        });

        let mut function = Function::default();
        let func_id = self.id_gen.next();
        function.signature = Some(Instruction::function(
            bool_type_id,
            func_id,
            spirv::FunctionControl::empty(),
            func_ty,
        ));

        let query_id = self.id_gen.next();
        let instruction = Instruction::function_parameter(ray_query_type_id, query_id);
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 0,
        });

        let init_tracker_id = self.id_gen.next();
        let instruction = Instruction::function_parameter(u32_ptr_ty, init_tracker_id);
        function.parameters.push(FunctionArgument {
            instruction,
            handle_id: 1,
        });

        let block_id = self.id_gen.next();
        let mut block = Block::new(block_id);

        // TODO: perhaps this could be replaced with an OpPhi?
        let proceeded_id = self.id_gen.next();
        let const_false = self.get_constant_scalar(crate::Literal::Bool(false));
        block.body.push(Instruction::variable(
            bool_ptr_ty,
            proceeded_id,
            spirv::StorageClass::Function,
            Some(const_false),
        ));

        let initialized_tracker_id = self.id_gen.next();
        block.body.push(Instruction::load(
            u32_ty,
            initialized_tracker_id,
            init_tracker_id,
            None,
        ));

        let is_initialized = write_ray_flags_contains_flags(
            self,
            &mut block,
            initialized_tracker_id,
            super::RayQueryPoint::INITIALIZED.bits(),
        );

        let merge_id = self.id_gen.next();
        let mut merge_block = Block::new(merge_id);

        let valid_block_id = self.id_gen.next();
        let mut valid_block = Block::new(valid_block_id);

        block.body.push(Instruction::selection_merge(
            merge_id,
            spirv::SelectionControl::NONE,
        ));

        function.consume(
            block,
            Instruction::branch_conditional(is_initialized, valid_block_id, merge_id),
        );

        let has_proceeded = self.id_gen.next();
        valid_block.body.push(Instruction::ray_query_proceed(
            bool_type_id,
            has_proceeded,
            query_id,
        ));

        valid_block
            .body
            .push(Instruction::store(proceeded_id, has_proceeded, None));

        let add_flag_finished = self.get_constant_scalar(crate::Literal::U32(
            (super::RayQueryPoint::PROCEED | super::RayQueryPoint::FINISHED_TRAVERSAL).bits(),
        ));
        let add_flag_continuing =
            self.get_constant_scalar(crate::Literal::U32(super::RayQueryPoint::PROCEED.bits()));

        let add_flags_id = self.id_gen.next();
        valid_block.body.push(Instruction::select(
            u32_ty,
            add_flags_id,
            has_proceeded,
            add_flag_continuing,
            add_flag_finished,
        ));
        let final_flags = self.id_gen.next();
        valid_block.body.push(Instruction::binary(
            spirv::Op::BitwiseOr,
            u32_ty,
            final_flags,
            initialized_tracker_id,
            add_flags_id,
        ));
        valid_block
            .body
            .push(Instruction::store(init_tracker_id, final_flags, None));

        function.consume(valid_block, Instruction::branch(merge_id));

        let loaded_proceeded_id = self.id_gen.next();
        merge_block.body.push(Instruction::load(
            bool_type_id,
            loaded_proceeded_id,
            proceeded_id,
            None,
        ));

        function.consume(merge_block, Instruction::return_value(loaded_proceeded_id));

        function.to_words(&mut self.logical_layout.function_definitions);

        self.ray_query_functions
            .insert(LookupRayQueryFunction::Proceed, func_id);
        func_id
    }
}

impl BlockContext<'_> {
    pub(super) fn write_ray_query_function(
        &mut self,
        query: Handle<crate::Expression>,
        function: &crate::RayQueryFunction,
        block: &mut Block,
    ) {
        let query_id = self.cached[query];
        let init_tracker_id = *self
            .ray_query_tracker_expr
            .get(&query)
            .expect("not a cached ray query");

        match *function {
            crate::RayQueryFunction::Initialize {
                acceleration_structure,
                descriptor,
            } => {
                let desc_id = self.cached[descriptor];
                let acc_struct_id = self.get_handle_id(acceleration_structure);

                let func = self.writer.write_ray_query_initialize(self.ir_module);

                let func_id = self.gen_id();
                block.body.push(Instruction::function_call(
                    self.writer.void_type,
                    func_id,
                    func,
                    &[query_id, acc_struct_id, desc_id, init_tracker_id],
                ));
            }
            crate::RayQueryFunction::Proceed { result } => {
                let id = self.gen_id();
                self.cached[result] = id;

                let bool_ty = self.writer.get_bool_type_id();

                let func_id = self.writer.write_ray_query_proceed();
                block.body.push(Instruction::function_call(
                    bool_ty,
                    id,
                    func_id,
                    &[query_id, init_tracker_id],
                ));
            }
            crate::RayQueryFunction::GenerateIntersection { hit_t } => {
                let hit_id = self.cached[hit_t];
                block
                    .body
                    .push(Instruction::ray_query_generate_intersection(
                        query_id, hit_id,
                    ));
            }
            crate::RayQueryFunction::ConfirmIntersection => {
                block
                    .body
                    .push(Instruction::ray_query_confirm_intersection(query_id));
            }
            crate::RayQueryFunction::Terminate => {}
        }
    }

    pub(super) fn write_ray_query_return_vertex_position(
        &mut self,
        query: Handle<crate::Expression>,
        block: &mut Block,
        is_committed: bool,
    ) -> spirv::Word {
        let query_id = self.cached[query];
        let id = self.gen_id();
        let ray_vertex_return_ty = self
            .ir_module
            .special_types
            .ray_vertex_return
            .expect("type should have been populated");
        let ray_vertex_return_ty_id = self.writer.get_handle_type_id(ray_vertex_return_ty);
        let intersection_id =
            self.writer
                .get_constant_scalar(crate::Literal::U32(if is_committed {
                    spirv::RayQueryIntersection::RayQueryCommittedIntersectionKHR
                } else {
                    spirv::RayQueryIntersection::RayQueryCandidateIntersectionKHR
                } as _));
        block
            .body
            .push(Instruction::ray_query_return_vertex_position(
                ray_vertex_return_ty_id,
                id,
                query_id,
                intersection_id,
            ));
        id
    }
}
