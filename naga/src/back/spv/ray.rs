/*!
Generating SPIR-V for ray query operations.
*/

use super::{Block, BlockContext, Instruction, LocalType, LookupType, NumericType};
use crate::arena::Handle;

impl BlockContext<'_> {
    pub(super) fn write_ray_query_function(
        &mut self,
        query: Handle<crate::Expression>,
        function: &crate::RayQueryFunction,
        block: &mut Block,
    ) {
        let query_id = self.cached[query];
        match *function {
            crate::RayQueryFunction::Initialize {
                acceleration_structure,
                descriptor,
            } => {
                //Note: composite extract indices and types must match `generate_ray_desc_type`
                let desc_id = self.cached[descriptor];
                let acc_struct_id = self.get_handle_id(acceleration_structure);

                let flag_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
                    NumericType::Scalar(crate::Scalar::U32),
                )));
                let ray_flags_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    flag_type_id,
                    ray_flags_id,
                    desc_id,
                    &[0],
                ));
                let cull_mask_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    flag_type_id,
                    cull_mask_id,
                    desc_id,
                    &[1],
                ));

                let scalar_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
                    NumericType::Scalar(crate::Scalar::F32),
                )));
                let tmin_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    scalar_type_id,
                    tmin_id,
                    desc_id,
                    &[2],
                ));
                let tmax_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    scalar_type_id,
                    tmax_id,
                    desc_id,
                    &[3],
                ));

                let vector_type_id =
                    self.get_type_id(LookupType::Local(LocalType::Numeric(NumericType::Vector {
                        size: crate::VectorSize::Tri,
                        scalar: crate::Scalar::F32,
                    })));
                let ray_origin_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    vector_type_id,
                    ray_origin_id,
                    desc_id,
                    &[4],
                ));
                let ray_dir_id = self.gen_id();
                block.body.push(Instruction::composite_extract(
                    vector_type_id,
                    ray_dir_id,
                    desc_id,
                    &[5],
                ));

                block.body.push(Instruction::ray_query_initialize(
                    query_id,
                    acc_struct_id,
                    ray_flags_id,
                    cull_mask_id,
                    ray_origin_id,
                    tmin_id,
                    ray_dir_id,
                    tmax_id,
                ));
            }
            crate::RayQueryFunction::Proceed { result } => {
                let id = self.gen_id();
                self.cached[result] = id;
                let result_type_id = self.get_expression_type_id(&self.fun_info[result].ty);

                block
                    .body
                    .push(Instruction::ray_query_proceed(result_type_id, id, query_id));
            }
            crate::RayQueryFunction::Terminate => {}
        }
    }

    pub(super) fn write_ray_query_get_intersection(
        &mut self,
        query: Handle<crate::Expression>,
        block: &mut Block,
        is_committed: bool,
    ) -> spirv::Word {
        let query_id = self.cached[query];
        let intersection_id =
            self.writer
                .get_constant_scalar(crate::Literal::U32(if is_committed {
                    spirv::RayQueryIntersection::RayQueryCommittedIntersectionKHR
                } else {
                    spirv::RayQueryIntersection::RayQueryCandidateIntersectionKHR
                } as _));

        let flag_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::U32),
        )));
        let raw_kind_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
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
            let condition_id = self.gen_id();
            let committed_triangle_kind_id = self.writer.get_constant_scalar(crate::Literal::U32(
                spirv::RayQueryCandidateIntersectionType::RayQueryCandidateIntersectionTriangleKHR
                    as _,
            ));
            block.body.push(Instruction::binary(
                spirv::Op::IEqual,
                self.writer.get_bool_type_id(),
                condition_id,
                raw_kind_id,
                committed_triangle_kind_id,
            ));
            let kind_id = self.gen_id();
            block.body.push(Instruction::select(
                flag_type_id,
                kind_id,
                condition_id,
                self.writer.get_constant_scalar(crate::Literal::U32(
                    crate::RayQueryIntersection::Triangle as _,
                )),
                self.writer.get_constant_scalar(crate::Literal::U32(
                    crate::RayQueryIntersection::Aabb as _,
                )),
            ));
            kind_id
        };

        let instance_custom_index_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionInstanceCustomIndexKHR,
            flag_type_id,
            instance_custom_index_id,
            query_id,
            intersection_id,
        ));
        let instance_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionInstanceIdKHR,
            flag_type_id,
            instance_id,
            query_id,
            intersection_id,
        ));
        let sbt_record_offset_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetKHR,
            flag_type_id,
            sbt_record_offset_id,
            query_id,
            intersection_id,
        ));
        let geometry_index_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionGeometryIndexKHR,
            flag_type_id,
            geometry_index_id,
            query_id,
            intersection_id,
        ));
        let primitive_index_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionPrimitiveIndexKHR,
            flag_type_id,
            primitive_index_id,
            query_id,
            intersection_id,
        ));

        let scalar_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::F32),
        )));
        let t_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionTKHR,
            scalar_type_id,
            t_id,
            query_id,
            intersection_id,
        ));

        let barycentrics_type_id =
            self.get_type_id(LookupType::Local(LocalType::Numeric(NumericType::Vector {
                size: crate::VectorSize::Bi,
                scalar: crate::Scalar::F32,
            })));
        let barycentrics_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionBarycentricsKHR,
            barycentrics_type_id,
            barycentrics_id,
            query_id,
            intersection_id,
        ));

        let bool_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::BOOL),
        )));
        let front_face_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionFrontFaceKHR,
            bool_type_id,
            front_face_id,
            query_id,
            intersection_id,
        ));
        //Note: there is also `OpRayQueryGetIntersectionCandidateAABBOpaqueKHR`,
        // but it's not a property of an intersection.

        let transform_type_id =
            self.get_type_id(LookupType::Local(LocalType::Numeric(NumericType::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Tri,
                scalar: crate::Scalar::F32,
            })));
        let object_to_world_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionObjectToWorldKHR,
            transform_type_id,
            object_to_world_id,
            query_id,
            intersection_id,
        ));
        let world_to_object_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionWorldToObjectKHR,
            transform_type_id,
            world_to_object_id,
            query_id,
            intersection_id,
        ));

        let id = self.gen_id();
        let intersection_type_id = self.get_type_id(LookupType::Handle(
            self.ir_module.special_types.ray_intersection.unwrap(),
        ));
        //Note: the arguments must match `generate_ray_intersection_type` layout
        block.body.push(Instruction::composite_construct(
            intersection_type_id,
            id,
            &[
                kind_id,
                t_id,
                instance_custom_index_id,
                instance_id,
                sbt_record_offset_id,
                geometry_index_id,
                primitive_index_id,
                barycentrics_id,
                front_face_id,
                object_to_world_id,
                world_to_object_id,
            ],
        ));
        id
    }
}
