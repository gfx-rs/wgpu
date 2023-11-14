/*!
Generating SPIR-V for ray query operations.
*/

use super::{Block, BlockContext, Instruction, LocalType, LookupType};
use crate::arena::Handle;

impl<'w> BlockContext<'w> {
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

                let flag_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
                    vector_size: None,
                    scalar: crate::Scalar::U32,
                    pointer_space: None,
                }));
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

                let scalar_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
                    vector_size: None,
                    scalar: crate::Scalar::F32,
                    pointer_space: None,
                }));
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

                let vector_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
                    vector_size: Some(crate::VectorSize::Tri),
                    scalar: crate::Scalar::F32,
                    pointer_space: None,
                }));
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
    ) -> spirv::Word {
        let query_id = self.cached[query];
        let intersection_id = self.writer.get_constant_scalar(crate::Literal::U32(
            spirv::RayQueryIntersection::RayQueryCommittedIntersectionKHR as _,
        ));

        let flag_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: None,
            scalar: crate::Scalar::U32,
            pointer_space: None,
        }));
        let kind_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionTypeKHR,
            flag_type_id,
            kind_id,
            query_id,
            intersection_id,
        ));
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

        let scalar_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: None,
            scalar: crate::Scalar::F32,
            pointer_space: None,
        }));
        let t_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionTKHR,
            scalar_type_id,
            t_id,
            query_id,
            intersection_id,
        ));

        let barycentrics_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: Some(crate::VectorSize::Bi),
            scalar: crate::Scalar::F32,
            pointer_space: None,
        }));
        let barycentrics_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionBarycentricsKHR,
            barycentrics_type_id,
            barycentrics_id,
            query_id,
            intersection_id,
        ));

        let bool_type_id = self.get_type_id(LookupType::Local(LocalType::Value {
            vector_size: None,
            scalar: crate::Scalar::BOOL,
            pointer_space: None,
        }));
        let front_face_id = self.gen_id();
        block.body.push(Instruction::ray_query_get_intersection(
            spirv::Op::RayQueryGetIntersectionFrontFaceKHR,
            bool_type_id,
            front_face_id,
            query_id,
            intersection_id,
        ));

        let transform_type_id = self.get_type_id(LookupType::Local(LocalType::Matrix {
            columns: crate::VectorSize::Quad,
            rows: crate::VectorSize::Tri,
            width: 4,
        }));
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
