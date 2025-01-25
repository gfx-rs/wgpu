/*!
Generating SPIR-V for ray query operations.
*/

use super::{
    Block, BlockContext, Function, FunctionArgument, Instruction, LocalType, LookupFunctionType,
    LookupType, NumericType, Writer,
};
use crate::arena::Handle;
use crate::{Type, TypeInner};

impl Writer {
    pub(super) fn write_ray_query_get_intersection_function(
        &mut self,
        is_committed: bool,
        ir_module: &crate::Module,
    ) -> spirv::Word {
        if let Some(func_id) = self.ray_get_intersection_function {
            return func_id;
        }
        let ray_intersection = ir_module.special_types.ray_intersection.unwrap();
        let intersection_type_id = self.get_type_id(LookupType::Handle(ray_intersection));
        let intersection_pointer_type_id =
            self.get_type_id(LookupType::Local(LocalType::Pointer {
                base: ray_intersection,
                class: spirv::StorageClass::Function,
            }));

        let flag_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::U32),
        )));
        let flag_type = ir_module
            .types
            .get(&Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::U32),
            })
            .unwrap();
        let flag_pointer_type_id = self.get_type_id(LookupType::Local(LocalType::Pointer {
            base: flag_type,
            class: spirv::StorageClass::Function,
        }));

        let transform_type_id =
            self.get_type_id(LookupType::Local(LocalType::Numeric(NumericType::Matrix {
                columns: crate::VectorSize::Quad,
                rows: crate::VectorSize::Tri,
                scalar: crate::Scalar::F32,
            })));
        let transform_type = ir_module
            .types
            .get(&Type {
                name: None,
                inner: TypeInner::Matrix {
                    columns: crate::VectorSize::Quad,
                    rows: crate::VectorSize::Tri,
                    scalar: crate::Scalar::F32,
                },
            })
            .unwrap();
        let transform_pointer_type_id = self.get_type_id(LookupType::Local(LocalType::Pointer {
            base: transform_type,
            class: spirv::StorageClass::Function,
        }));

        let barycentrics_type_id =
            self.get_type_id(LookupType::Local(LocalType::Numeric(NumericType::Vector {
                size: crate::VectorSize::Bi,
                scalar: crate::Scalar::F32,
            })));
        let barycentrics_type = ir_module
            .types
            .get(&Type {
                name: None,
                inner: TypeInner::Vector {
                    size: crate::VectorSize::Bi,
                    scalar: crate::Scalar::F32,
                },
            })
            .unwrap();
        let barycentrics_pointer_type_id =
            self.get_type_id(LookupType::Local(LocalType::Pointer {
                base: barycentrics_type,
                class: spirv::StorageClass::Function,
            }));

        let bool_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::BOOL),
        )));
        let bool_type = ir_module
            .types
            .get(&Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::BOOL),
            })
            .unwrap();
        let bool_pointer_type_id = self.get_type_id(LookupType::Local(LocalType::Pointer {
            base: bool_type,
            class: spirv::StorageClass::Function,
        }));

        let scalar_type_id = self.get_type_id(LookupType::Local(LocalType::Numeric(
            NumericType::Scalar(crate::Scalar::F32),
        )));
        let float_type = ir_module
            .types
            .get(&Type {
                name: None,
                inner: TypeInner::Scalar(crate::Scalar::F32),
            })
            .unwrap();
        let float_pointer_type_id = self.get_type_id(LookupType::Local(LocalType::Pointer {
            base: float_type,
            class: spirv::StorageClass::Function,
        }));

        let rq_ty = ir_module
            .types
            .get(&Type {
                name: None,
                inner: TypeInner::RayQuery,
            })
            .expect("ray_query type should have been populated by the variable passed into this!");
        let argument_type_id = self.get_type_id(LookupType::Local(LocalType::Pointer {
            base: rq_ty,
            class: spirv::StorageClass::Function,
        }));
        let func_ty = self.get_function_type(LookupFunctionType {
            parameter_type_ids: vec![argument_type_id],
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

        let label_id = self.id_gen.next();
        let mut block = Block::new(label_id);

        let blank_intersection_id = self.id_gen.next();
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
        let raw_kind_id = self.id_gen.next();
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
            let condition_id = self.id_gen.next();
            let committed_triangle_kind_id = self.get_constant_scalar(crate::Literal::U32(
                spirv::RayQueryCandidateIntersectionType::RayQueryCandidateIntersectionTriangleKHR
                    as _,
            ));
            block.body.push(Instruction::binary(
                spirv::Op::IEqual,
                self.get_bool_type_id(),
                condition_id,
                raw_kind_id,
                committed_triangle_kind_id,
            ));
            let kind_id = self.id_gen.next();
            block.body.push(Instruction::select(
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
        block.body.push(Instruction::access_chain(
            flag_pointer_type_id,
            access_idx,
            blank_intersection_id,
            &[idx_id],
        ));
        block
            .body
            .push(Instruction::store(access_idx, kind_id, None));

        let not_none_comp_id = self.id_gen.next();
        let none_id =
            self.get_constant_scalar(crate::Literal::U32(crate::RayQueryIntersection::None as _));
        block.body.push(Instruction::binary(
            spirv::Op::INotEqual,
            self.get_bool_type_id(),
            not_none_comp_id,
            kind_id,
            none_id,
        ));

        let not_none_label_id = self.id_gen.next();
        let mut not_none_block = Block::new(not_none_label_id);

        let final_label_id = self.id_gen.next();
        let mut final_block = Block::new(final_label_id);

        block.body.push(Instruction::selection_merge(
            final_label_id,
            spirv::SelectionControl::NONE,
        ));
        function.consume(
            block,
            Instruction::branch_conditional(not_none_comp_id, not_none_label_id, final_label_id),
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
        function.consume(merge_block, Instruction::branch(final_label_id));

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
        Instruction::function_end().to_words(&mut self.logical_layout.function_definitions);
        self.ray_get_intersection_function = Some(func_id);
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
}
