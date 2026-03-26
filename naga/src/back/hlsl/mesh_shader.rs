use core::fmt;

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    back::{
        self,
        hlsl::{
            writer::{EntryPointBinding, EpStructMember, Io, NestedEntryPointArgs},
            BackendResult, Error,
        },
    },
    proc::NameKey,
    Handle, Module, ShaderStage, TypeInner,
};

impl NestedEntryPointArgs {
    pub fn write_call_args(&self, out: &mut impl fmt::Write) -> fmt::Result {
        let all_args = self
            .user_args
            .iter()
            .map(String::as_str)
            .chain(self.task_payload.as_deref())
            .chain(core::iter::once(self.local_invocation_index.as_str()));
        for (i, arg) in all_args.enumerate() {
            if i != 0 {
                write!(out, ", ")?;
            }
            write!(out, "{arg}")?;
        }
        Ok(())
    }
}

impl<W: fmt::Write> super::Writer<'_, W> {
    #[expect(clippy::too_many_arguments)]
    fn write_mesh_shader_wrapper(
        &mut self,
        module: &Module,
        func_ctx: &back::FunctionCtx,
        need_workgroup_variables_initialization: bool,
        nested_name: &str,
        entry_point: &crate::EntryPoint,
        args: NestedEntryPointArgs,
        mut separator_if_needed: impl FnMut() -> &'static str,
    ) -> BackendResult {
        let Some(ref mesh_info) = entry_point.mesh_info else {
            unreachable!()
        };
        let back::FunctionType::EntryPoint(ep_index) = func_ctx.ty else {
            unreachable!()
        };
        // Mesh shader wrapper
        let mesh_interface = self.entry_point_io.get(&(ep_index as usize)).unwrap();
        let vert_info = mesh_interface.mesh_vertices.as_ref().unwrap();
        let prim_info = mesh_interface.mesh_primitives.as_ref().unwrap();
        let indices_info = mesh_interface.mesh_indices.as_ref().unwrap();
        // Write something of the form `out indices uint3 indices_var[num_primitives]`
        write!(
            self.out,
            "{}out indices {} {}[{}]",
            separator_if_needed(),
            indices_info.ty_name,
            indices_info.arg_name,
            mesh_info.max_primitives
        )?;
        // Write something of the form `out vertices VertexType vertices_var[num_vertices]`
        write!(
            self.out,
            ", out vertices {} {}[{}]",
            vert_info.ty_name, vert_info.arg_name, mesh_info.max_vertices
        )?;
        // Write something of the form `out primitives PrimitiveType} primitives_var[num_primitives]`
        write!(
            self.out,
            ", out primitives {} {}[{}]",
            prim_info.ty_name, prim_info.arg_name, mesh_info.max_primitives
        )?;
        if let Some(task_payload) = entry_point.task_payload {
            // Write the outer-function `in payload` arg.  The name is already in
            // args.task_payload, having been collected when the inner function
            // signature was written in write_function (writer.rs).
            write!(self.out, ", in payload ")?;
            let var = &module.global_variables[task_payload];
            self.write_type(module, var.ty)?;
            let name = &self.names[&NameKey::GlobalVariable(task_payload)];
            write!(self.out, " {name}")?;
            if let TypeInner::Array { base, size, .. } = module.types[var.ty].inner {
                self.write_array_size(module, base, size)?;
            }
        }
        writeln!(self.out, ") {{")?;
        if need_workgroup_variables_initialization {
            writeln!(
                self.out,
                "{}if ({} == 0) {{",
                back::INDENT,
                args.local_invocation_index,
            )?;
            self.write_workgroup_variables_initialization(
                func_ctx,
                module,
                module.entry_points[ep_index as usize].stage,
            )?;
            writeln!(self.out, "{}}}", back::INDENT)?;
            self.write_control_barrier(crate::Barrier::WORK_GROUP, back::Level(1))?;
        }
        write!(self.out, "{}{nested_name}(", back::INDENT)?;
        args.write_call_args(&mut self.out)?;
        writeln!(self.out, ");")?;
        writeln!(
            self.out,
            "{}GroupMemoryBarrierWithGroupSync();",
            back::INDENT
        )?;

        let ep = &module.entry_points[ep_index as usize];
        let mesh_info = ep.mesh_info.as_ref().unwrap();
        let io = self.entry_point_io.get(&(ep_index as usize)).unwrap();

        let var_name = &self.names[&NameKey::GlobalVariable(mesh_info.output_variable)];
        let var_type = module.global_variables[mesh_info.output_variable].ty;
        let wg_size: u32 = ep.workgroup_size.iter().product();

        let get_var_member_name = |bi, var_type| {
            // The mesh shader output type must be a struct with exactly 4 members.
            let TypeInner::Struct { ref members, .. } = module.types[var_type].inner else {
                unreachable!()
            };
            let idx = members
                .iter()
                .position(|f| f.binding == Some(crate::Binding::BuiltIn(bi)))
                .unwrap();
            self.names[&NameKey::StructMember(var_type, idx as u32)].clone()
        };

        let vert_count = format!(
            "{var_name}.{}",
            get_var_member_name(crate::BuiltIn::VertexCount, var_type),
        );
        let prim_count = format!(
            "{var_name}.{}",
            get_var_member_name(crate::BuiltIn::PrimitiveCount, var_type),
        );

        let level = back::Level(1);

        writeln!(
            self.out,
            "{level}SetMeshOutputCounts({vert_count}, {prim_count});"
        )?;

        // We need separate loops for vertices and primitives writing
        struct OutputArray<'a> {
            array_bi: crate::BuiltIn,
            count: String,
            io_interface: &'a EntryPointBinding,
            is_primitive: bool,
            index_name: &'static str,
            ty: Handle<crate::Type>,
        }
        let output_arrays = [
            OutputArray {
                array_bi: crate::BuiltIn::Vertices,
                count: vert_count,
                io_interface: io.mesh_vertices.as_ref().unwrap(),
                is_primitive: false,
                index_name: "vertIndex",
                ty: mesh_info.vertex_output_type,
            },
            OutputArray {
                array_bi: crate::BuiltIn::Primitives,
                count: prim_count,
                io_interface: io.mesh_primitives.as_ref().unwrap(),
                is_primitive: true,
                index_name: "primIndex",
                ty: mesh_info.primitive_output_type,
            },
        ];

        for output in output_arrays {
            let OutputArray {
                array_bi,
                count,
                io_interface,
                is_primitive,
                index_name,
                ty,
            } = output;
            let out_var_name = &io_interface.arg_name;
            let index_name = self.namer.call(index_name);
            let array_name = get_var_member_name(array_bi, var_type);
            let item_name = format!("{var_name}.{array_name}[{index_name}]");
            writeln!(
                self.out,
                "{level}for (int {index_name} = {}; {index_name} < {count}; {index_name} += {}) {{",
                args.local_invocation_index, wg_size
            )?;

            // Loop body, uses more indentation
            {
                let level = level.next();
                for member in &io_interface.members {
                    let out_member_name = &member.name;
                    let in_member_name = &self.names[&NameKey::StructMember(ty, member.index)];
                    writeln!(self.out, "{level}{out_var_name}[{index_name}].{out_member_name} = {item_name}.{in_member_name};",)?;
                }
                if is_primitive {
                    let indices_member_name = get_var_member_name(
                        mesh_info.topology.to_builtin(),
                        mesh_info.primitive_output_type,
                    );
                    let indices_var_name = &io.mesh_indices.as_ref().unwrap().arg_name;
                    writeln!(
                                self.out,
                                "{level}{indices_var_name}[{index_name}] = {item_name}.{indices_member_name};",
                            )?;
                }
            }

            writeln!(self.out, "{level}}}")?;
        }
        Ok(())
    }

    fn write_task_shader_wrapper(
        &mut self,
        module: &Module,
        func_ctx: &back::FunctionCtx,
        need_workgroup_variables_initialization: bool,
        nested_name: &str,
        entry_point: &crate::EntryPoint,
        args: NestedEntryPointArgs,
    ) -> BackendResult {
        let back::FunctionType::EntryPoint(ep_index) = func_ctx.ty else {
            unreachable!()
        };
        // Task shader wrapper
        writeln!(self.out, ") {{")?;
        if need_workgroup_variables_initialization {
            writeln!(
                self.out,
                "{}if ({} == 0) {{",
                back::INDENT,
                args.local_invocation_index,
            )?;
            self.write_workgroup_variables_initialization(
                func_ctx,
                module,
                module.entry_points[ep_index as usize].stage,
            )?;
            writeln!(self.out, "{}}}", back::INDENT)?;
            self.write_control_barrier(crate::Barrier::WORK_GROUP, back::Level(1))?;
        }
        let grid_size = self.namer.call("gridSize");
        write!(
            self.out,
            "{}uint3 {grid_size} = {nested_name}(",
            back::INDENT
        )?;
        args.write_call_args(&mut self.out)?;
        writeln!(self.out, ");")?;
        writeln!(
            self.out,
            "{}GroupMemoryBarrierWithGroupSync();",
            back::INDENT
        )?;
        if let Some(limits) = self.options.task_dispatch_limits {
            let level = back::Level(2);
            writeln!(self.out, "{}if (", back::INDENT)?;

            let max_per_dim = limits.max_mesh_workgroups_per_dim.min(2 << 21);
            let max_total = limits.max_mesh_workgroups_total;
            for i in 0..3 {
                writeln!(
                    self.out,
                    "{level}{grid_size}.{} > {max_per_dim} ||",
                    back::COMPONENTS[i],
                )?;
            }
            writeln!(
                self.out,
                "{level}((uint64_t){grid_size}.x) * ((uint64_t){grid_size}.y) > 0xffffffffull ||"
            )?;
            writeln!(
                    self.out,
                    "{level}((uint64_t){grid_size}.x) * ((uint64_t){grid_size}.y) * ((uint64_t){grid_size}.z) > {max_total}",
                )?;

            writeln!(self.out, "{}) {{", back::INDENT)?;
            writeln!(self.out, "{level}{grid_size} = uint3(0, 0, 0);")?;
            writeln!(self.out, "{}}}", back::INDENT)?;
        }
        writeln!(
            self.out,
            "{}DispatchMesh({grid_size}.x, {grid_size}.y, {grid_size}.z, {});",
            back::INDENT,
            self.names[&NameKey::GlobalVariable(entry_point.task_payload.unwrap())]
        )?;
        Ok(())
    }
    /// Mesh and task entry points must all return at the same `return` statement,
    /// so we have a nested function that can return wherever. This writes the caller,
    /// or the actual entry point.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn write_nested_function_outer(
        &mut self,
        module: &Module,
        func_ctx: &back::FunctionCtx,
        header: &str,
        name: &str,
        need_workgroup_variables_initialization: bool,
        nested_name: &str,
        entry_point: &crate::EntryPoint,
        // Built in write_function alongside the inner function signature, so the
        // call-site argument order is guaranteed to match the declaration order.
        args: NestedEntryPointArgs,
    ) -> BackendResult {
        let mut any_args_written = false;
        let mut separator_if_needed = || {
            if any_args_written {
                ", "
            } else {
                any_args_written = true;
                ""
            }
        };

        let back::FunctionType::EntryPoint(ep_index) = func_ctx.ty else {
            unreachable!();
        };
        let stage = module.entry_points[ep_index as usize].stage;
        write!(self.out, "{header}")?;
        write!(self.out, "void {name}(")?;
        // Write the outer function's argument list with full type annotations and
        // semantics.  Arg names come from self.names and are the same names that
        // were collected into `args` when writing the inner function signature.
        if let Some(ref ep_input) = self.entry_point_io.get(&(ep_index as usize)).unwrap().input {
            write!(self.out, "{} {}", ep_input.ty_name, ep_input.arg_name)?;
        } else {
            for (index, arg) in entry_point.function.arguments.iter().enumerate() {
                write!(self.out, "{}", separator_if_needed())?;
                self.write_type(module, arg.ty)?;

                let argument_name =
                    &self.names[&NameKey::EntryPointArgument(ep_index, index as u32)];

                write!(self.out, " {argument_name}")?;
                if let TypeInner::Array { base, size, .. } = module.types[arg.ty].inner {
                    self.write_array_size(module, base, size)?;
                }

                self.write_semantic(&arg.binding, Some((stage, Io::Input)))?;
            }
        }
        if need_workgroup_variables_initialization || stage == ShaderStage::Mesh {
            write!(
                self.out,
                "{}uint {} : SV_GroupIndex",
                separator_if_needed(),
                args.local_invocation_index,
            )?;
        }
        if entry_point.stage == ShaderStage::Mesh {
            self.write_mesh_shader_wrapper(
                module,
                func_ctx,
                need_workgroup_variables_initialization,
                nested_name,
                entry_point,
                args,
                separator_if_needed,
            )?;
        } else {
            self.write_task_shader_wrapper(
                module,
                func_ctx,
                need_workgroup_variables_initialization,
                nested_name,
                entry_point,
                args,
            )?;
        }

        writeln!(self.out, "}}")?;
        Ok(())
    }

    pub(super) fn write_ep_mesh_output_struct(
        &mut self,
        module: &Module,
        entry_point_name: &str,
        is_primitive: bool,
        mesh_info: &crate::MeshStageInfo,
    ) -> Result<EntryPointBinding, Error> {
        let (in_type, io, var_prefix, arg_name) = if is_primitive {
            (
                mesh_info.primitive_output_type,
                Io::MeshPrimitives,
                "Primitive",
                "primitives",
            )
        } else {
            (
                mesh_info.vertex_output_type,
                Io::MeshVertices,
                "Vertex",
                "vertices",
            )
        };
        let struct_name = format!("Mesh{var_prefix}Output_{entry_point_name}",);

        // Mesh shader output types must be structs; this is validated by naga
        let members = match module.types[in_type].inner {
            TypeInner::Struct { ref members, .. } => members,
            _ => unreachable!(),
        };
        let mut out_members = Vec::new();
        for (index, member) in members.iter().enumerate() {
            if matches!(
                member.binding,
                Some(crate::Binding::BuiltIn(
                    crate::BuiltIn::PointIndex
                        | crate::BuiltIn::LineIndices
                        | crate::BuiltIn::TriangleIndices
                ))
            ) {
                continue;
            }
            let member_name = self.namer.call_or(&member.name, "member");
            out_members.push(EpStructMember {
                name: member_name,
                ty: member.ty,
                binding: member.binding.clone(),
                index: index as u32,
            })
        }
        self.write_interface_struct(
            module,
            (ShaderStage::Mesh, io),
            struct_name,
            Some(arg_name),
            out_members,
        )
    }

    pub(super) fn write_ep_mesh_output_indices(
        &mut self,
        topology: crate::MeshOutputTopology,
    ) -> Result<EntryPointBinding, Error> {
        let (indices_name, indices_type) = match topology {
            // Points require a capability that isn't supported in the HLSL writer
            crate::MeshOutputTopology::Points => unreachable!(),
            crate::MeshOutputTopology::Lines => (self.namer.call("lineIndices"), "uint2"),
            crate::MeshOutputTopology::Triangles => (self.namer.call("triangleIndices"), "uint3"),
        };
        Ok(EntryPointBinding {
            ty_name: indices_type.to_string(),
            arg_name: indices_name,
            members: Vec::new(),
            local_invocation_index_name: None,
        })
    }
}
