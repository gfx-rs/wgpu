use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

use crate::{
    back::{
        self,
        msl::{
            writer::{TypeContext, TypedGlobalVariable},
            BackendResult, EntryPointArgument, Error, NAMESPACE, WRAPPED_ARRAY_FIELD,
        },
    },
    proc::NameKey,
};

pub(super) struct MeshOutputInfo {
    out_vertex_ty_name: String,
    out_primitive_ty_name: String,
    out_vertex_member_names: Vec<Option<String>>,
    out_primitive_member_names: Vec<Option<String>>,
}

pub(super) struct NestedFunctionInfo<'a> {
    pub(super) options: &'a super::Options,
    pub(super) ep: &'a crate::EntryPoint,
    pub(super) module: &'a crate::Module,
    pub(super) mod_info: &'a crate::valid::ModuleInfo,
    pub(super) fun_info: &'a crate::valid::FunctionInfo,
    pub(super) args: Vec<EntryPointArgument>,
    pub(super) local_invocation_index: Option<&'a NameKey>,
    pub(super) nested_name: &'a str,
    pub(super) outer_name: &'a str,
    pub(super) out_mesh_info: Option<MeshOutputInfo>,
}

impl<W: core::fmt::Write> super::Writer<W> {
    /// This writes the output vertex and primitive structs given the reflection information about them.
    pub(super) fn write_mesh_output_types(
        &mut self,
        mesh_info: &crate::MeshStageInfo,
        fun_name: &str,
        module: &crate::Module,
        // See `PipelineOptions::allow_and_force_point_size`
        allow_and_force_point_size: bool,
        options: &super::Options,
    ) -> Result<MeshOutputInfo, Error> {
        let mut vertex_member_names = Vec::new();
        let mut primitive_member_names = Vec::new();
        let vertex_out_name = self.namer.call(&format!("{fun_name}VertexOutput"));
        let primitive_out_name = self.namer.call(&format!("{fun_name}PrimitiveOutput"));
        let mut existing_names = Vec::new();
        for (out_name, struct_ty, is_primitive, member_names) in [
            (
                &vertex_out_name,
                mesh_info.vertex_output_type,
                false,
                &mut vertex_member_names,
            ),
            (
                &primitive_out_name,
                mesh_info.primitive_output_type,
                true,
                &mut primitive_member_names,
            ),
        ] {
            writeln!(self.out, "struct {out_name} {{")?;
            // Mesh output types are guaranteed to be user defined structs. This is validated by naga.
            let crate::TypeInner::Struct { ref members, .. } = module.types[struct_ty].inner else {
                unreachable!()
            };
            let mut has_point_size = false;
            for (index, member) in members.iter().enumerate() {
                member_names.push(None);
                let ty_name = TypeContext {
                    handle: member.ty,
                    gctx: module.to_ctx(),
                    names: &self.names,
                    access: crate::StorageAccess::empty(),
                    first_time: true,
                };
                let binding = member
                    .binding
                    .clone()
                    .ok_or_else(|| Error::GenericValidation("Expected binding, got None".into()))?;

                if let crate::Binding::BuiltIn(crate::BuiltIn::PointSize) = binding {
                    has_point_size = true;
                    if !allow_and_force_point_size {
                        continue;
                    }
                }
                if let crate::Binding::BuiltIn(
                    crate::BuiltIn::PointIndex
                    | crate::BuiltIn::LineIndices
                    | crate::BuiltIn::TriangleIndices,
                ) = binding
                {
                    continue;
                }

                // Names of struct members must be unique across vertex and primitive output.
                // Therefore, when writing the primitive output struct, we might need to rename some fields.
                let mut name = self.names[&NameKey::StructMember(struct_ty, index as u32)].clone();
                if existing_names.contains(&name) {
                    name = self.namer.call(&name);
                } else {
                    // Let the namer know this is illegal to use again
                    let _ = self.namer.call(&name);
                }

                let array_len = match module.types[member.ty].inner {
                    crate::TypeInner::Array {
                        size: crate::ArraySize::Constant(size),
                        ..
                    } => Some(size),
                    _ => None,
                };
                let resolved =
                    options.resolve_local_binding(&binding, back::msl::LocationMode::MeshOutput)?;
                write!(self.out, "{}{} {}", back::INDENT, ty_name, name)?;
                if let Some(array_len) = array_len {
                    write!(self.out, " [{array_len}]")?;
                }
                resolved.try_fmt(&mut self.out)?;
                writeln!(self.out, ";")?;
                *member_names.last_mut().unwrap() = Some(name.clone());
                existing_names.push(name);
            }
            if allow_and_force_point_size && !has_point_size && !is_primitive {
                // inject the point size output last
                writeln!(
                    self.out,
                    "{}float _point_size [[point_size]];",
                    back::INDENT
                )?;
            }
            writeln!(self.out, "}};")?;
        }
        Ok(MeshOutputInfo {
            out_vertex_ty_name: vertex_out_name,
            out_primitive_ty_name: primitive_out_name,
            out_vertex_member_names: vertex_member_names,
            out_primitive_member_names: primitive_member_names,
        })
    }

    pub(super) fn write_wrapper_function(&mut self, info: NestedFunctionInfo<'_>) -> BackendResult {
        let NestedFunctionInfo {
            options,
            ep,
            module,
            mod_info,
            fun_info,
            args,
            local_invocation_index: local_invocation_index_key,
            nested_name,
            outer_name,
            out_mesh_info,
        } = info;
        let indent = back::INDENT;

        let em_str = match ep.stage {
            crate::ShaderStage::Mesh => "[[mesh]]",
            crate::ShaderStage::Task => "[[object]]",
            _ => unreachable!(),
        };
        writeln!(self.out, "{em_str} void {outer_name}(")?;

        // Arguments

        let mut mesh_out_name: Option<String> = None;
        let mut mesh_variable_name = None;
        let mut task_grid_name = None;
        if let Some(ref info) = ep.mesh_info {
            let mesh_out = out_mesh_info.as_ref().unwrap();
            let mesh_name = self.namer.call("meshOutput");
            let topology_name = match info.topology {
                crate::MeshOutputTopology::Points => "point",
                crate::MeshOutputTopology::Lines => "line",
                crate::MeshOutputTopology::Triangles => "triangle",
            };
            let num_verts = info.max_vertices;
            let num_prims = info.max_primitives;
            writeln!(self.out,
                "  {NAMESPACE}::mesh<{}, {}, {num_verts}, {num_prims}, metal::topology::{topology_name}> {mesh_name}",
                mesh_out.out_vertex_ty_name,
                mesh_out.out_primitive_ty_name,
            )?;
            mesh_out_name = Some(mesh_name);
            mesh_variable_name = Some(
                self.names
                    [&NameKey::GlobalVariable(ep.mesh_info.as_ref().unwrap().output_variable)]
                    .clone(),
            );
        } else if ep.stage == crate::ShaderStage::Task {
            let grid_name = self.namer.call("nagaMeshGrid");
            writeln!(self.out, "  {NAMESPACE}::mesh_grid_properties {grid_name}")?;
            task_grid_name = Some(grid_name);
        }
        let local_invocation_index = if let Some(key) = local_invocation_index_key {
            self.names[key].clone()
        } else {
            "__local_invocation_index".to_string()
        };

        for arg in &args {
            write!(self.out, ", {} {}{}", arg.ty_name, arg.name, arg.binding)?;
            if let Some(init) = arg.init {
                write!(self.out, " = ")?;
                self.put_const_expression(init, module, mod_info, &module.global_expressions)?;
            }
            writeln!(self.out)?;
        }

        writeln!(self.out, ") {{")?;

        // Function body
        if ep.stage == crate::ShaderStage::Mesh || ep.stage == crate::ShaderStage::Task {
            for (handle, var) in module.global_variables.iter() {
                if var.space != crate::AddressSpace::WorkGroup || fun_info[handle].is_empty() {
                    continue;
                }
                let tyvar = TypedGlobalVariable {
                    module,
                    names: &self.names,
                    handle,
                    usage: crate::valid::GlobalUse::WRITE | crate::valid::GlobalUse::READ,
                    reference: false,
                };
                write!(self.out, "{}", back::INDENT)?;
                tyvar.try_fmt(&mut self.out)?;
                writeln!(self.out, ";")?;
            }
        }
        write!(self.out, "{indent}")?;
        let result_name = if ep.stage == crate::ShaderStage::Task {
            let name = self.namer.call("nagaGridSize");
            write!(self.out, "uint3 {} = ", name)?;
            Some(name)
        } else {
            None
        };
        write!(self.out, "{nested_name}(")?;
        {
            let mut is_first = true;
            for arg in &args {
                if !is_first {
                    write!(self.out, ", ")?;
                }
                is_first = false;
                write!(self.out, "{}", arg.name)?;
            }
            if ep.stage == crate::ShaderStage::Mesh || ep.stage == crate::ShaderStage::Task {
                for (handle, var) in module.global_variables.iter() {
                    if var.space != crate::AddressSpace::WorkGroup || fun_info[handle].is_empty() {
                        continue;
                    }
                    if !is_first {
                        write!(self.out, ", ")?;
                    }
                    let name = &self.names[&NameKey::GlobalVariable(handle)];
                    write!(self.out, "&{name}")?;
                }
            }
        }
        writeln!(self.out, ");")?;
        self.write_barrier(crate::Barrier::WORK_GROUP, back::Level(1))?;

        if let Some(grid_name) = task_grid_name {
            let result_name = result_name.unwrap();
            writeln!(self.out, "{indent}if ({local_invocation_index} == 0u) {{")?;
            {
                let level2 = back::Level(2);
                if let Some(limits) = options.task_dispatch_limits {
                    let level3 = back::Level(3);
                    let max_per_dim = limits.max_mesh_workgroups_per_dim;
                    let max_total = limits.max_mesh_workgroups_total;
                    writeln!(self.out, "{level2}if (")?;

                    writeln!(self.out, "{level3}{result_name}.x > {max_per_dim}u ||")?;
                    writeln!(self.out, "{level3}{result_name}.y > {max_per_dim}u ||")?;
                    writeln!(self.out, "{level3}{result_name}.z > {max_per_dim}u ||")?;
                    writeln!(
                        self.out,
                        "{level3}{NAMESPACE}::mulhi({result_name}.x, {result_name}.y) != 0u ||"
                    )?;
                    writeln!(
                        self.out,
                        "{level3}{NAMESPACE}::mulhi({result_name}.x * {result_name}.y, {result_name}.z) != 0u ||"
                    )?;
                    writeln!(self.out, "{level3}({result_name}.x * {result_name}.y * {result_name}.z) > {max_total}u")?;

                    writeln!(self.out, "{level2}) {{")?;
                    writeln!(self.out, "{level3}{result_name} = {NAMESPACE}::uint3(0u);")?;
                    writeln!(self.out, "{level2}}}")?;
                }
                writeln!(
                    self.out,
                    "{level2}{grid_name}.set_threadgroups_per_grid({result_name});"
                )?;
            }
            writeln!(self.out, "{indent}}}")?;
            writeln!(self.out, "{indent}return;")?;
        } else if let Some(ref info) = ep.mesh_info {
            let mesh_out = out_mesh_info.as_ref().unwrap();
            let out_ty = module.global_variables[info.output_variable].ty;
            let mesh_out_name = mesh_out_name.unwrap();
            let mesh_variable_name = mesh_variable_name.unwrap();
            // The output type is guaranteed to be a struct with exactly 4 members
            let crate::TypeInner::Struct { ref members, .. } = module.types[out_ty].inner else {
                unreachable!();
            };
            let get_out_value = |bi| {
                let member_idx = members
                    .iter()
                    .position(|a| a.binding == Some(crate::Binding::BuiltIn(bi)))
                    .unwrap() as u32;
                format!(
                    "{}.{}",
                    mesh_variable_name,
                    self.names[&NameKey::StructMember(out_ty, member_idx)]
                )
            };
            let vert_count = format!(
                "{NAMESPACE}::min({}, {}u)",
                get_out_value(crate::BuiltIn::VertexCount),
                info.max_vertices
            );
            let prim_count = format!(
                "{NAMESPACE}::min({}, {}u)",
                get_out_value(crate::BuiltIn::PrimitiveCount),
                info.max_primitives
            );
            let workgroup_size: u32 = ep.workgroup_size.iter().product();
            {
                let vert_index = self.namer.call("vertexIndex");
                let in_array = get_out_value(crate::BuiltIn::Vertices);
                writeln!(
                    self.out,
                    "{indent}for(uint {vert_index} = {local_invocation_index}; {vert_index} < {vert_count}; {vert_index} += {workgroup_size}) {{"
                )?;
                let out_vert = self.namer.call("vertex");
                writeln!(
                    self.out,
                    "{indent}{indent}{} {out_vert};",
                    mesh_out.out_vertex_ty_name,
                )?;
                for (member_idx, new_name) in mesh_out.out_vertex_member_names.iter().enumerate() {
                    let in_value = format!(
                        "{in_array}.{WRAPPED_ARRAY_FIELD}[{vert_index}].{}",
                        self.names
                            [&NameKey::StructMember(info.vertex_output_type, member_idx as u32)]
                    );
                    let out_value = format!("{out_vert}.{}", new_name.as_ref().unwrap());
                    writeln!(self.out, "{indent}{indent}{out_value} = {in_value};")?;
                }
                writeln!(
                    self.out,
                    "{indent}{indent}{}.set_vertex({vert_index}, {out_vert});",
                    mesh_out_name
                )?;
                writeln!(self.out, "{indent}}}")?;
            }
            {
                let prim_index = self.namer.call("primitiveIndex");
                let in_array = get_out_value(crate::BuiltIn::Primitives);
                writeln!(
                    self.out,
                    "{indent}for(uint {prim_index} = {local_invocation_index}; {prim_index} < {prim_count}; {prim_index} += {workgroup_size}) {{"
                )?;
                let out_prim = self.namer.call("primitive");
                writeln!(
                    self.out,
                    "{indent}{indent}{} {out_prim};",
                    mesh_out.out_primitive_ty_name
                )?;
                for (member_idx, new_name) in mesh_out.out_primitive_member_names.iter().enumerate()
                {
                    let in_value = format!(
                        "{in_array}.{WRAPPED_ARRAY_FIELD}[{prim_index}].{}",
                        self.names
                            [&NameKey::StructMember(info.primitive_output_type, member_idx as u32)]
                    );
                    if let Some(new_name) = new_name.as_ref() {
                        let out_value = format!("{out_prim}.{new_name}");
                        writeln!(
                            self.out,
                            "{indent}{}{out_value} = {in_value};",
                            back::INDENT
                        )?;
                    } else {
                        let num_indices = match info.topology {
                            crate::MeshOutputTopology::Points => 1,
                            crate::MeshOutputTopology::Lines => 2,
                            crate::MeshOutputTopology::Triangles => 3,
                        };
                        for i in 0..num_indices {
                            let component = if num_indices == 1 {
                                "".to_string()
                            } else {
                                format!(".{}", back::COMPONENTS[i])
                            };
                            writeln!(
                                self.out,
                                "{indent}{}{}.set_index({prim_index} * {num_indices} + {i}, {in_value}{component});",
                                back::INDENT,
                                mesh_out_name,
                            )?;
                        }
                    }
                }
                writeln!(
                    self.out,
                    "{indent}{}{}.set_primitive({prim_index}, {out_prim});",
                    back::INDENT,
                    mesh_out_name
                )?;
                writeln!(self.out, "{indent}}}")?;
            }

            writeln!(self.out, "{indent}if ({local_invocation_index} == 0u) {{")?;
            writeln!(
                self.out,
                "{indent}{indent}{}.set_primitive_count({prim_count});",
                mesh_out_name,
            )?;
            writeln!(self.out, "{indent}}}")?;
        } else {
            // Must either have task output grid (task shader) or mesh output info (mesh shader)
            unreachable!()
        }

        writeln!(self.out, "}}")?;
        Ok(())
    }
}
