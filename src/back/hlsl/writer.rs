use super::{
    help::{MipLevelCoordinate, WrappedArrayLength, WrappedImageQuery},
    storage::StoreValue,
    BackendResult, Error, Options,
};
use crate::{
    back,
    proc::{self, NameKey},
    valid, Handle, Module, ShaderStage, TypeInner,
};
use std::fmt;

const LOCATION_SEMANTIC: &str = "LOC";
const SPECIAL_CBUF_TYPE: &str = "NagaConstants";
const SPECIAL_CBUF_VAR: &str = "_NagaConstants";
const SPECIAL_BASE_VERTEX: &str = "base_vertex";
const SPECIAL_BASE_INSTANCE: &str = "base_instance";

/// Structure contains information required for generating
/// wrapped structure of all entry points arguments
pub(super) struct EntryPointBinding {
    /// Generated structure name
    name: String,
    /// Members of generated structure
    members: Vec<EpStructMember>,
}

struct EpStructMember {
    name: String,
    ty: Handle<crate::Type>,
    binding: Option<crate::Binding>,
}

#[derive(Copy, Clone, PartialEq)]
enum Io {
    Input,
    Output,
}

impl<'a, W: fmt::Write> super::Writer<'a, W> {
    pub fn new(out: W, options: &'a Options) -> Self {
        Self {
            out,
            names: crate::FastHashMap::default(),
            namer: proc::Namer::default(),
            options,
            ep_inputs: Vec::new(),
            named_expressions: crate::NamedExpressions::default(),
            wrapped_array_lengths: crate::FastHashSet::default(),
            wrapped_image_queries: crate::FastHashSet::default(),
            temp_access_chain: Vec::new(),
        }
    }

    fn reset(&mut self, module: &Module) {
        self.names.clear();
        self.namer
            .reset(module, super::keywords::RESERVED, &[], &mut self.names);
        self.ep_inputs.clear();
        self.named_expressions.clear();
        self.wrapped_array_lengths.clear();
        self.wrapped_image_queries.clear();
    }

    pub fn write(
        &mut self,
        module: &Module,
        module_info: &valid::ModuleInfo,
    ) -> Result<super::ReflectionInfo, Error> {
        self.reset(module);

        // Write special constants, if needed
        if let Some(ref bt) = self.options.special_constants_binding {
            writeln!(self.out, "struct {} {{", SPECIAL_CBUF_TYPE)?;
            writeln!(self.out, "{}int {};", back::INDENT, SPECIAL_BASE_VERTEX)?;
            writeln!(self.out, "{}int {};", back::INDENT, SPECIAL_BASE_INSTANCE)?;
            writeln!(self.out, "}};")?;
            write!(
                self.out,
                "ConstantBuffer<{}> {}: register(b{}",
                SPECIAL_CBUF_TYPE, SPECIAL_CBUF_VAR, bt.register
            )?;
            if bt.space != 0 {
                write!(self.out, ", space{}", bt.space)?;
            }
            writeln!(self.out, ");")?;
        }

        // Write all constants
        // For example, input wgsl shader:
        // ```wgsl
        // let c_scale: f32 = 1.2;
        // return VertexOutput(uv, vec4<f32>(c_scale * pos, 0.0, 1.0));
        // ```
        //
        // Output shader:
        // ```hlsl
        // static const float c_scale = 1.2;
        // const VertexOutput vertexoutput1 = { vertexinput.uv3, float4((c_scale * vertexinput.pos1), 0.0, 1.0) };
        // ```
        //
        // If we remove `write_global_constant` `c_scale` will be inlined.
        for (handle, constant) in module.constants.iter() {
            if constant.name.is_some() {
                self.write_global_constant(module, &constant.inner, handle)?;
            }
        }

        // Extra newline for readability
        writeln!(self.out)?;

        // Save all entry point output types
        let ep_results = module
            .entry_points
            .iter()
            .map(|ep| (ep.stage, ep.function.result.clone()))
            .collect::<Vec<(ShaderStage, Option<crate::FunctionResult>)>>();

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct {
                top_level,
                ref members,
                ..
            } = ty.inner
            {
                if let Some(member) = members.last() {
                    if let TypeInner::Array {
                        size: crate::ArraySize::Dynamic,
                        ..
                    } = module.types[member.ty].inner
                    {
                        // unsized arrays can only be in storage buffers, for which we use `ByteAddressBuffer` anyway.
                        continue;
                    }
                }

                let ep_result = ep_results.iter().find(|e| {
                    if let Some(ref result) = e.1 {
                        result.ty == handle
                    } else {
                        false
                    }
                });

                self.write_struct(
                    module,
                    handle,
                    top_level,
                    members,
                    ep_result.map(|r| (r.0, Io::Output)),
                )?;
                writeln!(self.out)?;
            }
        }

        // Write all globals
        for (ty, _) in module.global_variables.iter() {
            self.write_global(module, ty)?;
        }

        if !module.global_variables.is_empty() {
            // Add extra newline for readability
            writeln!(self.out)?;
        }

        // Write all entry points wrapped structs
        for ep in module.entry_points.iter() {
            self.write_ep_input_struct(module, &ep.function, ep.stage, &ep.name)?;
        }

        // Write all regular functions
        for (handle, function) in module.functions.iter() {
            let info = &module_info[handle];

            // Check if all of the globals are accessible
            if !self.options.fake_missing_bindings {
                if let Some((var_handle, _)) =
                    module
                        .global_variables
                        .iter()
                        .find(|&(var_handle, var)| match var.binding {
                            Some(ref binding) if !info[var_handle].is_empty() => {
                                self.options.resolve_resource_binding(binding).is_err()
                            }
                            _ => false,
                        })
                {
                    log::info!(
                        "Skipping function {:?} (name {:?}) because global {:?} is inaccessible",
                        handle,
                        function.name,
                        var_handle
                    );
                    continue;
                }
            }

            let ctx = back::FunctionCtx {
                ty: back::FunctionType::Function(handle),
                info,
                expressions: &function.expressions,
                named_expressions: &function.named_expressions,
            };
            let name = self.names[&NameKey::Function(handle)].clone();

            // Write wrapped function for `Expression::ImageQuery` and `Expressions::ArrayLength`
            // before writing all statements and expressions.
            self.write_wrapped_functions(module, &ctx)?;

            self.write_function(module, name.as_str(), function, &ctx)?;

            writeln!(self.out)?;
        }

        let mut entry_point_names = Vec::with_capacity(module.entry_points.len());

        // Write all entry points
        for (index, ep) in module.entry_points.iter().enumerate() {
            let info = module_info.get_entry_point(index);

            if !self.options.fake_missing_bindings {
                let mut ep_error = None;
                for (var_handle, var) in module.global_variables.iter() {
                    match var.binding {
                        Some(ref binding) if !info[var_handle].is_empty() => {
                            if let Err(err) = self.options.resolve_resource_binding(binding) {
                                ep_error = Some(err);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(err) = ep_error {
                    entry_point_names.push(Err(err));
                    continue;
                }
            }

            let ctx = back::FunctionCtx {
                ty: back::FunctionType::EntryPoint(index as u16),
                info,
                expressions: &ep.function.expressions,
                named_expressions: &ep.function.named_expressions,
            };

            // Write wrapped function for `Expression::ImageQuery` and `Expressions::ArrayLength`
            // before writing all statements and expressions.
            self.write_wrapped_functions(module, &ctx)?;

            if ep.stage == ShaderStage::Compute {
                // HLSL is calling workgroup size "num threads"
                let num_threads = ep.workgroup_size;
                writeln!(
                    self.out,
                    "[numthreads({}, {}, {})]",
                    num_threads[0], num_threads[1], num_threads[2]
                )?;
            }

            let name = self.names[&NameKey::EntryPoint(index as u16)].clone();
            self.write_function(module, &name, &ep.function, &ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(self.out)?;
            }

            entry_point_names.push(Ok(name));
        }

        Ok(super::ReflectionInfo { entry_point_names })
    }

    fn write_semantic(
        &mut self,
        binding: &crate::Binding,
        stage: Option<(ShaderStage, Io)>,
    ) -> BackendResult {
        match *binding {
            crate::Binding::BuiltIn(builtin) => {
                let builtin_str = builtin.to_hlsl_str()?;
                write!(self.out, " : {}", builtin_str)?;
            }
            crate::Binding::Location { location, .. } => {
                if stage == Some((crate::ShaderStage::Fragment, Io::Output)) {
                    write!(self.out, " : SV_Target{}", location)?;
                } else {
                    write!(self.out, " : {}{}", LOCATION_SEMANTIC, location)?;
                }
            }
        }

        Ok(())
    }

    fn write_ep_input_struct(
        &mut self,
        module: &Module,
        func: &crate::Function,
        stage: ShaderStage,
        entry_point_name: &str,
    ) -> BackendResult {
        if !func.arguments.is_empty() {
            let struct_name_prefix = match stage {
                ShaderStage::Vertex => "VertexInput",
                ShaderStage::Fragment => "FragmentInput",
                ShaderStage::Compute => "ComputeInput",
            };
            let struct_name = format!("{}_{}", struct_name_prefix, entry_point_name);

            let mut members = Vec::with_capacity(func.arguments.len());

            write!(self.out, "struct {}", &struct_name)?;
            writeln!(self.out, " {{")?;

            for arg in func.arguments.iter() {
                let member_name = if let Some(ref name) = arg.name {
                    name
                } else {
                    "member"
                };
                let member = EpStructMember {
                    name: self.namer.call_unique(member_name),
                    ty: arg.ty,
                    binding: arg.binding.clone(),
                };

                write!(self.out, "{}", back::INDENT)?;
                self.write_type(module, member.ty)?;
                write!(self.out, " {}", &member.name)?;
                if let Some(ref binding) = member.binding {
                    self.write_semantic(binding, Some((stage, Io::Input)))?;
                }
                write!(self.out, ";")?;
                writeln!(self.out)?;

                members.push(member);
            }

            writeln!(self.out, "}};")?;
            writeln!(self.out)?;

            let ep_input = EntryPointBinding {
                name: struct_name,
                members,
            };

            self.ep_inputs.push(Some(ep_input));
        } else {
            self.ep_inputs.push(None);
        }

        Ok(())
    }

    /// Helper method used to write global variables
    /// # Notes
    /// Always adds a newline
    fn write_global(
        &mut self,
        module: &Module,
        handle: Handle<crate::GlobalVariable>,
    ) -> BackendResult {
        let global = &module.global_variables[handle];
        let inner = &module.types[global.ty].inner;

        if let Some(ref binding) = global.binding {
            if let Err(err) = self.options.resolve_resource_binding(binding) {
                log::info!(
                    "Skipping global {:?} (name {:?}) for being inaccessible: {}",
                    handle,
                    global.name,
                    err,
                );
                return Ok(());
            }
        }

        // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-variable-register
        let register_ty = match global.class {
            crate::StorageClass::Function => unreachable!("Function storage class"),
            crate::StorageClass::Private => {
                write!(self.out, "static ")?;
                self.write_type(module, global.ty)?;
                ""
            }
            crate::StorageClass::WorkGroup => {
                write!(self.out, "groupshared ")?;
                self.write_type(module, global.ty)?;
                ""
            }
            crate::StorageClass::Uniform => {
                // constant buffer declarations are expected to be inlined, e.g.
                // `cbuffer foo: register(b0) { field1: type1; }`
                write!(self.out, "cbuffer")?;
                "b"
            }
            crate::StorageClass::Storage => {
                let (prefix, register) =
                    if global.storage_access.contains(crate::StorageAccess::STORE) {
                        ("RW", "u")
                    } else {
                        ("", "t")
                    };
                write!(self.out, "{}ByteAddressBuffer", prefix)?;
                register
            }
            crate::StorageClass::Handle => {
                let register = if let TypeInner::Sampler { .. } = *inner {
                    "s"
                } else if global.storage_access.contains(crate::StorageAccess::STORE) {
                    write!(self.out, "RW")?;
                    "u"
                } else {
                    "t"
                };
                self.write_type(module, global.ty)?;
                register
            }
            crate::StorageClass::PushConstant => unimplemented!("Push constants"),
        };

        let name = &self.names[&NameKey::GlobalVariable(handle)];
        write!(self.out, " {}", name)?;
        if let TypeInner::Array { size, .. } = module.types[global.ty].inner {
            self.write_array_size(module, size)?;
        }

        if let Some(ref binding) = global.binding {
            // this was already resolved earlier when we started evaluating an entry point.
            let bt = self.options.resolve_resource_binding(binding).unwrap();
            write!(self.out, " : register({}{}", register_ty, bt.register)?;
            if bt.space != 0 {
                write!(self.out, ", space{}", bt.space)?;
            }
            write!(self.out, ")")?;
        } else if global.class == crate::StorageClass::Private {
            write!(self.out, " = ")?;
            if let Some(init) = global.init {
                self.write_constant(module, init)?;
            } else {
                self.write_default_init(module, global.ty)?;
            }
        }

        if global.class == crate::StorageClass::Uniform {
            write!(self.out, " {{ ")?;
            self.write_type(module, global.ty)?;
            let name = &self.names[&NameKey::GlobalVariable(handle)];
            writeln!(self.out, " {}; }}", name)?;
        } else {
            writeln!(self.out, ";")?;
        }

        Ok(())
    }

    /// Helper method used to write global constants
    ///
    /// # Notes
    /// Ends in a newline
    fn write_global_constant(
        &mut self,
        module: &Module,
        inner: &crate::ConstantInner,
        handle: Handle<crate::Constant>,
    ) -> BackendResult {
        write!(self.out, "static const ")?;
        match *inner {
            crate::ConstantInner::Scalar {
                width: _,
                ref value,
            } => {
                // Write type
                let ty_str = match *value {
                    crate::ScalarValue::Sint(_) => "int",
                    crate::ScalarValue::Uint(_) => "uint",
                    crate::ScalarValue::Float(_) => "float",
                    crate::ScalarValue::Bool(_) => "bool",
                };
                let name = &self.names[&NameKey::Constant(handle)];
                write!(self.out, "{} {} = ", ty_str, name)?;

                // Second match required to avoid heap allocation by `format!()`
                match *value {
                    crate::ScalarValue::Sint(value) => write!(self.out, "{}", value)?,
                    crate::ScalarValue::Uint(value) => write!(self.out, "{}", value)?,
                    crate::ScalarValue::Float(value) => {
                        // Floats are written using `Debug` instead of `Display` because it always appends the
                        // decimal part even it's zero
                        write!(self.out, "{:?}", value)?
                    }
                    crate::ScalarValue::Bool(value) => write!(self.out, "{}", value)?,
                };
            }
            crate::ConstantInner::Composite { ty, ref components } => {
                self.write_type(module, ty)?;
                let name = &self.names[&NameKey::Constant(handle)];
                write!(self.out, " {} = ", name)?;
                self.write_composite_constant(module, ty, components)?;
            }
        }
        writeln!(self.out, ";")?;
        Ok(())
    }

    pub(super) fn write_array_size(
        &mut self,
        module: &Module,
        size: crate::ArraySize,
    ) -> BackendResult {
        write!(self.out, "[")?;

        // Write the array size
        // Writes nothing if `ArraySize::Dynamic`
        // Panics if `ArraySize::Constant` has a constant that isn't an sint or uint
        match size {
            crate::ArraySize::Constant(const_handle) => {
                let size = module.constants[const_handle].to_array_length().unwrap();
                write!(self.out, "{}", size)?;
            }
            crate::ArraySize::Dynamic => unreachable!(),
        }

        write!(self.out, "]")?;
        Ok(())
    }

    /// Helper method used to write structs
    ///
    /// # Notes
    /// Ends in a newline
    fn write_struct(
        &mut self,
        module: &Module,
        handle: Handle<crate::Type>,
        _block: bool,
        members: &[crate::StructMember],
        shader_stage: Option<(ShaderStage, Io)>,
    ) -> BackendResult {
        // Write struct name
        write!(self.out, "struct {}", self.names[&NameKey::Type(handle)])?;
        writeln!(self.out, " {{")?;

        for (index, member) in members.iter().enumerate() {
            // The indentation is only for readability
            write!(self.out, "{}", back::INDENT)?;

            match module.types[member.ty].inner {
                TypeInner::Array {
                    base,
                    size,
                    stride: _,
                } => {
                    // HLSL arrays are written as `type name[size]`
                    let (ty_name, vec_size) = match module.types[base].inner {
                        // Write scalar type by backend so as not to depend on the front-end implementation
                        // Name returned from frontend can be generated (type1, float1, etc.)
                        TypeInner::Scalar { kind, width } => (kind.to_hlsl_str(width)?, None),
                        // Similarly, write vector types directly.
                        TypeInner::Vector { size, kind, width } => {
                            (kind.to_hlsl_str(width)?, Some(size))
                        }
                        _ => (self.names[&NameKey::Type(base)].as_str(), None),
                    };

                    // Write `type` and `name`
                    write!(self.out, "{}", ty_name)?;
                    if let Some(s) = vec_size {
                        write!(self.out, "{}", s as usize)?;
                    }
                    write!(
                        self.out,
                        " {}",
                        &self.names[&NameKey::StructMember(handle, index as u32)]
                    )?;
                    // Write [size]
                    self.write_array_size(module, size)?;
                }
                _ => {
                    // Write interpolation modifier before type
                    if let Some(crate::Binding::Location {
                        interpolation,
                        sampling,
                        ..
                    }) = member.binding
                    {
                        if let Some(interpolation) = interpolation {
                            write!(self.out, "{} ", interpolation.to_hlsl_str())?
                        }

                        if let Some(sampling) = sampling {
                            if let Some(string) = sampling.to_hlsl_str() {
                                write!(self.out, "{} ", string)?
                            }
                        }
                    }

                    // Write the member type and name
                    self.write_type(module, member.ty)?;
                    write!(
                        self.out,
                        " {}",
                        &self.names[&NameKey::StructMember(handle, index as u32)]
                    )?;
                }
            }

            if let Some(ref binding) = member.binding {
                self.write_semantic(binding, shader_stage)?;
            };
            write!(self.out, ";")?;
            writeln!(self.out)?;
        }

        writeln!(self.out, "}};")?;
        Ok(())
    }

    /// Helper method used to write non image/sampler types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_type(&mut self, module: &Module, ty: Handle<crate::Type>) -> BackendResult {
        let inner = &module.types[ty].inner;
        match *inner {
            TypeInner::Struct { .. } => write!(self.out, "{}", self.names[&NameKey::Type(ty)])?,
            // hlsl array has the size separated from the base type
            TypeInner::Array { base, .. } => self.write_type(module, base)?,
            ref other => self.write_value_type(module, other)?,
        }

        Ok(())
    }

    /// Helper method used to write value types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    pub(super) fn write_value_type(&mut self, module: &Module, inner: &TypeInner) -> BackendResult {
        match *inner {
            TypeInner::Scalar { kind, width } => {
                write!(self.out, "{}", kind.to_hlsl_str(width)?)?;
            }
            TypeInner::Vector { size, kind, width } => {
                write!(
                    self.out,
                    "{}{}",
                    kind.to_hlsl_str(width)?,
                    back::vector_size_str(size)
                )?;
            }
            TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                // The IR supports only float matrix
                // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-matrix
                write!(
                    self.out,
                    "{}{}x{}",
                    crate::ScalarKind::Float.to_hlsl_str(width)?,
                    back::vector_size_str(columns),
                    back::vector_size_str(rows),
                )?;
            }
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                self.write_image_type(dim, arrayed, class)?;
            }
            TypeInner::Sampler { comparison } => {
                let sampler = if comparison {
                    "SamplerComparisonState"
                } else {
                    "SamplerState"
                };
                write!(self.out, "{}", sampler)?;
            }
            // HLSL arrays are written as `type name[size]`
            // Current code is written arrays only as `[size]`
            // Base `type` and `name` should be written outside
            TypeInner::Array { size, .. } => {
                self.write_array_size(module, size)?;
            }
            _ => {
                return Err(Error::Unimplemented(format!(
                    "write_value_type {:?}",
                    inner
                )))
            }
        }

        Ok(())
    }

    /// Helper method used to write structs
    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        module: &Module,
        name: &str,
        func: &crate::Function,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        // Function Declaration Syntax - https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-function-syntax
        if let Some(ref result) = func.result {
            self.write_type(module, result.ty)?;
        } else {
            write!(self.out, "void")?;
        }

        // Write function name
        write!(self.out, " {}(", name)?;

        // Write function arguments for non entry point functions
        match func_ctx.ty {
            back::FunctionType::Function(handle) => {
                for (index, arg) in func.arguments.iter().enumerate() {
                    // Write argument type
                    self.write_type(module, arg.ty)?;

                    let argument_name =
                        &self.names[&NameKey::FunctionArgument(handle, index as u32)];

                    // Write argument name. Space is important.
                    write!(self.out, " {}", argument_name)?;
                    if index < func.arguments.len() - 1 {
                        // Add a separator between args
                        write!(self.out, ", ")?;
                    }
                }
            }
            back::FunctionType::EntryPoint(index) => {
                // EntryPoint arguments wrapped into structure
                // We need to ensure that entry points have arguments too.
                // For the case when we working with multiple entry points
                // for example vertex shader with arguments and fragment shader without arguments.
                if !self.ep_inputs.is_empty()
                    && !module.entry_points[index as usize]
                        .function
                        .arguments
                        .is_empty()
                {
                    if let Some(ref ep_input) = self.ep_inputs[index as usize] {
                        write!(
                            self.out,
                            "{} {}",
                            ep_input.name,
                            self.namer
                                .call_unique(ep_input.name.to_lowercase().as_str())
                        )?;
                    }
                }
            }
        }
        // Ends of arguments
        write!(self.out, ")")?;

        // Write semantic if it present
        let stage = match func_ctx.ty {
            back::FunctionType::EntryPoint(index) => {
                Some(module.entry_points[index as usize].stage)
            }
            _ => None,
        };
        if let Some(ref result) = func.result {
            if let Some(ref binding) = result.binding {
                self.write_semantic(binding, stage.map(|s| (s, Io::Output)))?;
            }
        }

        // Function body start
        writeln!(self.out)?;
        writeln!(self.out, "{{")?;
        // Write function local variables
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability)
            write!(self.out, "{}", back::INDENT)?;

            // Write the local name
            // The leading space is important
            self.write_type(module, local.ty)?;
            write!(self.out, " {}", self.names[&func_ctx.name_key(handle)])?;
            // Write size for array type
            if let TypeInner::Array { size, .. } = module.types[local.ty].inner {
                self.write_array_size(module, size)?;
            }

            write!(self.out, " = ")?;
            // Write the local initializer if needed
            if let Some(init) = local.init {
                // Put the equal signal only if there's a initializer
                // The leading and trailing spaces aren't needed but help with readability

                // Write the constant
                // `write_constant` adds no trailing or leading space/newline
                self.write_constant(module, init)?;
            } else {
                // Zero initialize local variables
                self.write_default_init(module, local.ty)?;
            }

            // Finish the local with `;` and add a newline (only for readability)
            writeln!(self.out, ";")?
        }

        if !func.local_variables.is_empty() {
            writeln!(self.out)?;
        }

        // Write the function body (statement list)
        for sta in func.body.iter() {
            // The indentation should always be 1 when writing the function body
            self.write_stmt(module, sta, func_ctx, 1)?;
        }

        writeln!(self.out, "}}")?;

        self.named_expressions.clear();

        Ok(())
    }

    /// Helper method used to write statements
    ///
    /// # Notes
    /// Always adds a newline
    fn write_stmt(
        &mut self,
        module: &Module,
        stmt: &crate::Statement,
        func_ctx: &back::FunctionCtx<'_>,
        indent: usize,
    ) -> BackendResult {
        use crate::Statement;
        use back::INDENT;

        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let expr_name = if let Some(name) = func_ctx.named_expressions.get(&handle) {
                        // Front end provides names for all variables at the start of writing.
                        // But we write them to step by step. We need to recache them
                        // Otherwise, we could accidentally write variable name instead of full expression.
                        // Also, we use sanitized names! It defense backend from generating variable with name from reserved keywords.
                        Some(self.namer.call_unique(name))
                    } else {
                        let min_ref_count = func_ctx.expressions[handle].bake_ref_count();
                        if min_ref_count <= func_ctx.info[handle].ref_count {
                            Some(format!("_expr{}", handle.index()))
                        } else {
                            None
                        }
                    };

                    if let Some(name) = expr_name {
                        write!(self.out, "{}", INDENT.repeat(indent))?;
                        self.write_named_expr(module, handle, name, func_ctx)?;
                    }
                }
            }
            // TODO: copy-paste from glsl-out
            Statement::Block(ref block) => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                writeln!(self.out, "{{")?;
                for sta in block.iter() {
                    // Increase the indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, indent + 1)?
                }
                writeln!(self.out, "{}}}", INDENT.repeat(indent))?
            }
            // TODO: copy-paste from glsl-out
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                write!(self.out, "if (")?;
                self.write_expr(module, condition, func_ctx)?;
                writeln!(self.out, ") {{")?;

                for sta in accept {
                    // Increase indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, indent + 1)?;
                }

                // If there are no statements in the reject block we skip writing it
                // This is only for readability
                if !reject.is_empty() {
                    writeln!(self.out, "{}}} else {{", INDENT.repeat(indent))?;

                    for sta in reject {
                        // Increase indentation to help with readability
                        self.write_stmt(module, sta, func_ctx, indent + 1)?;
                    }
                }

                writeln!(self.out, "{}}}", INDENT.repeat(indent))?
            }
            // TODO: copy-paste from glsl-out
            Statement::Kill => writeln!(self.out, "{}discard;", INDENT.repeat(indent))?,
            Statement::Return { value: None } => {
                writeln!(self.out, "{}return;", INDENT.repeat(indent))?;
            }
            Statement::Return { value: Some(expr) } => {
                let base_ty_res = &func_ctx.info[expr].ty;
                let mut resolved = base_ty_res.inner_with(&module.types);
                if let TypeInner::Pointer { base, class: _ } = *resolved {
                    resolved = &module.types[base].inner;
                }

                if let TypeInner::Struct { .. } = *resolved {
                    // We can safery unwrap here, since we now we working with struct
                    let ty = base_ty_res.handle().unwrap();
                    let struct_name = &self.names[&NameKey::Type(ty)];
                    let variable_name = self.namer.call_unique(struct_name.as_str()).to_lowercase();
                    write!(
                        self.out,
                        "{}const {} {} = ",
                        INDENT.repeat(indent),
                        struct_name,
                        variable_name
                    )?;
                    self.write_expr(module, expr, func_ctx)?;
                    writeln!(self.out, ";")?;
                    writeln!(
                        self.out,
                        "{}return {};",
                        INDENT.repeat(indent),
                        variable_name
                    )?;
                } else {
                    write!(self.out, "{}return ", INDENT.repeat(indent))?;
                    self.write_expr(module, expr, func_ctx)?;
                    writeln!(self.out, ";")?
                }
            }
            Statement::Store { pointer, value } => {
                let array_info = match *func_ctx.info[pointer].ty.inner_with(&module.types) {
                    TypeInner::Pointer { base, .. } => match module.types[base].inner {
                        crate::TypeInner::Array {
                            size: crate::ArraySize::Constant(ch),
                            ..
                        } => Some((ch, base)),
                        _ => None,
                    },
                    _ => None,
                };

                if func_ctx.info[pointer]
                    .ty
                    .inner_with(&module.types)
                    .pointer_class()
                    == Some(crate::StorageClass::Storage)
                {
                    let var_handle = self.fill_access_chain(module, pointer, func_ctx)?;
                    self.write_storage_store(
                        module,
                        var_handle,
                        StoreValue::Expression(value),
                        func_ctx,
                        indent,
                    )?;
                } else if let Some((const_handle, base_ty)) = array_info {
                    let size = module.constants[const_handle].to_array_length().unwrap();
                    writeln!(self.out, "{}{{", INDENT.repeat(indent))?;
                    write!(self.out, "{}", INDENT.repeat(indent + 1))?;
                    self.write_type(module, base_ty)?;
                    write!(self.out, " _result[{}]=", size)?;
                    self.write_expr(module, value, func_ctx)?;
                    writeln!(self.out, ";")?;
                    write!(
                        self.out,
                        "{}for(int _i=0; _i<{}; ++_i) ",
                        INDENT.repeat(indent + 1),
                        size
                    )?;
                    self.write_expr(module, pointer, func_ctx)?;
                    writeln!(self.out, "[_i] = _result[_i];")?;
                    writeln!(self.out, "{}}}", INDENT.repeat(indent))?;
                } else {
                    write!(self.out, "{}", INDENT.repeat(indent))?;
                    self.write_expr(module, pointer, func_ctx)?;
                    write!(self.out, " = ")?;
                    self.write_expr(module, value, func_ctx)?;
                    writeln!(self.out, ";")?
                }
            }
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                if let Some(expr) = result {
                    write!(self.out, "const ")?;
                    let name = format!("{}{}", back::BAKE_PREFIX, expr.index());
                    let expr_ty = &func_ctx.info[expr].ty;
                    match *expr_ty {
                        proc::TypeResolution::Handle(handle) => self.write_type(module, handle)?,
                        proc::TypeResolution::Value(ref value) => {
                            self.write_value_type(module, value)?
                        }
                    };
                    write!(self.out, " {} = ", name)?;
                    self.write_expr(module, expr, func_ctx)?;
                    self.named_expressions.insert(expr, name);
                }
                let func_name = &self.names[&NameKey::Function(function)];
                write!(self.out, "{}(", func_name)?;
                for (index, argument) in arguments.iter().enumerate() {
                    self.write_expr(module, *argument, func_ctx)?;
                    // Only write a comma if isn't the last element
                    if index != arguments.len().saturating_sub(1) {
                        // The leading space is for readability only
                        write!(self.out, ", ")?;
                    }
                }
                writeln!(self.out, ");")?
            }
            Statement::Loop {
                ref body,
                ref continuing,
            } => {
                if !continuing.is_empty() {
                    let gate_name = self.namer.call("loop_init");
                    writeln!(
                        self.out,
                        "{}bool {} = true;",
                        INDENT.repeat(indent),
                        gate_name
                    )?;
                    writeln!(self.out, "{}while(true) {{", INDENT.repeat(indent))?;
                    writeln!(
                        self.out,
                        "{}if (!{}) {{",
                        INDENT.repeat(indent + 1),
                        gate_name
                    )?;
                    for sta in continuing.iter() {
                        self.write_stmt(module, sta, func_ctx, indent + 1)?;
                    }
                    writeln!(self.out, "{}}}", INDENT.repeat(indent + 1))?;
                    writeln!(
                        self.out,
                        "{}{} = false;",
                        INDENT.repeat(indent + 1),
                        gate_name
                    )?;
                } else {
                    writeln!(self.out, "{}while(true) {{", INDENT.repeat(indent))?;
                }

                for sta in body.iter() {
                    self.write_stmt(module, sta, func_ctx, indent + 1)?;
                }
                writeln!(self.out, "{}}}", INDENT.repeat(indent))?
            }
            Statement::Break => writeln!(self.out, "{}break;", INDENT.repeat(indent))?,
            Statement::Continue => writeln!(self.out, "{}continue;", INDENT.repeat(indent))?,
            Statement::Barrier(barrier) => {
                if barrier.contains(crate::Barrier::STORAGE) {
                    writeln!(
                        self.out,
                        "{}DeviceMemoryBarrierWithGroupSync();",
                        INDENT.repeat(indent)
                    )?;
                }

                if barrier.contains(crate::Barrier::WORK_GROUP) {
                    writeln!(
                        self.out,
                        "{}GroupMemoryBarrierWithGroupSync();",
                        INDENT.repeat(indent)
                    )?;
                }
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                self.write_expr(module, image, func_ctx)?;

                write!(self.out, "[")?;
                if let Some(index) = array_index {
                    // Array index accepted only for texture_storage_2d_array, so we can safety use int3(coordinate, array_index) here
                    write!(self.out, "int3(")?;
                    self.write_expr(module, coordinate, func_ctx)?;
                    write!(self.out, ", ")?;
                    self.write_expr(module, index, func_ctx)?;
                    write!(self.out, ")")?;
                } else {
                    self.write_expr(module, coordinate, func_ctx)?;
                }
                write!(self.out, "]")?;

                write!(self.out, " = ")?;
                self.write_expr(module, value, func_ctx)?;
                writeln!(self.out, ";")?;
            }
            _ => return Err(Error::Unimplemented(format!("write_stmt {:?}", stmt))),
        }

        Ok(())
    }

    /// Helper method to write expressions
    ///
    /// # Notes
    /// Doesn't add any newlines or leading/trailing spaces
    pub(super) fn write_expr(
        &mut self,
        module: &Module,
        expr: Handle<crate::Expression>,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        use crate::Expression;

        // Handle the special semantics for base vertex/instance
        let ff_input = if self.options.special_constants_binding.is_some() {
            func_ctx.is_fixed_function_input(expr, module)
        } else {
            None
        };
        let closing_bracket = match ff_input {
            Some(crate::BuiltIn::VertexIndex) => {
                write!(self.out, "({}.{} + ", SPECIAL_CBUF_VAR, SPECIAL_BASE_VERTEX)?;
                ")"
            }
            Some(crate::BuiltIn::InstanceIndex) => {
                write!(
                    self.out,
                    "({}.{} + ",
                    SPECIAL_CBUF_VAR, SPECIAL_BASE_INSTANCE
                )?;
                ")"
            }
            _ => "",
        };

        if let Some(name) = self.named_expressions.get(&expr) {
            write!(self.out, "{}{}", name, closing_bracket)?;
            return Ok(());
        }

        let expression = &func_ctx.expressions[expr];

        match *expression {
            Expression::Constant(constant) => self.write_constant(module, constant)?,
            Expression::Compose { ty, ref components } => {
                let braces_init = match module.types[ty].inner {
                    TypeInner::Struct { .. } | TypeInner::Array { .. } => true,
                    _ => false,
                };
                if braces_init {
                    write!(self.out, "{{ ")?;
                } else {
                    self.write_type(module, ty)?;
                    write!(self.out, "(")?;
                }
                for (index, component) in components.iter().enumerate() {
                    self.write_expr(module, *component, func_ctx)?;
                    // Only write a comma if isn't the last element
                    if index != components.len().saturating_sub(1) {
                        // The leading space is for readability only
                        write!(self.out, ", ")?;
                    }
                }
                if braces_init {
                    write!(self.out, " }}")?
                } else {
                    write!(self.out, ")")?
                }
            }
            // All of the multiplication can be expressed as `mul`,
            // except vector * vector, which needs to use the "*" operator.
            Expression::Binary {
                op: crate::BinaryOperator::Multiply,
                left,
                right,
            } if func_ctx.info[left].ty.inner_with(&module.types).is_matrix()
                || func_ctx.info[right]
                    .ty
                    .inner_with(&module.types)
                    .is_matrix() =>
            {
                write!(self.out, "mul(")?;
                self.write_expr(module, left, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, right, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Binary { op, left, right } => {
                write!(self.out, "(")?;
                self.write_expr(module, left, func_ctx)?;
                write!(self.out, " {} ", crate::back::binary_operation_str(op))?;
                self.write_expr(module, right, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Access { base, index } => {
                if func_ctx.info[expr]
                    .ty
                    .inner_with(&module.types)
                    .pointer_class()
                    == Some(crate::StorageClass::Storage)
                {
                    // do nothing, the chain is written on `Load`/`Store`
                } else {
                    self.write_expr(module, base, func_ctx)?;
                    write!(self.out, "[")?;
                    self.write_expr(module, index, func_ctx)?;
                    write!(self.out, "]")?;
                }
            }
            Expression::AccessIndex { base, index } => {
                if func_ctx.info[expr]
                    .ty
                    .inner_with(&module.types)
                    .pointer_class()
                    == Some(crate::StorageClass::Storage)
                {
                    // do nothing, the chain is written on `Load`/`Store`
                } else {
                    self.write_expr(module, base, func_ctx)?;

                    let base_ty_res = &func_ctx.info[base].ty;
                    let mut resolved = base_ty_res.inner_with(&module.types);
                    let base_ty_handle = match *resolved {
                        TypeInner::Pointer { base, class: _ } => {
                            resolved = &module.types[base].inner;
                            Some(base)
                        }
                        _ => base_ty_res.handle(),
                    };

                    match *resolved {
                        TypeInner::Vector { .. } => {
                            // Write vector access as a swizzle
                            write!(self.out, ".{}", back::COMPONENTS[index as usize])?
                        }
                        TypeInner::Matrix { .. }
                        | TypeInner::Array { .. }
                        | TypeInner::ValuePointer { .. } => write!(self.out, "[{}]", index)?,
                        TypeInner::Struct { .. } => {
                            // This will never panic in case the type is a `Struct`, this is not true
                            // for other types so we can only check while inside this match arm
                            let ty = base_ty_handle.unwrap();

                            write!(
                                self.out,
                                ".{}",
                                &self.names[&NameKey::StructMember(ty, index)]
                            )?
                        }
                        ref other => {
                            return Err(Error::Custom(format!("Cannot index {:?}", other)))
                        }
                    }
                }
            }
            Expression::FunctionArgument(pos) => {
                match func_ctx.ty {
                    back::FunctionType::Function(handle) => {
                        let name = &self.names[&NameKey::FunctionArgument(handle, pos)];
                        write!(self.out, "{}", name)?;
                    }
                    back::FunctionType::EntryPoint(index) => {
                        // EntryPoint arguments wrapped into structure
                        // We can safery unwrap here, because if we write function arguments it means, that ep_input struct already exists
                        let ep_input = self.ep_inputs[index as usize].as_ref().unwrap();
                        let member_name = &ep_input.members[pos as usize].name;
                        write!(
                            self.out,
                            "{}.{}",
                            &ep_input.name.to_lowercase(),
                            member_name
                        )?
                    }
                };
            }
            Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                use crate::SampleLevel as Sl;

                let texture_func = match level {
                    Sl::Auto => {
                        if depth_ref.is_some() {
                            "SampleCmp"
                        } else {
                            "Sample"
                        }
                    }
                    Sl::Zero => "SampleCmpLevelZero",
                    Sl::Exact(_) => "SampleLevel",
                    Sl::Bias(_) => "SampleBias",
                    Sl::Gradient { .. } => "SampleGrad",
                };

                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ".{}(", texture_func)?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_texture_coordinates(
                    "float",
                    coordinate,
                    array_index,
                    MipLevelCoordinate::NotApplicable,
                    module,
                    func_ctx,
                )?;

                if let Some(depth_ref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.write_expr(module, depth_ref, func_ctx)?;
                }

                match level {
                    Sl::Auto | Sl::Zero => {}
                    Sl::Exact(expr) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, expr, func_ctx)?;
                    }
                    Sl::Bias(expr) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, expr, func_ctx)?;
                    }
                    Sl::Gradient { x, y } => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, x, func_ctx)?;
                        write!(self.out, ", ")?;
                        self.write_expr(module, y, func_ctx)?;
                    }
                }

                if let Some(offset) = offset {
                    write!(self.out, ", ")?;
                    self.write_constant(module, offset)?;
                }

                write!(self.out, ")")?;
            }
            Expression::ImageQuery { image, query } => {
                // use wrapped image query function
                if let TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } = *func_ctx.info[image].ty.inner_with(&module.types)
                {
                    let wrapped_image_query = WrappedImageQuery {
                        dim,
                        arrayed,
                        class,
                        query: query.into(),
                    };

                    self.write_wrapped_image_query_function_name(wrapped_image_query)?;
                    write!(self.out, "(")?;
                    // Image always first param
                    self.write_expr(module, image, func_ctx)?;
                    if let crate::ImageQuery::Size { level: Some(level) } = query {
                        write!(self.out, ", ")?;
                        self.write_expr(module, level, func_ctx)?;
                    }
                    write!(self.out, ")")?;
                }
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-load
                let ms = match *func_ctx.info[image].ty.inner_with(&module.types) {
                    TypeInner::Image {
                        class: crate::ImageClass::Sampled { multi, .. },
                        ..
                    }
                    | TypeInner::Image {
                        class: crate::ImageClass::Depth { multi },
                        ..
                    } => multi,
                    _ => false,
                };

                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ".Load(")?;

                let mip_level = if ms {
                    MipLevelCoordinate::NotApplicable
                } else {
                    match index {
                        Some(expr) => MipLevelCoordinate::Expression(expr),
                        None => MipLevelCoordinate::Zero,
                    }
                };

                self.write_texture_coordinates(
                    "int",
                    coordinate,
                    array_index,
                    mip_level,
                    module,
                    func_ctx,
                )?;

                if ms {
                    write!(self.out, ", ")?;
                    self.write_expr(module, index.unwrap(), func_ctx)?;
                }

                // close bracket for Load function
                write!(self.out, ")")?;

                // return x component if return type is scalar
                if let TypeInner::Scalar { .. } = *func_ctx.info[expr].ty.inner_with(&module.types)
                {
                    write!(self.out, ".x")?;
                }
            }
            Expression::GlobalVariable(handle) => {
                if module.global_variables[handle].class != crate::StorageClass::Storage {
                    let name = &self.names[&NameKey::GlobalVariable(handle)];
                    write!(self.out, "{}", name)?;
                }
            }
            Expression::LocalVariable(handle) => {
                write!(self.out, "{}", self.names[&func_ctx.name_key(handle)])?
            }
            Expression::Load { pointer } => {
                if func_ctx.info[pointer]
                    .ty
                    .inner_with(&module.types)
                    .pointer_class()
                    == Some(crate::StorageClass::Storage)
                {
                    let var_handle = self.fill_access_chain(module, pointer, func_ctx)?;
                    let result_ty = func_ctx.info[expr].ty.clone();
                    self.write_storage_load(module, var_handle, result_ty, func_ctx)?;
                } else {
                    self.write_expr(module, pointer, func_ctx)?;
                }
            }
            Expression::Unary { op, expr } => {
                // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-operators#unary-operators
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => "!",
                };
                write!(self.out, "{}", op_str)?;
                self.write_expr(module, expr, func_ctx)?;
            }
            Expression::As { expr, kind, .. } => {
                let inner = func_ctx.info[expr].ty.inner_with(&module.types);
                match *inner {
                    TypeInner::Vector { size, width, .. } => {
                        write!(
                            self.out,
                            "{}{}",
                            kind.to_hlsl_str(width)?,
                            back::vector_size_str(size),
                        )?;
                    }
                    TypeInner::Scalar { width, .. } => {
                        write!(self.out, "{}", kind.to_hlsl_str(width)?)?
                    }
                    _ => {
                        return Err(Error::Unimplemented(format!(
                            "write_expr expression::as {:?}",
                            inner
                        )));
                    }
                };
                write!(self.out, "(")?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
            } => {
                use crate::MathFunction as Mf;

                let fun_name = match fun {
                    // comparison
                    Mf::Abs => "abs",
                    Mf::Min => "min",
                    Mf::Max => "max",
                    Mf::Clamp => "clamp",
                    // trigonometry
                    Mf::Cos => "cos",
                    Mf::Cosh => "cosh",
                    Mf::Sin => "sin",
                    Mf::Sinh => "sinh",
                    Mf::Tan => "tan",
                    Mf::Tanh => "tanh",
                    Mf::Acos => "acos",
                    Mf::Asin => "asin",
                    Mf::Atan => "atan",
                    Mf::Atan2 => "atan2",
                    // decomposition
                    Mf::Ceil => "ceil",
                    Mf::Floor => "floor",
                    Mf::Round => "round",
                    Mf::Fract => "frac",
                    Mf::Trunc => "trunc",
                    Mf::Modf => "modf",
                    Mf::Frexp => "frexp",
                    Mf::Ldexp => "ldexp",
                    // exponent
                    Mf::Exp => "exp",
                    Mf::Exp2 => "exp2",
                    Mf::Log => "log",
                    Mf::Log2 => "log2",
                    Mf::Pow => "pow",
                    // geometry
                    Mf::Dot => "dot",
                    //Mf::Outer => ,
                    Mf::Cross => "cross",
                    Mf::Distance => "distance",
                    Mf::Length => "length",
                    Mf::Normalize => "normalize",
                    Mf::FaceForward => "faceforward",
                    Mf::Reflect => "reflect",
                    Mf::Refract => "refract",
                    // computational
                    Mf::Sign => "sign",
                    Mf::Fma => "fma",
                    Mf::Mix => "lerp",
                    Mf::Step => "step",
                    Mf::SmoothStep => "smoothstep",
                    Mf::Sqrt => "sqrt",
                    Mf::InverseSqrt => "rsqrt",
                    //Mf::Inverse =>,
                    Mf::Transpose => "transpose",
                    Mf::Determinant => "determinant",
                    // bits
                    Mf::CountOneBits => "countbits",
                    Mf::ReverseBits => "reversebits",
                    _ => return Err(Error::Unimplemented(format!("write_expr_math {:?}", fun))),
                };

                write!(self.out, "{}(", fun_name)?;
                self.write_expr(module, arg, func_ctx)?;
                if let Some(arg) = arg1 {
                    write!(self.out, ", ")?;
                    self.write_expr(module, arg, func_ctx)?;
                }
                if let Some(arg) = arg2 {
                    write!(self.out, ", ")?;
                    self.write_expr(module, arg, func_ctx)?;
                }
                write!(self.out, ")")?
            }
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                self.write_expr(module, vector, func_ctx)?;
                write!(self.out, ".")?;
                for &sc in pattern[..size as usize].iter() {
                    self.out.write_char(back::COMPONENTS[sc as usize])?;
                }
            }
            Expression::ArrayLength(expr) => {
                let var_handle = match func_ctx.expressions[expr] {
                    Expression::AccessIndex { base, index: _ } => {
                        match func_ctx.expressions[base] {
                            Expression::GlobalVariable(handle) => handle,
                            _ => unreachable!(),
                        }
                    }
                    Expression::GlobalVariable(handle) => handle,
                    _ => unreachable!(),
                };

                let var = &module.global_variables[var_handle];
                let (offset, stride) = match module.types[var.ty].inner {
                    TypeInner::Array { stride, .. } => (0, stride),
                    TypeInner::Struct {
                        top_level: true,
                        ref members,
                        ..
                    } => {
                        let last = members.last().unwrap();
                        let stride = match module.types[last.ty].inner {
                            TypeInner::Array { stride, .. } => stride,
                            _ => unreachable!(),
                        };
                        (last.offset, stride)
                    }
                    _ => unreachable!(),
                };

                let wrapped_array_length = WrappedArrayLength {
                    writable: var.storage_access.contains(crate::StorageAccess::STORE),
                };

                write!(self.out, "((")?;
                self.write_wrapped_array_length_function_name(wrapped_array_length)?;
                let var_name = &self.names[&NameKey::GlobalVariable(var_handle)];
                write!(self.out, "({}) - {}) / {})", var_name, offset, stride)?
            }
            Expression::Derivative { axis, expr } => {
                use crate::DerivativeAxis as Da;

                write!(
                    self.out,
                    "{}(",
                    match axis {
                        Da::X => "ddx",
                        Da::Y => "ddy",
                        Da::Width => "fwidth",
                    }
                )?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?
            }
            Expression::Splat { size, value } => {
                // hlsl is not supported one value constructor
                // if we write, for example, int4(0), dxc returns error:
                // error: too few elements in vector initialization (expected 4 elements, have 1)
                let number_of_components = match size {
                    crate::VectorSize::Bi => "xx",
                    crate::VectorSize::Tri => "xxx",
                    crate::VectorSize::Quad => "xxxx",
                };
                let resolved = func_ctx.info[expr].ty.inner_with(&module.types);
                self.write_value_type(module, resolved)?;
                write!(self.out, "(")?;
                self.write_expr(module, value, func_ctx)?;
                write!(self.out, ".{})", number_of_components)?
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                write!(self.out, "(")?;
                self.write_expr(module, condition, func_ctx)?;
                write!(self.out, " ? ")?;
                self.write_expr(module, accept, func_ctx)?;
                write!(self.out, " : ")?;
                self.write_expr(module, reject, func_ctx)?;
                write!(self.out, ")")?
            }
            // Nothing to do here, since call expression already cached
            Expression::Call(_) => {}
            _ => return Err(Error::Unimplemented(format!("write_expr {:?}", expression))),
        }

        if !closing_bracket.is_empty() {
            write!(self.out, "{}", closing_bracket)?;
        }
        Ok(())
    }

    /// Helper method used to write constants
    ///
    /// # Notes
    /// Doesn't add any newlines or leading/trailing spaces
    fn write_constant(
        &mut self,
        module: &Module,
        handle: Handle<crate::Constant>,
    ) -> BackendResult {
        let constant = &module.constants[handle];
        match constant.inner {
            crate::ConstantInner::Scalar {
                width: _,
                ref value,
            } => {
                if constant.name.is_some() {
                    write!(self.out, "{}", &self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_scalar_value(*value)?;
                }
            }
            crate::ConstantInner::Composite { ty, ref components } => {
                self.write_composite_constant(module, ty, components)?;
            }
        }

        Ok(())
    }

    fn write_composite_constant(
        &mut self,
        module: &Module,
        ty: Handle<crate::Type>,
        components: &[Handle<crate::Constant>],
    ) -> BackendResult {
        let (open_b, close_b) = match module.types[ty].inner {
            TypeInner::Struct { .. } => ("{ ", " }"),
            _ => {
                // We should write type only for non struct constants
                self.write_type(module, ty)?;
                ("(", ")")
            }
        };
        write!(self.out, "{}", open_b)?;
        for (index, constant) in components.iter().enumerate() {
            self.write_constant(module, *constant)?;
            // Only write a comma if isn't the last element
            if index != components.len().saturating_sub(1) {
                // The leading space is for readability only
                write!(self.out, ", ")?;
            }
        }
        write!(self.out, "{}", close_b)?;

        Ok(())
    }

    /// Helper method used to write [`ScalarValue`](ScalarValue)
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_scalar_value(&mut self, value: crate::ScalarValue) -> BackendResult {
        use crate::ScalarValue as Sv;

        match value {
            Sv::Sint(value) => write!(self.out, "{}", value)?,
            Sv::Uint(value) => write!(self.out, "{}u", value)?,
            // Floats are written using `Debug` instead of `Display` because it always appends the
            // decimal part even it's zero
            Sv::Float(value) => write!(self.out, "{:?}", value)?,
            Sv::Bool(value) => write!(self.out, "{}", value)?,
        }

        Ok(())
    }

    fn write_named_expr(
        &mut self,
        module: &Module,
        handle: Handle<crate::Expression>,
        name: String,
        ctx: &back::FunctionCtx,
    ) -> BackendResult {
        match ctx.info[handle].ty {
            proc::TypeResolution::Handle(ty_handle) => match module.types[ty_handle].inner {
                TypeInner::Struct { .. } => {
                    let ty_name = &self.names[&NameKey::Type(ty_handle)];
                    write!(self.out, "{}", ty_name)?;
                }
                _ => {
                    self.write_type(module, ty_handle)?;
                }
            },
            proc::TypeResolution::Value(ref inner) => {
                self.write_value_type(module, inner)?;
            }
        }

        let base_ty_res = &ctx.info[handle].ty;
        let resolved = base_ty_res.inner_with(&module.types);

        write!(self.out, " {}", name)?;
        // If rhs is a array type, we should write array size
        if let TypeInner::Array { size, .. } = *resolved {
            self.write_array_size(module, size)?;
        }
        write!(self.out, " = ")?;
        self.write_expr(module, handle, ctx)?;
        writeln!(self.out, ";")?;
        self.named_expressions.insert(handle, name);

        Ok(())
    }

    /// Helper function that write default zero initialization
    fn write_default_init(&mut self, module: &Module, ty: Handle<crate::Type>) -> BackendResult {
        match module.types[ty].inner {
            TypeInner::Array {
                size: crate::ArraySize::Constant(const_handle),
                base,
                ..
            } => {
                write!(self.out, "{{")?;
                let count = module.constants[const_handle].to_array_length().unwrap();
                for i in 0..count {
                    if i != 0 {
                        write!(self.out, ",")?;
                    }
                    self.write_default_init(module, base)?;
                }
                write!(self.out, "}}")?;
            }
            _ => {
                write!(self.out, "(")?;
                self.write_type(module, ty)?;
                write!(self.out, ")0")?;
            }
        }
        Ok(())
    }
}
