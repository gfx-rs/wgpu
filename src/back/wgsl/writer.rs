use super::Error;
use crate::{
    back,
    proc::{self, NameKey},
    valid, Handle, Module, ShaderStage, TypeInner,
};
use std::fmt::Write;

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

/// WGSL [attribute](https://gpuweb.github.io/gpuweb/wgsl/#attributes)
enum Attribute {
    Binding(u32),
    BuiltIn(crate::BuiltIn),
    Group(u32),
    Invariant,
    Interpolate(Option<crate::Interpolation>, Option<crate::Sampling>),
    Location(u32),
    Stage(ShaderStage),
    WorkGroupSize([u32; 3]),
}

/// The WGSL form that `write_expr_with_indirection` should use to render a Naga
/// expression.
///
/// Sometimes a Naga `Expression` alone doesn't provide enough information to
/// choose the right rendering for it in WGSL. For example, one natural WGSL
/// rendering of a Naga `LocalVariable(x)` expression might be `&x`, since
/// `LocalVariable` produces a pointer to the local variable's storage. But when
/// rendering a `Store` statement, the `pointer` operand must be the left hand
/// side of a WGSL assignment, so the proper rendering is `x`.
///
/// The caller of `write_expr_with_indirection` must provide an `Expected` value
/// to indicate how ambiguous expressions should be rendered.
#[derive(Clone, Copy, Debug)]
enum Indirection {
    /// Render pointer-construction expressions as WGSL `ptr`-typed expressions.
    ///
    /// This is the right choice for most cases. Whenever a Naga pointer
    /// expression is not the `pointer` operand of a `Load` or `Store`, it
    /// must be a WGSL pointer expression.
    Ordinary,

    /// Render pointer-construction expressions as WGSL reference-typed
    /// expressions.
    ///
    /// For example, this is the right choice for the `pointer` operand when
    /// rendering a `Store` statement as a WGSL assignment.
    Reference,
}

bitflags::bitflags! {
    #[cfg_attr(feature = "serialize", derive(serde::Serialize))]
    #[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
    pub struct WriterFlags: u32 {
        /// Always annotate the type information instead of inferring.
        const EXPLICIT_TYPES = 0x1;
    }
}

pub struct Writer<W> {
    out: W,
    flags: WriterFlags,
    names: crate::FastHashMap<NameKey, String>,
    namer: proc::Namer,
    named_expressions: crate::NamedExpressions,
    ep_results: Vec<(ShaderStage, Handle<crate::Type>)>,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W, flags: WriterFlags) -> Self {
        Writer {
            out,
            flags,
            names: crate::FastHashMap::default(),
            namer: proc::Namer::default(),
            named_expressions: crate::NamedExpressions::default(),
            ep_results: vec![],
        }
    }

    fn reset(&mut self, module: &Module) {
        self.names.clear();
        self.namer.reset(
            module,
            crate::keywords::wgsl::RESERVED,
            // an identifier must not start with two underscore
            &["__"],
            &mut self.names,
        );
        self.named_expressions.clear();
        self.ep_results.clear();
    }

    pub fn write(&mut self, module: &Module, info: &valid::ModuleInfo) -> BackendResult {
        self.reset(module);

        // Save all ep result types
        for (_, ep) in module.entry_points.iter().enumerate() {
            if let Some(ref result) = ep.function.result {
                self.ep_results.push((ep.stage, result.ty));
            }
        }

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct {
                ref members,
                span: _,
            } = ty.inner
            {
                self.write_struct(module, handle, members)?;
                writeln!(self.out)?;
            }
        }

        // Write all constants
        for (handle, constant) in module.constants.iter() {
            if constant.name.is_some() {
                self.write_global_constant(module, &constant.inner, handle)?;
            }
        }

        // Write all globals
        for (ty, global) in module.global_variables.iter() {
            self.write_global(module, global, ty)?;
        }

        if !module.global_variables.is_empty() {
            // Add extra newline for readability
            writeln!(self.out)?;
        }

        // Write all regular functions
        for (handle, function) in module.functions.iter() {
            let fun_info = &info[handle];

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::Function(handle),
                info: fun_info,
                expressions: &function.expressions,
                named_expressions: &function.named_expressions,
            };

            // Write the function
            self.write_function(module, function, &func_ctx)?;

            writeln!(self.out)?;
        }

        // Write all entry points
        for (index, ep) in module.entry_points.iter().enumerate() {
            let attributes = match ep.stage {
                ShaderStage::Vertex | ShaderStage::Fragment => vec![Attribute::Stage(ep.stage)],
                ShaderStage::Compute => vec![
                    Attribute::Stage(ShaderStage::Compute),
                    Attribute::WorkGroupSize(ep.workgroup_size),
                ],
            };

            self.write_attributes(&attributes)?;
            // Add a newline after attribute
            writeln!(self.out)?;

            let func_ctx = back::FunctionCtx {
                ty: back::FunctionType::EntryPoint(index as u16),
                info: info.get_entry_point(index),
                expressions: &ep.function.expressions,
                named_expressions: &ep.function.named_expressions,
            };
            self.write_function(module, &ep.function, &func_ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(self.out)?;
            }
        }

        Ok(())
    }

    /// Helper method used to write [`ScalarValue`](crate::ScalarValue)
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

    /// Helper method used to write struct name
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_struct_name(&mut self, module: &Module, handle: Handle<crate::Type>) -> BackendResult {
        if module.types[handle].name.is_none() {
            if let Some(&(stage, _)) = self.ep_results.iter().find(|&&(_, ty)| ty == handle) {
                let name = match stage {
                    ShaderStage::Compute => "ComputeOutput",
                    ShaderStage::Fragment => "FragmentOutput",
                    ShaderStage::Vertex => "VertexOutput",
                };

                write!(self.out, "{}", name)?;
                return Ok(());
            }
        }

        write!(self.out, "{}", self.names[&NameKey::Type(handle)])?;

        Ok(())
    }

    /// Helper method used to write
    /// [functions](https://gpuweb.github.io/gpuweb/wgsl/#functions)
    ///
    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        module: &Module,
        func: &crate::Function,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        let func_name = match func_ctx.ty {
            back::FunctionType::EntryPoint(index) => &self.names[&NameKey::EntryPoint(index)],
            back::FunctionType::Function(handle) => &self.names[&NameKey::Function(handle)],
        };

        // Write function name
        write!(self.out, "fn {}(", func_name)?;

        // Write function arguments
        for (index, arg) in func.arguments.iter().enumerate() {
            // Write argument attribute if a binding is present
            if let Some(ref binding) = arg.binding {
                self.write_attributes(&map_binding_to_attribute(
                    binding,
                    module.types[arg.ty].inner.scalar_kind(),
                ))?;
            }
            // Write argument name
            let argument_name = match func_ctx.ty {
                back::FunctionType::Function(handle) => {
                    &self.names[&NameKey::FunctionArgument(handle, index as u32)]
                }
                back::FunctionType::EntryPoint(ep_index) => {
                    &self.names[&NameKey::EntryPointArgument(ep_index, index as u32)]
                }
            };

            write!(self.out, "{}: ", argument_name)?;
            // Write argument type
            self.write_type(module, arg.ty)?;
            if index < func.arguments.len() - 1 {
                // Add a separator between args
                write!(self.out, ", ")?;
            }
        }

        write!(self.out, ")")?;

        // Write function return type
        if let Some(ref result) = func.result {
            write!(self.out, " -> ")?;
            if let Some(ref binding) = result.binding {
                self.write_attributes(&map_binding_to_attribute(
                    binding,
                    module.types[result.ty].inner.scalar_kind(),
                ))?;
            }
            self.write_type(module, result.ty)?;
        }

        write!(self.out, " {{")?;
        writeln!(self.out)?;

        // Write function local variables
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability)
            write!(self.out, "{}", back::INDENT)?;

            // Write the local name
            // The leading space is important
            write!(self.out, "var {}: ", self.names[&func_ctx.name_key(handle)])?;

            // Write the local type
            self.write_type(module, local.ty)?;

            // Write the local initializer if needed
            if let Some(init) = local.init {
                // Put the equal signal only if there's a initializer
                // The leading and trailing spaces aren't needed but help with readability
                write!(self.out, " = ")?;

                // Write the constant
                // `write_constant` adds no trailing or leading space/newline
                self.write_constant(module, init)?;
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
            self.write_stmt(module, sta, func_ctx, back::Level(1))?;
        }

        writeln!(self.out, "}}")?;

        self.named_expressions.clear();

        Ok(())
    }

    /// Helper method to write a attribute
    fn write_attributes(&mut self, attributes: &[Attribute]) -> BackendResult {
        for attribute in attributes {
            match *attribute {
                Attribute::Location(id) => write!(self.out, "@location({}) ", id)?,
                Attribute::BuiltIn(builtin_attrib) => {
                    if let Some(builtin) = builtin_str(builtin_attrib) {
                        write!(self.out, "@builtin({}) ", builtin)?;
                    } else {
                        log::warn!("Unsupported builtin attribute: {:?}", builtin_attrib);
                    }
                }
                Attribute::Stage(shader_stage) => {
                    let stage_str = match shader_stage {
                        ShaderStage::Vertex => "vertex",
                        ShaderStage::Fragment => "fragment",
                        ShaderStage::Compute => "compute",
                    };
                    write!(self.out, "@{} ", stage_str)?;
                }
                Attribute::WorkGroupSize(size) => {
                    write!(
                        self.out,
                        "@workgroup_size({}, {}, {}) ",
                        size[0], size[1], size[2]
                    )?;
                }
                Attribute::Binding(id) => write!(self.out, "@binding({}) ", id)?,
                Attribute::Group(id) => write!(self.out, "@group({}) ", id)?,
                Attribute::Invariant => write!(self.out, "@invariant ")?,
                Attribute::Interpolate(interpolation, sampling) => {
                    if sampling.is_some() && sampling != Some(crate::Sampling::Center) {
                        write!(
                            self.out,
                            "@interpolate({}, {}) ",
                            interpolation_str(
                                interpolation.unwrap_or(crate::Interpolation::Perspective)
                            ),
                            sampling_str(sampling.unwrap_or(crate::Sampling::Center))
                        )?;
                    } else if interpolation.is_some()
                        && interpolation != Some(crate::Interpolation::Perspective)
                    {
                        write!(
                            self.out,
                            "@interpolate({}) ",
                            interpolation_str(
                                interpolation.unwrap_or(crate::Interpolation::Perspective)
                            )
                        )?;
                    }
                }
            };
        }
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
        members: &[crate::StructMember],
    ) -> BackendResult {
        write!(self.out, "struct ")?;
        self.write_struct_name(module, handle)?;
        write!(self.out, " {{")?;
        writeln!(self.out)?;
        for (index, member) in members.iter().enumerate() {
            // Skip struct member with unsupported built in
            if let Some(crate::Binding::BuiltIn(built_in)) = member.binding {
                if builtin_str(built_in).is_none() {
                    log::warn!("Skip member with unsupported builtin {:?}", built_in);
                    continue;
                }
            }

            // The indentation is only for readability
            write!(self.out, "{}", back::INDENT)?;
            if let Some(ref binding) = member.binding {
                self.write_attributes(&map_binding_to_attribute(
                    binding,
                    module.types[member.ty].inner.scalar_kind(),
                ))?;
            }
            // Write struct member name and type
            let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];
            write!(self.out, "{}: ", member_name)?;
            self.write_type(module, member.ty)?;
            write!(self.out, ",")?;
            writeln!(self.out)?;
        }

        write!(self.out, "}}")?;

        writeln!(self.out)?;

        Ok(())
    }

    /// Helper method used to write non image/sampler types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_type(&mut self, module: &Module, ty: Handle<crate::Type>) -> BackendResult {
        let inner = &module.types[ty].inner;
        match *inner {
            TypeInner::Struct { .. } => self.write_struct_name(module, ty)?,
            ref other => self.write_value_type(module, other)?,
        }

        Ok(())
    }

    /// Helper method used to write value types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_value_type(&mut self, module: &Module, inner: &TypeInner) -> BackendResult {
        match *inner {
            TypeInner::Vector { size, kind, .. } => write!(
                self.out,
                "vec{}<{}>",
                back::vector_size_str(size),
                scalar_kind_str(kind),
            )?,
            TypeInner::Sampler { comparison: false } => {
                write!(self.out, "sampler")?;
            }
            TypeInner::Sampler { comparison: true } => {
                write!(self.out, "sampler_comparison")?;
            }
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                // More about texture types: https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
                use crate::ImageClass as Ic;

                let dim_str = image_dimension_str(dim);
                let arrayed_str = if arrayed { "_array" } else { "" };
                let (class_str, multisampled_str, format_str, storage_str) = match class {
                    Ic::Sampled { kind, multi } => (
                        "",
                        if multi { "multisampled_" } else { "" },
                        scalar_kind_str(kind),
                        "",
                    ),
                    Ic::Depth { multi } => {
                        ("depth_", if multi { "multisampled_" } else { "" }, "", "")
                    }
                    Ic::Storage { format, access } => (
                        "storage_",
                        "",
                        storage_format_str(format),
                        if access.contains(crate::StorageAccess::LOAD | crate::StorageAccess::STORE)
                        {
                            ",read_write"
                        } else if access.contains(crate::StorageAccess::LOAD) {
                            ",read"
                        } else {
                            ",write"
                        },
                    ),
                };
                write!(
                    self.out,
                    "texture_{}{}{}{}",
                    class_str, multisampled_str, dim_str, arrayed_str
                )?;

                if !format_str.is_empty() {
                    write!(self.out, "<{}{}>", format_str, storage_str)?;
                }
            }
            TypeInner::Scalar { kind, .. } => {
                write!(self.out, "{}", scalar_kind_str(kind))?;
            }
            TypeInner::Atomic { kind, .. } => {
                write!(self.out, "atomic<{}>", scalar_kind_str(kind))?;
            }
            TypeInner::Array {
                base,
                size,
                stride: _,
            } => {
                // More info https://gpuweb.github.io/gpuweb/wgsl/#array-types
                // array<A, 3> -- Constant array
                // array<A> -- Dynamic array
                write!(self.out, "array<")?;
                match size {
                    crate::ArraySize::Constant(handle) => {
                        self.write_type(module, base)?;
                        write!(self.out, ",")?;
                        self.write_constant(module, handle)?;
                    }
                    crate::ArraySize::Dynamic => {
                        self.write_type(module, base)?;
                    }
                }
                write!(self.out, ">")?;
            }
            TypeInner::BindingArray { base, size } => {
                // More info https://github.com/gpuweb/gpuweb/issues/2105
                write!(self.out, "binding_array<")?;
                match size {
                    crate::ArraySize::Constant(handle) => {
                        self.write_type(module, base)?;
                        write!(self.out, ",")?;
                        self.write_constant(module, handle)?;
                    }
                    crate::ArraySize::Dynamic => {
                        self.write_type(module, base)?;
                    }
                }
                write!(self.out, ">")?;
            }
            TypeInner::Matrix {
                columns,
                rows,
                width: _,
            } => {
                write!(
                    self.out,
                    //TODO: Can matrix be other than f32?
                    "mat{}x{}<f32>",
                    back::vector_size_str(columns),
                    back::vector_size_str(rows),
                )?;
            }
            TypeInner::Pointer { base, space } => {
                let (address, maybe_access) = address_space_str(space);
                // Everything but `AddressSpace::Handle` gives us a `address` name, but
                // Naga IR never produces pointers to handles, so it doesn't matter much
                // how we write such a type. Just write it as the base type alone.
                if let Some(space) = address {
                    write!(self.out, "ptr<{}, ", space)?;
                }
                self.write_type(module, base)?;
                if address.is_some() {
                    if let Some(access) = maybe_access {
                        write!(self.out, ", {}", access)?;
                    }
                    write!(self.out, ">")?;
                }
            }
            TypeInner::ValuePointer {
                size: None,
                kind,
                width: _,
                space,
            } => {
                let (address, maybe_access) = address_space_str(space);
                if let Some(space) = address {
                    write!(self.out, "ptr<{}, {}", space, scalar_kind_str(kind))?;
                    if let Some(access) = maybe_access {
                        write!(self.out, ", {}", access)?;
                    }
                    write!(self.out, ">")?;
                } else {
                    return Err(Error::Unimplemented(format!(
                        "ValuePointer to AddressSpace::Handle {:?}",
                        inner
                    )));
                }
            }
            TypeInner::ValuePointer {
                size: Some(size),
                kind,
                width: _,
                space,
            } => {
                let (address, maybe_access) = address_space_str(space);
                if let Some(space) = address {
                    write!(
                        self.out,
                        "ptr<{}, vec{}<{}>",
                        space,
                        back::vector_size_str(size),
                        scalar_kind_str(kind)
                    )?;
                    if let Some(access) = maybe_access {
                        write!(self.out, ", {}", access)?;
                    }
                    write!(self.out, ">")?;
                } else {
                    return Err(Error::Unimplemented(format!(
                        "ValuePointer to AddressSpace::Handle {:?}",
                        inner
                    )));
                }
                write!(self.out, ">")?;
            }
            _ => {
                return Err(Error::Unimplemented(format!(
                    "write_value_type {:?}",
                    inner
                )));
            }
        }

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
        level: back::Level,
    ) -> BackendResult {
        use crate::{Expression, Statement};

        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let info = &func_ctx.info[handle];
                    let expr_name = if let Some(name) = func_ctx.named_expressions.get(&handle) {
                        // Front end provides names for all variables at the start of writing.
                        // But we write them to step by step. We need to recache them
                        // Otherwise, we could accidentally write variable name instead of full expression.
                        // Also, we use sanitized names! It defense backend from generating variable with name from reserved keywords.
                        Some(self.namer.call(name))
                    } else if info.ref_count == 0 {
                        write!(self.out, "{}_ = ", level)?;
                        self.write_expr(module, handle, func_ctx)?;
                        writeln!(self.out, ";")?;
                        continue;
                    } else {
                        let expr = &func_ctx.expressions[handle];
                        let min_ref_count = expr.bake_ref_count();
                        // Forcefully creating baking expressions in some cases to help with readability
                        let required_baking_expr = match *expr {
                            Expression::ImageLoad { .. }
                            | Expression::ImageQuery { .. }
                            | Expression::ImageSample { .. } => true,
                            _ => false,
                        };
                        if min_ref_count <= info.ref_count || required_baking_expr {
                            // If expression contains unsupported builtin we should skip it
                            if let Expression::Load { pointer } = func_ctx.expressions[handle] {
                                if let Expression::AccessIndex { base, index } =
                                    func_ctx.expressions[pointer]
                                {
                                    if access_to_unsupported_builtin(
                                        base,
                                        index,
                                        module,
                                        func_ctx.info,
                                    ) {
                                        return Ok(());
                                    }
                                }
                            }

                            Some(format!("{}{}", back::BAKE_PREFIX, handle.index()))
                        } else {
                            None
                        }
                    };

                    if let Some(name) = expr_name {
                        write!(self.out, "{}", level)?;
                        self.start_named_expr(module, handle, func_ctx, &name)?;
                        self.write_expr(module, handle, func_ctx)?;
                        self.named_expressions.insert(handle, name);
                        writeln!(self.out, ";")?;
                    }
                }
            }
            // TODO: copy-paste from glsl-out
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                write!(self.out, "{}", level)?;
                write!(self.out, "if ")?;
                self.write_expr(module, condition, func_ctx)?;
                writeln!(self.out, " {{")?;

                let l2 = level.next();
                for sta in accept {
                    // Increase indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, l2)?;
                }

                // If there are no statements in the reject block we skip writing it
                // This is only for readability
                if !reject.is_empty() {
                    writeln!(self.out, "{}}} else {{", level)?;

                    for sta in reject {
                        // Increase indentation to help with readability
                        self.write_stmt(module, sta, func_ctx, l2)?;
                    }
                }

                writeln!(self.out, "{}}}", level)?
            }
            Statement::Return { value } => {
                write!(self.out, "{}", level)?;
                write!(self.out, "return")?;
                if let Some(return_value) = value {
                    // The leading space is important
                    write!(self.out, " ")?;
                    self.write_expr(module, return_value, func_ctx)?;
                }
                writeln!(self.out, ";")?;
            }
            // TODO: copy-paste from glsl-out
            Statement::Kill => {
                write!(self.out, "{}", level)?;
                writeln!(self.out, "discard;")?
            }
            Statement::Store { pointer, value } => {
                // WGSL does not support all SPIR-V builtins and we should skip it in generated shaders.
                // We already skip them when we generate struct type.
                // Now we need to find expression that used struct with ignored builtins
                if let Expression::AccessIndex { base, index } = func_ctx.expressions[pointer] {
                    if access_to_unsupported_builtin(base, index, module, func_ctx.info) {
                        return Ok(());
                    }
                }
                write!(self.out, "{}", level)?;

                let is_atomic = match *func_ctx.info[pointer].ty.inner_with(&module.types) {
                    crate::TypeInner::Pointer { base, .. } => match module.types[base].inner {
                        crate::TypeInner::Atomic { .. } => true,
                        _ => false,
                    },
                    _ => false,
                };
                if is_atomic {
                    write!(self.out, "atomicStore(")?;
                    self.write_expr(module, pointer, func_ctx)?;
                    write!(self.out, ", ")?;
                    self.write_expr(module, value, func_ctx)?;
                    write!(self.out, ")")?;
                } else {
                    self.write_expr_with_indirection(
                        module,
                        pointer,
                        func_ctx,
                        Indirection::Reference,
                    )?;
                    write!(self.out, " = ")?;
                    self.write_expr(module, value, func_ctx)?;
                }
                writeln!(self.out, ";")?
            }
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                write!(self.out, "{}", level)?;
                if let Some(expr) = result {
                    let name = format!("{}{}", back::BAKE_PREFIX, expr.index());
                    self.start_named_expr(module, expr, func_ctx, &name)?;
                    self.named_expressions.insert(expr, name);
                }
                let func_name = &self.names[&NameKey::Function(function)];
                write!(self.out, "{}(", func_name)?;
                for (index, &argument) in arguments.iter().enumerate() {
                    self.write_expr(module, argument, func_ctx)?;
                    // Only write a comma if isn't the last element
                    if index != arguments.len().saturating_sub(1) {
                        // The leading space is for readability only
                        write!(self.out, ", ")?;
                    }
                }
                writeln!(self.out, ");")?
            }
            Statement::Atomic {
                pointer,
                ref fun,
                value,
                result,
            } => {
                write!(self.out, "{}", level)?;
                let res_name = format!("{}{}", back::BAKE_PREFIX, result.index());
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);

                let fun_str = fun.to_wgsl();
                write!(self.out, "atomic{}(", fun_str)?;
                self.write_expr(module, pointer, func_ctx)?;
                if let crate::AtomicFunction::Exchange { compare: Some(cmp) } = *fun {
                    write!(self.out, ", ")?;
                    self.write_expr(module, cmp, func_ctx)?;
                }
                write!(self.out, ", ")?;
                self.write_expr(module, value, func_ctx)?;
                writeln!(self.out, ");")?
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                write!(self.out, "{}", level)?;
                write!(self.out, "textureStore(")?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;
                if let Some(array_index_expr) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index_expr, func_ctx)?;
                }
                write!(self.out, ", ")?;
                self.write_expr(module, value, func_ctx)?;
                writeln!(self.out, ");")?;
            }
            // TODO: copy-paste from glsl-out
            Statement::Block(ref block) => {
                write!(self.out, "{}", level)?;
                writeln!(self.out, "{{")?;
                for sta in block.iter() {
                    // Increase the indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, level.next())?
                }
                writeln!(self.out, "{}}}", level)?
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Start the switch
                write!(self.out, "{}", level)?;
                write!(self.out, "switch ")?;
                self.write_expr(module, selector, func_ctx)?;
                writeln!(self.out, " {{")?;

                let type_postfix = match *func_ctx.info[selector].ty.inner_with(&module.types) {
                    crate::TypeInner::Scalar {
                        kind: crate::ScalarKind::Uint,
                        ..
                    } => "u",
                    _ => "",
                };

                let l2 = level.next();
                if !cases.is_empty() {
                    for case in cases {
                        match case.value {
                            crate::SwitchValue::Integer(value) => {
                                writeln!(self.out, "{}case {}{}: {{", l2, value, type_postfix)?;
                            }
                            crate::SwitchValue::Default => {
                                writeln!(self.out, "{}default: {{", l2)?;
                            }
                        }

                        for sta in case.body.iter() {
                            self.write_stmt(module, sta, func_ctx, l2.next())?;
                        }

                        if case.fall_through {
                            writeln!(self.out, "{}fallthrough;", l2.next())?;
                        }

                        writeln!(self.out, "{}}}", l2)?;
                    }
                }

                writeln!(self.out, "{}}}", level)?
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                write!(self.out, "{}", level)?;
                writeln!(self.out, "loop {{")?;

                let l2 = level.next();
                for sta in body.iter() {
                    self.write_stmt(module, sta, func_ctx, l2)?;
                }

                // The continuing is optional so we don't need to write it if
                // it is empty, but the `break if` counts as a continuing statement
                // so even if `continuing` is empty we must generate it if a
                // `break if` exists
                if !continuing.is_empty() || break_if.is_some() {
                    writeln!(self.out, "{}continuing {{", l2)?;
                    for sta in continuing.iter() {
                        self.write_stmt(module, sta, func_ctx, l2.next())?;
                    }

                    // The `break if` is always the last
                    // statement of the `continuing` block
                    if let Some(condition) = break_if {
                        // The trailing space is important
                        write!(self.out, "{}break if ", l2.next())?;
                        self.write_expr(module, condition, func_ctx)?;
                        // Close the `break if` statement
                        writeln!(self.out, ";")?;
                    }

                    writeln!(self.out, "{}}}", l2)?;
                }

                writeln!(self.out, "{}}}", level)?
            }
            Statement::Break => {
                writeln!(self.out, "{}break;", level)?;
            }
            Statement::Continue => {
                writeln!(self.out, "{}continue;", level)?;
            }
            Statement::Barrier(barrier) => {
                if barrier.contains(crate::Barrier::STORAGE) {
                    writeln!(self.out, "{}storageBarrier();", level)?;
                }

                if barrier.contains(crate::Barrier::WORK_GROUP) {
                    writeln!(self.out, "{}workgroupBarrier();", level)?;
                }
            }
        }

        Ok(())
    }

    /// Return the sort of indirection that `expr`'s plain form evaluates to.
    ///
    /// An expression's 'plain form' is the most general rendition of that
    /// expression into WGSL, lacking `&` or `*` operators:
    ///
    /// - The plain form of `LocalVariable(x)` is simply `x`, which is a reference
    ///   to the local variable's storage.
    ///
    /// - The plain form of `GlobalVariable(g)` is simply `g`, which is usually a
    ///   reference to the global variable's storage. However, globals in the
    ///   `Handle` address space are immutable, and `GlobalVariable` expressions for
    ///   those produce the value directly, not a pointer to it. Such
    ///   `GlobalVariable` expressions are `Ordinary`.
    ///
    /// - `Access` and `AccessIndex` are `Reference` when their `base` operand is a
    ///   pointer. If they are applied directly to a composite value, they are
    ///   `Ordinary`.
    ///
    /// Note that `FunctionArgument` expressions are never `Reference`, even when
    /// the argument's type is `Pointer`. `FunctionArgument` always evaluates to the
    /// argument's value directly, so any pointer it produces is merely the value
    /// passed by the caller.
    fn plain_form_indirection(
        &self,
        expr: Handle<crate::Expression>,
        module: &Module,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> Indirection {
        use crate::Expression as Ex;

        // Named expressions are `let` expressions, which apply the Load Rule,
        // so if their type is a Naga pointer, then that must be a WGSL pointer
        // as well.
        if self.named_expressions.contains_key(&expr) {
            return Indirection::Ordinary;
        }

        match func_ctx.expressions[expr] {
            Ex::LocalVariable(_) => Indirection::Reference,
            Ex::GlobalVariable(handle) => {
                let global = &module.global_variables[handle];
                match global.space {
                    crate::AddressSpace::Handle => Indirection::Ordinary,
                    _ => Indirection::Reference,
                }
            }
            Ex::Access { base, .. } | Ex::AccessIndex { base, .. } => {
                let base_ty = func_ctx.info[base].ty.inner_with(&module.types);
                match *base_ty {
                    crate::TypeInner::Pointer { .. } | crate::TypeInner::ValuePointer { .. } => {
                        Indirection::Reference
                    }
                    _ => Indirection::Ordinary,
                }
            }
            _ => Indirection::Ordinary,
        }
    }

    fn start_named_expr(
        &mut self,
        module: &Module,
        handle: Handle<crate::Expression>,
        func_ctx: &back::FunctionCtx,
        name: &str,
    ) -> BackendResult {
        // Write variable name
        write!(self.out, "let {}", name)?;
        if self.flags.contains(WriterFlags::EXPLICIT_TYPES) {
            write!(self.out, ": ")?;
            let ty = &func_ctx.info[handle].ty;
            // Write variable type
            match *ty {
                proc::TypeResolution::Handle(handle) => {
                    self.write_type(module, handle)?;
                }
                proc::TypeResolution::Value(ref inner) => {
                    self.write_value_type(module, inner)?;
                }
            }
        }

        write!(self.out, " = ")?;
        Ok(())
    }

    /// Write the ordinary WGSL form of `expr`.
    ///
    /// See `write_expr_with_indirection` for details.
    fn write_expr(
        &mut self,
        module: &Module,
        expr: Handle<crate::Expression>,
        func_ctx: &back::FunctionCtx<'_>,
    ) -> BackendResult {
        self.write_expr_with_indirection(module, expr, func_ctx, Indirection::Ordinary)
    }

    /// Write `expr` as a WGSL expression with the requested indirection.
    ///
    /// In terms of the WGSL grammar, the resulting expression is a
    /// `singular_expression`. It may be parenthesized. This makes it suitable
    /// for use as the operand of a unary or binary operator without worrying
    /// about precedence.
    ///
    /// This does not produce newlines or indentation.
    ///
    /// The `requested` argument indicates (roughly) whether Naga
    /// `Pointer`-valued expressions represent WGSL references or pointers. See
    /// `Indirection` for details.
    fn write_expr_with_indirection(
        &mut self,
        module: &Module,
        expr: Handle<crate::Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        requested: Indirection,
    ) -> BackendResult {
        // If the plain form of the expression is not what we need, emit the
        // operator necessary to correct that.
        let plain = self.plain_form_indirection(expr, module, func_ctx);
        match (requested, plain) {
            (Indirection::Ordinary, Indirection::Reference) => {
                write!(self.out, "(&")?;
                self.write_expr_plain_form(module, expr, func_ctx, plain)?;
                write!(self.out, ")")?;
            }
            (Indirection::Reference, Indirection::Ordinary) => {
                write!(self.out, "(*")?;
                self.write_expr_plain_form(module, expr, func_ctx, plain)?;
                write!(self.out, ")")?;
            }
            (_, _) => self.write_expr_plain_form(module, expr, func_ctx, plain)?,
        }

        Ok(())
    }

    /// Write the 'plain form' of `expr`.
    ///
    /// An expression's 'plain form' is the most general rendition of that
    /// expression into WGSL, lacking `&` or `*` operators. The plain forms of
    /// `LocalVariable(x)` and `GlobalVariable(g)` are simply `x` and `g`. Such
    /// Naga expressions represent both WGSL pointers and references; it's the
    /// caller's responsibility to distinguish those cases appropriately.
    fn write_expr_plain_form(
        &mut self,
        module: &Module,
        expr: Handle<crate::Expression>,
        func_ctx: &back::FunctionCtx<'_>,
        indirection: Indirection,
    ) -> BackendResult {
        use crate::Expression;

        if let Some(name) = self.named_expressions.get(&expr) {
            write!(self.out, "{}", name)?;
            return Ok(());
        }

        let expression = &func_ctx.expressions[expr];

        // Write the plain WGSL form of a Naga expression.
        //
        // The plain form of `LocalVariable` and `GlobalVariable` expressions is
        // simply the variable name; `*` and `&` operators are never emitted.
        //
        // The plain form of `Access` and `AccessIndex` expressions are WGSL
        // `postfix_expression` forms for member/component access and
        // subscripting.
        match *expression {
            Expression::Constant(constant) => self.write_constant(module, constant)?,
            Expression::Compose { ty, ref components } => {
                self.write_type(module, ty)?;
                write!(self.out, "(")?;
                // !spv-in specific notes!
                // WGSL does not support all SPIR-V builtins and we should skip it in generated shaders.
                // We already skip them when we generate struct type.
                // Now we need to find components that used struct with ignored builtins.

                // So, why we can't just return the error to a user?
                // We can, but otherwise, we can't generate WGSL shader from any glslang SPIR-V shaders.
                // glslang generates gl_PerVertex struct with gl_CullDistance, gl_ClipDistance and gl_PointSize builtin inside by default.
                // All of them are not supported by WGSL.

                // We need to copy components to another vec because we don't know which of them we should write.
                let mut components_to_write = Vec::with_capacity(components.len());
                for component in components {
                    let mut skip_component = false;
                    if let Expression::Load { pointer } = func_ctx.expressions[*component] {
                        if let Expression::AccessIndex { base, index } =
                            func_ctx.expressions[pointer]
                        {
                            if access_to_unsupported_builtin(base, index, module, func_ctx.info) {
                                skip_component = true;
                            }
                        }
                    }
                    if skip_component {
                        continue;
                    } else {
                        components_to_write.push(*component);
                    }
                }

                // non spv-in specific notes!
                // Real `Expression::Compose` logic generates here.
                for (index, component) in components_to_write.iter().enumerate() {
                    self.write_expr(module, *component, func_ctx)?;
                    // Only write a comma if isn't the last element
                    if index != components_to_write.len().saturating_sub(1) {
                        // The leading space is for readability only
                        write!(self.out, ", ")?;
                    }
                }
                write!(self.out, ")")?
            }
            Expression::FunctionArgument(pos) => {
                let name_key = func_ctx.argument_key(pos);
                let name = &self.names[&name_key];
                write!(self.out, "{}", name)?;
            }
            Expression::Binary { op, left, right } => {
                write!(self.out, "(")?;
                self.write_expr(module, left, func_ctx)?;
                write!(self.out, " {} ", back::binary_operation_str(op))?;
                self.write_expr(module, right, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Access { base, index } => {
                self.write_expr_with_indirection(module, base, func_ctx, indirection)?;
                write!(self.out, "[")?;
                self.write_expr(module, index, func_ctx)?;
                write!(self.out, "]")?
            }
            Expression::AccessIndex { base, index } => {
                let base_ty_res = &func_ctx.info[base].ty;
                let mut resolved = base_ty_res.inner_with(&module.types);

                self.write_expr_with_indirection(module, base, func_ctx, indirection)?;

                let base_ty_handle = match *resolved {
                    TypeInner::Pointer { base, space: _ } => {
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
                    | TypeInner::BindingArray { .. }
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
                    ref other => return Err(Error::Custom(format!("Cannot index {:?}", other))),
                }
            }
            Expression::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                use crate::SampleLevel as Sl;

                let suffix_cmp = match depth_ref {
                    Some(_) => "Compare",
                    None => "",
                };
                let suffix_level = match level {
                    Sl::Auto => "",
                    Sl::Zero | Sl::Exact(_) => "Level",
                    Sl::Bias(_) => "Bias",
                    Sl::Gradient { .. } => "Grad",
                };

                write!(self.out, "textureSample{}{}(", suffix_cmp, suffix_level)?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;

                if let Some(array_index) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index, func_ctx)?;
                }

                if let Some(depth_ref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.write_expr(module, depth_ref, func_ctx)?;
                }

                match level {
                    Sl::Auto => {}
                    Sl::Zero => {
                        // Level 0 is implied for depth comparison
                        if depth_ref.is_none() {
                            write!(self.out, ", 0.0")?;
                        }
                    }
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
            Expression::ImageSample {
                image,
                sampler,
                gather: Some(component),
                coordinate,
                array_index,
                offset,
                level: _,
                depth_ref,
            } => {
                let suffix_cmp = match depth_ref {
                    Some(_) => "Compare",
                    None => "",
                };

                write!(self.out, "textureGather{}(", suffix_cmp)?;
                match *func_ctx.info[image].ty.inner_with(&module.types) {
                    TypeInner::Image {
                        class: crate::ImageClass::Depth { multi: _ },
                        ..
                    } => {}
                    _ => {
                        write!(self.out, "{}, ", component as u8)?;
                    }
                }
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;

                if let Some(array_index) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index, func_ctx)?;
                }

                if let Some(depth_ref) = depth_ref {
                    write!(self.out, ", ")?;
                    self.write_expr(module, depth_ref, func_ctx)?;
                }

                if let Some(offset) = offset {
                    write!(self.out, ", ")?;
                    self.write_constant(module, offset)?;
                }

                write!(self.out, ")")?;
            }
            Expression::ImageQuery { image, query } => {
                use crate::ImageQuery as Iq;

                let texture_function = match query {
                    Iq::Size { .. } => "textureDimensions",
                    Iq::NumLevels => "textureNumLevels",
                    Iq::NumLayers => "textureNumLayers",
                    Iq::NumSamples => "textureNumSamples",
                };

                write!(self.out, "{}(", texture_function)?;
                self.write_expr(module, image, func_ctx)?;
                if let Iq::Size { level: Some(level) } = query {
                    write!(self.out, ", ")?;
                    self.write_expr(module, level, func_ctx)?;
                };
                write!(self.out, ")")?;
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                write!(self.out, "textureLoad(")?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;
                if let Some(array_index) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index, func_ctx)?;
                }
                if let Some(index) = sample.or(level) {
                    write!(self.out, ", ")?;
                    self.write_expr(module, index, func_ctx)?;
                }
                write!(self.out, ")")?;
            }
            Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{}", name)?;
            }
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let inner = func_ctx.info[expr].ty.inner_with(&module.types);
                match *inner {
                    TypeInner::Matrix { columns, rows, .. } => {
                        write!(
                            self.out,
                            "mat{}x{}<f32>",
                            back::vector_size_str(columns),
                            back::vector_size_str(rows)
                        )?;
                    }
                    TypeInner::Vector { size, .. } => {
                        let vector_size_str = back::vector_size_str(size);
                        let scalar_kind_str = scalar_kind_str(kind);
                        if convert.is_some() {
                            write!(self.out, "vec{}<{}>", vector_size_str, scalar_kind_str)?;
                        } else {
                            write!(
                                self.out,
                                "bitcast<vec{}<{}>>",
                                vector_size_str, scalar_kind_str
                            )?;
                        }
                    }
                    TypeInner::Scalar { .. } => {
                        if convert.is_some() {
                            write!(self.out, "{}", scalar_kind_str(kind))?
                        } else {
                            write!(self.out, "bitcast<{}>", scalar_kind_str(kind))?
                        }
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
            Expression::Splat { size, value } => {
                let inner = func_ctx.info[value].ty.inner_with(&module.types);
                let scalar_kind = match *inner {
                    crate::TypeInner::Scalar { kind, .. } => kind,
                    _ => {
                        return Err(Error::Unimplemented(format!(
                            "write_expr expression::splat {:?}",
                            inner
                        )));
                    }
                };
                let scalar = scalar_kind_str(scalar_kind);
                let size = back::vector_size_str(size);

                write!(self.out, "vec{}<{}>(", size, scalar)?;
                self.write_expr(module, value, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Load { pointer } => {
                let is_atomic = match *func_ctx.info[pointer].ty.inner_with(&module.types) {
                    crate::TypeInner::Pointer { base, .. } => match module.types[base].inner {
                        crate::TypeInner::Atomic { .. } => true,
                        _ => false,
                    },
                    _ => false,
                };

                if is_atomic {
                    write!(self.out, "atomicLoad(")?;
                    self.write_expr(module, pointer, func_ctx)?;
                    write!(self.out, ")")?;
                } else {
                    self.write_expr_with_indirection(
                        module,
                        pointer,
                        func_ctx,
                        Indirection::Reference,
                    )?;
                }
            }
            Expression::LocalVariable(handle) => {
                write!(self.out, "{}", self.names[&func_ctx.name_key(handle)])?
            }
            Expression::ArrayLength(expr) => {
                write!(self.out, "arrayLength(")?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                use crate::MathFunction as Mf;

                enum Function {
                    Asincosh { is_sin: bool },
                    Atanh,
                    Regular(&'static str),
                }

                // NOTE: If https://github.com/gpuweb/gpuweb/issues/1622 ever is
                // accepted, replace this with the builtin functions
                let function = match fun {
                    Mf::Abs => Function::Regular("abs"),
                    Mf::Min => Function::Regular("min"),
                    Mf::Max => Function::Regular("max"),
                    Mf::Clamp => Function::Regular("clamp"),
                    // trigonometry
                    Mf::Cos => Function::Regular("cos"),
                    Mf::Cosh => Function::Regular("cosh"),
                    Mf::Sin => Function::Regular("sin"),
                    Mf::Sinh => Function::Regular("sinh"),
                    Mf::Tan => Function::Regular("tan"),
                    Mf::Tanh => Function::Regular("tanh"),
                    Mf::Acos => Function::Regular("acos"),
                    Mf::Asin => Function::Regular("asin"),
                    Mf::Atan => Function::Regular("atan"),
                    Mf::Atan2 => Function::Regular("atan2"),
                    Mf::Asinh => Function::Asincosh { is_sin: true },
                    Mf::Acosh => Function::Asincosh { is_sin: false },
                    Mf::Atanh => Function::Atanh,
                    Mf::Radians => Function::Regular("radians"),
                    Mf::Degrees => Function::Regular("degrees"),
                    // decomposition
                    Mf::Ceil => Function::Regular("ceil"),
                    Mf::Floor => Function::Regular("floor"),
                    Mf::Round => Function::Regular("round"),
                    Mf::Fract => Function::Regular("fract"),
                    Mf::Trunc => Function::Regular("trunc"),
                    Mf::Modf => Function::Regular("modf"),
                    Mf::Frexp => Function::Regular("frexp"),
                    Mf::Ldexp => Function::Regular("ldexp"),
                    // exponent
                    Mf::Exp => Function::Regular("exp"),
                    Mf::Exp2 => Function::Regular("exp2"),
                    Mf::Log => Function::Regular("log"),
                    Mf::Log2 => Function::Regular("log2"),
                    Mf::Pow => Function::Regular("pow"),
                    // geometry
                    Mf::Dot => Function::Regular("dot"),
                    Mf::Outer => Function::Regular("outerProduct"),
                    Mf::Cross => Function::Regular("cross"),
                    Mf::Distance => Function::Regular("distance"),
                    Mf::Length => Function::Regular("length"),
                    Mf::Normalize => Function::Regular("normalize"),
                    Mf::FaceForward => Function::Regular("faceForward"),
                    Mf::Reflect => Function::Regular("reflect"),
                    // computational
                    Mf::Sign => Function::Regular("sign"),
                    Mf::Fma => Function::Regular("fma"),
                    Mf::Mix => Function::Regular("mix"),
                    Mf::Step => Function::Regular("step"),
                    Mf::SmoothStep => Function::Regular("smoothstep"),
                    Mf::Sqrt => Function::Regular("sqrt"),
                    Mf::InverseSqrt => Function::Regular("inverseSqrt"),
                    Mf::Transpose => Function::Regular("transpose"),
                    Mf::Determinant => Function::Regular("determinant"),
                    // bits
                    Mf::CountOneBits => Function::Regular("countOneBits"),
                    Mf::ReverseBits => Function::Regular("reverseBits"),
                    Mf::ExtractBits => Function::Regular("extractBits"),
                    Mf::InsertBits => Function::Regular("insertBits"),
                    Mf::FindLsb => Function::Regular("firstTrailingBit"),
                    Mf::FindMsb => Function::Regular("firstLeadingBit"),
                    // data packing
                    Mf::Pack4x8snorm => Function::Regular("pack4x8snorm"),
                    Mf::Pack4x8unorm => Function::Regular("pack4x8unorm"),
                    Mf::Pack2x16snorm => Function::Regular("pack2x16snorm"),
                    Mf::Pack2x16unorm => Function::Regular("pack2x16unorm"),
                    Mf::Pack2x16float => Function::Regular("pack2x16float"),
                    // data unpacking
                    Mf::Unpack4x8snorm => Function::Regular("unpack4x8snorm"),
                    Mf::Unpack4x8unorm => Function::Regular("unpack4x8unorm"),
                    Mf::Unpack2x16snorm => Function::Regular("unpack2x16snorm"),
                    Mf::Unpack2x16unorm => Function::Regular("unpack2x16unorm"),
                    Mf::Unpack2x16float => Function::Regular("unpack2x16float"),
                    _ => {
                        return Err(Error::UnsupportedMathFunction(fun));
                    }
                };

                match function {
                    Function::Asincosh { is_sin } => {
                        write!(self.out, "log(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " + sqrt(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " * ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        match is_sin {
                            true => write!(self.out, " + 1.0))")?,
                            false => write!(self.out, " - 1.0))")?,
                        }
                    }
                    Function::Atanh => {
                        write!(self.out, "0.5 * log((1.0 + ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, ") / (1.0 - ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "))")?;
                    }
                    Function::Regular(fun_name) => {
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
                        if let Some(arg) = arg3 {
                            write!(self.out, ", ")?;
                            self.write_expr(module, arg, func_ctx)?;
                        }
                        write!(self.out, ")")?
                    }
                }
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
            Expression::Unary { op, expr } => {
                let unary = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => {
                        match *func_ctx.info[expr].ty.inner_with(&module.types) {
                            TypeInner::Scalar {
                                kind: crate::ScalarKind::Bool,
                                ..
                            }
                            | TypeInner::Vector { .. } => "!",
                            _ => "~",
                        }
                    }
                };

                write!(self.out, "{}(", unary)?;
                self.write_expr(module, expr, func_ctx)?;

                write!(self.out, ")")?
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                write!(self.out, "select(")?;
                self.write_expr(module, reject, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, accept, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, condition, func_ctx)?;
                write!(self.out, ")")?
            }
            Expression::Derivative { axis, expr } => {
                use crate::DerivativeAxis as Da;

                let op = match axis {
                    Da::X => "dpdx",
                    Da::Y => "dpdy",
                    Da::Width => "fwidth",
                };
                write!(self.out, "{}(", op)?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?
            }
            Expression::Relational { fun, argument } => {
                use crate::RelationalFunction as Rf;

                let fun_name = match fun {
                    Rf::IsFinite => "isFinite",
                    Rf::IsNormal => "isNormal",
                    Rf::All => "all",
                    Rf::Any => "any",
                    _ => return Err(Error::UnsupportedRelationalFunction(fun)),
                };
                write!(self.out, "{}(", fun_name)?;

                self.write_expr(module, argument, func_ctx)?;

                write!(self.out, ")")?
            }
            // Nothing to do here, since call expression already cached
            Expression::CallResult(_) | Expression::AtomicResult { .. } => {}
        }

        Ok(())
    }

    /// Helper method used to write global variables
    /// # Notes
    /// Always adds a newline
    fn write_global(
        &mut self,
        module: &Module,
        global: &crate::GlobalVariable,
        handle: Handle<crate::GlobalVariable>,
    ) -> BackendResult {
        // Write group and binding attributes if present
        if let Some(ref binding) = global.binding {
            self.write_attributes(&[
                Attribute::Group(binding.group),
                Attribute::Binding(binding.binding),
            ])?;
            writeln!(self.out)?;
        }

        // First write global name and address space if supported
        write!(self.out, "var")?;
        let (address, maybe_access) = address_space_str(global.space);
        if let Some(space) = address {
            write!(self.out, "<{}", space)?;
            if let Some(access) = maybe_access {
                write!(self.out, ", {}", access)?;
            }
            write!(self.out, ">")?;
        }
        write!(
            self.out,
            " {}: ",
            &self.names[&NameKey::GlobalVariable(handle)]
        )?;

        // Write global type
        self.write_type(module, global.ty)?;

        // Write initializer
        if let Some(init) = global.init {
            write!(self.out, " = ")?;
            self.write_constant(module, init)?;
        }

        // End with semicolon
        writeln!(self.out, ";")?;

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
                    write!(self.out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_scalar_value(*value)?;
                }
            }
            crate::ConstantInner::Composite { ty, ref components } => {
                self.write_type(module, ty)?;
                write!(self.out, "(")?;

                let members = match module.types[ty].inner {
                    TypeInner::Struct { ref members, .. } => Some(members),
                    _ => None,
                };

                // Write the comma separated constants
                for (index, constant) in components.iter().enumerate() {
                    if let Some(&crate::Binding::BuiltIn(built_in)) =
                        members.and_then(|members| members.get(index)?.binding.as_ref())
                    {
                        if builtin_str(built_in).is_none() {
                            log::warn!(
                                "Skip constant for struct member with unsupported builtin {:?}",
                                built_in
                            );
                            continue;
                        }
                    }

                    self.write_constant(module, *constant)?;
                    // Only write a comma if isn't the last element
                    if index != components.len().saturating_sub(1) {
                        // The leading space is for readability only
                        write!(self.out, ", ")?;
                    }
                }
                write!(self.out, ")")?
            }
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
        match *inner {
            crate::ConstantInner::Scalar {
                width: _,
                ref value,
            } => {
                let name = &self.names[&NameKey::Constant(handle)];
                // First write only constant name
                write!(self.out, "let {}: ", name)?;
                // Next write constant type and value
                match *value {
                    crate::ScalarValue::Sint(value) => {
                        write!(self.out, "i32 = {}", value)?;
                    }
                    crate::ScalarValue::Uint(value) => {
                        write!(self.out, "u32 = {}u", value)?;
                    }
                    crate::ScalarValue::Float(value) => {
                        // Floats are written using `Debug` instead of `Display` because it always appends the
                        // decimal part even it's zero
                        write!(self.out, "f32 = {:?}", value)?;
                    }
                    crate::ScalarValue::Bool(value) => {
                        write!(self.out, "bool = {}", value)?;
                    }
                };
                // End with semicolon
                writeln!(self.out, ";")?;
            }
            crate::ConstantInner::Composite { ty, ref components } => {
                let name = &self.names[&NameKey::Constant(handle)];
                // First write only constant name
                write!(self.out, "let {}: ", name)?;
                // Next write constant type
                self.write_type(module, ty)?;

                write!(self.out, " = ")?;
                self.write_type(module, ty)?;

                write!(self.out, "(")?;
                for (index, constant) in components.iter().enumerate() {
                    self.write_constant(module, *constant)?;
                    // Only write a comma if isn't the last element
                    if index != components.len().saturating_sub(1) {
                        // The leading space is for readability only
                        write!(self.out, ", ")?;
                    }
                }
                write!(self.out, ");")?;
            }
        }
        // End with extra newline for readability
        writeln!(self.out)?;
        Ok(())
    }

    // See https://github.com/rust-lang/rust-clippy/issues/4979.
    #[allow(clippy::missing_const_for_fn)]
    pub fn finish(self) -> W {
        self.out
    }
}

const fn builtin_str(built_in: crate::BuiltIn) -> Option<&'static str> {
    use crate::BuiltIn as Bi;

    match built_in {
        Bi::VertexIndex => Some("vertex_index"),
        Bi::InstanceIndex => Some("instance_index"),
        Bi::Position { .. } => Some("position"),
        Bi::FrontFacing => Some("front_facing"),
        Bi::FragDepth => Some("frag_depth"),
        Bi::LocalInvocationId => Some("local_invocation_id"),
        Bi::LocalInvocationIndex => Some("local_invocation_index"),
        Bi::GlobalInvocationId => Some("global_invocation_id"),
        Bi::WorkGroupId => Some("workgroup_id"),
        Bi::WorkGroupSize => Some("workgroup_size"),
        Bi::NumWorkGroups => Some("num_workgroups"),
        Bi::SampleIndex => Some("sample_index"),
        Bi::SampleMask => Some("sample_mask"),
        Bi::PrimitiveIndex => Some("primitive_index"),
        Bi::ViewIndex => Some("view_index"),
        _ => None,
    }
}

const fn image_dimension_str(dim: crate::ImageDimension) -> &'static str {
    use crate::ImageDimension as IDim;

    match dim {
        IDim::D1 => "1d",
        IDim::D2 => "2d",
        IDim::D3 => "3d",
        IDim::Cube => "cube",
    }
}

const fn scalar_kind_str(kind: crate::ScalarKind) -> &'static str {
    use crate::ScalarKind as Sk;

    match kind {
        Sk::Float => "f32",
        Sk::Sint => "i32",
        Sk::Uint => "u32",
        Sk::Bool => "bool",
    }
}

const fn storage_format_str(format: crate::StorageFormat) -> &'static str {
    use crate::StorageFormat as Sf;

    match format {
        Sf::R8Unorm => "r8unorm",
        Sf::R8Snorm => "r8snorm",
        Sf::R8Uint => "r8uint",
        Sf::R8Sint => "r8sint",
        Sf::R16Uint => "r16uint",
        Sf::R16Sint => "r16sint",
        Sf::R16Float => "r16float",
        Sf::Rg8Unorm => "rg8unorm",
        Sf::Rg8Snorm => "rg8snorm",
        Sf::Rg8Uint => "rg8uint",
        Sf::Rg8Sint => "rg8sint",
        Sf::R32Uint => "r32uint",
        Sf::R32Sint => "r32sint",
        Sf::R32Float => "r32float",
        Sf::Rg16Uint => "rg16uint",
        Sf::Rg16Sint => "rg16sint",
        Sf::Rg16Float => "rg16float",
        Sf::Rgba8Unorm => "rgba8unorm",
        Sf::Rgba8Snorm => "rgba8snorm",
        Sf::Rgba8Uint => "rgba8uint",
        Sf::Rgba8Sint => "rgba8sint",
        Sf::Rgb10a2Unorm => "rgb10a2unorm",
        Sf::Rg11b10Float => "rg11b10float",
        Sf::Rg32Uint => "rg32uint",
        Sf::Rg32Sint => "rg32sint",
        Sf::Rg32Float => "rg32float",
        Sf::Rgba16Uint => "rgba16uint",
        Sf::Rgba16Sint => "rgba16sint",
        Sf::Rgba16Float => "rgba16float",
        Sf::Rgba32Uint => "rgba32uint",
        Sf::Rgba32Sint => "rgba32sint",
        Sf::Rgba32Float => "rgba32float",
    }
}

/// Helper function that returns the string corresponding to the WGSL interpolation qualifier
const fn interpolation_str(interpolation: crate::Interpolation) -> &'static str {
    use crate::Interpolation as I;

    match interpolation {
        I::Perspective => "perspective",
        I::Linear => "linear",
        I::Flat => "flat",
    }
}

/// Return the WGSL auxiliary qualifier for the given sampling value.
const fn sampling_str(sampling: crate::Sampling) -> &'static str {
    use crate::Sampling as S;

    match sampling {
        S::Center => "",
        S::Centroid => "centroid",
        S::Sample => "sample",
    }
}

const fn address_space_str(
    space: crate::AddressSpace,
) -> (Option<&'static str>, Option<&'static str>) {
    use crate::AddressSpace as As;

    (
        Some(match space {
            As::Private => "private",
            As::Uniform => "uniform",
            As::Storage { access } => {
                if access.contains(crate::StorageAccess::STORE) {
                    return (Some("storage"), Some("read_write"));
                } else {
                    "storage"
                }
            }
            As::PushConstant => "push_constant",
            As::WorkGroup => "workgroup",
            As::Handle => return (None, None),
            As::Function => "function",
        }),
        None,
    )
}

fn map_binding_to_attribute(
    binding: &crate::Binding,
    scalar_kind: Option<crate::ScalarKind>,
) -> Vec<Attribute> {
    match *binding {
        crate::Binding::BuiltIn(built_in) => {
            if let crate::BuiltIn::Position { invariant: true } = built_in {
                vec![Attribute::BuiltIn(built_in), Attribute::Invariant]
            } else {
                vec![Attribute::BuiltIn(built_in)]
            }
        }
        crate::Binding::Location {
            location,
            interpolation,
            sampling,
        } => match scalar_kind {
            Some(crate::ScalarKind::Float) => vec![
                Attribute::Location(location),
                Attribute::Interpolate(interpolation, sampling),
            ],
            _ => vec![Attribute::Location(location)],
        },
    }
}

/// Helper function that check that expression don't access to structure member with unsupported builtin.
fn access_to_unsupported_builtin(
    expr: Handle<crate::Expression>,
    index: u32,
    module: &Module,
    info: &valid::FunctionInfo,
) -> bool {
    let base_ty_res = &info[expr].ty;
    let resolved = base_ty_res.inner_with(&module.types);
    if let TypeInner::Pointer {
        base: pointer_base_handle,
        ..
    } = *resolved
    {
        // Let's check that we try to access a struct member with unsupported built-in and skip it.
        if let TypeInner::Struct { ref members, .. } = module.types[pointer_base_handle].inner {
            if let Some(crate::Binding::BuiltIn(built_in)) = members[index as usize].binding {
                if builtin_str(built_in).is_none() {
                    log::warn!("Skip component with unsupported builtin {:?}", built_in);
                    return true;
                }
            }
        }
    }

    false
}
