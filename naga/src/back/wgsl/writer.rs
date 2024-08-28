use super::Error;
use crate::{
    back::{self, Baked},
    proc::{self, ExpressionKindTracker, NameKey},
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
    SecondBlendSource,
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
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
            &[],
            &[],
            &["__"],
            &mut self.names,
        );
        self.named_expressions.clear();
        self.ep_results.clear();
    }

    fn is_builtin_wgsl_struct(&self, module: &Module, handle: Handle<crate::Type>) -> bool {
        module
            .special_types
            .predeclared_types
            .values()
            .any(|t| *t == handle)
    }

    pub fn write(&mut self, module: &Module, info: &valid::ModuleInfo) -> BackendResult {
        if !module.overrides.is_empty() {
            return Err(Error::Unimplemented(
                "Pipeline constants are not yet supported for this back-end".to_string(),
            ));
        }

        self.reset(module);

        // Save all ep result types
        for ep in &module.entry_points {
            if let Some(ref result) = ep.function.result {
                self.ep_results.push((ep.stage, result.ty));
            }
        }

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct { ref members, .. } = ty.inner {
                {
                    if !self.is_builtin_wgsl_struct(module, handle) {
                        self.write_struct(module, handle, members)?;
                        writeln!(self.out)?;
                    }
                }
            }
        }

        // Write all named constants
        let mut constants = module
            .constants
            .iter()
            .filter(|&(_, c)| c.name.is_some())
            .peekable();
        while let Some((handle, _)) = constants.next() {
            self.write_global_constant(module, handle)?;
            // Add extra newline for readability on last iteration
            if constants.peek().is_none() {
                writeln!(self.out)?;
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
                expr_kind_tracker: ExpressionKindTracker::from_arena(&function.expressions),
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
                expr_kind_tracker: ExpressionKindTracker::from_arena(&ep.function.expressions),
            };
            self.write_function(module, &ep.function, &func_ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(self.out)?;
            }
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

                write!(self.out, "{name}")?;
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
        write!(self.out, "fn {func_name}(")?;

        // Write function arguments
        for (index, arg) in func.arguments.iter().enumerate() {
            // Write argument attribute if a binding is present
            if let Some(ref binding) = arg.binding {
                self.write_attributes(&map_binding_to_attribute(binding))?;
            }
            // Write argument name
            let argument_name = &self.names[&func_ctx.argument_key(index as u32)];

            write!(self.out, "{argument_name}: ")?;
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
                self.write_attributes(&map_binding_to_attribute(binding))?;
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
                self.write_expr(module, init, func_ctx)?;
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
                Attribute::Location(id) => write!(self.out, "@location({id}) ")?,
                Attribute::SecondBlendSource => write!(self.out, "@second_blend_source ")?,
                Attribute::BuiltIn(builtin_attrib) => {
                    let builtin = builtin_str(builtin_attrib)?;
                    write!(self.out, "@builtin({builtin}) ")?;
                }
                Attribute::Stage(shader_stage) => {
                    let stage_str = match shader_stage {
                        ShaderStage::Vertex => "vertex",
                        ShaderStage::Fragment => "fragment",
                        ShaderStage::Compute => "compute",
                    };
                    write!(self.out, "@{stage_str} ")?;
                }
                Attribute::WorkGroupSize(size) => {
                    write!(
                        self.out,
                        "@workgroup_size({}, {}, {}) ",
                        size[0], size[1], size[2]
                    )?;
                }
                Attribute::Binding(id) => write!(self.out, "@binding({id}) ")?,
                Attribute::Group(id) => write!(self.out, "@group({id}) ")?,
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
            // The indentation is only for readability
            write!(self.out, "{}", back::INDENT)?;
            if let Some(ref binding) = member.binding {
                self.write_attributes(&map_binding_to_attribute(binding))?;
            }
            // Write struct member name and type
            let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];
            write!(self.out, "{member_name}: ")?;
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
            TypeInner::Vector { size, scalar } => write!(
                self.out,
                "vec{}<{}>",
                back::vector_size_str(size),
                scalar_kind_str(scalar),
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
                        scalar_kind_str(crate::Scalar { kind, width: 4 }),
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
                    "texture_{class_str}{multisampled_str}{dim_str}{arrayed_str}"
                )?;

                if !format_str.is_empty() {
                    write!(self.out, "<{format_str}{storage_str}>")?;
                }
            }
            TypeInner::Scalar(scalar) => {
                write!(self.out, "{}", scalar_kind_str(scalar))?;
            }
            TypeInner::Atomic(scalar) => {
                write!(self.out, "atomic<{}>", scalar_kind_str(scalar))?;
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
                    crate::ArraySize::Constant(len) => {
                        self.write_type(module, base)?;
                        write!(self.out, ", {len}")?;
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
                    crate::ArraySize::Constant(len) => {
                        self.write_type(module, base)?;
                        write!(self.out, ", {len}")?;
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
                scalar,
            } => {
                write!(
                    self.out,
                    "mat{}x{}<{}>",
                    back::vector_size_str(columns),
                    back::vector_size_str(rows),
                    scalar_kind_str(scalar)
                )?;
            }
            TypeInner::Pointer { base, space } => {
                let (address, maybe_access) = address_space_str(space);
                // Everything but `AddressSpace::Handle` gives us a `address` name, but
                // Naga IR never produces pointers to handles, so it doesn't matter much
                // how we write such a type. Just write it as the base type alone.
                if let Some(space) = address {
                    write!(self.out, "ptr<{space}, ")?;
                }
                self.write_type(module, base)?;
                if address.is_some() {
                    if let Some(access) = maybe_access {
                        write!(self.out, ", {access}")?;
                    }
                    write!(self.out, ">")?;
                }
            }
            TypeInner::ValuePointer {
                size: None,
                scalar,
                space,
            } => {
                let (address, maybe_access) = address_space_str(space);
                if let Some(space) = address {
                    write!(self.out, "ptr<{}, {}", space, scalar_kind_str(scalar))?;
                    if let Some(access) = maybe_access {
                        write!(self.out, ", {access}")?;
                    }
                    write!(self.out, ">")?;
                } else {
                    return Err(Error::Unimplemented(format!(
                        "ValuePointer to AddressSpace::Handle {inner:?}"
                    )));
                }
            }
            TypeInner::ValuePointer {
                size: Some(size),
                scalar,
                space,
            } => {
                let (address, maybe_access) = address_space_str(space);
                if let Some(space) = address {
                    write!(
                        self.out,
                        "ptr<{}, vec{}<{}>",
                        space,
                        back::vector_size_str(size),
                        scalar_kind_str(scalar)
                    )?;
                    if let Some(access) = maybe_access {
                        write!(self.out, ", {access}")?;
                    }
                    write!(self.out, ">")?;
                } else {
                    return Err(Error::Unimplemented(format!(
                        "ValuePointer to AddressSpace::Handle {inner:?}"
                    )));
                }
                write!(self.out, ">")?;
            }
            TypeInner::AccelerationStructure => write!(self.out, "acceleration_structure")?,
            _ => {
                return Err(Error::Unimplemented(format!("write_value_type {inner:?}")));
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
                            Some(Baked(handle).to_string())
                        } else {
                            None
                        }
                    };

                    if let Some(name) = expr_name {
                        write!(self.out, "{level}")?;
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
                write!(self.out, "{level}")?;
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
                    writeln!(self.out, "{level}}} else {{")?;

                    for sta in reject {
                        // Increase indentation to help with readability
                        self.write_stmt(module, sta, func_ctx, l2)?;
                    }
                }

                writeln!(self.out, "{level}}}")?
            }
            Statement::Return { value } => {
                write!(self.out, "{level}")?;
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
                write!(self.out, "{level}")?;
                writeln!(self.out, "discard;")?
            }
            Statement::Store { pointer, value } => {
                write!(self.out, "{level}")?;

                let is_atomic_pointer = func_ctx
                    .resolve_type(pointer, &module.types)
                    .is_atomic_pointer(&module.types);

                if is_atomic_pointer {
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
                write!(self.out, "{level}")?;
                if let Some(expr) = result {
                    let name = Baked(expr).to_string();
                    self.start_named_expr(module, expr, func_ctx, &name)?;
                    self.named_expressions.insert(expr, name);
                }
                let func_name = &self.names[&NameKey::Function(function)];
                write!(self.out, "{func_name}(")?;
                for (index, &argument) in arguments.iter().enumerate() {
                    if index != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.write_expr(module, argument, func_ctx)?;
                }
                writeln!(self.out, ");")?
            }
            Statement::Atomic {
                pointer,
                ref fun,
                value,
                result,
            } => {
                write!(self.out, "{level}")?;
                if let Some(result) = result {
                    let res_name = Baked(result).to_string();
                    self.start_named_expr(module, result, func_ctx, &res_name)?;
                    self.named_expressions.insert(result, res_name);
                }

                let fun_str = fun.to_wgsl();
                write!(self.out, "atomic{fun_str}(")?;
                self.write_expr(module, pointer, func_ctx)?;
                if let crate::AtomicFunction::Exchange { compare: Some(cmp) } = *fun {
                    write!(self.out, ", ")?;
                    self.write_expr(module, cmp, func_ctx)?;
                }
                write!(self.out, ", ")?;
                self.write_expr(module, value, func_ctx)?;
                writeln!(self.out, ");")?
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                write!(self.out, "{level}")?;
                // TODO: Obey named expressions here.
                let res_name = Baked(result).to_string();
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);
                write!(self.out, "workgroupUniformLoad(")?;
                self.write_expr(module, pointer, func_ctx)?;
                writeln!(self.out, ");")?;
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                write!(self.out, "{level}")?;
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
                write!(self.out, "{level}")?;
                writeln!(self.out, "{{")?;
                for sta in block.iter() {
                    // Increase the indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, level.next())?
                }
                writeln!(self.out, "{level}}}")?
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Start the switch
                write!(self.out, "{level}")?;
                write!(self.out, "switch ")?;
                self.write_expr(module, selector, func_ctx)?;
                writeln!(self.out, " {{")?;

                let l2 = level.next();
                let mut new_case = true;
                for case in cases {
                    if case.fall_through && !case.body.is_empty() {
                        // TODO: we could do the same workaround as we did for the HLSL backend
                        return Err(Error::Unimplemented(
                            "fall-through switch case block".into(),
                        ));
                    }

                    match case.value {
                        crate::SwitchValue::I32(value) => {
                            if new_case {
                                write!(self.out, "{l2}case ")?;
                            }
                            write!(self.out, "{value}")?;
                        }
                        crate::SwitchValue::U32(value) => {
                            if new_case {
                                write!(self.out, "{l2}case ")?;
                            }
                            write!(self.out, "{value}u")?;
                        }
                        crate::SwitchValue::Default => {
                            if new_case {
                                if case.fall_through {
                                    write!(self.out, "{l2}case ")?;
                                } else {
                                    write!(self.out, "{l2}")?;
                                }
                            }
                            write!(self.out, "default")?;
                        }
                    }

                    new_case = !case.fall_through;

                    if case.fall_through {
                        write!(self.out, ", ")?;
                    } else {
                        writeln!(self.out, ": {{")?;
                    }

                    for sta in case.body.iter() {
                        self.write_stmt(module, sta, func_ctx, l2.next())?;
                    }

                    if !case.fall_through {
                        writeln!(self.out, "{l2}}}")?;
                    }
                }

                writeln!(self.out, "{level}}}")?
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                write!(self.out, "{level}")?;
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
                    writeln!(self.out, "{l2}continuing {{")?;
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

                    writeln!(self.out, "{l2}}}")?;
                }

                writeln!(self.out, "{level}}}")?
            }
            Statement::Break => {
                writeln!(self.out, "{level}break;")?;
            }
            Statement::Continue => {
                writeln!(self.out, "{level}continue;")?;
            }
            Statement::Barrier(barrier) => {
                if barrier.contains(crate::Barrier::STORAGE) {
                    writeln!(self.out, "{level}storageBarrier();")?;
                }

                if barrier.contains(crate::Barrier::WORK_GROUP) {
                    writeln!(self.out, "{level}workgroupBarrier();")?;
                }

                if barrier.contains(crate::Barrier::SUB_GROUP) {
                    writeln!(self.out, "{level}subgroupBarrier();")?;
                }
            }
            Statement::RayQuery { .. } => unreachable!(),
            Statement::SubgroupBallot { result, predicate } => {
                write!(self.out, "{level}")?;
                let res_name = Baked(result).to_string();
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);

                write!(self.out, "subgroupBallot(")?;
                if let Some(predicate) = predicate {
                    self.write_expr(module, predicate, func_ctx)?;
                }
                writeln!(self.out, ");")?;
            }
            Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => {
                write!(self.out, "{level}")?;
                let res_name = Baked(result).to_string();
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);

                match (collective_op, op) {
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::All) => {
                        write!(self.out, "subgroupAll(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Any) => {
                        write!(self.out, "subgroupAny(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Add) => {
                        write!(self.out, "subgroupAdd(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Mul) => {
                        write!(self.out, "subgroupMul(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Max) => {
                        write!(self.out, "subgroupMax(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Min) => {
                        write!(self.out, "subgroupMin(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::And) => {
                        write!(self.out, "subgroupAnd(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Or) => {
                        write!(self.out, "subgroupOr(")?
                    }
                    (crate::CollectiveOperation::Reduce, crate::SubgroupOperation::Xor) => {
                        write!(self.out, "subgroupXor(")?
                    }
                    (crate::CollectiveOperation::ExclusiveScan, crate::SubgroupOperation::Add) => {
                        write!(self.out, "subgroupExclusiveAdd(")?
                    }
                    (crate::CollectiveOperation::ExclusiveScan, crate::SubgroupOperation::Mul) => {
                        write!(self.out, "subgroupExclusiveMul(")?
                    }
                    (crate::CollectiveOperation::InclusiveScan, crate::SubgroupOperation::Add) => {
                        write!(self.out, "subgroupInclusiveAdd(")?
                    }
                    (crate::CollectiveOperation::InclusiveScan, crate::SubgroupOperation::Mul) => {
                        write!(self.out, "subgroupInclusiveMul(")?
                    }
                    _ => unimplemented!(),
                }
                self.write_expr(module, argument, func_ctx)?;
                writeln!(self.out, ");")?;
            }
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                write!(self.out, "{level}")?;
                let res_name = Baked(result).to_string();
                self.start_named_expr(module, result, func_ctx, &res_name)?;
                self.named_expressions.insert(result, res_name);

                match mode {
                    crate::GatherMode::BroadcastFirst => {
                        write!(self.out, "subgroupBroadcastFirst(")?;
                    }
                    crate::GatherMode::Broadcast(_) => {
                        write!(self.out, "subgroupBroadcast(")?;
                    }
                    crate::GatherMode::Shuffle(_) => {
                        write!(self.out, "subgroupShuffle(")?;
                    }
                    crate::GatherMode::ShuffleDown(_) => {
                        write!(self.out, "subgroupShuffleDown(")?;
                    }
                    crate::GatherMode::ShuffleUp(_) => {
                        write!(self.out, "subgroupShuffleUp(")?;
                    }
                    crate::GatherMode::ShuffleXor(_) => {
                        write!(self.out, "subgroupShuffleXor(")?;
                    }
                }
                self.write_expr(module, argument, func_ctx)?;
                match mode {
                    crate::GatherMode::BroadcastFirst => {}
                    crate::GatherMode::Broadcast(index)
                    | crate::GatherMode::Shuffle(index)
                    | crate::GatherMode::ShuffleDown(index)
                    | crate::GatherMode::ShuffleUp(index)
                    | crate::GatherMode::ShuffleXor(index) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, index, func_ctx)?;
                    }
                }
                writeln!(self.out, ");")?;
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
                let base_ty = func_ctx.resolve_type(base, &module.types);
                match *base_ty {
                    TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => {
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
        // Some functions are marked as const, but are not yet implemented as constant expression
        let quantifier = if func_ctx.expr_kind_tracker.is_impl_const(handle) {
            "const"
        } else {
            "let"
        };
        // Write variable name
        write!(self.out, "{quantifier} {name}")?;
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

    fn write_const_expression(
        &mut self,
        module: &Module,
        expr: Handle<crate::Expression>,
    ) -> BackendResult {
        self.write_possibly_const_expression(
            module,
            expr,
            &module.global_expressions,
            |writer, expr| writer.write_const_expression(module, expr),
        )
    }

    fn write_possibly_const_expression<E>(
        &mut self,
        module: &Module,
        expr: Handle<crate::Expression>,
        expressions: &crate::Arena<crate::Expression>,
        write_expression: E,
    ) -> BackendResult
    where
        E: Fn(&mut Self, Handle<crate::Expression>) -> BackendResult,
    {
        use crate::Expression;

        match expressions[expr] {
            Expression::Literal(literal) => match literal {
                crate::Literal::F32(value) => write!(self.out, "{}f", value)?,
                crate::Literal::U32(value) => write!(self.out, "{}u", value)?,
                crate::Literal::I32(value) => {
                    // `-2147483648i` is not valid WGSL. The most negative `i32`
                    // value can only be expressed in WGSL using AbstractInt and
                    // a unary negation operator.
                    if value == i32::MIN {
                        write!(self.out, "i32({})", value)?;
                    } else {
                        write!(self.out, "{}i", value)?;
                    }
                }
                crate::Literal::Bool(value) => write!(self.out, "{}", value)?,
                crate::Literal::F64(value) => write!(self.out, "{:?}lf", value)?,
                crate::Literal::I64(value) => {
                    // `-9223372036854775808li` is not valid WGSL. The most negative `i64`
                    // value can only be expressed in WGSL using AbstractInt and
                    // a unary negation operator.
                    if value == i64::MIN {
                        write!(self.out, "i64({})", value)?;
                    } else {
                        write!(self.out, "{}li", value)?;
                    }
                }
                crate::Literal::U64(value) => write!(self.out, "{:?}lu", value)?,
                crate::Literal::AbstractInt(_) | crate::Literal::AbstractFloat(_) => {
                    return Err(Error::Custom(
                        "Abstract types should not appear in IR presented to backends".into(),
                    ));
                }
            },
            Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    write!(self.out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_const_expression(module, constant.init)?;
                }
            }
            Expression::ZeroValue(ty) => {
                self.write_type(module, ty)?;
                write!(self.out, "()")?;
            }
            Expression::Compose { ty, ref components } => {
                self.write_type(module, ty)?;
                write!(self.out, "(")?;
                for (index, component) in components.iter().enumerate() {
                    if index != 0 {
                        write!(self.out, ", ")?;
                    }
                    write_expression(self, *component)?;
                }
                write!(self.out, ")")?
            }
            Expression::Splat { size, value } => {
                let size = back::vector_size_str(size);
                write!(self.out, "vec{size}(")?;
                write_expression(self, value)?;
                write!(self.out, ")")?;
            }
            _ => unreachable!(),
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
            write!(self.out, "{name}")?;
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
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::ZeroValue(_)
            | Expression::Compose { .. }
            | Expression::Splat { .. } => {
                self.write_possibly_const_expression(
                    module,
                    expr,
                    func_ctx.expressions,
                    |writer, expr| writer.write_expr(module, expr, func_ctx),
                )?;
            }
            Expression::Override(_) => unreachable!(),
            Expression::FunctionArgument(pos) => {
                let name_key = func_ctx.argument_key(pos);
                let name = &self.names[&name_key];
                write!(self.out, "{name}")?;
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
                    | TypeInner::ValuePointer { .. } => write!(self.out, "[{index}]")?,
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
                    ref other => return Err(Error::Custom(format!("Cannot index {other:?}"))),
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

                write!(self.out, "textureSample{suffix_cmp}{suffix_level}(")?;
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
                    self.write_const_expression(module, offset)?;
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

                write!(self.out, "textureGather{suffix_cmp}(")?;
                match *func_ctx.resolve_type(image, &module.types) {
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
                    self.write_const_expression(module, offset)?;
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

                write!(self.out, "{texture_function}(")?;
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
                write!(self.out, "{name}")?;
            }

            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let inner = func_ctx.resolve_type(expr, &module.types);
                match *inner {
                    TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    } => {
                        let scalar = crate::Scalar {
                            kind,
                            width: convert.unwrap_or(scalar.width),
                        };
                        let scalar_kind_str = scalar_kind_str(scalar);
                        write!(
                            self.out,
                            "mat{}x{}<{}>",
                            back::vector_size_str(columns),
                            back::vector_size_str(rows),
                            scalar_kind_str
                        )?;
                    }
                    TypeInner::Vector {
                        size,
                        scalar: crate::Scalar { width, .. },
                    } => {
                        let scalar = crate::Scalar {
                            kind,
                            width: convert.unwrap_or(width),
                        };
                        let vector_size_str = back::vector_size_str(size);
                        let scalar_kind_str = scalar_kind_str(scalar);
                        if convert.is_some() {
                            write!(self.out, "vec{vector_size_str}<{scalar_kind_str}>")?;
                        } else {
                            write!(self.out, "bitcast<vec{vector_size_str}<{scalar_kind_str}>>")?;
                        }
                    }
                    TypeInner::Scalar(crate::Scalar { width, .. }) => {
                        let scalar = crate::Scalar {
                            kind,
                            width: convert.unwrap_or(width),
                        };
                        let scalar_kind_str = scalar_kind_str(scalar);
                        if convert.is_some() {
                            write!(self.out, "{scalar_kind_str}")?
                        } else {
                            write!(self.out, "bitcast<{scalar_kind_str}>")?
                        }
                    }
                    _ => {
                        return Err(Error::Unimplemented(format!(
                            "write_expr expression::as {inner:?}"
                        )));
                    }
                };
                write!(self.out, "(")?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Load { pointer } => {
                let is_atomic_pointer = func_ctx
                    .resolve_type(pointer, &module.types)
                    .is_atomic_pointer(&module.types);

                if is_atomic_pointer {
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
                    Regular(&'static str),
                }

                let function = match fun {
                    Mf::Abs => Function::Regular("abs"),
                    Mf::Min => Function::Regular("min"),
                    Mf::Max => Function::Regular("max"),
                    Mf::Clamp => Function::Regular("clamp"),
                    Mf::Saturate => Function::Regular("saturate"),
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
                    Mf::Asinh => Function::Regular("asinh"),
                    Mf::Acosh => Function::Regular("acosh"),
                    Mf::Atanh => Function::Regular("atanh"),
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
                    Mf::Cross => Function::Regular("cross"),
                    Mf::Distance => Function::Regular("distance"),
                    Mf::Length => Function::Regular("length"),
                    Mf::Normalize => Function::Regular("normalize"),
                    Mf::FaceForward => Function::Regular("faceForward"),
                    Mf::Reflect => Function::Regular("reflect"),
                    Mf::Refract => Function::Regular("refract"),
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
                    Mf::CountTrailingZeros => Function::Regular("countTrailingZeros"),
                    Mf::CountLeadingZeros => Function::Regular("countLeadingZeros"),
                    Mf::CountOneBits => Function::Regular("countOneBits"),
                    Mf::ReverseBits => Function::Regular("reverseBits"),
                    Mf::ExtractBits => Function::Regular("extractBits"),
                    Mf::InsertBits => Function::Regular("insertBits"),
                    Mf::FirstTrailingBit => Function::Regular("firstTrailingBit"),
                    Mf::FirstLeadingBit => Function::Regular("firstLeadingBit"),
                    // data packing
                    Mf::Pack4x8snorm => Function::Regular("pack4x8snorm"),
                    Mf::Pack4x8unorm => Function::Regular("pack4x8unorm"),
                    Mf::Pack2x16snorm => Function::Regular("pack2x16snorm"),
                    Mf::Pack2x16unorm => Function::Regular("pack2x16unorm"),
                    Mf::Pack2x16float => Function::Regular("pack2x16float"),
                    Mf::Pack4xI8 => Function::Regular("pack4xI8"),
                    Mf::Pack4xU8 => Function::Regular("pack4xU8"),
                    // data unpacking
                    Mf::Unpack4x8snorm => Function::Regular("unpack4x8snorm"),
                    Mf::Unpack4x8unorm => Function::Regular("unpack4x8unorm"),
                    Mf::Unpack2x16snorm => Function::Regular("unpack2x16snorm"),
                    Mf::Unpack2x16unorm => Function::Regular("unpack2x16unorm"),
                    Mf::Unpack2x16float => Function::Regular("unpack2x16float"),
                    Mf::Unpack4xI8 => Function::Regular("unpack4xI8"),
                    Mf::Unpack4xU8 => Function::Regular("unpack4xU8"),
                    Mf::Inverse | Mf::Outer => {
                        return Err(Error::UnsupportedMathFunction(fun));
                    }
                };

                match function {
                    Function::Regular(fun_name) => {
                        write!(self.out, "{fun_name}(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        for arg in IntoIterator::into_iter([arg1, arg2, arg3]).flatten() {
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
                    crate::UnaryOperator::LogicalNot => "!",
                    crate::UnaryOperator::BitwiseNot => "~",
                };

                write!(self.out, "{unary}(")?;
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
            Expression::Derivative { axis, ctrl, expr } => {
                use crate::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
                let op = match (axis, ctrl) {
                    (Axis::X, Ctrl::Coarse) => "dpdxCoarse",
                    (Axis::X, Ctrl::Fine) => "dpdxFine",
                    (Axis::X, Ctrl::None) => "dpdx",
                    (Axis::Y, Ctrl::Coarse) => "dpdyCoarse",
                    (Axis::Y, Ctrl::Fine) => "dpdyFine",
                    (Axis::Y, Ctrl::None) => "dpdy",
                    (Axis::Width, Ctrl::Coarse) => "fwidthCoarse",
                    (Axis::Width, Ctrl::Fine) => "fwidthFine",
                    (Axis::Width, Ctrl::None) => "fwidth",
                };
                write!(self.out, "{op}(")?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?
            }
            Expression::Relational { fun, argument } => {
                use crate::RelationalFunction as Rf;

                let fun_name = match fun {
                    Rf::All => "all",
                    Rf::Any => "any",
                    _ => return Err(Error::UnsupportedRelationalFunction(fun)),
                };
                write!(self.out, "{fun_name}(")?;

                self.write_expr(module, argument, func_ctx)?;

                write!(self.out, ")")?
            }
            // Not supported yet
            Expression::RayQueryGetIntersection { .. } => unreachable!(),
            // Nothing to do here, since call expression already cached
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. } => {}
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
            write!(self.out, "<{space}")?;
            if let Some(access) = maybe_access {
                write!(self.out, ", {access}")?;
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
            self.write_const_expression(module, init)?;
        }

        // End with semicolon
        writeln!(self.out, ";")?;

        Ok(())
    }

    /// Helper method used to write global constants
    ///
    /// # Notes
    /// Ends in a newline
    fn write_global_constant(
        &mut self,
        module: &Module,
        handle: Handle<crate::Constant>,
    ) -> BackendResult {
        let name = &self.names[&NameKey::Constant(handle)];
        // First write only constant name
        write!(self.out, "const {name}: ")?;
        self.write_type(module, module.constants[handle].ty)?;
        write!(self.out, " = ")?;
        let init = module.constants[handle].init;
        self.write_const_expression(module, init)?;
        writeln!(self.out, ";")?;

        Ok(())
    }

    // See https://github.com/rust-lang/rust-clippy/issues/4979.
    #[allow(clippy::missing_const_for_fn)]
    pub fn finish(self) -> W {
        self.out
    }
}

fn builtin_str(built_in: crate::BuiltIn) -> Result<&'static str, Error> {
    use crate::BuiltIn as Bi;

    Ok(match built_in {
        Bi::VertexIndex => "vertex_index",
        Bi::InstanceIndex => "instance_index",
        Bi::Position { .. } => "position",
        Bi::FrontFacing => "front_facing",
        Bi::FragDepth => "frag_depth",
        Bi::LocalInvocationId => "local_invocation_id",
        Bi::LocalInvocationIndex => "local_invocation_index",
        Bi::GlobalInvocationId => "global_invocation_id",
        Bi::WorkGroupId => "workgroup_id",
        Bi::NumWorkGroups => "num_workgroups",
        Bi::SampleIndex => "sample_index",
        Bi::SampleMask => "sample_mask",
        Bi::PrimitiveIndex => "primitive_index",
        Bi::ViewIndex => "view_index",
        Bi::NumSubgroups => "num_subgroups",
        Bi::SubgroupId => "subgroup_id",
        Bi::SubgroupSize => "subgroup_size",
        Bi::SubgroupInvocationId => "subgroup_invocation_id",
        Bi::BaseInstance
        | Bi::BaseVertex
        | Bi::ClipDistance
        | Bi::CullDistance
        | Bi::PointSize
        | Bi::PointCoord
        | Bi::WorkGroupSize => {
            return Err(Error::Custom(format!("Unsupported builtin {built_in:?}")))
        }
    })
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

const fn scalar_kind_str(scalar: crate::Scalar) -> &'static str {
    use crate::Scalar;
    use crate::ScalarKind as Sk;

    match scalar {
        Scalar {
            kind: Sk::Float,
            width: 8,
        } => "f64",
        Scalar {
            kind: Sk::Float,
            width: 4,
        } => "f32",
        Scalar {
            kind: Sk::Sint,
            width: 4,
        } => "i32",
        Scalar {
            kind: Sk::Uint,
            width: 4,
        } => "u32",
        Scalar {
            kind: Sk::Sint,
            width: 8,
        } => "i64",
        Scalar {
            kind: Sk::Uint,
            width: 8,
        } => "u64",
        Scalar {
            kind: Sk::Bool,
            width: 1,
        } => "bool",
        _ => unreachable!(),
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
        Sf::Bgra8Unorm => "bgra8unorm",
        Sf::Rgb10a2Uint => "rgb10a2uint",
        Sf::Rgb10a2Unorm => "rgb10a2unorm",
        Sf::Rg11b10UFloat => "rg11b10float",
        Sf::Rg32Uint => "rg32uint",
        Sf::Rg32Sint => "rg32sint",
        Sf::Rg32Float => "rg32float",
        Sf::Rgba16Uint => "rgba16uint",
        Sf::Rgba16Sint => "rgba16sint",
        Sf::Rgba16Float => "rgba16float",
        Sf::Rgba32Uint => "rgba32uint",
        Sf::Rgba32Sint => "rgba32sint",
        Sf::Rgba32Float => "rgba32float",
        Sf::R16Unorm => "r16unorm",
        Sf::R16Snorm => "r16snorm",
        Sf::Rg16Unorm => "rg16unorm",
        Sf::Rg16Snorm => "rg16snorm",
        Sf::Rgba16Unorm => "rgba16unorm",
        Sf::Rgba16Snorm => "rgba16snorm",
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
        S::First => "first",
        S::Either => "either",
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

fn map_binding_to_attribute(binding: &crate::Binding) -> Vec<Attribute> {
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
            second_blend_source: false,
        } => vec![
            Attribute::Location(location),
            Attribute::Interpolate(interpolation, sampling),
        ],
        crate::Binding::Location {
            location,
            interpolation,
            sampling,
            second_blend_source: true,
        } => vec![
            Attribute::Location(location),
            Attribute::SecondBlendSource,
            Attribute::Interpolate(interpolation, sampling),
        ],
    }
}
