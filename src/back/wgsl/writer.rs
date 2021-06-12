use super::Error;
use crate::{
    back::{binary_operation_str, vector_size_str, wgsl::keywords::RESERVED},
    proc::{EntryPointIndex, NameKey, Namer, TypeResolution},
    valid::{FunctionInfo, ModuleInfo},
    Arena, ArraySize, Binding, Constant, ConstantInner, Expression, FastHashMap, Function,
    GlobalVariable, Handle, ImageClass, ImageDimension, Interpolation, LocalVariable, Module,
    SampleLevel, Sampling, ScalarKind, ScalarValue, ShaderStage, Statement, StorageClass,
    StorageFormat, StructMember, Type, TypeInner,
};
use std::fmt::Write;

const INDENT: &str = "    ";
const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];
const BAKE_PREFIX: &str = "_e";

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

/// WGSL attribute
/// https://gpuweb.github.io/gpuweb/wgsl/#attributes
enum Attribute {
    Access(crate::StorageAccess),
    Binding(u32),
    Block,
    BuiltIn(crate::BuiltIn),
    Group(u32),
    Interpolate(Option<Interpolation>, Option<Sampling>),
    Location(u32),
    Stage(ShaderStage),
    Stride(u32),
    WorkGroupSize([u32; 3]),
}

/// Stores the current function type (either a regular function or an entry point)
///
/// Also stores data needed to identify it (handle for a regular function or index for an entry point)
// TODO: copy-paste from glsl-out
enum FunctionType {
    /// A regular function and it's handle
    Function(Handle<Function>),
    /// A entry point and it's index
    EntryPoint(EntryPointIndex),
}

/// Helper structure that stores data needed when writing the function
// TODO: copy-paste from glsl-out
struct FunctionCtx<'a> {
    /// The current function type being written
    ty: FunctionType,
    /// Analysis about the function
    info: &'a FunctionInfo,
    /// The expression arena of the current function being written
    expressions: &'a Arena<Expression>,
    /// Map of expressions that have associated variable names
    named_expressions: &'a crate::NamedExpressions,
}

impl<'a> FunctionCtx<'_> {
    /// Helper method that generates a [`NameKey`](crate::proc::NameKey) for a local in the current function
    fn name_key(&self, local: Handle<LocalVariable>) -> NameKey {
        match self.ty {
            FunctionType::Function(handle) => NameKey::FunctionLocal(handle, local),
            FunctionType::EntryPoint(idx) => NameKey::EntryPointLocal(idx, local),
        }
    }
}

pub struct Writer<W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    namer: Namer,
    named_expressions: crate::NamedExpressions,
    ep_results: Vec<(ShaderStage, Handle<Type>)>,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W) -> Self {
        Writer {
            out,
            names: FastHashMap::default(),
            namer: Namer::default(),
            named_expressions: crate::NamedExpressions::default(),
            ep_results: vec![],
        }
    }

    fn reset(&mut self, module: &Module) {
        self.names.clear();
        self.namer.reset(module, RESERVED, &[], &mut self.names);
        self.named_expressions.clear();
        self.ep_results.clear();
    }

    pub fn write(&mut self, module: &Module, info: &ModuleInfo) -> BackendResult {
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
                top_level,
                ref members,
                ..
            } = ty.inner
            {
                self.write_struct(module, handle, top_level, members)?;
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

            let func_ctx = FunctionCtx {
                ty: FunctionType::Function(handle),
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

            self.write_attributes(&attributes, false)?;
            // Add a newline after attribute
            writeln!(self.out)?;

            let func_ctx = FunctionCtx {
                ty: FunctionType::EntryPoint(index as u16),
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

    /// Helper method used to write [`ScalarValue`](ScalarValue)
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_scalar_value(&mut self, value: ScalarValue) -> BackendResult {
        match value {
            ScalarValue::Sint(value) => write!(self.out, "{}", value)?,
            ScalarValue::Uint(value) => write!(self.out, "{}u", value)?,
            // Floats are written using `Debug` instead of `Display` because it always appends the
            // decimal part even it's zero
            ScalarValue::Float(value) => write!(self.out, "{:?}", value)?,
            ScalarValue::Bool(value) => write!(self.out, "{}", value)?,
        }

        Ok(())
    }

    /// Helper method used to write stuct name
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_struct_name(&mut self, module: &Module, handle: Handle<Type>) -> BackendResult {
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

    /// Helper method used to write structs
    /// https://gpuweb.github.io/gpuweb/wgsl/#functions
    ///
    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        module: &Module,
        func: &Function,
        func_ctx: &FunctionCtx<'_>,
    ) -> BackendResult {
        let func_name = match func_ctx.ty {
            FunctionType::EntryPoint(index) => self.names[&NameKey::EntryPoint(index)].clone(),
            FunctionType::Function(handle) => self.names[&NameKey::Function(handle)].clone(),
        };

        // Write function name
        write!(self.out, "fn {}(", func_name)?;

        // Write function arguments
        for (index, arg) in func.arguments.iter().enumerate() {
            // Write argument attribute if a binding is present
            if let Some(ref binding) = arg.binding {
                self.write_attributes(&map_binding_to_attribute(binding), false)?;
                write!(self.out, " ")?;
            }
            // Write argument name
            let argument_name = match func_ctx.ty {
                FunctionType::Function(handle) => {
                    self.names[&NameKey::FunctionArgument(handle, index as u32)].clone()
                }
                FunctionType::EntryPoint(ep_index) => {
                    self.names[&NameKey::EntryPointArgument(ep_index, index as u32)].clone()
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
                self.write_attributes(&map_binding_to_attribute(binding), true)?;
            }
            self.write_type(module, result.ty)?;
        }

        write!(self.out, " {{")?;
        writeln!(self.out)?;

        // Write function local variables
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability)
            write!(self.out, "{}", INDENT)?;

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
            self.write_stmt(module, sta, func_ctx, 1)?;
        }

        writeln!(self.out, "}}")?;

        self.named_expressions.clear();

        Ok(())
    }

    /// Helper method to write a attribute
    ///
    /// # Notes
    /// Adds an extra space if required
    fn write_attributes(&mut self, attributes: &[Attribute], extra_space: bool) -> BackendResult {
        let mut attributes_str = String::new();
        for (index, attribute) in attributes.iter().enumerate() {
            let attribute_str = match *attribute {
                Attribute::Access(access) => {
                    let access_str = if access.is_all() {
                        "read_write"
                    } else if access.contains(crate::StorageAccess::LOAD) {
                        "read"
                    } else {
                        "write"
                    };
                    format!("access({})", access_str)
                }
                Attribute::Block => String::from("block"),
                Attribute::Location(id) => format!("location({})", id),
                Attribute::BuiltIn(builtin_attrib) => {
                    let builtin_str = builtin_str(builtin_attrib);
                    if let Some(builtin) = builtin_str {
                        format!("builtin({})", builtin)
                    } else {
                        log::warn!("Unsupported builtin attribute: {:?}", builtin_attrib);
                        String::from("")
                    }
                }
                Attribute::Stage(shader_stage) => match shader_stage {
                    ShaderStage::Vertex => String::from("stage(vertex)"),
                    ShaderStage::Fragment => String::from("stage(fragment)"),
                    ShaderStage::Compute => String::from("stage(compute)"),
                },
                Attribute::Stride(stride) => format!("stride({})", stride),
                Attribute::WorkGroupSize(size) => {
                    format!("workgroup_size({}, {}, {})", size[0], size[1], size[2])
                }
                Attribute::Binding(id) => format!("binding({})", id),
                Attribute::Group(id) => format!("group({})", id),
                Attribute::Interpolate(interpolation, sampling) => {
                    if interpolation.is_some() || sampling.is_some() {
                        let interpolation_str = if let Some(interpolation) = interpolation {
                            interpolation_str(interpolation)
                        } else {
                            ""
                        };
                        let sampling_str = if let Some(sampling) = sampling {
                            // Center sampling is the default
                            if sampling == Sampling::Center {
                                String::from("")
                            } else {
                                format!(",{}", sampling_str(sampling))
                            }
                        } else {
                            String::from("")
                        };
                        format!("interpolate({}{})", interpolation_str, sampling_str)
                    } else {
                        String::from("")
                    }
                }
            };
            if !attribute_str.is_empty() {
                // Add a separator between args
                let separator = if index < attributes.len() - 1 {
                    ", "
                } else {
                    ""
                };
                attributes_str = format!("{}{}{}", attributes_str, attribute_str, separator);
            }
        }
        if !attributes_str.is_empty() {
            //TODO: looks ugly
            if attributes_str.ends_with(", ") {
                attributes_str = attributes_str[0..attributes_str.len() - 2].to_string();
            }
            let extra_space_str = if extra_space { " " } else { "" };
            write!(self.out, "[[{}]]{}", attributes_str, extra_space_str)?;
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
        handle: Handle<Type>,
        block: bool,
        members: &[StructMember],
    ) -> BackendResult {
        if block {
            self.write_attributes(&[Attribute::Block], false)?;
            writeln!(self.out)?;
        }

        write!(self.out, "struct ")?;
        self.write_struct_name(module, handle)?;
        write!(self.out, " {{")?;
        writeln!(self.out)?;
        for (index, member) in members.iter().enumerate() {
            // Skip struct member with unsupported built in
            if let Some(Binding::BuiltIn(builtin)) = member.binding {
                if builtin_str(builtin).is_none() {
                    log::warn!("Skip member with unsupported builtin {:?}", builtin);
                    continue;
                }
            }

            // The indentation is only for readability
            write!(self.out, "{}", INDENT)?;
            if let Some(ref binding) = member.binding {
                self.write_attributes(&map_binding_to_attribute(binding), true)?;
            }
            // Write struct member name and type
            let member_name = &self.names[&NameKey::StructMember(handle, index as u32)];
            write!(self.out, "{}: ", member_name)?;
            // Write stride attribute for array struct member
            if let TypeInner::Array {
                base: _,
                size: _,
                stride,
            } = module.types[member.ty].inner
            {
                self.write_attributes(&[Attribute::Stride(stride)], true)?;
            }
            self.write_type(module, member.ty)?;
            write!(self.out, ";")?;
            writeln!(self.out)?;
        }

        write!(self.out, "}};")?;

        writeln!(self.out)?;

        Ok(())
    }

    /// Helper method used to write non image/sampler types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    fn write_type(&mut self, module: &Module, ty: Handle<Type>) -> BackendResult {
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
                "{}",
                format!("vec{}<{}>", vector_size_str(size), scalar_kind_str(kind),)
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
                let dim_str = image_dimension_str(dim);
                let arrayed_str = if arrayed { "_array" } else { "" };
                let (class_str, multisampled_str, scalar_str) = match class {
                    ImageClass::Sampled { kind, multi } => (
                        "",
                        if multi { "multisampled_" } else { "" },
                        format!("<{}>", scalar_kind_str(kind)),
                    ),
                    ImageClass::Depth => ("depth_", "", String::from("")),
                    ImageClass::Storage(storage_format) => (
                        "storage_",
                        "",
                        format!("<{}>", storage_format_str(storage_format)),
                    ),
                };
                let ty_str = format!(
                    "texture_{}{}{}{}{}",
                    class_str, multisampled_str, dim_str, arrayed_str, scalar_str
                );
                write!(self.out, "{}", ty_str)?;
            }
            TypeInner::Scalar { kind, .. } => {
                write!(self.out, "{}", scalar_kind_str(kind))?;
            }
            TypeInner::Array { base, size, .. } => {
                // More info https://gpuweb.github.io/gpuweb/wgsl/#array-types
                // array<A, 3> -- Constant array
                // array<A> -- Dynamic array
                write!(self.out, "array<")?;
                match size {
                    ArraySize::Constant(handle) => {
                        self.write_type(module, base)?;
                        write!(self.out, ",")?;
                        self.write_constant(module, handle)?;
                    }
                    ArraySize::Dynamic => {
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
                    vector_size_str(columns),
                    vector_size_str(rows),
                )?;
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
        stmt: &Statement,
        func_ctx: &FunctionCtx<'_>,
        indent: usize,
    ) -> BackendResult {
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
                        let expr = &func_ctx.expressions[handle];
                        let min_ref_count = expr.bake_ref_count();
                        // Forcefully creating baking expressions in some cases to help with readability
                        let required_baking_expr = match *expr {
                            Expression::ImageLoad { .. }
                            | Expression::ImageQuery { .. }
                            | Expression::ImageSample { .. } => true,
                            _ => false,
                        };
                        if min_ref_count <= func_ctx.info[handle].ref_count || required_baking_expr
                        {
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

                            Some(format!("{}{}", BAKE_PREFIX, handle.index()))
                        } else {
                            None
                        }
                    };

                    if let Some(name) = expr_name {
                        write!(self.out, "{}", INDENT.repeat(indent))?;
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
            Statement::Return { value } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
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
                write!(self.out, "{}", INDENT.repeat(indent))?;
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
                write!(self.out, "{}", INDENT.repeat(indent))?;
                self.write_expr(module, pointer, func_ctx)?;
                write!(self.out, " = ")?;
                self.write_expr(module, value, func_ctx)?;
                writeln!(self.out, ";")?
            }
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                if let Some(expr) = result {
                    let name = format!("{}{}", BAKE_PREFIX, expr.index());
                    self.start_named_expr(module, expr, func_ctx, &name)?;
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
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
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
                write!(self.out, "{}", INDENT.repeat(indent))?;
                writeln!(self.out, "{{")?;
                for sta in block.iter() {
                    // Increase the indentation to help with readability
                    self.write_stmt(module, sta, func_ctx, indent + 1)?
                }
                writeln!(self.out, "{}}}", INDENT.repeat(indent))?
            }
            Statement::Switch {
                selector,
                ref cases,
                ref default,
            } => {
                // Start the switch
                write!(self.out, "{}", INDENT.repeat(indent))?;
                write!(self.out, "switch(")?;
                self.write_expr(module, selector, func_ctx)?;
                writeln!(self.out, ") {{")?;

                // Write all cases
                let mut write_case = true;
                let all_fall_through = cases
                    .iter()
                    .all(|case| case.fall_through && case.body.is_empty());
                if !cases.is_empty() {
                    for case in cases {
                        if write_case {
                            write!(self.out, "{}case ", INDENT.repeat(indent + 1))?;
                        }
                        if !all_fall_through && case.fall_through && case.body.is_empty() {
                            write_case = false;
                            write!(self.out, "{}, ", case.value)?;
                            continue;
                        } else {
                            write_case = true;
                            writeln!(self.out, "{}: {{", case.value)?;
                        }

                        for sta in case.body.iter() {
                            self.write_stmt(module, sta, func_ctx, indent + 2)?;
                        }

                        if case.fall_through {
                            writeln!(self.out, "{}fallthrough;", INDENT.repeat(indent + 2))?;
                        }
                    }

                    writeln!(self.out, "{}}}", INDENT.repeat(indent + 1))?;
                }

                if !default.is_empty() {
                    writeln!(self.out, "{}default: {{", INDENT.repeat(indent + 1))?;

                    for sta in default {
                        self.write_stmt(module, sta, func_ctx, indent + 2)?;
                    }

                    writeln!(self.out, "{}}}", INDENT.repeat(indent + 1))?
                }

                writeln!(self.out, "{}}}", INDENT.repeat(indent))?
            }
            Statement::Loop {
                ref body,
                ref continuing,
            } => {
                write!(self.out, "{}", INDENT.repeat(indent))?;
                writeln!(self.out, "loop {{")?;

                for sta in body.iter() {
                    self.write_stmt(module, sta, func_ctx, indent + 1)?;
                }

                if !continuing.is_empty() {
                    writeln!(self.out, "{}continuing {{", INDENT.repeat(indent + 1))?;
                    for sta in continuing.iter() {
                        self.write_stmt(module, sta, func_ctx, indent + 2)?;
                    }
                    writeln!(self.out, "{}}}", INDENT.repeat(indent + 1))?;
                }

                writeln!(self.out, "{}}}", INDENT.repeat(indent))?
            }
            Statement::Break => {
                writeln!(self.out, "{}break;", INDENT.repeat(indent))?;
            }
            Statement::Continue => {
                writeln!(self.out, "{}continue;", INDENT.repeat(indent))?;
            }
            Statement::Barrier(barrier) => {
                if barrier.contains(crate::Barrier::STORAGE) {
                    writeln!(self.out, "{}storageBarrier();", INDENT.repeat(indent))?;
                }

                if barrier.contains(crate::Barrier::WORK_GROUP) {
                    writeln!(self.out, "{}workgroupBarrier();", INDENT.repeat(indent))?;
                }
            }
        }

        Ok(())
    }

    fn start_named_expr(
        &mut self,
        module: &Module,
        handle: Handle<Expression>,
        func_ctx: &FunctionCtx,
        name: &str,
    ) -> BackendResult {
        // Write variable name
        write!(self.out, "let {}: ", name)?;
        let ty = &func_ctx.info[handle].ty;
        // Write variable type
        match *ty {
            TypeResolution::Handle(handle) => {
                self.write_type(module, handle)?;
            }
            TypeResolution::Value(ref inner) => {
                self.write_value_type(module, inner)?;
            }
        }

        write!(self.out, " = ")?;
        Ok(())
    }

    /// Helper method to write expressions
    ///
    /// # Notes
    /// Doesn't add any newlines or leading/trailing spaces
    fn write_expr(
        &mut self,
        module: &Module,
        expr: Handle<Expression>,
        func_ctx: &FunctionCtx<'_>,
    ) -> BackendResult {
        if let Some(name) = self.named_expressions.get(&expr) {
            write!(self.out, "{}", name)?;
            return Ok(());
        }

        let expression = &func_ctx.expressions[expr];

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
                let name_key = match func_ctx.ty {
                    FunctionType::Function(handle) => NameKey::FunctionArgument(handle, pos),
                    FunctionType::EntryPoint(ep_index) => {
                        NameKey::EntryPointArgument(ep_index, pos)
                    }
                };
                let name = &self.names[&name_key];
                write!(self.out, "{}", name)?;
            }
            Expression::Binary { op, left, right } => {
                write!(self.out, "(")?;
                self.write_expr(module, left, func_ctx)?;
                write!(self.out, " {} ", binary_operation_str(op))?;
                self.write_expr(module, right, func_ctx)?;
                write!(self.out, ")")?;
            }
            // TODO: copy-paste from glsl-out
            Expression::Access { base, index } => {
                self.write_expr(module, base, func_ctx)?;
                write!(self.out, "[")?;
                self.write_expr(module, index, func_ctx)?;
                write!(self.out, "]")?
            }
            // TODO: copy-paste from glsl-out
            Expression::AccessIndex { base, index } => {
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
                        write!(self.out, ".{}", COMPONENTS[index as usize])?
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
                    ref other => return Err(Error::Custom(format!("Cannot index {:?}", other))),
                }
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
                let suffix_cmp = match depth_ref {
                    Some(_) => "Compare",
                    None => "",
                };
                let suffix_level = match level {
                    SampleLevel::Auto => "",
                    SampleLevel::Zero | SampleLevel::Exact(_) => "Level",
                    SampleLevel::Bias(_) => "Bias",
                    SampleLevel::Gradient { .. } => "Grad",
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
                    SampleLevel::Auto => {}
                    SampleLevel::Zero => {
                        // Level 0 is implied for depth comparison
                        if depth_ref.is_none() {
                            write!(self.out, ", 0.0")?;
                        }
                    }
                    SampleLevel::Exact(expr) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, expr, func_ctx)?;
                    }
                    SampleLevel::Bias(expr) => {
                        write!(self.out, ", ")?;
                        self.write_expr(module, expr, func_ctx)?;
                    }
                    SampleLevel::Gradient { x, y } => {
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
                let texture_function = match query {
                    crate::ImageQuery::Size { .. } => "textureDimensions",
                    crate::ImageQuery::NumLevels => "textureNumLevels",
                    crate::ImageQuery::NumLayers => "textureNumLayers",
                    crate::ImageQuery::NumSamples => "textureNumSamples",
                };

                write!(self.out, "{}(", texture_function)?;
                self.write_expr(module, image, func_ctx)?;
                if let crate::ImageQuery::Size { level: Some(level) } = query {
                    write!(self.out, ", ")?;
                    self.write_expr(module, level, func_ctx)?;
                };
                write!(self.out, ")")?;
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                write!(self.out, "textureLoad(")?;
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;
                if let Some(array_index) = array_index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, array_index, func_ctx)?;
                }
                if let Some(index) = index {
                    write!(self.out, ", ")?;
                    self.write_expr(module, index, func_ctx)?;
                }
                write!(self.out, ")")?;
            }
            Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{}", name)?;
            }
            Expression::As { expr, kind, .. } => {
                let inner = func_ctx.info[expr].ty.inner_with(&module.types);
                match *inner {
                    TypeInner::Matrix { columns, rows, .. } => {
                        write!(
                            self.out,
                            "mat{}x{}<f32>",
                            vector_size_str(columns),
                            vector_size_str(rows)
                        )?;
                    }
                    TypeInner::Vector { size, .. } => {
                        write!(
                            self.out,
                            "vec{}<{}>",
                            vector_size_str(size),
                            scalar_kind_str(kind)
                        )?;
                    }
                    TypeInner::Scalar { .. } => write!(self.out, "{}", scalar_kind_str(kind))?,
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
                let size = vector_size_str(size);

                write!(self.out, "vec{}<{}>(", size, scalar)?;
                self.write_expr(module, value, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::Load { pointer } => self.write_expr(module, pointer, func_ctx)?,
            Expression::LocalVariable(handle) => {
                write!(self.out, "{}", self.names[&func_ctx.name_key(handle)])?
            }
            Expression::ArrayLength(expr) => {
                write!(self.out, "arrayLength(")?;
                if is_deref_required(expr, module, func_ctx.info) {
                    write!(self.out, "&")?;
                };
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
                    Mf::Fract => "fract",
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
                    Mf::Outer => "outerProduct",
                    Mf::Cross => "cross",
                    Mf::Distance => "distance",
                    Mf::Length => "length",
                    Mf::Normalize => "normalize",
                    Mf::FaceForward => "faceForward",
                    Mf::Reflect => "reflect",
                    // computational
                    Mf::Sign => "sign",
                    Mf::Fma => "fma",
                    Mf::Mix => "mix",
                    Mf::Step => "step",
                    Mf::SmoothStep => "smoothStep",
                    Mf::Sqrt => "sqrt",
                    Mf::InverseSqrt => "inverseSqrt",
                    Mf::Transpose => "transpose",
                    Mf::Determinant => "determinant",
                    // bits
                    Mf::CountOneBits => "countOneBits",
                    Mf::ReverseBits => "reverseBits",
                    _ => {
                        return Err(Error::UnsupportedMathFunction(fun));
                    }
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
                    self.out.write_char(COMPONENTS[sc as usize])?;
                }
            }
            Expression::Unary { op, expr } => {
                let unary = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::Not => {
                        match *func_ctx.info[expr].ty.inner_with(&module.types) {
                            TypeInner::Scalar {
                                kind: ScalarKind::Bool,
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
                self.write_expr(module, accept, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, reject, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, condition, func_ctx)?;
                write!(self.out, ")")?
            }
            Expression::Derivative { axis, expr } => {
                let op = match axis {
                    crate::DerivativeAxis::X => "dpdx",
                    crate::DerivativeAxis::Y => "dpdy",
                    crate::DerivativeAxis::Width => "fwidth",
                };
                write!(self.out, "{}(", op)?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?
            }
            Expression::Relational { fun, argument } => {
                let fun_name = match fun {
                    crate::RelationalFunction::IsFinite => "isFinite",
                    crate::RelationalFunction::IsInf => "isInf",
                    crate::RelationalFunction::IsNan => "isNan",
                    crate::RelationalFunction::IsNormal => "isNormal",
                    crate::RelationalFunction::All => "all",
                    crate::RelationalFunction::Any => "any",
                };
                write!(self.out, "{}(", fun_name)?;

                self.write_expr(module, argument, func_ctx)?;

                write!(self.out, ")")?
            }
            // Nothing to do here, since call expression already cached
            Expression::Call(_) => {}
        }

        Ok(())
    }

    /// Helper method used to write global variables
    /// # Notes
    /// Always adds a newline
    fn write_global(
        &mut self,
        module: &Module,
        global: &GlobalVariable,
        handle: Handle<GlobalVariable>,
    ) -> BackendResult {
        let name = self.names[&NameKey::GlobalVariable(handle)].clone();
        // Write group and dinding attributes if present
        if let Some(ref binding) = global.binding {
            self.write_attributes(
                &[
                    Attribute::Group(binding.group),
                    Attribute::Binding(binding.binding),
                ],
                false,
            )?;
            writeln!(self.out)?;
        }

        // First write global name and storage class if supported
        write!(self.out, "var")?;
        if let Some(storage_class) = storage_class_str(global.class) {
            write!(self.out, "<{}>", storage_class)?;
        }
        write!(self.out, " {}: ", name)?;
        // Write access attribute if present
        if !global.storage_access.is_empty() {
            self.write_attributes(&[Attribute::Access(global.storage_access)], true)?;
        }
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
    fn write_constant(&mut self, module: &Module, handle: Handle<Constant>) -> BackendResult {
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
                    if let Some(&Binding::BuiltIn(builtin)) =
                        members.and_then(|members| members.get(index)?.binding.as_ref())
                    {
                        if builtin_str(builtin).is_none() {
                            log::warn!(
                                "Skip constant for struct member with unsupported builtin {:?}",
                                builtin
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
        inner: &ConstantInner,
        handle: Handle<Constant>,
    ) -> BackendResult {
        match *inner {
            crate::ConstantInner::Scalar {
                width: _,
                ref value,
            } => {
                let name = self.names[&NameKey::Constant(handle)].clone();
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
            ConstantInner::Composite { ty, ref components } => {
                let name = self.names[&NameKey::Constant(handle)].clone();
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

    pub fn finish(self) -> W {
        self.out
    }
}

fn builtin_str(built_in: crate::BuiltIn) -> Option<&'static str> {
    use crate::BuiltIn;
    match built_in {
        BuiltIn::VertexIndex => Some("vertex_index"),
        BuiltIn::InstanceIndex => Some("instance_index"),
        BuiltIn::Position => Some("position"),
        BuiltIn::FrontFacing => Some("front_facing"),
        BuiltIn::FragDepth => Some("frag_depth"),
        BuiltIn::LocalInvocationId => Some("local_invocation_id"),
        BuiltIn::LocalInvocationIndex => Some("local_invocation_index"),
        BuiltIn::GlobalInvocationId => Some("global_invocation_id"),
        BuiltIn::WorkGroupId => Some("workgroup_id"),
        BuiltIn::WorkGroupSize => Some("workgroup_size"),
        BuiltIn::SampleIndex => Some("sample_index"),
        BuiltIn::SampleMask => Some("sample_mask"),
        _ => None,
    }
}

fn image_dimension_str(dim: ImageDimension) -> &'static str {
    match dim {
        ImageDimension::D1 => "1d",
        ImageDimension::D2 => "2d",
        ImageDimension::D3 => "3d",
        ImageDimension::Cube => "cube",
    }
}

fn scalar_kind_str(kind: ScalarKind) -> &'static str {
    match kind {
        crate::ScalarKind::Float => "f32",
        crate::ScalarKind::Sint => "i32",
        crate::ScalarKind::Uint => "u32",
        crate::ScalarKind::Bool => "bool",
    }
}

fn storage_format_str(format: StorageFormat) -> &'static str {
    match format {
        StorageFormat::R8Unorm => "r8unorm",
        StorageFormat::R8Snorm => "r8snorm",
        StorageFormat::R8Uint => "r8uint",
        StorageFormat::R8Sint => "r8sint",
        StorageFormat::R16Uint => "r16uint",
        StorageFormat::R16Sint => "r16sint",
        StorageFormat::R16Float => "r16float",
        StorageFormat::Rg8Unorm => "rg8unorm",
        StorageFormat::Rg8Snorm => "rg8snorm",
        StorageFormat::Rg8Uint => "rg8uint",
        StorageFormat::Rg8Sint => "rg8sint",
        StorageFormat::R32Uint => "r32uint",
        StorageFormat::R32Sint => "r32sint",
        StorageFormat::R32Float => "r32float",
        StorageFormat::Rg16Uint => "rg16uint",
        StorageFormat::Rg16Sint => "rg16sint",
        StorageFormat::Rg16Float => "rg16float",
        StorageFormat::Rgba8Unorm => "rgba8unorm",
        StorageFormat::Rgba8Snorm => "rgba8snorm",
        StorageFormat::Rgba8Uint => "rgba8uint",
        StorageFormat::Rgba8Sint => "rgba8sint",
        StorageFormat::Rgb10a2Unorm => "rgb10a2unorm",
        StorageFormat::Rg11b10Float => "rg11b10float",
        StorageFormat::Rg32Uint => "rg32uint",
        StorageFormat::Rg32Sint => "rg32sint",
        StorageFormat::Rg32Float => "rg32float",
        StorageFormat::Rgba16Uint => "rgba16uint",
        StorageFormat::Rgba16Sint => "rgba16sint",
        StorageFormat::Rgba16Float => "rgba16float",
        StorageFormat::Rgba32Uint => "rgba32uint",
        StorageFormat::Rgba32Sint => "rgba32sint",
        StorageFormat::Rgba32Float => "rgba32float",
    }
}

/// Helper function that returns the string corresponding to the WGSL interpolation qualifier
fn interpolation_str(interpolation: Interpolation) -> &'static str {
    match interpolation {
        Interpolation::Perspective => "perspective",
        Interpolation::Linear => "linear",
        Interpolation::Flat => "flat",
    }
}

/// Return the WGSL auxiliary qualifier for the given sampling value.
fn sampling_str(sampling: Sampling) -> &'static str {
    match sampling {
        Sampling::Center => "",
        Sampling::Centroid => "centroid",
        Sampling::Sample => "sample",
    }
}

fn storage_class_str(storage_class: StorageClass) -> Option<&'static str> {
    match storage_class {
        StorageClass::Private => Some("private"),
        StorageClass::Uniform => Some("uniform"),
        StorageClass::Storage => Some("storage"),
        StorageClass::PushConstant => Some("push_constant"),
        StorageClass::WorkGroup => Some("workgroup"),
        StorageClass::Function | StorageClass::Handle => None,
    }
}

fn map_binding_to_attribute(binding: &Binding) -> Vec<Attribute> {
    match *binding {
        Binding::BuiltIn(built_in) => vec![Attribute::BuiltIn(built_in)],
        Binding::Location {
            location,
            interpolation,
            sampling,
        } => vec![
            Attribute::Location(location),
            Attribute::Interpolate(interpolation, sampling),
        ],
    }
}

fn is_deref_required(expr: Handle<Expression>, module: &Module, info: &FunctionInfo) -> bool {
    let base_ty_res = &info[expr].ty;
    let resolved = base_ty_res.inner_with(&module.types);
    match *resolved {
        TypeInner::Pointer { base, class: _ } => match module.types[base].inner {
            TypeInner::Scalar { .. } | TypeInner::Vector { .. } | TypeInner::Array { .. } => true,
            _ => false,
        },
        TypeInner::ValuePointer { .. } => true,
        _ => false,
    }
}

/// Helper function that check that expression don't access to structure member with unsupported builtin.
fn access_to_unsupported_builtin(
    expr: Handle<Expression>,
    index: u32,
    module: &Module,
    info: &FunctionInfo,
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
            if let Some(Binding::BuiltIn(builtin)) = members[index as usize].binding {
                if builtin_str(builtin).is_none() {
                    log::warn!("Skip component with unsupported builtin {:?}", builtin);
                    return true;
                }
            }
        }
    }

    false
}
