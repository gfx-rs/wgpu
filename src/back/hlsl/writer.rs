//TODO: temp
#![allow(dead_code)]
use super::{Error, Options, ShaderModel};
use crate::{
    back::{hlsl::keywords::RESERVED, vector_size_str},
    proc::{EntryPointIndex, NameKey, Namer, TypeResolution},
    valid::{FunctionInfo, ModuleInfo},
    Arena, ArraySize, BuiltIn, Bytes, Constant, ConstantInner, Expression, FastHashMap, Function,
    GlobalVariable, Handle, ImageDimension, LocalVariable, Module, ScalarKind, ScalarValue,
    ShaderStage, Statement, StructMember, Type, TypeInner,
};
use std::fmt::Write;

const INDENT: &str = "    ";
const COMPONENTS: &[char] = &['x', 'y', 'z', 'w'];
const LOCATION_SEMANTIC: &str = "LOC";
const BAKE_PREFIX: &str = "_e";

/// Shorthand result used internally by the backend
type BackendResult = Result<(), Error>;

/// Stores the current function type (either a regular function or an entry point)
///
/// Also stores data needed to identify it (handle for a regular function or index for an entry point)
// TODO: copy-paste from glsl-out, wgsl-out
enum FunctionType {
    /// A regular function and it's handle
    Function(Handle<Function>),
    /// A entry point and it's index
    EntryPoint(EntryPointIndex),
}

/// Helper structure that stores data needed when writing the function
// TODO: copy-paste from glsl-out, wgsl-out
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

struct EntryPointBinding {
    stage: ShaderStage,
    name: String,
    members: Vec<EpStructMember>,
}

struct EpStructMember {
    pub name: String,
    pub ty: Handle<Type>,
    pub binding: Option<crate::Binding>,
}

pub struct Writer<'a, W> {
    out: W,
    names: FastHashMap<NameKey, String>,
    namer: Namer,
    options: &'a Options,
    ep_inputs: Vec<Option<EntryPointBinding>>,
    named_expressions: crate::NamedExpressions,
}

impl<'a, W: Write> Writer<'a, W> {
    pub fn new(out: W, options: &'a Options) -> Self {
        Self {
            out,
            names: FastHashMap::default(),
            namer: Namer::default(),
            options,
            ep_inputs: Vec::with_capacity(3),
            named_expressions: crate::NamedExpressions::default(),
        }
    }

    fn reset(&mut self, module: &Module) {
        self.names.clear();
        self.namer.reset(module, RESERVED, &[], &mut self.names);
        self.named_expressions.clear();
        self.ep_inputs.clear();
    }

    pub fn write(&mut self, module: &Module, info: &ModuleInfo) -> BackendResult {
        if self.options.shader_model < ShaderModel::default() {
            return Err(Error::UnsupportedShaderModel(self.options.shader_model));
        }

        self.reset(module);

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

        // Write all globals
        for (ty, _) in module.global_variables.iter() {
            self.write_global(module, ty)?;
        }

        if !module.global_variables.is_empty() {
            // Add extra newline for readability
            writeln!(self.out)?;
        }

        // Write all entry points wrapped structs
        for (index, ep) in module.entry_points.iter().enumerate() {
            self.write_ep_input_struct(module, &ep.function, ep.stage, index)?;
        }

        // Write all regular functions
        for (handle, function) in module.functions.iter() {
            let info = &info[handle];
            let ctx = FunctionCtx {
                ty: FunctionType::Function(handle),
                info,
                expressions: &function.expressions,
                named_expressions: &function.named_expressions,
            };
            let name = self.names[&NameKey::Function(handle)].clone();

            self.write_function(module, name.as_str(), function, &ctx)?;

            writeln!(self.out)?;
        }

        // Write all entry points
        for (index, ep) in module.entry_points.iter().enumerate() {
            let ctx = FunctionCtx {
                ty: FunctionType::EntryPoint(index as u16),
                info: info.get_entry_point(index),
                expressions: &ep.function.expressions,
                named_expressions: &ep.function.named_expressions,
            };

            if ep.stage == ShaderStage::Compute {
                // HLSL is calling workgroup size, num threads
                let num_threads = ep.workgroup_size;
                writeln!(
                    self.out,
                    "[numthreads({}, {}, {})]",
                    num_threads[0], num_threads[1], num_threads[2]
                )?;
            }

            let name = match ep.stage {
                ShaderStage::Vertex => &self.options.vertex_entry_point_name,
                ShaderStage::Fragment => &self.options.fragment_entry_point_name,
                ShaderStage::Compute => &self.options.compute_entry_point_name,
            };

            self.write_function(module, name, &ep.function, &ctx)?;

            if index < module.entry_points.len() - 1 {
                writeln!(self.out)?;
            }
        }

        Ok(())
    }

    fn write_binding(&mut self, binding: &crate::Binding) -> BackendResult {
        match *binding {
            crate::Binding::BuiltIn(builtin) => {
                write!(self.out, " : {}", builtin_str(builtin))?;
            }
            crate::Binding::Location { location, .. } => {
                write!(self.out, " : {}{}", LOCATION_SEMANTIC, location)?;
            }
        }

        Ok(())
    }

    fn write_ep_input_struct(
        &mut self,
        module: &Module,
        func: &Function,
        stage: ShaderStage,
        index: usize,
    ) -> BackendResult {
        if !func.arguments.is_empty() {
            let struct_name = self.namer.call_unique(match stage {
                ShaderStage::Vertex => "VertexInput",
                ShaderStage::Fragment => "FragmentInput",
                ShaderStage::Compute => "ComputeInput",
            });

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

                write!(self.out, "{}", INDENT)?;
                self.write_type(module, member.ty)?;
                write!(self.out, " {}", &member.name)?;
                if let Some(ref binding) = member.binding {
                    self.write_binding(binding)?;
                }
                write!(self.out, ";")?;
                writeln!(self.out)?;

                members.push(member);
            }

            writeln!(self.out, "}};")?;
            writeln!(self.out)?;

            let ep_input = EntryPointBinding {
                stage,
                name: struct_name,
                members,
            };

            self.ep_inputs.insert(index, Some(ep_input));
        }

        Ok(())
    }

    /// Helper method used to write global variables
    /// # Notes
    /// Always adds a newline
    fn write_global(&mut self, module: &Module, handle: Handle<GlobalVariable>) -> BackendResult {
        let global = &module.global_variables[handle];
        let inner = &module.types[global.ty].inner;

        let (storage_class, register_ty) = match *inner {
            TypeInner::Image { .. } => ("", "t"),
            TypeInner::Sampler { .. } => ("", "s"),
            TypeInner::Struct { .. } | TypeInner::Vector { .. } => ("static ", ""),
            // TODO: other register ty https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-variable-register
            _ => return Err(Error::Unimplemented(format!("register_ty {:?}", inner))),
        };

        write!(self.out, "{}", storage_class)?;
        self.write_type(module, global.ty)?;
        if let TypeInner::Array { size, .. } = module.types[global.ty].inner {
            self.write_array_size(module, size)?;
        }
        write!(
            self.out,
            " {}",
            &self.names[&NameKey::GlobalVariable(handle)]
        )?;

        if let Some(ref binding) = global.binding {
            writeln!(self.out, " : register({}{});", register_ty, binding.binding)?;
        } else {
            if let Some(init) = global.init {
                write!(self.out, " = ")?;
                self.write_constant(module, init)?;
            }
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
        _module: &Module,
        inner: &ConstantInner,
        handle: Handle<Constant>,
    ) -> BackendResult {
        match *inner {
            ConstantInner::Scalar {
                width: _,
                ref value,
            } => {
                write!(self.out, "static const ")?;
                // Write type
                match *value {
                    crate::ScalarValue::Sint(_) => write!(self.out, "int")?,
                    crate::ScalarValue::Uint(_) => write!(self.out, "uint")?,
                    crate::ScalarValue::Float(_) => write!(self.out, "float")?,
                    crate::ScalarValue::Bool(_) => write!(self.out, "bool")?,
                };
                let name = &self.names[&NameKey::Constant(handle)];
                write!(self.out, " {} = ", name)?;

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
                writeln!(self.out, ";")?;
            }
            ConstantInner::Composite { .. } => {
                return Err(Error::Unimplemented(format!(
                    "write_global_constant Composite {:?}",
                    inner
                )))
            }
        }
        // End with extra newline for readability
        writeln!(self.out)?;
        Ok(())
    }

    // copy-paste from glsl-out
    fn write_array_size(&mut self, module: &Module, size: ArraySize) -> BackendResult {
        write!(self.out, "[")?;

        // Write the array size
        // Writes nothing if `ArraySize::Dynamic`
        // Panics if `ArraySize::Constant` has a constant that isn't an uint
        match size {
            ArraySize::Constant(const_handle) => match module.constants[const_handle].inner {
                ConstantInner::Scalar {
                    width: _,
                    value: ScalarValue::Uint(size),
                } => write!(self.out, "{}", size)?,
                _ => unreachable!(),
            },
            ArraySize::Dynamic => (),
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
        handle: Handle<Type>,
        _block: bool,
        members: &[StructMember],
    ) -> BackendResult {
        // Write struct name
        write!(self.out, "struct {}", self.names[&NameKey::Type(handle)])?;
        writeln!(self.out, " {{")?;

        for (index, member) in members.iter().enumerate() {
            // The indentation is only for readability
            write!(self.out, "{}", INDENT)?;

            match module.types[member.ty].inner {
                TypeInner::Array {
                    base,
                    size,
                    stride: _,
                } => {
                    // HLSL arrays are written as `type name[size]`
                    let ty_name = match module.types[base].inner {
                        // Write scalar type by backend so as not to depend on the front-end implementation
                        // Name returned from frontend can be generated (type1, float1, etc.)
                        TypeInner::Scalar { kind, width } => scalar_kind_str(kind, width)?,
                        _ => &self.names[&NameKey::Type(base)],
                    };

                    // Write `type` and `name`
                    write!(self.out, "{}", ty_name)?;
                    write!(
                        self.out,
                        " {}",
                        &self.names[&NameKey::StructMember(handle, index as u32)]
                    )?;
                    // Write [size]
                    self.write_array_size(module, size)?;
                }
                _ => {
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
                self.write_binding(binding)?;
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
    fn write_type(&mut self, module: &Module, ty: Handle<Type>) -> BackendResult {
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
    fn write_value_type(&mut self, module: &Module, inner: &TypeInner) -> BackendResult {
        match *inner {
            TypeInner::Scalar { kind, width } => {
                write!(self.out, "{}", scalar_kind_str(kind, width)?)?;
            }
            TypeInner::Vector { size, kind, width } => {
                write!(
                    self.out,
                    "{}{}",
                    scalar_kind_str(kind, width)?,
                    vector_size_str(size)
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
                    scalar_kind_str(ScalarKind::Float, width)?,
                    vector_size_str(columns),
                    vector_size_str(rows),
                )?;
            }
            TypeInner::Image {
                dim,
                arrayed: _, //TODO:
                class,
            } => {
                let dim_str = image_dimension_str(dim);
                if let crate::ImageClass::Sampled { kind, multi: false } = class {
                    write!(
                        self.out,
                        "Texture{}<{}4>",
                        dim_str,
                        scalar_kind_str(kind, 4)?
                    )?
                } else {
                    return Err(Error::Unimplemented(format!(
                        "write_value_type {:?}",
                        inner
                    )));
                }
            }
            TypeInner::Sampler { comparison: false } => {
                write!(self.out, "SamplerState")?;
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
        func: &Function,
        func_ctx: &FunctionCtx<'_>,
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
            FunctionType::Function(handle) => {
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
            FunctionType::EntryPoint(index) => {
                // EntryPoint arguments wrapped into structure
                if !self.ep_inputs.is_empty() {
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
            FunctionType::EntryPoint(index) => Some(module.entry_points[index as usize].stage),
            _ => None,
        };
        if let Some(ref result) = func.result {
            if let Some(ref binding) = result.binding {
                match *binding {
                    crate::Binding::BuiltIn(builtin) => {
                        write!(self.out, " : {}", builtin_str(builtin))?;
                    }
                    crate::Binding::Location { location, .. } => {
                        if stage == Some(ShaderStage::Fragment) {
                            write!(self.out, " : SV_Target{}", location)?;
                        }
                    }
                }
            }
        }

        // Function body start
        writeln!(self.out)?;
        writeln!(self.out, "{{")?;
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
                    writeln!(self.out)?;
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
                    write!(self.out, "const {} = ", name)?;
                    self.write_expr(module, expr, func_ctx)?;
                    self.named_expressions.insert(expr, name);
                    writeln!(self.out, ";")?
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
            _ => return Err(Error::Unimplemented(format!("write_stmt {:?}", stmt))),
        }

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
                let is_struct = if let TypeInner::Struct { .. } = module.types[ty].inner {
                    true
                } else {
                    false
                };
                if is_struct {
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
                if is_struct {
                    write!(self.out, " }};")?
                } else {
                    write!(self.out, ")")?
                }
            }
            // TODO: copy-paste from wgsl-out
            Expression::Binary { op, left, right } => {
                write!(self.out, "(")?;
                self.write_expr(module, left, func_ctx)?;
                write!(self.out, " {} ", crate::back::binary_operation_str(op))?;
                self.write_expr(module, right, func_ctx)?;
                write!(self.out, ")")?;
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
            Expression::FunctionArgument(pos) => {
                match func_ctx.ty {
                    FunctionType::Function(handle) => {
                        let name = &self.names[&NameKey::FunctionArgument(handle, pos)];
                        write!(self.out, "{}", name)?;
                    }
                    FunctionType::EntryPoint(index) => {
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
                sampler,        // TODO:
                coordinate,     // TODO:
                array_index: _, // TODO:
                offset: _,      // TODO:
                level: _,       // TODO:
                depth_ref: _,   // TODO:
            } => {
                // TODO: others
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ".Sample(")?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, coordinate, func_ctx)?;
                write!(self.out, ")")?;
            }
            // TODO: copy-paste from wgsl-out
            Expression::GlobalVariable(handle) => {
                let name = &self.names[&NameKey::GlobalVariable(handle)];
                write!(self.out, "{}", name)?;
            }
            Expression::Load { pointer } => self.write_expr(module, pointer, func_ctx)?,
            _ => return Err(Error::Unimplemented(format!("write_expr {:?}", expression))),
        }

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
                    write!(self.out, "{}", &self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_scalar_value(*value)?;
                }
            }
            crate::ConstantInner::Composite { ty, ref components } => {
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

    fn write_named_expr(
        &mut self,
        module: &Module,
        handle: Handle<Expression>,
        name: String,
        ctx: &FunctionCtx,
    ) -> BackendResult {
        match ctx.info[handle].ty {
            TypeResolution::Handle(ty_handle) => match module.types[ty_handle].inner {
                TypeInner::Struct { .. } => {
                    let ty_name = &self.names[&NameKey::Type(ty_handle)];
                    write!(self.out, "{}", ty_name)?;
                }
                _ => {
                    self.write_type(module, ty_handle)?;
                }
            },
            TypeResolution::Value(ref inner) => {
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

    pub fn finish(self) -> W {
        self.out
    }
}

fn image_dimension_str(dim: ImageDimension) -> &'static str {
    match dim {
        ImageDimension::D1 => "1D",
        ImageDimension::D2 => "2D",
        ImageDimension::D3 => "3D",
        ImageDimension::Cube => "Cube",
    }
}

fn builtin_str(built_in: BuiltIn) -> &'static str {
    match built_in {
        BuiltIn::Position => "SV_Position",
        // vertex
        BuiltIn::ClipDistance => "SV_ClipDistance",
        BuiltIn::CullDistance => "SV_CullDistance",
        BuiltIn::InstanceIndex => "SV_InstanceID",
        // based on this page https://docs.microsoft.com/en-us/windows/uwp/gaming/glsl-to-hlsl-reference#comparing-opengl-es-20-with-direct3d-11
        // No meaning unless you target Direct3D 9
        BuiltIn::PointSize => "PSIZE",
        BuiltIn::VertexIndex => "SV_VertexID",
        // fragment
        BuiltIn::FragDepth => "SV_Depth",
        BuiltIn::FrontFacing => "SV_IsFrontFace",
        BuiltIn::SampleIndex => "SV_SampleIndex",
        BuiltIn::SampleMask => "SV_Coverage",
        // compute
        BuiltIn::GlobalInvocationId => "SV_DispatchThreadID",
        BuiltIn::LocalInvocationId => "SV_GroupThreadID",
        BuiltIn::LocalInvocationIndex => "SV_GroupIndex",
        BuiltIn::WorkGroupId => "SV_GroupID",
        _ => todo!("builtin_str {:?}", built_in),
    }
}

/// Helper function that returns scalar related strings
/// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar
fn scalar_kind_str(kind: ScalarKind, width: Bytes) -> Result<&'static str, Error> {
    match kind {
        ScalarKind::Sint => Ok("int"),
        ScalarKind::Uint => Ok("uint"),
        ScalarKind::Float => match width {
            2 => Ok("half"),
            4 => Ok("float"),
            8 => Ok("double"),
            _ => Err(Error::UnsupportedScalar(kind, width)),
        },
        ScalarKind::Bool => Ok("bool"),
    }
}
