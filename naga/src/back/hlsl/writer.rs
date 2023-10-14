use super::{
    help::{WrappedArrayLength, WrappedConstructor, WrappedImageQuery, WrappedStructMatrixAccess},
    storage::StoreValue,
    BackendResult, Error, Options,
};
use crate::{
    back,
    proc::{self, NameKey},
    valid, Handle, Module, ScalarKind, ShaderStage, TypeInner,
};
use std::{fmt, mem};

const LOCATION_SEMANTIC: &str = "LOC";
const SPECIAL_CBUF_TYPE: &str = "NagaConstants";
const SPECIAL_CBUF_VAR: &str = "_NagaConstants";
const SPECIAL_BASE_VERTEX: &str = "base_vertex";
const SPECIAL_BASE_INSTANCE: &str = "base_instance";
const SPECIAL_OTHER: &str = "other";

pub(crate) const MODF_FUNCTION: &str = "naga_modf";
pub(crate) const FREXP_FUNCTION: &str = "naga_frexp";

struct EpStructMember {
    name: String,
    ty: Handle<crate::Type>,
    // technically, this should always be `Some`
    binding: Option<crate::Binding>,
    index: u32,
}

/// Structure contains information required for generating
/// wrapped structure of all entry points arguments
struct EntryPointBinding {
    /// Name of the fake EP argument that contains the struct
    /// with all the flattened input data.
    arg_name: String,
    /// Generated structure name
    ty_name: String,
    /// Members of generated structure
    members: Vec<EpStructMember>,
}

pub(super) struct EntryPointInterface {
    /// If `Some`, the input of an entry point is gathered in a special
    /// struct with members sorted by binding.
    /// The `EntryPointBinding::members` array is sorted by index,
    /// so that we can walk it in `write_ep_arguments_initialization`.
    input: Option<EntryPointBinding>,
    /// If `Some`, the output of an entry point is flattened.
    /// The `EntryPointBinding::members` array is sorted by binding,
    /// So that we can walk it in `Statement::Return` handler.
    output: Option<EntryPointBinding>,
}

#[derive(Clone, Eq, PartialEq, PartialOrd, Ord)]
enum InterfaceKey {
    Location(u32),
    BuiltIn(crate::BuiltIn),
    Other,
}

impl InterfaceKey {
    const fn new(binding: Option<&crate::Binding>) -> Self {
        match binding {
            Some(&crate::Binding::Location { location, .. }) => Self::Location(location),
            Some(&crate::Binding::BuiltIn(built_in)) => Self::BuiltIn(built_in),
            None => Self::Other,
        }
    }
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
            entry_point_io: Vec::new(),
            named_expressions: crate::NamedExpressions::default(),
            wrapped: super::Wrapped::default(),
            temp_access_chain: Vec::new(),
            need_bake_expressions: Default::default(),
        }
    }

    fn reset(&mut self, module: &Module) {
        self.names.clear();
        self.namer.reset(
            module,
            super::keywords::RESERVED,
            super::keywords::TYPES,
            super::keywords::RESERVED_CASE_INSENSITIVE,
            &[],
            &mut self.names,
        );
        self.entry_point_io.clear();
        self.named_expressions.clear();
        self.wrapped.clear();
        self.need_bake_expressions.clear();
    }

    /// Helper method used to find which expressions of a given function require baking
    ///
    /// # Notes
    /// Clears `need_bake_expressions` set before adding to it
    fn update_expressions_to_bake(
        &mut self,
        module: &Module,
        func: &crate::Function,
        info: &valid::FunctionInfo,
    ) {
        use crate::Expression;
        self.need_bake_expressions.clear();
        for (fun_handle, expr) in func.expressions.iter() {
            let expr_info = &info[fun_handle];
            let min_ref_count = func.expressions[fun_handle].bake_ref_count();
            if min_ref_count <= expr_info.ref_count {
                self.need_bake_expressions.insert(fun_handle);
            }

            if let Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } = *expr
            {
                match fun {
                    crate::MathFunction::Asinh
                    | crate::MathFunction::Acosh
                    | crate::MathFunction::Atanh
                    | crate::MathFunction::Unpack2x16float
                    | crate::MathFunction::Unpack2x16snorm
                    | crate::MathFunction::Unpack2x16unorm
                    | crate::MathFunction::Unpack4x8snorm
                    | crate::MathFunction::Unpack4x8unorm
                    | crate::MathFunction::Pack2x16float
                    | crate::MathFunction::Pack2x16snorm
                    | crate::MathFunction::Pack2x16unorm
                    | crate::MathFunction::Pack4x8snorm
                    | crate::MathFunction::Pack4x8unorm => {
                        self.need_bake_expressions.insert(arg);
                    }
                    crate::MathFunction::ExtractBits => {
                        self.need_bake_expressions.insert(arg);
                        self.need_bake_expressions.insert(arg1.unwrap());
                        self.need_bake_expressions.insert(arg2.unwrap());
                    }
                    crate::MathFunction::InsertBits => {
                        self.need_bake_expressions.insert(arg);
                        self.need_bake_expressions.insert(arg1.unwrap());
                        self.need_bake_expressions.insert(arg2.unwrap());
                        self.need_bake_expressions.insert(arg3.unwrap());
                    }
                    crate::MathFunction::CountLeadingZeros => {
                        let inner = info[fun_handle].ty.inner_with(&module.types);
                        if let Some(crate::ScalarKind::Sint) = inner.scalar_kind() {
                            self.need_bake_expressions.insert(arg);
                        }
                    }
                    _ => {}
                }
            }

            if let Expression::Derivative { axis, ctrl, expr } = *expr {
                use crate::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
                if axis == Axis::Width && (ctrl == Ctrl::Coarse || ctrl == Ctrl::Fine) {
                    self.need_bake_expressions.insert(expr);
                }
            }
        }
    }

    pub fn write(
        &mut self,
        module: &Module,
        module_info: &valid::ModuleInfo,
    ) -> Result<super::ReflectionInfo, Error> {
        self.reset(module);

        // Write special constants, if needed
        if let Some(ref bt) = self.options.special_constants_binding {
            writeln!(self.out, "struct {SPECIAL_CBUF_TYPE} {{")?;
            writeln!(self.out, "{}int {};", back::INDENT, SPECIAL_BASE_VERTEX)?;
            writeln!(self.out, "{}int {};", back::INDENT, SPECIAL_BASE_INSTANCE)?;
            writeln!(self.out, "{}uint {};", back::INDENT, SPECIAL_OTHER)?;
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

            // Extra newline for readability
            writeln!(self.out)?;
        }

        // Save all entry point output types
        let ep_results = module
            .entry_points
            .iter()
            .map(|ep| (ep.stage, ep.function.result.clone()))
            .collect::<Vec<(ShaderStage, Option<crate::FunctionResult>)>>();

        self.write_all_mat_cx2_typedefs_and_functions(module)?;

        // Write all structs
        for (handle, ty) in module.types.iter() {
            if let TypeInner::Struct { ref members, span } = ty.inner {
                if module.types[members.last().unwrap().ty]
                    .inner
                    .is_dynamically_sized(&module.types)
                {
                    // unsized arrays can only be in storage buffers,
                    // for which we use `ByteAddressBuffer` anyway.
                    continue;
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
                    members,
                    span,
                    ep_result.map(|r| (r.0, Io::Output)),
                )?;
                writeln!(self.out)?;
            }
        }

        self.write_special_functions(module)?;

        self.write_wrapped_compose_functions(module, &module.const_expressions)?;

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
        for (ty, _) in module.global_variables.iter() {
            self.write_global(module, ty)?;
        }

        if !module.global_variables.is_empty() {
            // Add extra newline for readability
            writeln!(self.out)?;
        }

        // Write all entry points wrapped structs
        for (index, ep) in module.entry_points.iter().enumerate() {
            let ep_name = self.names[&NameKey::EntryPoint(index as u16)].clone();
            let ep_io = self.write_ep_interface(module, &ep.function, ep.stage, &ep_name)?;
            self.entry_point_io.push(ep_io);
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

            self.write_wrapped_functions(module, &ctx)?;

            self.write_function(module, name.as_str(), function, &ctx, info)?;

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
            self.write_function(module, &name, &ep.function, &ctx, info)?;

            if index < module.entry_points.len() - 1 {
                writeln!(self.out)?;
            }

            entry_point_names.push(Ok(name));
        }

        Ok(super::ReflectionInfo { entry_point_names })
    }

    fn write_modifier(&mut self, binding: &crate::Binding) -> BackendResult {
        match *binding {
            crate::Binding::BuiltIn(crate::BuiltIn::Position { invariant: true }) => {
                write!(self.out, "precise ")?;
            }
            crate::Binding::Location {
                interpolation,
                sampling,
                ..
            } => {
                if let Some(interpolation) = interpolation {
                    if let Some(string) = interpolation.to_hlsl_str() {
                        write!(self.out, "{string} ")?
                    }
                }

                if let Some(sampling) = sampling {
                    if let Some(string) = sampling.to_hlsl_str() {
                        write!(self.out, "{string} ")?
                    }
                }
            }
            crate::Binding::BuiltIn(_) => {}
        }

        Ok(())
    }

    //TODO: we could force fragment outputs to always go through `entry_point_io.output` path
    // if they are struct, so that the `stage` argument here could be omitted.
    fn write_semantic(
        &mut self,
        binding: &crate::Binding,
        stage: Option<(ShaderStage, Io)>,
    ) -> BackendResult {
        match *binding {
            crate::Binding::BuiltIn(builtin) => {
                let builtin_str = builtin.to_hlsl_str()?;
                write!(self.out, " : {builtin_str}")?;
            }
            crate::Binding::Location {
                second_blend_source: true,
                ..
            } => {
                write!(self.out, " : SV_Target1")?;
            }
            crate::Binding::Location {
                location,
                second_blend_source: false,
                ..
            } => {
                if stage == Some((crate::ShaderStage::Fragment, Io::Output)) {
                    write!(self.out, " : SV_Target{location}")?;
                } else {
                    write!(self.out, " : {LOCATION_SEMANTIC}{location}")?;
                }
            }
        }

        Ok(())
    }

    fn write_interface_struct(
        &mut self,
        module: &Module,
        shader_stage: (ShaderStage, Io),
        struct_name: String,
        mut members: Vec<EpStructMember>,
    ) -> Result<EntryPointBinding, Error> {
        // Sort the members so that first come the user-defined varyings
        // in ascending locations, and then built-ins. This allows VS and FS
        // interfaces to match with regards to order.
        members.sort_by_key(|m| InterfaceKey::new(m.binding.as_ref()));

        write!(self.out, "struct {struct_name}")?;
        writeln!(self.out, " {{")?;
        for m in members.iter() {
            write!(self.out, "{}", back::INDENT)?;
            if let Some(ref binding) = m.binding {
                self.write_modifier(binding)?;
            }
            self.write_type(module, m.ty)?;
            write!(self.out, " {}", &m.name)?;
            if let Some(ref binding) = m.binding {
                self.write_semantic(binding, Some(shader_stage))?;
            }
            writeln!(self.out, ";")?;
        }
        writeln!(self.out, "}};")?;
        writeln!(self.out)?;

        match shader_stage.1 {
            Io::Input => {
                // bring back the original order
                members.sort_by_key(|m| m.index);
            }
            Io::Output => {
                // keep it sorted by binding
            }
        }

        Ok(EntryPointBinding {
            arg_name: self.namer.call(struct_name.to_lowercase().as_str()),
            ty_name: struct_name,
            members,
        })
    }

    /// Flatten all entry point arguments into a single struct.
    /// This is needed since we need to re-order them: first placing user locations,
    /// then built-ins.
    fn write_ep_input_struct(
        &mut self,
        module: &Module,
        func: &crate::Function,
        stage: ShaderStage,
        entry_point_name: &str,
    ) -> Result<EntryPointBinding, Error> {
        let struct_name = format!("{stage:?}Input_{entry_point_name}");

        let mut fake_members = Vec::new();
        for arg in func.arguments.iter() {
            match module.types[arg.ty].inner {
                TypeInner::Struct { ref members, .. } => {
                    for member in members.iter() {
                        let name = self.namer.call_or(&member.name, "member");
                        let index = fake_members.len() as u32;
                        fake_members.push(EpStructMember {
                            name,
                            ty: member.ty,
                            binding: member.binding.clone(),
                            index,
                        });
                    }
                }
                _ => {
                    let member_name = self.namer.call_or(&arg.name, "member");
                    let index = fake_members.len() as u32;
                    fake_members.push(EpStructMember {
                        name: member_name,
                        ty: arg.ty,
                        binding: arg.binding.clone(),
                        index,
                    });
                }
            }
        }

        self.write_interface_struct(module, (stage, Io::Input), struct_name, fake_members)
    }

    /// Flatten all entry point results into a single struct.
    /// This is needed since we need to re-order them: first placing user locations,
    /// then built-ins.
    fn write_ep_output_struct(
        &mut self,
        module: &Module,
        result: &crate::FunctionResult,
        stage: ShaderStage,
        entry_point_name: &str,
    ) -> Result<EntryPointBinding, Error> {
        let struct_name = format!("{stage:?}Output_{entry_point_name}");

        let mut fake_members = Vec::new();
        let empty = [];
        let members = match module.types[result.ty].inner {
            TypeInner::Struct { ref members, .. } => members,
            ref other => {
                log::error!("Unexpected {:?} output type without a binding", other);
                &empty[..]
            }
        };

        for member in members.iter() {
            let member_name = self.namer.call_or(&member.name, "member");
            let index = fake_members.len() as u32;
            fake_members.push(EpStructMember {
                name: member_name,
                ty: member.ty,
                binding: member.binding.clone(),
                index,
            });
        }

        self.write_interface_struct(module, (stage, Io::Output), struct_name, fake_members)
    }

    /// Writes special interface structures for an entry point. The special structures have
    /// all the fields flattened into them and sorted by binding. They are only needed for
    /// VS outputs and FS inputs, so that these interfaces match.
    fn write_ep_interface(
        &mut self,
        module: &Module,
        func: &crate::Function,
        stage: ShaderStage,
        ep_name: &str,
    ) -> Result<EntryPointInterface, Error> {
        Ok(EntryPointInterface {
            input: if !func.arguments.is_empty() && stage == ShaderStage::Fragment {
                Some(self.write_ep_input_struct(module, func, stage, ep_name)?)
            } else {
                None
            },
            output: match func.result {
                Some(ref fr) if fr.binding.is_none() && stage == ShaderStage::Vertex => {
                    Some(self.write_ep_output_struct(module, fr, stage, ep_name)?)
                }
                _ => None,
            },
        })
    }

    /// Write an entry point preface that initializes the arguments as specified in IR.
    fn write_ep_arguments_initialization(
        &mut self,
        module: &Module,
        func: &crate::Function,
        ep_index: u16,
    ) -> BackendResult {
        let ep_input = match self.entry_point_io[ep_index as usize].input.take() {
            Some(ep_input) => ep_input,
            None => return Ok(()),
        };
        let mut fake_iter = ep_input.members.iter();
        for (arg_index, arg) in func.arguments.iter().enumerate() {
            write!(self.out, "{}", back::INDENT)?;
            self.write_type(module, arg.ty)?;
            let arg_name = &self.names[&NameKey::EntryPointArgument(ep_index, arg_index as u32)];
            write!(self.out, " {arg_name}")?;
            match module.types[arg.ty].inner {
                TypeInner::Array { base, size, .. } => {
                    self.write_array_size(module, base, size)?;
                    let fake_member = fake_iter.next().unwrap();
                    writeln!(self.out, " = {}.{};", ep_input.arg_name, fake_member.name)?;
                }
                TypeInner::Struct { ref members, .. } => {
                    write!(self.out, " = {{ ")?;
                    for index in 0..members.len() {
                        if index != 0 {
                            write!(self.out, ", ")?;
                        }
                        let fake_member = fake_iter.next().unwrap();
                        write!(self.out, "{}.{}", ep_input.arg_name, fake_member.name)?;
                    }
                    writeln!(self.out, " }};")?;
                }
                _ => {
                    let fake_member = fake_iter.next().unwrap();
                    writeln!(self.out, " = {}.{};", ep_input.arg_name, fake_member.name)?;
                }
            }
        }
        assert!(fake_iter.next().is_none());
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
        let register_ty = match global.space {
            crate::AddressSpace::Function => unreachable!("Function address space"),
            crate::AddressSpace::Private => {
                write!(self.out, "static ")?;
                self.write_type(module, global.ty)?;
                ""
            }
            crate::AddressSpace::WorkGroup => {
                write!(self.out, "groupshared ")?;
                self.write_type(module, global.ty)?;
                ""
            }
            crate::AddressSpace::Uniform => {
                // constant buffer declarations are expected to be inlined, e.g.
                // `cbuffer foo: register(b0) { field1: type1; }`
                write!(self.out, "cbuffer")?;
                "b"
            }
            crate::AddressSpace::Storage { access } => {
                let (prefix, register) = if access.contains(crate::StorageAccess::STORE) {
                    ("RW", "u")
                } else {
                    ("", "t")
                };
                write!(self.out, "{prefix}ByteAddressBuffer")?;
                register
            }
            crate::AddressSpace::Handle => {
                let handle_ty = match *inner {
                    TypeInner::BindingArray { ref base, .. } => &module.types[*base].inner,
                    _ => inner,
                };

                let register = match *handle_ty {
                    TypeInner::Sampler { .. } => "s",
                    // all storage textures are UAV, unconditionally
                    TypeInner::Image {
                        class: crate::ImageClass::Storage { .. },
                        ..
                    } => "u",
                    _ => "t",
                };
                self.write_type(module, global.ty)?;
                register
            }
            crate::AddressSpace::PushConstant => {
                // The type of the push constants will be wrapped in `ConstantBuffer`
                write!(self.out, "ConstantBuffer<")?;
                "b"
            }
        };

        // If the global is a push constant write the type now because it will be a
        // generic argument to `ConstantBuffer`
        if global.space == crate::AddressSpace::PushConstant {
            self.write_global_type(module, global.ty)?;

            // need to write the array size if the type was emitted with `write_type`
            if let TypeInner::Array { base, size, .. } = module.types[global.ty].inner {
                self.write_array_size(module, base, size)?;
            }

            // Close the angled brackets for the generic argument
            write!(self.out, ">")?;
        }

        let name = &self.names[&NameKey::GlobalVariable(handle)];
        write!(self.out, " {name}")?;

        // Push constants need to be assigned a binding explicitly by the consumer
        // since naga has no way to know the binding from the shader alone
        if global.space == crate::AddressSpace::PushConstant {
            let target = self
                .options
                .push_constants_target
                .as_ref()
                .expect("No bind target was defined for the push constants block");
            write!(self.out, ": register(b{}", target.register)?;
            if target.space != 0 {
                write!(self.out, ", space{}", target.space)?;
            }
            write!(self.out, ")")?;
        }

        if let Some(ref binding) = global.binding {
            // this was already resolved earlier when we started evaluating an entry point.
            let bt = self.options.resolve_resource_binding(binding).unwrap();

            // need to write the binding array size if the type was emitted with `write_type`
            if let TypeInner::BindingArray { base, size, .. } = module.types[global.ty].inner {
                if let Some(overridden_size) = bt.binding_array_size {
                    write!(self.out, "[{overridden_size}]")?;
                } else {
                    self.write_array_size(module, base, size)?;
                }
            }

            write!(self.out, " : register({}{}", register_ty, bt.register)?;
            if bt.space != 0 {
                write!(self.out, ", space{}", bt.space)?;
            }
            write!(self.out, ")")?;
        } else {
            // need to write the array size if the type was emitted with `write_type`
            if let TypeInner::Array { base, size, .. } = module.types[global.ty].inner {
                self.write_array_size(module, base, size)?;
            }
            if global.space == crate::AddressSpace::Private {
                write!(self.out, " = ")?;
                if let Some(init) = global.init {
                    self.write_const_expression(module, init)?;
                } else {
                    self.write_default_init(module, global.ty)?;
                }
            }
        }

        if global.space == crate::AddressSpace::Uniform {
            write!(self.out, " {{ ")?;

            self.write_global_type(module, global.ty)?;

            write!(
                self.out,
                " {}",
                &self.names[&NameKey::GlobalVariable(handle)]
            )?;

            // need to write the array size if the type was emitted with `write_type`
            if let TypeInner::Array { base, size, .. } = module.types[global.ty].inner {
                self.write_array_size(module, base, size)?;
            }

            writeln!(self.out, "; }}")?;
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
        handle: Handle<crate::Constant>,
    ) -> BackendResult {
        write!(self.out, "static const ")?;
        let constant = &module.constants[handle];
        self.write_type(module, constant.ty)?;
        let name = &self.names[&NameKey::Constant(handle)];
        write!(self.out, " {}", name)?;
        // Write size for array type
        if let TypeInner::Array { base, size, .. } = module.types[constant.ty].inner {
            self.write_array_size(module, base, size)?;
        }
        write!(self.out, " = ")?;
        self.write_const_expression(module, constant.init)?;
        writeln!(self.out, ";")?;
        Ok(())
    }

    pub(super) fn write_array_size(
        &mut self,
        module: &Module,
        base: Handle<crate::Type>,
        size: crate::ArraySize,
    ) -> BackendResult {
        write!(self.out, "[")?;

        match size {
            crate::ArraySize::Constant(size) => {
                write!(self.out, "{size}")?;
            }
            crate::ArraySize::Dynamic => unreachable!(),
        }

        write!(self.out, "]")?;

        if let TypeInner::Array {
            base: next_base,
            size: next_size,
            ..
        } = module.types[base].inner
        {
            self.write_array_size(module, next_base, next_size)?;
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
        span: u32,
        shader_stage: Option<(ShaderStage, Io)>,
    ) -> BackendResult {
        // Write struct name
        let struct_name = &self.names[&NameKey::Type(handle)];
        writeln!(self.out, "struct {struct_name} {{")?;

        let mut last_offset = 0;
        for (index, member) in members.iter().enumerate() {
            if member.binding.is_none() && member.offset > last_offset {
                // using int as padding should work as long as the backend
                // doesn't support a type that's less than 4 bytes in size
                // (Error::UnsupportedScalar catches this)
                let padding = (member.offset - last_offset) / 4;
                for i in 0..padding {
                    writeln!(self.out, "{}int _pad{}_{};", back::INDENT, index, i)?;
                }
            }
            let ty_inner = &module.types[member.ty].inner;
            last_offset = member.offset + ty_inner.size_hlsl(module.to_ctx());

            // The indentation is only for readability
            write!(self.out, "{}", back::INDENT)?;

            match module.types[member.ty].inner {
                TypeInner::Array { base, size, .. } => {
                    // HLSL arrays are written as `type name[size]`

                    self.write_global_type(module, member.ty)?;

                    // Write `name`
                    write!(
                        self.out,
                        " {}",
                        &self.names[&NameKey::StructMember(handle, index as u32)]
                    )?;
                    // Write [size]
                    self.write_array_size(module, base, size)?;
                }
                // We treat matrices of the form `matCx2` as a sequence of C `vec2`s.
                // See the module-level block comment in mod.rs for details.
                TypeInner::Matrix {
                    rows,
                    columns,
                    width,
                } if member.binding.is_none() && rows == crate::VectorSize::Bi => {
                    let vec_ty = crate::TypeInner::Vector {
                        size: rows,
                        kind: crate::ScalarKind::Float,
                        width,
                    };
                    let field_name_key = NameKey::StructMember(handle, index as u32);

                    for i in 0..columns as u8 {
                        if i != 0 {
                            write!(self.out, "; ")?;
                        }
                        self.write_value_type(module, &vec_ty)?;
                        write!(self.out, " {}_{}", &self.names[&field_name_key], i)?;
                    }
                }
                _ => {
                    // Write modifier before type
                    if let Some(ref binding) = member.binding {
                        self.write_modifier(binding)?;
                    }

                    // Even though Naga IR matrices are column-major, we must describe
                    // matrices passed from the CPU as being in row-major order.
                    // See the module-level block comment in mod.rs for details.
                    if let TypeInner::Matrix { .. } = module.types[member.ty].inner {
                        write!(self.out, "row_major ")?;
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
            writeln!(self.out, ";")?;
        }

        // add padding at the end since sizes of types don't get rounded up to their alignment in HLSL
        if members.last().unwrap().binding.is_none() && span > last_offset {
            let padding = (span - last_offset) / 4;
            for i in 0..padding {
                writeln!(self.out, "{}int _end_pad_{};", back::INDENT, i)?;
            }
        }

        writeln!(self.out, "}};")?;
        Ok(())
    }

    /// Helper method used to write global/structs non image/sampler types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    pub(super) fn write_global_type(
        &mut self,
        module: &Module,
        ty: Handle<crate::Type>,
    ) -> BackendResult {
        let matrix_data = get_inner_matrix_data(module, ty);

        // We treat matrices of the form `matCx2` as a sequence of C `vec2`s.
        // See the module-level block comment in mod.rs for details.
        if let Some(MatrixType {
            columns,
            rows: crate::VectorSize::Bi,
            width: 4,
        }) = matrix_data
        {
            write!(self.out, "__mat{}x2", columns as u8)?;
        } else {
            // Even though Naga IR matrices are column-major, we must describe
            // matrices passed from the CPU as being in row-major order.
            // See the module-level block comment in mod.rs for details.
            if matrix_data.is_some() {
                write!(self.out, "row_major ")?;
            }

            self.write_type(module, ty)?;
        }

        Ok(())
    }

    /// Helper method used to write non image/sampler types
    ///
    /// # Notes
    /// Adds no trailing or leading whitespace
    pub(super) fn write_type(&mut self, module: &Module, ty: Handle<crate::Type>) -> BackendResult {
        let inner = &module.types[ty].inner;
        match *inner {
            TypeInner::Struct { .. } => write!(self.out, "{}", self.names[&NameKey::Type(ty)])?,
            // hlsl array has the size separated from the base type
            TypeInner::Array { base, .. } | TypeInner::BindingArray { base, .. } => {
                self.write_type(module, base)?
            }
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
            TypeInner::Scalar { kind, width } | TypeInner::Atomic { kind, width } => {
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

                // Because of the implicit transpose all matrices have in HLSL, we need to transpose the size as well.
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
                write!(self.out, "{sampler}")?;
            }
            // HLSL arrays are written as `type name[size]`
            // Current code is written arrays only as `[size]`
            // Base `type` and `name` should be written outside
            TypeInner::Array { base, size, .. } | TypeInner::BindingArray { base, size } => {
                self.write_array_size(module, base, size)?;
            }
            _ => return Err(Error::Unimplemented(format!("write_value_type {inner:?}"))),
        }

        Ok(())
    }

    /// Helper method used to write functions
    /// # Notes
    /// Ends in a newline
    fn write_function(
        &mut self,
        module: &Module,
        name: &str,
        func: &crate::Function,
        func_ctx: &back::FunctionCtx<'_>,
        info: &valid::FunctionInfo,
    ) -> BackendResult {
        // Function Declaration Syntax - https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-function-syntax

        self.update_expressions_to_bake(module, func, info);

        // Write modifier
        if let Some(crate::FunctionResult {
            binding:
                Some(
                    ref binding @ crate::Binding::BuiltIn(crate::BuiltIn::Position {
                        invariant: true,
                    }),
                ),
            ..
        }) = func.result
        {
            self.write_modifier(binding)?;
        }

        // Write return type
        if let Some(ref result) = func.result {
            match func_ctx.ty {
                back::FunctionType::Function(_) => {
                    self.write_type(module, result.ty)?;
                }
                back::FunctionType::EntryPoint(index) => {
                    if let Some(ref ep_output) = self.entry_point_io[index as usize].output {
                        write!(self.out, "{}", ep_output.ty_name)?;
                    } else {
                        self.write_type(module, result.ty)?;
                    }
                }
            }
        } else {
            write!(self.out, "void")?;
        }

        // Write function name
        write!(self.out, " {name}(")?;

        let need_workgroup_variables_initialization =
            self.need_workgroup_variables_initialization(func_ctx, module);

        // Write function arguments for non entry point functions
        match func_ctx.ty {
            back::FunctionType::Function(handle) => {
                for (index, arg) in func.arguments.iter().enumerate() {
                    if index != 0 {
                        write!(self.out, ", ")?;
                    }
                    // Write argument type
                    let arg_ty = match module.types[arg.ty].inner {
                        // pointers in function arguments are expected and resolve to `inout`
                        TypeInner::Pointer { base, .. } => {
                            //TODO: can we narrow this down to just `in` when possible?
                            write!(self.out, "inout ")?;
                            base
                        }
                        _ => arg.ty,
                    };
                    self.write_type(module, arg_ty)?;

                    let argument_name =
                        &self.names[&NameKey::FunctionArgument(handle, index as u32)];

                    // Write argument name. Space is important.
                    write!(self.out, " {argument_name}")?;
                    if let TypeInner::Array { base, size, .. } = module.types[arg_ty].inner {
                        self.write_array_size(module, base, size)?;
                    }
                }
            }
            back::FunctionType::EntryPoint(ep_index) => {
                if let Some(ref ep_input) = self.entry_point_io[ep_index as usize].input {
                    write!(self.out, "{} {}", ep_input.ty_name, ep_input.arg_name,)?;
                } else {
                    let stage = module.entry_points[ep_index as usize].stage;
                    for (index, arg) in func.arguments.iter().enumerate() {
                        if index != 0 {
                            write!(self.out, ", ")?;
                        }
                        self.write_type(module, arg.ty)?;

                        let argument_name =
                            &self.names[&NameKey::EntryPointArgument(ep_index, index as u32)];

                        write!(self.out, " {argument_name}")?;
                        if let TypeInner::Array { base, size, .. } = module.types[arg.ty].inner {
                            self.write_array_size(module, base, size)?;
                        }

                        if let Some(ref binding) = arg.binding {
                            self.write_semantic(binding, Some((stage, Io::Input)))?;
                        }
                    }

                    if need_workgroup_variables_initialization {
                        if !func.arguments.is_empty() {
                            write!(self.out, ", ")?;
                        }
                        write!(self.out, "uint3 __local_invocation_id : SV_GroupThreadID")?;
                    }
                }
            }
        }
        // Ends of arguments
        write!(self.out, ")")?;

        // Write semantic if it present
        if let back::FunctionType::EntryPoint(index) = func_ctx.ty {
            let stage = module.entry_points[index as usize].stage;
            if let Some(crate::FunctionResult {
                binding: Some(ref binding),
                ..
            }) = func.result
            {
                self.write_semantic(binding, Some((stage, Io::Output)))?;
            }
        }

        // Function body start
        writeln!(self.out)?;
        writeln!(self.out, "{{")?;

        if need_workgroup_variables_initialization {
            self.write_workgroup_variables_initialization(func_ctx, module)?;
        }

        if let back::FunctionType::EntryPoint(index) = func_ctx.ty {
            self.write_ep_arguments_initialization(module, func, index)?;
        }

        // Write function local variables
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability)
            write!(self.out, "{}", back::INDENT)?;

            // Write the local name
            // The leading space is important
            self.write_type(module, local.ty)?;
            write!(self.out, " {}", self.names[&func_ctx.name_key(handle)])?;
            // Write size for array type
            if let TypeInner::Array { base, size, .. } = module.types[local.ty].inner {
                self.write_array_size(module, base, size)?;
            }

            write!(self.out, " = ")?;
            // Write the local initializer if needed
            if let Some(init) = local.init {
                self.write_expr(module, init, func_ctx)?;
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
            self.write_stmt(module, sta, func_ctx, back::Level(1))?;
        }

        writeln!(self.out, "}}")?;

        self.named_expressions.clear();

        Ok(())
    }

    fn need_workgroup_variables_initialization(
        &mut self,
        func_ctx: &back::FunctionCtx,
        module: &Module,
    ) -> bool {
        self.options.zero_initialize_workgroup_memory
            && func_ctx.ty.is_compute_entry_point(module)
            && module.global_variables.iter().any(|(handle, var)| {
                !func_ctx.info[handle].is_empty() && var.space == crate::AddressSpace::WorkGroup
            })
    }

    fn write_workgroup_variables_initialization(
        &mut self,
        func_ctx: &back::FunctionCtx,
        module: &Module,
    ) -> BackendResult {
        let level = back::Level(1);

        writeln!(
            self.out,
            "{level}if (all(__local_invocation_id == uint3(0u, 0u, 0u))) {{"
        )?;

        let vars = module.global_variables.iter().filter(|&(handle, var)| {
            !func_ctx.info[handle].is_empty() && var.space == crate::AddressSpace::WorkGroup
        });

        for (handle, var) in vars {
            let name = &self.names[&NameKey::GlobalVariable(handle)];
            write!(self.out, "{}{} = ", level.next(), name)?;
            self.write_default_init(module, var.ty)?;
            writeln!(self.out, ";")?;
        }

        writeln!(self.out, "{level}}}")?;
        self.write_barrier(crate::Barrier::WORK_GROUP, level)
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
        use crate::Statement;

        match *stmt {
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let ptr_class = func_ctx.resolve_type(handle, &module.types).pointer_space();
                    let expr_name = if ptr_class.is_some() {
                        // HLSL can't save a pointer-valued expression in a variable,
                        // but we shouldn't ever need to: they should never be named expressions,
                        // and none of the expression types flagged by bake_ref_count can be pointer-valued.
                        None
                    } else if let Some(name) = func_ctx.named_expressions.get(&handle) {
                        // Front end provides names for all variables at the start of writing.
                        // But we write them to step by step. We need to recache them
                        // Otherwise, we could accidentally write variable name instead of full expression.
                        // Also, we use sanitized names! It defense backend from generating variable with name from reserved keywords.
                        Some(self.namer.call(name))
                    } else if self.need_bake_expressions.contains(&handle) {
                        Some(format!("_expr{}", handle.index()))
                    } else {
                        None
                    };

                    if let Some(name) = expr_name {
                        write!(self.out, "{level}")?;
                        self.write_named_expr(module, handle, name, handle, func_ctx)?;
                    }
                }
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
            // TODO: copy-paste from glsl-out
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                write!(self.out, "{level}")?;
                write!(self.out, "if (")?;
                self.write_expr(module, condition, func_ctx)?;
                writeln!(self.out, ") {{")?;

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
            // TODO: copy-paste from glsl-out
            Statement::Kill => writeln!(self.out, "{level}discard;")?,
            Statement::Return { value: None } => {
                writeln!(self.out, "{level}return;")?;
            }
            Statement::Return { value: Some(expr) } => {
                let base_ty_res = &func_ctx.info[expr].ty;
                let mut resolved = base_ty_res.inner_with(&module.types);
                if let TypeInner::Pointer { base, space: _ } = *resolved {
                    resolved = &module.types[base].inner;
                }

                if let TypeInner::Struct { .. } = *resolved {
                    // We can safely unwrap here, since we now we working with struct
                    let ty = base_ty_res.handle().unwrap();
                    let struct_name = &self.names[&NameKey::Type(ty)];
                    let variable_name = self.namer.call(&struct_name.to_lowercase());
                    write!(self.out, "{level}const {struct_name} {variable_name} = ",)?;
                    self.write_expr(module, expr, func_ctx)?;
                    writeln!(self.out, ";")?;

                    // for entry point returns, we may need to reshuffle the outputs into a different struct
                    let ep_output = match func_ctx.ty {
                        back::FunctionType::Function(_) => None,
                        back::FunctionType::EntryPoint(index) => {
                            self.entry_point_io[index as usize].output.as_ref()
                        }
                    };
                    let final_name = match ep_output {
                        Some(ep_output) => {
                            let final_name = self.namer.call(&variable_name);
                            write!(
                                self.out,
                                "{}const {} {} = {{ ",
                                level, ep_output.ty_name, final_name,
                            )?;
                            for (index, m) in ep_output.members.iter().enumerate() {
                                if index != 0 {
                                    write!(self.out, ", ")?;
                                }
                                let member_name = &self.names[&NameKey::StructMember(ty, m.index)];
                                write!(self.out, "{variable_name}.{member_name}")?;
                            }
                            writeln!(self.out, " }};")?;
                            final_name
                        }
                        None => variable_name,
                    };
                    writeln!(self.out, "{level}return {final_name};")?;
                } else {
                    write!(self.out, "{level}return ")?;
                    self.write_expr(module, expr, func_ctx)?;
                    writeln!(self.out, ";")?
                }
            }
            Statement::Store { pointer, value } => {
                let ty_inner = func_ctx.resolve_type(pointer, &module.types);
                if let Some(crate::AddressSpace::Storage { .. }) = ty_inner.pointer_space() {
                    let var_handle = self.fill_access_chain(module, pointer, func_ctx)?;
                    self.write_storage_store(
                        module,
                        var_handle,
                        StoreValue::Expression(value),
                        func_ctx,
                        level,
                    )?;
                } else {
                    // We treat matrices of the form `matCx2` as a sequence of C `vec2`s.
                    // See the module-level block comment in mod.rs for details.
                    //
                    // We handle matrix Stores here directly (including sub accesses for Vectors and Scalars).
                    // Loads are handled by `Expression::AccessIndex` (since sub accesses work fine for Loads).
                    struct MatrixAccess {
                        base: Handle<crate::Expression>,
                        index: u32,
                    }
                    enum Index {
                        Expression(Handle<crate::Expression>),
                        Static(u32),
                    }

                    let get_members = |expr: Handle<crate::Expression>| {
                        let resolved = func_ctx.resolve_type(expr, &module.types);
                        match *resolved {
                            TypeInner::Pointer { base, .. } => match module.types[base].inner {
                                TypeInner::Struct { ref members, .. } => Some(members),
                                _ => None,
                            },
                            _ => None,
                        }
                    };

                    let mut matrix = None;
                    let mut vector = None;
                    let mut scalar = None;

                    let mut current_expr = pointer;
                    for _ in 0..3 {
                        let resolved = func_ctx.resolve_type(current_expr, &module.types);

                        match (resolved, &func_ctx.expressions[current_expr]) {
                            (
                                &TypeInner::Pointer { base: ty, .. },
                                &crate::Expression::AccessIndex { base, index },
                            ) if matches!(
                                module.types[ty].inner,
                                TypeInner::Matrix {
                                    rows: crate::VectorSize::Bi,
                                    ..
                                }
                            ) && get_members(base)
                                .map(|members| members[index as usize].binding.is_none())
                                == Some(true) =>
                            {
                                matrix = Some(MatrixAccess { base, index });
                                break;
                            }
                            (
                                &TypeInner::ValuePointer {
                                    size: Some(crate::VectorSize::Bi),
                                    ..
                                },
                                &crate::Expression::Access { base, index },
                            ) => {
                                vector = Some(Index::Expression(index));
                                current_expr = base;
                            }
                            (
                                &TypeInner::ValuePointer {
                                    size: Some(crate::VectorSize::Bi),
                                    ..
                                },
                                &crate::Expression::AccessIndex { base, index },
                            ) => {
                                vector = Some(Index::Static(index));
                                current_expr = base;
                            }
                            (
                                &TypeInner::ValuePointer { size: None, .. },
                                &crate::Expression::Access { base, index },
                            ) => {
                                scalar = Some(Index::Expression(index));
                                current_expr = base;
                            }
                            (
                                &TypeInner::ValuePointer { size: None, .. },
                                &crate::Expression::AccessIndex { base, index },
                            ) => {
                                scalar = Some(Index::Static(index));
                                current_expr = base;
                            }
                            _ => break,
                        }
                    }

                    write!(self.out, "{level}")?;

                    if let Some(MatrixAccess { index, base }) = matrix {
                        let base_ty_res = &func_ctx.info[base].ty;
                        let resolved = base_ty_res.inner_with(&module.types);
                        let ty = match *resolved {
                            TypeInner::Pointer { base, .. } => base,
                            _ => base_ty_res.handle().unwrap(),
                        };

                        if let Some(Index::Static(vec_index)) = vector {
                            self.write_expr(module, base, func_ctx)?;
                            write!(
                                self.out,
                                ".{}_{}",
                                &self.names[&NameKey::StructMember(ty, index)],
                                vec_index
                            )?;

                            if let Some(scalar_index) = scalar {
                                write!(self.out, "[")?;
                                match scalar_index {
                                    Index::Static(index) => {
                                        write!(self.out, "{index}")?;
                                    }
                                    Index::Expression(index) => {
                                        self.write_expr(module, index, func_ctx)?;
                                    }
                                }
                                write!(self.out, "]")?;
                            }

                            write!(self.out, " = ")?;
                            self.write_expr(module, value, func_ctx)?;
                            writeln!(self.out, ";")?;
                        } else {
                            let access = WrappedStructMatrixAccess { ty, index };
                            match (&vector, &scalar) {
                                (&Some(_), &Some(_)) => {
                                    self.write_wrapped_struct_matrix_set_scalar_function_name(
                                        access,
                                    )?;
                                }
                                (&Some(_), &None) => {
                                    self.write_wrapped_struct_matrix_set_vec_function_name(access)?;
                                }
                                (&None, _) => {
                                    self.write_wrapped_struct_matrix_set_function_name(access)?;
                                }
                            }

                            write!(self.out, "(")?;
                            self.write_expr(module, base, func_ctx)?;
                            write!(self.out, ", ")?;
                            self.write_expr(module, value, func_ctx)?;

                            if let Some(Index::Expression(vec_index)) = vector {
                                write!(self.out, ", ")?;
                                self.write_expr(module, vec_index, func_ctx)?;

                                if let Some(scalar_index) = scalar {
                                    write!(self.out, ", ")?;
                                    match scalar_index {
                                        Index::Static(index) => {
                                            write!(self.out, "{index}")?;
                                        }
                                        Index::Expression(index) => {
                                            self.write_expr(module, index, func_ctx)?;
                                        }
                                    }
                                }
                            }
                            writeln!(self.out, ");")?;
                        }
                    } else {
                        // We handle `Store`s to __matCx2 column vectors and scalar elements via
                        // the previously injected functions __set_col_of_matCx2 / __set_el_of_matCx2.
                        struct MatrixData {
                            columns: crate::VectorSize,
                            base: Handle<crate::Expression>,
                        }

                        enum Index {
                            Expression(Handle<crate::Expression>),
                            Static(u32),
                        }

                        let mut matrix = None;
                        let mut vector = None;
                        let mut scalar = None;

                        let mut current_expr = pointer;
                        for _ in 0..3 {
                            let resolved = func_ctx.resolve_type(current_expr, &module.types);
                            match (resolved, &func_ctx.expressions[current_expr]) {
                                (
                                    &TypeInner::ValuePointer {
                                        size: Some(crate::VectorSize::Bi),
                                        ..
                                    },
                                    &crate::Expression::Access { base, index },
                                ) => {
                                    vector = Some(index);
                                    current_expr = base;
                                }
                                (
                                    &TypeInner::ValuePointer { size: None, .. },
                                    &crate::Expression::Access { base, index },
                                ) => {
                                    scalar = Some(Index::Expression(index));
                                    current_expr = base;
                                }
                                (
                                    &TypeInner::ValuePointer { size: None, .. },
                                    &crate::Expression::AccessIndex { base, index },
                                ) => {
                                    scalar = Some(Index::Static(index));
                                    current_expr = base;
                                }
                                _ => {
                                    if let Some(MatrixType {
                                        columns,
                                        rows: crate::VectorSize::Bi,
                                        width: 4,
                                    }) = get_inner_matrix_of_struct_array_member(
                                        module,
                                        current_expr,
                                        func_ctx,
                                        true,
                                    ) {
                                        matrix = Some(MatrixData {
                                            columns,
                                            base: current_expr,
                                        });
                                    }

                                    break;
                                }
                            }
                        }

                        if let (Some(MatrixData { columns, base }), Some(vec_index)) =
                            (matrix, vector)
                        {
                            if scalar.is_some() {
                                write!(self.out, "__set_el_of_mat{}x2", columns as u8)?;
                            } else {
                                write!(self.out, "__set_col_of_mat{}x2", columns as u8)?;
                            }
                            write!(self.out, "(")?;
                            self.write_expr(module, base, func_ctx)?;
                            write!(self.out, ", ")?;
                            self.write_expr(module, vec_index, func_ctx)?;

                            if let Some(scalar_index) = scalar {
                                write!(self.out, ", ")?;
                                match scalar_index {
                                    Index::Static(index) => {
                                        write!(self.out, "{index}")?;
                                    }
                                    Index::Expression(index) => {
                                        self.write_expr(module, index, func_ctx)?;
                                    }
                                }
                            }

                            write!(self.out, ", ")?;
                            self.write_expr(module, value, func_ctx)?;

                            writeln!(self.out, ");")?;
                        } else {
                            self.write_expr(module, pointer, func_ctx)?;
                            write!(self.out, " = ")?;

                            // We cast the RHS of this store in cases where the LHS
                            // is a struct member with type:
                            //  - matCx2 or
                            //  - a (possibly nested) array of matCx2's
                            if let Some(MatrixType {
                                columns,
                                rows: crate::VectorSize::Bi,
                                width: 4,
                            }) = get_inner_matrix_of_struct_array_member(
                                module, pointer, func_ctx, false,
                            ) {
                                let mut resolved = func_ctx.resolve_type(pointer, &module.types);
                                if let TypeInner::Pointer { base, .. } = *resolved {
                                    resolved = &module.types[base].inner;
                                }

                                write!(self.out, "(__mat{}x2", columns as u8)?;
                                if let TypeInner::Array { base, size, .. } = *resolved {
                                    self.write_array_size(module, base, size)?;
                                }
                                write!(self.out, ")")?;
                            }

                            self.write_expr(module, value, func_ctx)?;
                            writeln!(self.out, ";")?
                        }
                    }
                }
            }
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                let l2 = level.next();
                if !continuing.is_empty() || break_if.is_some() {
                    let gate_name = self.namer.call("loop_init");
                    writeln!(self.out, "{level}bool {gate_name} = true;")?;
                    writeln!(self.out, "{level}while(true) {{")?;
                    writeln!(self.out, "{l2}if (!{gate_name}) {{")?;
                    let l3 = l2.next();
                    for sta in continuing.iter() {
                        self.write_stmt(module, sta, func_ctx, l3)?;
                    }
                    if let Some(condition) = break_if {
                        write!(self.out, "{l3}if (")?;
                        self.write_expr(module, condition, func_ctx)?;
                        writeln!(self.out, ") {{")?;
                        writeln!(self.out, "{}break;", l3.next())?;
                        writeln!(self.out, "{l3}}}")?;
                    }
                    writeln!(self.out, "{l2}}}")?;
                    writeln!(self.out, "{l2}{gate_name} = false;")?;
                } else {
                    writeln!(self.out, "{level}while(true) {{")?;
                }

                for sta in body.iter() {
                    self.write_stmt(module, sta, func_ctx, l2)?;
                }
                writeln!(self.out, "{level}}}")?
            }
            Statement::Break => writeln!(self.out, "{level}break;")?,
            Statement::Continue => writeln!(self.out, "{level}continue;")?,
            Statement::Barrier(barrier) => {
                self.write_barrier(barrier, level)?;
            }
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                write!(self.out, "{level}")?;
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
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                write!(self.out, "{level}")?;
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
                    write!(self.out, " {name} = ")?;
                    self.named_expressions.insert(expr, name);
                }
                let func_name = &self.names[&NameKey::Function(function)];
                write!(self.out, "{func_name}(")?;
                for (index, argument) in arguments.iter().enumerate() {
                    if index != 0 {
                        write!(self.out, ", ")?;
                    }
                    self.write_expr(module, *argument, func_ctx)?;
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
                let res_name = format!("{}{}", back::BAKE_PREFIX, result.index());
                match func_ctx.info[result].ty {
                    proc::TypeResolution::Handle(handle) => self.write_type(module, handle)?,
                    proc::TypeResolution::Value(ref value) => {
                        self.write_value_type(module, value)?
                    }
                };

                // Validation ensures that `pointer` has a `Pointer` type.
                let pointer_space = func_ctx
                    .resolve_type(pointer, &module.types)
                    .pointer_space()
                    .unwrap();

                let fun_str = fun.to_hlsl_suffix();
                write!(self.out, " {res_name}; ")?;
                match pointer_space {
                    crate::AddressSpace::WorkGroup => {
                        write!(self.out, "Interlocked{fun_str}(")?;
                        self.write_expr(module, pointer, func_ctx)?;
                    }
                    crate::AddressSpace::Storage { .. } => {
                        let var_handle = self.fill_access_chain(module, pointer, func_ctx)?;
                        // The call to `self.write_storage_address` wants
                        // mutable access to all of `self`, so temporarily take
                        // ownership of our reusable access chain buffer.
                        let chain = mem::take(&mut self.temp_access_chain);
                        let var_name = &self.names[&NameKey::GlobalVariable(var_handle)];
                        write!(self.out, "{var_name}.Interlocked{fun_str}(")?;
                        self.write_storage_address(module, &chain, func_ctx)?;
                        self.temp_access_chain = chain;
                    }
                    ref other => {
                        return Err(Error::Custom(format!(
                            "invalid address space {other:?} for atomic statement"
                        )))
                    }
                }
                write!(self.out, ", ")?;
                // handle the special cases
                match *fun {
                    crate::AtomicFunction::Subtract => {
                        // we just wrote `InterlockedAdd`, so negate the argument
                        write!(self.out, "-")?;
                    }
                    crate::AtomicFunction::Exchange { compare: Some(_) } => {
                        return Err(Error::Unimplemented("atomic CompareExchange".to_string()));
                    }
                    _ => {}
                }
                self.write_expr(module, value, func_ctx)?;
                writeln!(self.out, ", {res_name});")?;
                self.named_expressions.insert(result, res_name);
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                self.write_barrier(crate::Barrier::WORK_GROUP, level)?;
                write!(self.out, "{level}")?;
                let name = format!("_expr{}", result.index());
                self.write_named_expr(module, pointer, name, result, func_ctx)?;

                self.write_barrier(crate::Barrier::WORK_GROUP, level)?;
            }
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Start the switch
                write!(self.out, "{level}")?;
                write!(self.out, "switch(")?;
                self.write_expr(module, selector, func_ctx)?;
                writeln!(self.out, ") {{")?;

                // Write all cases
                let indent_level_1 = level.next();
                let indent_level_2 = indent_level_1.next();

                for (i, case) in cases.iter().enumerate() {
                    match case.value {
                        crate::SwitchValue::I32(value) => {
                            write!(self.out, "{indent_level_1}case {value}:")?
                        }
                        crate::SwitchValue::U32(value) => {
                            write!(self.out, "{indent_level_1}case {value}u:")?
                        }
                        crate::SwitchValue::Default => {
                            write!(self.out, "{indent_level_1}default:")?
                        }
                    }

                    // The new block is not only stylistic, it plays a role here:
                    // We might end up having to write the same case body
                    // multiple times due to FXC not supporting fallthrough.
                    // Therefore, some `Expression`s written by `Statement::Emit`
                    // will end up having the same name (`_expr<handle_index>`).
                    // So we need to put each case in its own scope.
                    let write_block_braces = !(case.fall_through && case.body.is_empty());
                    if write_block_braces {
                        writeln!(self.out, " {{")?;
                    } else {
                        writeln!(self.out)?;
                    }

                    // Although FXC does support a series of case clauses before
                    // a block[^yes], it does not support fallthrough from a
                    // non-empty case block to the next[^no]. If this case has a
                    // non-empty body with a fallthrough, emulate that by
                    // duplicating the bodies of all the cases it would fall
                    // into as extensions of this case's own body. This makes
                    // the HLSL output potentially quadratic in the size of the
                    // Naga IR.
                    //
                    // [^yes]: ```hlsl
                    // case 1:
                    // case 2: do_stuff()
                    // ```
                    // [^no]: ```hlsl
                    // case 1: do_this();
                    // case 2: do_that();
                    // ```
                    if case.fall_through && !case.body.is_empty() {
                        let curr_len = i + 1;
                        let end_case_idx = curr_len
                            + cases
                                .iter()
                                .skip(curr_len)
                                .position(|case| !case.fall_through)
                                .unwrap();
                        let indent_level_3 = indent_level_2.next();
                        for case in &cases[i..=end_case_idx] {
                            writeln!(self.out, "{indent_level_2}{{")?;
                            let prev_len = self.named_expressions.len();
                            for sta in case.body.iter() {
                                self.write_stmt(module, sta, func_ctx, indent_level_3)?;
                            }
                            // Clear all named expressions that were previously inserted by the statements in the block
                            self.named_expressions.truncate(prev_len);
                            writeln!(self.out, "{indent_level_2}}}")?;
                        }

                        let last_case = &cases[end_case_idx];
                        if last_case.body.last().map_or(true, |s| !s.is_terminator()) {
                            writeln!(self.out, "{indent_level_2}break;")?;
                        }
                    } else {
                        for sta in case.body.iter() {
                            self.write_stmt(module, sta, func_ctx, indent_level_2)?;
                        }
                        if !case.fall_through
                            && case.body.last().map_or(true, |s| !s.is_terminator())
                        {
                            writeln!(self.out, "{indent_level_2}break;")?;
                        }
                    }

                    if write_block_braces {
                        writeln!(self.out, "{indent_level_1}}}")?;
                    }
                }

                writeln!(self.out, "{level}}}")?
            }
            Statement::RayQuery { .. } => unreachable!(),
            Statement::SubgroupBallot { result, predicate } => {
                write!(self.out, "{level}")?;

                let name = format!("{}{}", back::BAKE_PREFIX, result.index());
                write!(self.out, "const uint4 {name} = ")?;
                self.named_expressions.insert(result, name);

                write!(self.out, "WaveActiveBallot(")?;
                match predicate {
                    Some(predicate) => self.write_expr(module, predicate, func_ctx)?,
                    None => write!(self.out, "true")?,
                }
                writeln!(self.out, ");")?;
            }
            Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => {
                unimplemented!(); // FIXME
            }
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                unimplemented!(); // FIXME
            }
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
            &module.const_expressions,
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
                // Floats are written using `Debug` instead of `Display` because it always appends the
                // decimal part even it's zero
                crate::Literal::F64(value) => write!(self.out, "{value:?}L")?,
                crate::Literal::F32(value) => write!(self.out, "{value:?}")?,
                crate::Literal::U32(value) => write!(self.out, "{}u", value)?,
                crate::Literal::I32(value) => write!(self.out, "{}", value)?,
                crate::Literal::Bool(value) => write!(self.out, "{}", value)?,
            },
            Expression::Constant(handle) => {
                let constant = &module.constants[handle];
                if constant.name.is_some() {
                    write!(self.out, "{}", self.names[&NameKey::Constant(handle)])?;
                } else {
                    self.write_const_expression(module, constant.init)?;
                }
            }
            Expression::ZeroValue(ty) => self.write_default_init(module, ty)?,
            Expression::Compose { ty, ref components } => {
                match module.types[ty].inner {
                    TypeInner::Struct { .. } | TypeInner::Array { .. } => {
                        self.write_wrapped_constructor_function_name(
                            module,
                            WrappedConstructor { ty },
                        )?;
                    }
                    _ => {
                        self.write_type(module, ty)?;
                    }
                };
                write!(self.out, "(")?;
                for (index, component) in components.iter().enumerate() {
                    if index != 0 {
                        write!(self.out, ", ")?;
                    }
                    write_expression(self, *component)?;
                }
                write!(self.out, ")")?;
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
                write!(self.out, "(")?;
                write_expression(self, value)?;
                write!(self.out, ").{number_of_components}")?
            }
            _ => unreachable!(),
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
                write!(self.out, "({SPECIAL_CBUF_VAR}.{SPECIAL_BASE_VERTEX} + ")?;
                ")"
            }
            Some(crate::BuiltIn::InstanceIndex) => {
                write!(self.out, "({SPECIAL_CBUF_VAR}.{SPECIAL_BASE_INSTANCE} + ",)?;
                ")"
            }
            Some(crate::BuiltIn::NumWorkGroups) => {
                //Note: despite their names (`BASE_VERTEX` and `BASE_INSTANCE`),
                // in compute shaders the special constants contain the number
                // of workgroups, which we are using here.
                write!(
                    self.out,
                    "uint3({SPECIAL_CBUF_VAR}.{SPECIAL_BASE_VERTEX}, {SPECIAL_CBUF_VAR}.{SPECIAL_BASE_INSTANCE}, {SPECIAL_CBUF_VAR}.{SPECIAL_OTHER})",
                )?;
                return Ok(());
            }
            _ => "",
        };

        if let Some(name) = self.named_expressions.get(&expr) {
            write!(self.out, "{name}{closing_bracket}")?;
            return Ok(());
        }

        let expression = &func_ctx.expressions[expr];

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
            // All of the multiplication can be expressed as `mul`,
            // except vector * vector, which needs to use the "*" operator.
            Expression::Binary {
                op: crate::BinaryOperator::Multiply,
                left,
                right,
            } if func_ctx.resolve_type(left, &module.types).is_matrix()
                || func_ctx.resolve_type(right, &module.types).is_matrix() =>
            {
                // We intentionally flip the order of multiplication as our matrices are implicitly transposed.
                write!(self.out, "mul(")?;
                self.write_expr(module, right, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_expr(module, left, func_ctx)?;
                write!(self.out, ")")?;
            }

            // TODO: handle undefined behavior of BinaryOperator::Modulo
            //
            // sint:
            // if right == 0 return 0
            // if left == min(type_of(left)) && right == -1 return 0
            // if sign(left) != sign(right) return result as defined by WGSL
            //
            // uint:
            // if right == 0 return 0
            //
            // float:
            // if right == 0 return ? see https://github.com/gpuweb/gpuweb/issues/2798

            // While HLSL supports float operands with the % operator it is only
            // defined in cases where both sides are either positive or negative.
            Expression::Binary {
                op: crate::BinaryOperator::Modulo,
                left,
                right,
            } if func_ctx.resolve_type(left, &module.types).scalar_kind()
                == Some(crate::ScalarKind::Float) =>
            {
                write!(self.out, "fmod(")?;
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
                if let Some(crate::AddressSpace::Storage { .. }) =
                    func_ctx.resolve_type(expr, &module.types).pointer_space()
                {
                    // do nothing, the chain is written on `Load`/`Store`
                } else {
                    // We use the function __get_col_of_matCx2 here in cases
                    // where `base`s type resolves to a matCx2 and is part of a
                    // struct member with type of (possibly nested) array of matCx2's.
                    //
                    // Note that this only works for `Load`s and we handle
                    // `Store`s differently in `Statement::Store`.
                    if let Some(MatrixType {
                        columns,
                        rows: crate::VectorSize::Bi,
                        width: 4,
                    }) = get_inner_matrix_of_struct_array_member(module, base, func_ctx, true)
                    {
                        write!(self.out, "__get_col_of_mat{}x2(", columns as u8)?;
                        self.write_expr(module, base, func_ctx)?;
                        write!(self.out, ", ")?;
                        self.write_expr(module, index, func_ctx)?;
                        write!(self.out, ")")?;
                        return Ok(());
                    }

                    let resolved = func_ctx.resolve_type(base, &module.types);

                    let non_uniform_qualifier = match *resolved {
                        TypeInner::BindingArray { .. } => {
                            let uniformity = &func_ctx.info[index].uniformity;

                            uniformity.non_uniform_result.is_some()
                        }
                        _ => false,
                    };

                    self.write_expr(module, base, func_ctx)?;
                    write!(self.out, "[")?;
                    if non_uniform_qualifier {
                        write!(self.out, "NonUniformResourceIndex(")?;
                    }
                    self.write_expr(module, index, func_ctx)?;
                    if non_uniform_qualifier {
                        write!(self.out, ")")?;
                    }
                    write!(self.out, "]")?;
                }
            }
            Expression::AccessIndex { base, index } => {
                if let Some(crate::AddressSpace::Storage { .. }) =
                    func_ctx.resolve_type(expr, &module.types).pointer_space()
                {
                    // do nothing, the chain is written on `Load`/`Store`
                } else {
                    fn write_access<W: fmt::Write>(
                        writer: &mut super::Writer<'_, W>,
                        resolved: &TypeInner,
                        base_ty_handle: Option<Handle<crate::Type>>,
                        index: u32,
                    ) -> BackendResult {
                        match *resolved {
                            // We specifcally lift the ValuePointer to this case. While `[0]` is valid
                            // HLSL for any vector behind a value pointer, FXC completely miscompiles
                            // it and generates completely nonsensical DXBC.
                            //
                            // See https://github.com/gfx-rs/naga/issues/2095 for more details.
                            TypeInner::Vector { .. } | TypeInner::ValuePointer { .. } => {
                                // Write vector access as a swizzle
                                write!(writer.out, ".{}", back::COMPONENTS[index as usize])?
                            }
                            TypeInner::Matrix { .. }
                            | TypeInner::Array { .. }
                            | TypeInner::BindingArray { .. } => write!(writer.out, "[{index}]")?,
                            TypeInner::Struct { .. } => {
                                // This will never panic in case the type is a `Struct`, this is not true
                                // for other types so we can only check while inside this match arm
                                let ty = base_ty_handle.unwrap();

                                write!(
                                    writer.out,
                                    ".{}",
                                    &writer.names[&NameKey::StructMember(ty, index)]
                                )?
                            }
                            ref other => {
                                return Err(Error::Custom(format!("Cannot index {other:?}")))
                            }
                        }
                        Ok(())
                    }

                    // We write the matrix column access in a special way since
                    // the type of `base` is our special __matCx2 struct.
                    if let Some(MatrixType {
                        rows: crate::VectorSize::Bi,
                        width: 4,
                        ..
                    }) = get_inner_matrix_of_struct_array_member(module, base, func_ctx, true)
                    {
                        self.write_expr(module, base, func_ctx)?;
                        write!(self.out, "._{index}")?;
                        return Ok(());
                    }

                    let base_ty_res = &func_ctx.info[base].ty;
                    let mut resolved = base_ty_res.inner_with(&module.types);
                    let base_ty_handle = match *resolved {
                        TypeInner::Pointer { base, .. } => {
                            resolved = &module.types[base].inner;
                            Some(base)
                        }
                        _ => base_ty_res.handle(),
                    };

                    // We treat matrices of the form `matCx2` as a sequence of C `vec2`s.
                    // See the module-level block comment in mod.rs for details.
                    //
                    // We handle matrix reconstruction here for Loads.
                    // Stores are handled directly by `Statement::Store`.
                    if let TypeInner::Struct { ref members, .. } = *resolved {
                        let member = &members[index as usize];

                        match module.types[member.ty].inner {
                            TypeInner::Matrix {
                                rows: crate::VectorSize::Bi,
                                ..
                            } if member.binding.is_none() => {
                                let ty = base_ty_handle.unwrap();
                                self.write_wrapped_struct_matrix_get_function_name(
                                    WrappedStructMatrixAccess { ty, index },
                                )?;
                                write!(self.out, "(")?;
                                self.write_expr(module, base, func_ctx)?;
                                write!(self.out, ")")?;
                                return Ok(());
                            }
                            _ => {}
                        }
                    }

                    self.write_expr(module, base, func_ctx)?;
                    write_access(self, resolved, base_ty_handle, index)?;
                }
            }
            Expression::FunctionArgument(pos) => {
                let key = func_ctx.argument_key(pos);
                let name = &self.names[&key];
                write!(self.out, "{name}")?;
            }
            Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                use crate::SampleLevel as Sl;
                const COMPONENTS: [&str; 4] = ["", "Green", "Blue", "Alpha"];

                let (base_str, component_str) = match gather {
                    Some(component) => ("Gather", COMPONENTS[component as usize]),
                    None => ("Sample", ""),
                };
                let cmp_str = match depth_ref {
                    Some(_) => "Cmp",
                    None => "",
                };
                let level_str = match level {
                    Sl::Zero if gather.is_none() => "LevelZero",
                    Sl::Auto | Sl::Zero => "",
                    Sl::Exact(_) => "Level",
                    Sl::Bias(_) => "Bias",
                    Sl::Gradient { .. } => "Grad",
                };

                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ".{base_str}{cmp_str}{component_str}{level_str}(")?;
                self.write_expr(module, sampler, func_ctx)?;
                write!(self.out, ", ")?;
                self.write_texture_coordinates(
                    "float",
                    coordinate,
                    array_index,
                    None,
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
                    write!(self.out, "int2(")?; // work around https://github.com/microsoft/DirectXShaderCompiler/issues/5082#issuecomment-1540147807
                    self.write_const_expression(module, offset)?;
                    write!(self.out, ")")?;
                }

                write!(self.out, ")")?;
            }
            Expression::ImageQuery { image, query } => {
                // use wrapped image query function
                if let TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } = *func_ctx.resolve_type(image, &module.types)
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
                sample,
                level,
            } => {
                // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-to-load
                self.write_expr(module, image, func_ctx)?;
                write!(self.out, ".Load(")?;

                self.write_texture_coordinates(
                    "int",
                    coordinate,
                    array_index,
                    level,
                    module,
                    func_ctx,
                )?;

                if let Some(sample) = sample {
                    write!(self.out, ", ")?;
                    self.write_expr(module, sample, func_ctx)?;
                }

                // close bracket for Load function
                write!(self.out, ")")?;

                // return x component if return type is scalar
                if let TypeInner::Scalar { .. } = *func_ctx.resolve_type(expr, &module.types) {
                    write!(self.out, ".x")?;
                }
            }
            Expression::GlobalVariable(handle) => match module.global_variables[handle].space {
                crate::AddressSpace::Storage { .. } => {}
                _ => {
                    let name = &self.names[&NameKey::GlobalVariable(handle)];
                    write!(self.out, "{name}")?;
                }
            },
            Expression::LocalVariable(handle) => {
                write!(self.out, "{}", self.names[&func_ctx.name_key(handle)])?
            }
            Expression::Load { pointer } => {
                match func_ctx
                    .resolve_type(pointer, &module.types)
                    .pointer_space()
                {
                    Some(crate::AddressSpace::Storage { .. }) => {
                        let var_handle = self.fill_access_chain(module, pointer, func_ctx)?;
                        let result_ty = func_ctx.info[expr].ty.clone();
                        self.write_storage_load(module, var_handle, result_ty, func_ctx)?;
                    }
                    _ => {
                        let mut close_paren = false;

                        // We cast the value loaded to a native HLSL floatCx2
                        // in cases where it is of type:
                        //  - __matCx2 or
                        //  - a (possibly nested) array of __matCx2's
                        if let Some(MatrixType {
                            rows: crate::VectorSize::Bi,
                            width: 4,
                            ..
                        }) = get_inner_matrix_of_struct_array_member(
                            module, pointer, func_ctx, false,
                        )
                        .or_else(|| get_inner_matrix_of_global_uniform(module, pointer, func_ctx))
                        {
                            let mut resolved = func_ctx.resolve_type(pointer, &module.types);
                            if let TypeInner::Pointer { base, .. } = *resolved {
                                resolved = &module.types[base].inner;
                            }

                            write!(self.out, "((")?;
                            if let TypeInner::Array { base, size, .. } = *resolved {
                                self.write_type(module, base)?;
                                self.write_array_size(module, base, size)?;
                            } else {
                                self.write_value_type(module, resolved)?;
                            }
                            write!(self.out, ")")?;
                            close_paren = true;
                        }

                        self.write_expr(module, pointer, func_ctx)?;

                        if close_paren {
                            write!(self.out, ")")?;
                        }
                    }
                }
            }
            Expression::Unary { op, expr } => {
                // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-operators#unary-operators
                let op_str = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::LogicalNot => "!",
                    crate::UnaryOperator::BitwiseNot => "~",
                };
                write!(self.out, "{op_str}(")?;
                self.write_expr(module, expr, func_ctx)?;
                write!(self.out, ")")?;
            }
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                let inner = func_ctx.resolve_type(expr, &module.types);
                match convert {
                    Some(dst_width) => {
                        match *inner {
                            TypeInner::Vector { size, .. } => {
                                write!(
                                    self.out,
                                    "{}{}(",
                                    kind.to_hlsl_str(dst_width)?,
                                    back::vector_size_str(size)
                                )?;
                            }
                            TypeInner::Scalar { .. } => {
                                write!(self.out, "{}(", kind.to_hlsl_str(dst_width)?,)?;
                            }
                            TypeInner::Matrix { columns, rows, .. } => {
                                write!(
                                    self.out,
                                    "{}{}x{}(",
                                    kind.to_hlsl_str(dst_width)?,
                                    back::vector_size_str(columns),
                                    back::vector_size_str(rows)
                                )?;
                            }
                            _ => {
                                return Err(Error::Unimplemented(format!(
                                    "write_expr expression::as {inner:?}"
                                )));
                            }
                        };
                    }
                    None => {
                        write!(self.out, "{}(", kind.to_hlsl_cast(),)?;
                    }
                }
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
                    ExtractBits,
                    InsertBits,
                    Pack2x16float,
                    Pack2x16snorm,
                    Pack2x16unorm,
                    Pack4x8snorm,
                    Pack4x8unorm,
                    Unpack2x16float,
                    Unpack2x16snorm,
                    Unpack2x16unorm,
                    Unpack4x8snorm,
                    Unpack4x8unorm,
                    Regular(&'static str),
                    MissingIntOverload(&'static str),
                    MissingIntReturnType(&'static str),
                    CountTrailingZeros,
                    CountLeadingZeros,
                }

                let fun = match fun {
                    // comparison
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
                    Mf::Asinh => Function::Asincosh { is_sin: true },
                    Mf::Acosh => Function::Asincosh { is_sin: false },
                    Mf::Atanh => Function::Atanh,
                    Mf::Radians => Function::Regular("radians"),
                    Mf::Degrees => Function::Regular("degrees"),
                    // decomposition
                    Mf::Ceil => Function::Regular("ceil"),
                    Mf::Floor => Function::Regular("floor"),
                    Mf::Round => Function::Regular("round"),
                    Mf::Fract => Function::Regular("frac"),
                    Mf::Trunc => Function::Regular("trunc"),
                    Mf::Modf => Function::Regular(MODF_FUNCTION),
                    Mf::Frexp => Function::Regular(FREXP_FUNCTION),
                    Mf::Ldexp => Function::Regular("ldexp"),
                    // exponent
                    Mf::Exp => Function::Regular("exp"),
                    Mf::Exp2 => Function::Regular("exp2"),
                    Mf::Log => Function::Regular("log"),
                    Mf::Log2 => Function::Regular("log2"),
                    Mf::Pow => Function::Regular("pow"),
                    // geometry
                    Mf::Dot => Function::Regular("dot"),
                    //Mf::Outer => ,
                    Mf::Cross => Function::Regular("cross"),
                    Mf::Distance => Function::Regular("distance"),
                    Mf::Length => Function::Regular("length"),
                    Mf::Normalize => Function::Regular("normalize"),
                    Mf::FaceForward => Function::Regular("faceforward"),
                    Mf::Reflect => Function::Regular("reflect"),
                    Mf::Refract => Function::Regular("refract"),
                    // computational
                    Mf::Sign => Function::Regular("sign"),
                    Mf::Fma => Function::Regular("mad"),
                    Mf::Mix => Function::Regular("lerp"),
                    Mf::Step => Function::Regular("step"),
                    Mf::SmoothStep => Function::Regular("smoothstep"),
                    Mf::Sqrt => Function::Regular("sqrt"),
                    Mf::InverseSqrt => Function::Regular("rsqrt"),
                    //Mf::Inverse =>,
                    Mf::Transpose => Function::Regular("transpose"),
                    Mf::Determinant => Function::Regular("determinant"),
                    // bits
                    Mf::CountTrailingZeros => Function::CountTrailingZeros,
                    Mf::CountLeadingZeros => Function::CountLeadingZeros,
                    Mf::CountOneBits => Function::MissingIntOverload("countbits"),
                    Mf::ReverseBits => Function::MissingIntOverload("reversebits"),
                    Mf::FindLsb => Function::MissingIntReturnType("firstbitlow"),
                    Mf::FindMsb => Function::MissingIntReturnType("firstbithigh"),
                    Mf::ExtractBits => Function::ExtractBits,
                    Mf::InsertBits => Function::InsertBits,
                    // Data Packing
                    Mf::Pack2x16float => Function::Pack2x16float,
                    Mf::Pack2x16snorm => Function::Pack2x16snorm,
                    Mf::Pack2x16unorm => Function::Pack2x16unorm,
                    Mf::Pack4x8snorm => Function::Pack4x8snorm,
                    Mf::Pack4x8unorm => Function::Pack4x8unorm,
                    // Data Unpacking
                    Mf::Unpack2x16float => Function::Unpack2x16float,
                    Mf::Unpack2x16snorm => Function::Unpack2x16snorm,
                    Mf::Unpack2x16unorm => Function::Unpack2x16unorm,
                    Mf::Unpack4x8snorm => Function::Unpack4x8snorm,
                    Mf::Unpack4x8unorm => Function::Unpack4x8unorm,
                    _ => return Err(Error::Unimplemented(format!("write_expr_math {fun:?}"))),
                };

                match fun {
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
                    Function::ExtractBits => {
                        // e: T,
                        // offset: u32,
                        // count: u32
                        // T is u32 or i32 or vecN<u32> or vecN<i32>
                        if let (Some(offset), Some(count)) = (arg1, arg2) {
                            let scalar_width: u8 = 32;
                            // Works for signed and unsigned
                            // (count == 0 ? 0 : (e << (32 - count - offset)) >> (32 - count))
                            write!(self.out, "(")?;
                            self.write_expr(module, count, func_ctx)?;
                            write!(self.out, " == 0 ? 0 : (")?;
                            self.write_expr(module, arg, func_ctx)?;
                            write!(self.out, " << ({scalar_width} - ")?;
                            self.write_expr(module, count, func_ctx)?;
                            write!(self.out, " - ")?;
                            self.write_expr(module, offset, func_ctx)?;
                            write!(self.out, ")) >> ({scalar_width} - ")?;
                            self.write_expr(module, count, func_ctx)?;
                            write!(self.out, "))")?;
                        }
                    }
                    Function::InsertBits => {
                        // e: T,
                        // newbits: T,
                        // offset: u32,
                        // count: u32
                        // returns T
                        // T is i32, u32, vecN<i32>, or vecN<u32>
                        if let (Some(newbits), Some(offset), Some(count)) = (arg1, arg2, arg3) {
                            let scalar_width: u8 = 32;
                            let scalar_max: u32 = 0xFFFFFFFF;
                            // mask = ((0xFFFFFFFFu >> (32 - count)) << offset)
                            // (count == 0 ? e : ((e & ~mask) | ((newbits << offset) & mask)))
                            write!(self.out, "(")?;
                            self.write_expr(module, count, func_ctx)?;
                            write!(self.out, " == 0 ? ")?;
                            self.write_expr(module, arg, func_ctx)?;
                            write!(self.out, " : ")?;
                            write!(self.out, "(")?;
                            self.write_expr(module, arg, func_ctx)?;
                            write!(self.out, " & ~")?;
                            // mask
                            write!(self.out, "(({scalar_max}u >> ({scalar_width}u - ")?;
                            self.write_expr(module, count, func_ctx)?;
                            write!(self.out, ")) << ")?;
                            self.write_expr(module, offset, func_ctx)?;
                            write!(self.out, ")")?;
                            // end mask
                            write!(self.out, ") | ((")?;
                            self.write_expr(module, newbits, func_ctx)?;
                            write!(self.out, " << ")?;
                            self.write_expr(module, offset, func_ctx)?;
                            write!(self.out, ") & ")?;
                            // // mask
                            write!(self.out, "(({scalar_max}u >> ({scalar_width}u - ")?;
                            self.write_expr(module, count, func_ctx)?;
                            write!(self.out, ")) << ")?;
                            self.write_expr(module, offset, func_ctx)?;
                            write!(self.out, ")")?;
                            // // end mask
                            write!(self.out, "))")?;
                        }
                    }
                    Function::Pack2x16float => {
                        write!(self.out, "(f32tof16(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[0]) | f32tof16(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[1]) << 16)")?;
                    }
                    Function::Pack2x16snorm => {
                        let scale = 32767;

                        write!(self.out, "uint((int(round(clamp(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(
                            self.out,
                            "[0], -1.0, 1.0) * {scale}.0)) & 0xFFFF) | ((int(round(clamp("
                        )?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[1], -1.0, 1.0) * {scale}.0)) & 0xFFFF) << 16))",)?;
                    }
                    Function::Pack2x16unorm => {
                        let scale = 65535;

                        write!(self.out, "(uint(round(clamp(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[0], 0.0, 1.0) * {scale}.0)) | uint(round(clamp(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[1], 0.0, 1.0) * {scale}.0)) << 16)")?;
                    }
                    Function::Pack4x8snorm => {
                        let scale = 127;

                        write!(self.out, "uint((int(round(clamp(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(
                            self.out,
                            "[0], -1.0, 1.0) * {scale}.0)) & 0xFF) | ((int(round(clamp("
                        )?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(
                            self.out,
                            "[1], -1.0, 1.0) * {scale}.0)) & 0xFF) << 8) | ((int(round(clamp("
                        )?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(
                            self.out,
                            "[2], -1.0, 1.0) * {scale}.0)) & 0xFF) << 16) | ((int(round(clamp("
                        )?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[3], -1.0, 1.0) * {scale}.0)) & 0xFF) << 24))",)?;
                    }
                    Function::Pack4x8unorm => {
                        let scale = 255;

                        write!(self.out, "(uint(round(clamp(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[0], 0.0, 1.0) * {scale}.0)) | uint(round(clamp(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(
                            self.out,
                            "[1], 0.0, 1.0) * {scale}.0)) << 8 | uint(round(clamp("
                        )?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(
                            self.out,
                            "[2], 0.0, 1.0) * {scale}.0)) << 16 | uint(round(clamp("
                        )?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "[3], 0.0, 1.0) * {scale}.0)) << 24)")?;
                    }

                    Function::Unpack2x16float => {
                        write!(self.out, "float2(f16tof32(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, "), f16tof32((")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, ") >> 16))")?;
                    }
                    Function::Unpack2x16snorm => {
                        let scale = 32767;

                        write!(self.out, "(float2(int2(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " << 16, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, ") >> 16) / {scale}.0)")?;
                    }
                    Function::Unpack2x16unorm => {
                        let scale = 65535;

                        write!(self.out, "(float2(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " & 0xFFFF, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " >> 16) / {scale}.0)")?;
                    }
                    Function::Unpack4x8snorm => {
                        let scale = 127;

                        write!(self.out, "(float4(int4(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " << 24, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " << 16, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " << 8, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, ") >> 24) / {scale}.0)")?;
                    }
                    Function::Unpack4x8unorm => {
                        let scale = 255;

                        write!(self.out, "(float4(")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " & 0xFF, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " >> 8 & 0xFF, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " >> 16 & 0xFF, ")?;
                        self.write_expr(module, arg, func_ctx)?;
                        write!(self.out, " >> 24) / {scale}.0)")?;
                    }
                    Function::Regular(fun_name) => {
                        write!(self.out, "{fun_name}(")?;
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
                    Function::MissingIntOverload(fun_name) => {
                        let scalar_kind = func_ctx.resolve_type(arg, &module.types).scalar_kind();
                        if let Some(ScalarKind::Sint) = scalar_kind {
                            write!(self.out, "asint({fun_name}(asuint(")?;
                            self.write_expr(module, arg, func_ctx)?;
                            write!(self.out, ")))")?;
                        } else {
                            write!(self.out, "{fun_name}(")?;
                            self.write_expr(module, arg, func_ctx)?;
                            write!(self.out, ")")?;
                        }
                    }
                    Function::MissingIntReturnType(fun_name) => {
                        let scalar_kind = func_ctx.resolve_type(arg, &module.types).scalar_kind();
                        if let Some(ScalarKind::Sint) = scalar_kind {
                            write!(self.out, "asint({fun_name}(")?;
                            self.write_expr(module, arg, func_ctx)?;
                            write!(self.out, "))")?;
                        } else {
                            write!(self.out, "{fun_name}(")?;
                            self.write_expr(module, arg, func_ctx)?;
                            write!(self.out, ")")?;
                        }
                    }
                    Function::CountTrailingZeros => {
                        match *func_ctx.resolve_type(arg, &module.types) {
                            TypeInner::Vector { size, kind, .. } => {
                                let s = match size {
                                    crate::VectorSize::Bi => ".xx",
                                    crate::VectorSize::Tri => ".xxx",
                                    crate::VectorSize::Quad => ".xxxx",
                                };

                                if let ScalarKind::Uint = kind {
                                    write!(self.out, "min((32u){s}, firstbitlow(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, "))")?;
                                } else {
                                    write!(self.out, "asint(min((32u){s}, firstbitlow(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, ")))")?;
                                }
                            }
                            TypeInner::Scalar { kind, .. } => {
                                if let ScalarKind::Uint = kind {
                                    write!(self.out, "min(32u, firstbitlow(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, "))")?;
                                } else {
                                    write!(self.out, "asint(min(32u, firstbitlow(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, ")))")?;
                                }
                            }
                            _ => unreachable!(),
                        }

                        return Ok(());
                    }
                    Function::CountLeadingZeros => {
                        match *func_ctx.resolve_type(arg, &module.types) {
                            TypeInner::Vector { size, kind, .. } => {
                                let s = match size {
                                    crate::VectorSize::Bi => ".xx",
                                    crate::VectorSize::Tri => ".xxx",
                                    crate::VectorSize::Quad => ".xxxx",
                                };

                                if let ScalarKind::Uint = kind {
                                    write!(self.out, "((31u){s} - firstbithigh(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, "))")?;
                                } else {
                                    write!(self.out, "(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(
                                        self.out,
                                        " < (0){s} ? (0){s} : (31){s} - asint(firstbithigh("
                                    )?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, ")))")?;
                                }
                            }
                            TypeInner::Scalar { kind, .. } => {
                                if let ScalarKind::Uint = kind {
                                    write!(self.out, "(31u - firstbithigh(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, "))")?;
                                } else {
                                    write!(self.out, "(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, " < 0 ? 0 : 31 - asint(firstbithigh(")?;
                                    self.write_expr(module, arg, func_ctx)?;
                                    write!(self.out, ")))")?;
                                }
                            }
                            _ => unreachable!(),
                        }

                        return Ok(());
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
                    TypeInner::Struct { ref members, .. } => {
                        let last = members.last().unwrap();
                        let stride = match module.types[last.ty].inner {
                            TypeInner::Array { stride, .. } => stride,
                            _ => unreachable!(),
                        };
                        (last.offset, stride)
                    }
                    _ => unreachable!(),
                };

                let storage_access = match var.space {
                    crate::AddressSpace::Storage { access } => access,
                    _ => crate::StorageAccess::default(),
                };
                let wrapped_array_length = WrappedArrayLength {
                    writable: storage_access.contains(crate::StorageAccess::STORE),
                };

                write!(self.out, "((")?;
                self.write_wrapped_array_length_function_name(wrapped_array_length)?;
                let var_name = &self.names[&NameKey::GlobalVariable(var_handle)];
                write!(self.out, "({var_name}) - {offset}) / {stride})")?
            }
            Expression::Derivative { axis, ctrl, expr } => {
                use crate::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
                if axis == Axis::Width && (ctrl == Ctrl::Coarse || ctrl == Ctrl::Fine) {
                    let tail = match ctrl {
                        Ctrl::Coarse => "coarse",
                        Ctrl::Fine => "fine",
                        Ctrl::None => unreachable!(),
                    };
                    write!(self.out, "abs(ddx_{tail}(")?;
                    self.write_expr(module, expr, func_ctx)?;
                    write!(self.out, ")) + abs(ddy_{tail}(")?;
                    self.write_expr(module, expr, func_ctx)?;
                    write!(self.out, "))")?
                } else {
                    let fun_str = match (axis, ctrl) {
                        (Axis::X, Ctrl::Coarse) => "ddx_coarse",
                        (Axis::X, Ctrl::Fine) => "ddx_fine",
                        (Axis::X, Ctrl::None) => "ddx",
                        (Axis::Y, Ctrl::Coarse) => "ddy_coarse",
                        (Axis::Y, Ctrl::Fine) => "ddy_fine",
                        (Axis::Y, Ctrl::None) => "ddy",
                        (Axis::Width, Ctrl::Coarse | Ctrl::Fine) => unreachable!(),
                        (Axis::Width, Ctrl::None) => "fwidth",
                    };
                    write!(self.out, "{fun_str}(")?;
                    self.write_expr(module, expr, func_ctx)?;
                    write!(self.out, ")")?
                }
            }
            Expression::Relational { fun, argument } => {
                use crate::RelationalFunction as Rf;

                let fun_str = match fun {
                    Rf::All => "all",
                    Rf::Any => "any",
                    Rf::IsNan => "isnan",
                    Rf::IsInf => "isinf",
                };
                write!(self.out, "{fun_str}(")?;
                self.write_expr(module, argument, func_ctx)?;
                write!(self.out, ")")?
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
            // Not supported yet
            Expression::RayQueryGetIntersection { .. } => unreachable!(),
            // Nothing to do here, since call expression already cached
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::SubgroupBallotResult
            | Expression::SubgroupOperationResult { .. } => {}
        }

        if !closing_bracket.is_empty() {
            write!(self.out, "{closing_bracket}")?;
        }
        Ok(())
    }

    fn write_named_expr(
        &mut self,
        module: &Module,
        handle: Handle<crate::Expression>,
        name: String,
        // The expression which is being named.
        // Generally, this is the same as handle, except in WorkGroupUniformLoad
        named: Handle<crate::Expression>,
        ctx: &back::FunctionCtx,
    ) -> BackendResult {
        match ctx.info[named].ty {
            proc::TypeResolution::Handle(ty_handle) => match module.types[ty_handle].inner {
                TypeInner::Struct { .. } => {
                    let ty_name = &self.names[&NameKey::Type(ty_handle)];
                    write!(self.out, "{ty_name}")?;
                }
                _ => {
                    self.write_type(module, ty_handle)?;
                }
            },
            proc::TypeResolution::Value(ref inner) => {
                self.write_value_type(module, inner)?;
            }
        }

        let resolved = ctx.resolve_type(named, &module.types);

        write!(self.out, " {name}")?;
        // If rhs is a array type, we should write array size
        if let TypeInner::Array { base, size, .. } = *resolved {
            self.write_array_size(module, base, size)?;
        }
        write!(self.out, " = ")?;
        self.write_expr(module, handle, ctx)?;
        writeln!(self.out, ";")?;
        self.named_expressions.insert(named, name);

        Ok(())
    }

    /// Helper function that write default zero initialization
    fn write_default_init(&mut self, module: &Module, ty: Handle<crate::Type>) -> BackendResult {
        write!(self.out, "(")?;
        self.write_type(module, ty)?;
        if let TypeInner::Array { base, size, .. } = module.types[ty].inner {
            self.write_array_size(module, base, size)?;
        }
        write!(self.out, ")0")?;
        Ok(())
    }

    fn write_barrier(&mut self, barrier: crate::Barrier, level: back::Level) -> BackendResult {
        if barrier.contains(crate::Barrier::STORAGE) {
            writeln!(self.out, "{level}DeviceMemoryBarrierWithGroupSync();")?;
        }
        if barrier.contains(crate::Barrier::WORK_GROUP) {
            writeln!(self.out, "{level}GroupMemoryBarrierWithGroupSync();")?;
        }
        if barrier.contains(crate::Barrier::SUB_GROUP) {
            unimplemented!() // FIXME
        }
        Ok(())
    }
}

pub(super) struct MatrixType {
    pub(super) columns: crate::VectorSize,
    pub(super) rows: crate::VectorSize,
    pub(super) width: crate::Bytes,
}

pub(super) fn get_inner_matrix_data(
    module: &Module,
    handle: Handle<crate::Type>,
) -> Option<MatrixType> {
    match module.types[handle].inner {
        TypeInner::Matrix {
            columns,
            rows,
            width,
        } => Some(MatrixType {
            columns,
            rows,
            width,
        }),
        TypeInner::Array { base, .. } => get_inner_matrix_data(module, base),
        _ => None,
    }
}

/// Returns the matrix data if the access chain starting at `base`:
/// - starts with an expression with resolved type of [`TypeInner::Matrix`] if `direct = true`
/// - contains one or more expressions with resolved type of [`TypeInner::Array`] of [`TypeInner::Matrix`]
/// - ends at an expression with resolved type of [`TypeInner::Struct`]
pub(super) fn get_inner_matrix_of_struct_array_member(
    module: &Module,
    base: Handle<crate::Expression>,
    func_ctx: &back::FunctionCtx<'_>,
    direct: bool,
) -> Option<MatrixType> {
    let mut mat_data = None;
    let mut array_base = None;

    let mut current_base = base;
    loop {
        let mut resolved = func_ctx.resolve_type(current_base, &module.types);
        if let TypeInner::Pointer { base, .. } = *resolved {
            resolved = &module.types[base].inner;
        };

        match *resolved {
            TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                mat_data = Some(MatrixType {
                    columns,
                    rows,
                    width,
                })
            }
            TypeInner::Array { base, .. } => {
                array_base = Some(base);
            }
            TypeInner::Struct { .. } => {
                if let Some(array_base) = array_base {
                    if direct {
                        return mat_data;
                    } else {
                        return get_inner_matrix_data(module, array_base);
                    }
                }

                break;
            }
            _ => break,
        }

        current_base = match func_ctx.expressions[current_base] {
            crate::Expression::Access { base, .. } => base,
            crate::Expression::AccessIndex { base, .. } => base,
            _ => break,
        };
    }
    None
}

/// Returns the matrix data if the access chain starting at `base`:
/// - starts with an expression with resolved type of [`TypeInner::Matrix`]
/// - contains zero or more expressions with resolved type of [`TypeInner::Array`] of [`TypeInner::Matrix`]
/// - ends with an [`Expression::GlobalVariable`](crate::Expression::GlobalVariable) in [`AddressSpace::Uniform`](crate::AddressSpace::Uniform)
fn get_inner_matrix_of_global_uniform(
    module: &Module,
    base: Handle<crate::Expression>,
    func_ctx: &back::FunctionCtx<'_>,
) -> Option<MatrixType> {
    let mut mat_data = None;
    let mut array_base = None;

    let mut current_base = base;
    loop {
        let mut resolved = func_ctx.resolve_type(current_base, &module.types);
        if let TypeInner::Pointer { base, .. } = *resolved {
            resolved = &module.types[base].inner;
        };

        match *resolved {
            TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                mat_data = Some(MatrixType {
                    columns,
                    rows,
                    width,
                })
            }
            TypeInner::Array { base, .. } => {
                array_base = Some(base);
            }
            _ => break,
        }

        current_base = match func_ctx.expressions[current_base] {
            crate::Expression::Access { base, .. } => base,
            crate::Expression::AccessIndex { base, .. } => base,
            crate::Expression::GlobalVariable(handle)
                if module.global_variables[handle].space == crate::AddressSpace::Uniform =>
            {
                return mat_data.or_else(|| {
                    array_base.and_then(|array_base| get_inner_matrix_data(module, array_base))
                })
            }
            _ => break,
        };
    }
    None
}
