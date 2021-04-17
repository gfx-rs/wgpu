use super::{
    helpers::map_storage_class, Instruction, LogicalLayout, Options, PhysicalLayout, WriterFlags,
};
use crate::{
    arena::{Arena, Handle},
    proc::TypeResolution,
    valid::{FunctionInfo, ModuleInfo},
};
use spirv::Word;
use std::{collections::hash_map::Entry, ops};
use thiserror::Error;

const BITS_PER_BYTE: crate::Bytes = 8;
const CACHED_CONSTANT_INDICES: usize = 8;

#[derive(Clone, Debug, Error)]
pub enum Error {
    #[error("target SPIRV-{0}.{1} is not supported")]
    UnsupportedVersion(u8, u8),
    #[error("one of the required capabilities {0:?} is missing")]
    MissingCapabilities(Vec<spirv::Capability>),
    #[error("unimplemented {0}")]
    FeatureNotImplemented(&'static str),
}

#[derive(Default)]
struct IdGenerator(Word);

impl IdGenerator {
    fn next(&mut self) -> Word {
        self.0 += 1;
        self.0
    }
}

struct Block {
    label_id: Word,
    body: Vec<Instruction>,
    termination: Option<Instruction>,
}

impl Block {
    fn new(label_id: Word) -> Self {
        Block {
            label_id,
            body: Vec::new(),
            termination: None,
        }
    }
}

struct LocalVariable {
    id: Word,
    instruction: Instruction,
}

struct ResultMember {
    id: Word,
    type_id: Word,
    built_in: Option<crate::BuiltIn>,
}

struct EntryPointContext {
    argument_ids: Vec<Word>,
    results: Vec<ResultMember>,
}

#[derive(Default)]
struct Function {
    signature: Option<Instruction>,
    parameters: Vec<Instruction>,
    variables: crate::FastHashMap<Handle<crate::LocalVariable>, LocalVariable>,
    internal_variables: Vec<LocalVariable>,
    blocks: Vec<Block>,
    entry_point_context: Option<EntryPointContext>,
}

impl Function {
    fn to_words(&self, sink: &mut impl Extend<Word>) {
        self.signature.as_ref().unwrap().to_words(sink);
        for instruction in self.parameters.iter() {
            instruction.to_words(sink);
        }
        for (index, block) in self.blocks.iter().enumerate() {
            Instruction::label(block.label_id).to_words(sink);
            if index == 0 {
                for local_var in self.variables.values() {
                    local_var.instruction.to_words(sink);
                }
                for internal_var in self.internal_variables.iter() {
                    internal_var.instruction.to_words(sink);
                }
            }
            for instruction in block.body.iter() {
                instruction.to_words(sink);
            }
            block.termination.as_ref().unwrap().to_words(sink);
        }
    }

    fn consume(&mut self, mut block: Block, termination: Instruction) {
        block.termination = Some(termination);
        self.blocks.push(block);
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LocalType {
    Value {
        vector_size: Option<crate::VectorSize>,
        kind: crate::ScalarKind,
        width: crate::Bytes,
        pointer_class: Option<spirv::StorageClass>,
    },
    Matrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
        width: crate::Bytes,
    },
    Pointer {
        base: Handle<crate::Type>,
        class: spirv::StorageClass,
    },
    SampledImage {
        image_type: Handle<crate::Type>,
    },
}

impl PhysicalLayout {
    fn make_local(&self, inner: &crate::TypeInner) -> Option<LocalType> {
        Some(match *inner {
            crate::TypeInner::Scalar { kind, width } => LocalType::Value {
                vector_size: None,
                kind,
                width,
                pointer_class: None,
            },
            crate::TypeInner::Vector { size, kind, width } => LocalType::Value {
                vector_size: Some(size),
                kind,
                width,
                pointer_class: None,
            },
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => LocalType::Matrix {
                columns,
                rows,
                width,
            },
            crate::TypeInner::Pointer { base, class } => LocalType::Pointer {
                base,
                class: map_storage_class(class),
            },
            crate::TypeInner::ValuePointer {
                size,
                kind,
                width,
                class,
            } => LocalType::Value {
                vector_size: size,
                kind,
                width,
                pointer_class: Some(map_storage_class(class)),
            },
            _ => return None,
        })
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LookupType {
    Handle(Handle<crate::Type>),
    Local(LocalType),
}

impl From<LocalType> for LookupType {
    fn from(local: LocalType) -> Self {
        Self::Local(local)
    }
}

fn map_dim(dim: crate::ImageDimension) -> spirv::Dim {
    match dim {
        crate::ImageDimension::D1 => spirv::Dim::Dim1D,
        crate::ImageDimension::D2 => spirv::Dim::Dim2D,
        crate::ImageDimension::D3 => spirv::Dim::Dim2D,
        crate::ImageDimension::Cube => spirv::Dim::DimCube,
    }
}

#[derive(Debug, PartialEq, Clone, Hash, Eq)]
struct LookupFunctionType {
    parameter_type_ids: Vec<Word>,
    return_type_id: Word,
}

#[derive(Debug)]
enum Dimension {
    Scalar,
    Vector,
    Matrix,
}

fn get_dimension(type_inner: &crate::TypeInner) -> Dimension {
    match *type_inner {
        crate::TypeInner::Scalar { .. } => Dimension::Scalar,
        crate::TypeInner::Vector { .. } => Dimension::Vector,
        crate::TypeInner::Matrix { .. } => Dimension::Matrix,
        _ => unreachable!(),
    }
}

#[derive(Clone, Copy, Default)]
struct LoopContext {
    continuing_id: Option<Word>,
    break_id: Option<Word>,
}

#[derive(Default)]
struct CachedExpressions {
    ids: Vec<Word>,
}
impl CachedExpressions {
    fn reset(&mut self, length: usize) {
        self.ids.clear();
        self.ids.resize(length, 0);
    }
}
impl ops::Index<Handle<crate::Expression>> for CachedExpressions {
    type Output = Word;
    fn index(&self, h: Handle<crate::Expression>) -> &Word {
        let id = &self.ids[h.index()];
        if *id == 0 {
            unreachable!("Expression {:?} is not cached!", h);
        }
        id
    }
}
impl ops::IndexMut<Handle<crate::Expression>> for CachedExpressions {
    fn index_mut(&mut self, h: Handle<crate::Expression>) -> &mut Word {
        let id = &mut self.ids[h.index()];
        if *id != 0 {
            unreachable!("Expression {:?} is already cached!", h);
        }
        id
    }
}

struct GlobalVariable {
    /// Actual ID of the variable.
    id: Word,
    /// For `StorageClass::Handle` variables, this ID is recorded in the function
    /// prelude block (and reset before every function) as `OpLoad` of the variable.
    /// It is then used for all the global ops, such as `OpImageSample`.
    handle_id: Word,
    /// SPIR-V storage class.
    class: spirv::StorageClass,
}

pub struct Writer {
    physical_layout: PhysicalLayout,
    logical_layout: LogicalLayout,
    id_gen: IdGenerator,
    capabilities: crate::FastHashSet<spirv::Capability>,
    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    flags: WriterFlags,
    void_type: u32,
    //TODO: convert most of these into vectors, addressable by handle indices
    lookup_type: crate::FastHashMap<LookupType, Word>,
    lookup_function: crate::FastHashMap<Handle<crate::Function>, Word>,
    lookup_function_type: crate::FastHashMap<LookupFunctionType, Word>,
    lookup_function_call: crate::FastHashMap<Handle<crate::Expression>, Word>,
    constant_ids: Vec<Word>,
    index_constant_ids: Vec<Word>,
    global_variables: Vec<GlobalVariable>,
    cached: CachedExpressions,
    gl450_ext_inst_id: Word,
    // Just a temporary list of SPIR-V ids
    temp_list: Vec<Word>,
}

impl Writer {
    pub fn new(options: &Options) -> Result<Self, Error> {
        let (major, minor) = options.lang_version;
        if major != 1 {
            return Err(Error::UnsupportedVersion(major, minor));
        }
        let raw_version = ((major as u32) << 16) | ((minor as u32) << 8);
        let mut id_gen = IdGenerator::default();
        let gl450_ext_inst_id = id_gen.next();
        let void_type = id_gen.next();

        Ok(Writer {
            physical_layout: PhysicalLayout::new(raw_version),
            logical_layout: LogicalLayout::default(),
            id_gen,
            capabilities: options.capabilities.clone(),
            debugs: vec![],
            annotations: vec![],
            flags: options.flags,
            void_type,
            lookup_type: crate::FastHashMap::default(),
            lookup_function: crate::FastHashMap::default(),
            lookup_function_type: crate::FastHashMap::default(),
            lookup_function_call: crate::FastHashMap::default(),
            constant_ids: Vec::new(),
            index_constant_ids: Vec::new(),
            global_variables: Vec::new(),
            cached: CachedExpressions::default(),
            gl450_ext_inst_id,
            temp_list: Vec::new(),
        })
    }

    fn check(&mut self, capabilities: &[spirv::Capability]) -> Result<(), Error> {
        if capabilities.is_empty()
            || capabilities
                .iter()
                .any(|cap| self.capabilities.contains(cap))
        {
            Ok(())
        } else {
            Err(Error::MissingCapabilities(capabilities.to_vec()))
        }
    }

    fn get_type_id(
        &mut self,
        arena: &Arena<crate::Type>,
        lookup_ty: LookupType,
    ) -> Result<Word, Error> {
        if let Entry::Occupied(e) = self.lookup_type.entry(lookup_ty) {
            Ok(*e.get())
        } else {
            match lookup_ty {
                LookupType::Handle(_handle) => unreachable!("Handles are populated at start"),
                LookupType::Local(local_ty) => self.write_type_declaration_local(arena, local_ty),
            }
        }
    }

    fn get_expression_type_id(
        &mut self,
        arena: &Arena<crate::Type>,
        tr: &TypeResolution,
    ) -> Result<Word, Error> {
        let lookup_ty = match *tr {
            TypeResolution::Handle(ty_handle) => LookupType::Handle(ty_handle),
            TypeResolution::Value(ref inner) => {
                LookupType::Local(self.physical_layout.make_local(inner).unwrap())
            }
        };
        self.get_type_id(arena, lookup_ty)
    }

    fn get_pointer_id(
        &mut self,
        arena: &Arena<crate::Type>,
        handle: Handle<crate::Type>,
        class: spirv::StorageClass,
    ) -> Result<Word, Error> {
        let ty_id = self.get_type_id(arena, LookupType::Handle(handle))?;
        if let crate::TypeInner::Pointer { .. } = arena[handle].inner {
            return Ok(ty_id);
        }
        let lookup_type = LookupType::Local(LocalType::Pointer {
            base: handle,
            class,
        });
        Ok(if let Some(&id) = self.lookup_type.get(&lookup_type) {
            id
        } else {
            let id = self.id_gen.next();
            let instruction = Instruction::type_pointer(id, class, ty_id);
            instruction.to_words(&mut self.logical_layout.declarations);
            self.lookup_type.insert(lookup_type, id);
            id
        })
    }

    fn create_constant(&mut self, type_id: Word, value: &[Word]) -> Word {
        let id = self.id_gen.next();
        let instruction = Instruction::constant(type_id, id, value);
        instruction.to_words(&mut self.logical_layout.declarations);
        id
    }

    fn get_index_constant(
        &mut self,
        index: Word,
        types: &Arena<crate::Type>,
    ) -> Result<Word, Error> {
        while self.index_constant_ids.len() <= index as usize {
            self.index_constant_ids.push(0);
        }
        let cached = self.index_constant_ids[index as usize];
        if cached != 0 {
            return Ok(cached);
        }

        let type_id = self.get_type_id(
            types,
            LookupType::Local(LocalType::Value {
                vector_size: None,
                kind: crate::ScalarKind::Sint,
                width: 4,
                pointer_class: None,
            }),
        )?;

        let id = self.create_constant(type_id, &[index]);
        self.index_constant_ids[index as usize] = id;
        Ok(id)
    }

    fn decorate(&mut self, id: Word, decoration: spirv::Decoration, operands: &[Word]) {
        self.annotations.push(Instruction::decorate(id, decoration, operands));
    }

    fn write_function(
        &mut self,
        ir_function: &crate::Function,
        info: &FunctionInfo,
        ir_module: &crate::Module,
        mut varying_ids: Option<&mut Vec<Word>>,
    ) -> Result<Word, Error> {
        let mut function = Function::default();

        for (handle, variable) in ir_function.local_variables.iter() {
            let id = self.id_gen.next();

            if self.flags.contains(WriterFlags::DEBUG) {
                if let Some(ref name) = variable.name {
                    self.debugs.push(Instruction::name(id, name));
                }
            }

            let init_word = variable
                .init
                .map(|constant| self.constant_ids[constant.index()]);
            let pointer_type_id =
                self.get_pointer_id(&ir_module.types, variable.ty, spirv::StorageClass::Function)?;
            let instruction = Instruction::variable(
                pointer_type_id,
                id,
                spirv::StorageClass::Function,
                init_word,
            );
            function
                .variables
                .insert(handle, LocalVariable { id, instruction });
        }

        let prelude_id = self.id_gen.next();
        let mut prelude = Block::new(prelude_id);
        let mut ep_context = EntryPointContext {
            argument_ids: Vec::new(),
            results: Vec::new(),
        };

        let mut parameter_type_ids = Vec::with_capacity(ir_function.arguments.len());
        for argument in ir_function.arguments.iter() {
            let class = spirv::StorageClass::Input;
            let argument_type_id =
                self.get_type_id(&ir_module.types, LookupType::Handle(argument.ty))?;
            if let Some(ref mut list) = varying_ids {
                let id = if let Some(ref binding) = argument.binding {
                    let name = argument.name.as_ref().map(AsRef::as_ref);
                    let varying_id =
                        self.write_varying(ir_module, class, name, argument.ty, binding)?;
                    list.push(varying_id);
                    let id = self.id_gen.next();
                    prelude
                        .body
                        .push(Instruction::load(argument_type_id, id, varying_id, None));
                    id
                } else if let crate::TypeInner::Struct { ref members, .. } =
                    ir_module.types[argument.ty].inner
                {
                    let struct_id = self.id_gen.next();
                    let mut constituent_ids = Vec::with_capacity(members.len());
                    for member in members {
                        let type_id =
                            self.get_type_id(&ir_module.types, LookupType::Handle(member.ty))?;
                        let name = member.name.as_ref().map(AsRef::as_ref);
                        let binding = member.binding.as_ref().unwrap();
                        let varying_id =
                            self.write_varying(ir_module, class, name, member.ty, binding)?;
                        list.push(varying_id);
                        let id = self.id_gen.next();
                        prelude
                            .body
                            .push(Instruction::load(type_id, id, varying_id, None));
                        constituent_ids.push(id);
                    }
                    prelude.body.push(Instruction::composite_construct(
                        argument_type_id,
                        struct_id,
                        &constituent_ids,
                    ));
                    struct_id
                } else {
                    unreachable!("Missing argument binding on an entry point");
                };
                ep_context.argument_ids.push(id);
            } else {
                let id = self.id_gen.next();
                let instruction = Instruction::function_parameter(argument_type_id, id);
                function.parameters.push(instruction);
                parameter_type_ids.push(argument_type_id);
            };
        }

        let return_type_id = match ir_function.result {
            Some(ref result) => {
                if let Some(ref mut list) = varying_ids {
                    let class = spirv::StorageClass::Output;
                    if let Some(ref binding) = result.binding {
                        let type_id =
                            self.get_type_id(&ir_module.types, LookupType::Handle(result.ty))?;
                        let varying_id =
                            self.write_varying(ir_module, class, None, result.ty, binding)?;
                        list.push(varying_id);
                        ep_context.results.push(ResultMember {
                            id: varying_id,
                            type_id,
                            built_in: binding.to_built_in(),
                        });
                    } else if let crate::TypeInner::Struct { ref members, .. } =
                        ir_module.types[result.ty].inner
                    {
                        for member in members {
                            let type_id =
                                self.get_type_id(&ir_module.types, LookupType::Handle(member.ty))?;
                            let name = member.name.as_ref().map(AsRef::as_ref);
                            let binding = member.binding.as_ref().unwrap();
                            let varying_id =
                                self.write_varying(ir_module, class, name, member.ty, binding)?;
                            list.push(varying_id);
                            ep_context.results.push(ResultMember {
                                id: varying_id,
                                type_id,
                                built_in: binding.to_built_in(),
                            });
                        }
                    } else {
                        unreachable!("Missing result binding on an entry point");
                    }
                    self.void_type
                } else {
                    self.get_type_id(&ir_module.types, LookupType::Handle(result.ty))?
                }
            }
            None => self.void_type,
        };

        let lookup_function_type = LookupFunctionType {
            return_type_id,
            parameter_type_ids,
        };

        let function_id = self.id_gen.next();
        if self.flags.contains(WriterFlags::DEBUG) {
            if let Some(ref name) = ir_function.name {
                self.debugs.push(Instruction::name(function_id, name));
            }
        }

        let function_type = self.get_function_type(lookup_function_type);
        function.signature = Some(Instruction::function(
            return_type_id,
            function_id,
            spirv::FunctionControl::empty(),
            function_type,
        ));

        if varying_ids.is_some() {
            function.entry_point_context = Some(ep_context);
        }

        // fill up the `GlobalVariable::handle_id`
        for gv in self.global_variables.iter_mut() {
            gv.handle_id = 0;
        }
        for (handle, var) in ir_module.global_variables.iter() {
            // Handle globals are pre-emitted and should be loaded automatically.
            if info[handle].is_empty() || var.class != crate::StorageClass::Handle {
                continue;
            }
            let id = self.id_gen.next();
            let result_type_id = self.get_type_id(&ir_module.types, LookupType::Handle(var.ty))?;
            let gv = &mut self.global_variables[handle.index()];
            prelude
                .body
                .push(Instruction::load(result_type_id, id, gv.id, None));
            gv.handle_id = id;
        }
        // fill up the pre-emitted expressions
        self.cached.reset(ir_function.expressions.len());
        for (handle, expr) in ir_function.expressions.iter() {
            if expr.needs_pre_emit() {
                self.cache_expression_value(
                    ir_module,
                    ir_function,
                    info,
                    handle,
                    &mut prelude,
                    &mut function,
                )?;
            }
        }

        let main_id = self.id_gen.next();
        function.consume(prelude, Instruction::branch(main_id));
        self.write_block(
            main_id,
            &ir_function.body,
            ir_module,
            ir_function,
            info,
            &mut function,
            None,
            LoopContext::default(),
        )?;

        function.to_words(&mut self.logical_layout.function_definitions);
        Instruction::function_end().to_words(&mut self.logical_layout.function_definitions);

        Ok(function_id)
    }

    // TODO Move to instructions module
    fn write_entry_point(
        &mut self,
        entry_point: &crate::EntryPoint,
        info: &FunctionInfo,
        ir_module: &crate::Module,
    ) -> Result<Instruction, Error> {
        let mut interface_ids = Vec::new();
        let function_id = self.write_function(
            &entry_point.function,
            info,
            ir_module,
            Some(&mut interface_ids),
        )?;

        let exec_model = match entry_point.stage {
            crate::ShaderStage::Vertex => spirv::ExecutionModel::Vertex,
            crate::ShaderStage::Fragment => {
                let execution_mode = spirv::ExecutionMode::OriginUpperLeft;
                self.check(execution_mode.required_capabilities())?;
                Instruction::execution_mode(function_id, execution_mode, &[])
                    .to_words(&mut self.logical_layout.execution_modes);
                spirv::ExecutionModel::Fragment
            }
            crate::ShaderStage::Compute => {
                let execution_mode = spirv::ExecutionMode::LocalSize;
                self.check(execution_mode.required_capabilities())?;
                Instruction::execution_mode(
                    function_id,
                    execution_mode,
                    &entry_point.workgroup_size,
                )
                .to_words(&mut self.logical_layout.execution_modes);
                spirv::ExecutionModel::GLCompute
            }
        };
        self.check(exec_model.required_capabilities())?;

        Ok(Instruction::entry_point(
            exec_model,
            function_id,
            &entry_point.name,
            interface_ids.as_slice(),
        ))
    }

    fn write_scalar(&self, id: Word, kind: crate::ScalarKind, width: crate::Bytes) -> Instruction {
        let bits = (width * BITS_PER_BYTE) as u32;
        match kind {
            crate::ScalarKind::Sint => {
                Instruction::type_int(id, bits, super::instructions::Signedness::Signed)
            }
            crate::ScalarKind::Uint => {
                Instruction::type_int(id, bits, super::instructions::Signedness::Unsigned)
            }
            crate::ScalarKind::Float => Instruction::type_float(id, bits),
            crate::ScalarKind::Bool => Instruction::type_bool(id),
        }
    }

    fn write_type_declaration_local(
        &mut self,
        arena: &Arena<crate::Type>,
        local_ty: LocalType,
    ) -> Result<Word, Error> {
        let id = self.id_gen.next();
        let instruction = match local_ty {
            LocalType::Value {
                vector_size: None,
                kind,
                width,
                pointer_class: None,
            } => self.write_scalar(id, kind, width),
            LocalType::Value {
                vector_size: Some(size),
                kind,
                width,
                pointer_class: None,
            } => {
                let scalar_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Value {
                        vector_size: None,
                        kind,
                        width,
                        pointer_class: None,
                    }),
                )?;
                Instruction::type_vector(id, scalar_id, size)
            }
            LocalType::Matrix {
                columns,
                rows,
                width,
            } => {
                let vector_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Value {
                        vector_size: Some(rows),
                        kind: crate::ScalarKind::Float,
                        width,
                        pointer_class: None,
                    }),
                )?;
                Instruction::type_matrix(id, vector_id, columns)
            }
            LocalType::Pointer { base, class } => {
                let type_id = self.get_type_id(arena, LookupType::Handle(base))?;
                Instruction::type_pointer(id, class, type_id)
            }
            LocalType::Value {
                vector_size,
                kind,
                width,
                pointer_class: Some(class),
            } => {
                let type_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Value {
                        vector_size,
                        kind,
                        width,
                        pointer_class: None,
                    }),
                )?;
                Instruction::type_pointer(id, class, type_id)
            }
            LocalType::SampledImage { image_type } => {
                let image_type_id = self.get_type_id(arena, LookupType::Handle(image_type))?;
                Instruction::type_sampled_image(id, image_type_id)
            }
        };

        self.lookup_type.insert(LookupType::Local(local_ty), id);
        instruction.to_words(&mut self.logical_layout.declarations);
        Ok(id)
    }

    fn write_type_declaration_arena(
        &mut self,
        arena: &Arena<crate::Type>,
        handle: Handle<crate::Type>,
    ) -> Result<Word, Error> {
        let ty = &arena[handle];
        let decorate_layout = true; //TODO?

        let id = if let Some(local) = self.physical_layout.make_local(&ty.inner) {
            match self.lookup_type.entry(LookupType::Local(local)) {
                // if it's already known as local, re-use it
                Entry::Occupied(e) => {
                    let id = *e.into_mut();
                    self.lookup_type.insert(LookupType::Handle(handle), id);
                    return Ok(id);
                }
                // also register the type as "local", to avoid duplication
                Entry::Vacant(e) => {
                    let id = self.id_gen.next();
                    *e.insert(id)
                }
            }
        } else {
            self.id_gen.next()
        };
        self.lookup_type.insert(LookupType::Handle(handle), id);

        if self.flags.contains(WriterFlags::DEBUG) {
            if let Some(ref name) = ty.name {
                self.debugs.push(Instruction::name(id, name));
            }
        }

        use spirv::Decoration;

        let instruction = match ty.inner {
            crate::TypeInner::Scalar { kind, width } => self.write_scalar(id, kind, width),
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Value {
                        vector_size: None,
                        kind,
                        width,
                        pointer_class: None,
                    }),
                )?;
                Instruction::type_vector(id, scalar_id, size)
            }
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                let vector_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Value {
                        vector_size: Some(rows),
                        kind: crate::ScalarKind::Float,
                        width,
                        pointer_class: None,
                    }),
                )?;
                Instruction::type_matrix(id, vector_id, columns)
            }
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let kind = match class {
                    crate::ImageClass::Sampled { kind, multi: _ } => kind,
                    crate::ImageClass::Depth => crate::ScalarKind::Float,
                    crate::ImageClass::Storage(format) => format.into(),
                };
                let local_type = LocalType::Value {
                    vector_size: None,
                    kind,
                    width: 4,
                    pointer_class: None,
                };
                let type_id = self.get_type_id(arena, LookupType::Local(local_type))?;
                let dim = map_dim(dim);
                self.check(dim.required_capabilities())?;
                Instruction::type_image(id, type_id, dim, arrayed, class)
            }
            crate::TypeInner::Sampler { comparison: _ } => Instruction::type_sampler(id),
            crate::TypeInner::Array { base, size, stride } => {
                if decorate_layout {
                    self.decorate(id, Decoration::ArrayStride, &[stride]);
                }

                let type_id = self.get_type_id(arena, LookupType::Handle(base))?;
                match size {
                    crate::ArraySize::Constant(const_handle) => {
                        let length_id = self.constant_ids[const_handle.index()];
                        Instruction::type_array(id, type_id, length_id)
                    }
                    crate::ArraySize::Dynamic => Instruction::type_runtime_array(id, type_id),
                }
            }
            crate::TypeInner::Struct {
                ref level,
                ref members,
                span: _,
            } => {
                if let crate::StructLevel::Root = *level {
                    self.decorate(id, Decoration::Block, &[]);
                }

                let mut member_ids = Vec::with_capacity(members.len());
                for (index, member) in members.iter().enumerate() {
                    if decorate_layout {
                        self.annotations.push(Instruction::member_decorate(
                            id,
                            index as u32,
                            Decoration::Offset,
                            &[member.offset],
                        ));
                    }

                    if self.flags.contains(WriterFlags::DEBUG) {
                        if let Some(ref name) = member.name {
                            self.debugs
                                .push(Instruction::member_name(id, index as u32, name));
                        }
                    }

                    // The matrix decorations also go on arrays of matrices,
                    // so lets check this first.
                    let member_array_subty_inner = match arena[member.ty].inner {
                        crate::TypeInner::Array { base, .. } => &arena[base].inner,
                        ref other => other,
                    };
                    if let crate::TypeInner::Matrix {
                        columns,
                        rows: _,
                        width,
                    } = *member_array_subty_inner
                    {
                        let byte_stride = match columns {
                            crate::VectorSize::Bi => 2 * width,
                            crate::VectorSize::Tri | crate::VectorSize::Quad => 4 * width,
                        };
                        self.annotations.push(Instruction::member_decorate(
                            id,
                            index as u32,
                            Decoration::ColMajor,
                            &[],
                        ));
                        self.annotations.push(Instruction::member_decorate(
                            id,
                            index as u32,
                            Decoration::MatrixStride,
                            &[byte_stride as u32],
                        ));
                    }

                    let member_id = self.get_type_id(arena, LookupType::Handle(member.ty))?;
                    member_ids.push(member_id);
                }
                Instruction::type_struct(id, member_ids.as_slice())
            }
            crate::TypeInner::Pointer { base, class } => {
                let type_id = self.get_type_id(arena, LookupType::Handle(base))?;
                let raw_class = map_storage_class(class);
                Instruction::type_pointer(id, raw_class, type_id)
            }
            crate::TypeInner::ValuePointer {
                size,
                kind,
                width,
                class,
            } => {
                let raw_class = map_storage_class(class);
                let type_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Value {
                        vector_size: size,
                        kind,
                        width,
                        pointer_class: None,
                    }),
                )?;
                Instruction::type_pointer(id, raw_class, type_id)
            }
        };

        instruction.to_words(&mut self.logical_layout.declarations);
        Ok(id)
    }

    fn write_constant_scalar(
        &mut self,
        id: Word,
        value: &crate::ScalarValue,
        width: crate::Bytes,
        debug_name: Option<&String>,
        types: &Arena<crate::Type>,
    ) -> Result<(), Error> {
        if self.flags.contains(WriterFlags::DEBUG) {
            if let Some(name) = debug_name {
                self.debugs.push(Instruction::name(id, name));
            }
        }
        let type_id = self.get_type_id(
            types,
            LookupType::Local(LocalType::Value {
                vector_size: None,
                kind: value.scalar_kind(),
                width,
                pointer_class: None,
            }),
        )?;
        let (solo, pair);
        let instruction = match *value {
            crate::ScalarValue::Sint(val) => {
                let words = match width {
                    4 => {
                        if debug_name.is_none()
                            && 0 <= val
                            && val < CACHED_CONSTANT_INDICES as i64
                            && self.index_constant_ids.get(val as usize).unwrap_or(&0) == &0
                        {
                            // cache this as an indexing constant
                            while self.index_constant_ids.len() <= val as usize {
                                self.index_constant_ids.push(0);
                            }
                            self.index_constant_ids[val as usize] = id;
                        }
                        solo = [val as u32];
                        &solo[..]
                    }
                    8 => {
                        pair = [(val >> 32) as u32, val as u32];
                        &pair
                    }
                    _ => unreachable!(),
                };
                Instruction::constant(type_id, id, words)
            }
            crate::ScalarValue::Uint(val) => {
                let words = match width {
                    4 => {
                        solo = [val as u32];
                        &solo[..]
                    }
                    8 => {
                        pair = [(val >> 32) as u32, val as u32];
                        &pair
                    }
                    _ => unreachable!(),
                };
                Instruction::constant(type_id, id, words)
            }
            crate::ScalarValue::Float(val) => {
                let words = match width {
                    4 => {
                        solo = [(val as f32).to_bits()];
                        &solo[..]
                    }
                    8 => {
                        let bits = f64::to_bits(val);
                        pair = [(bits >> 32) as u32, bits as u32];
                        &pair
                    }
                    _ => unreachable!(),
                };
                Instruction::constant(type_id, id, words)
            }
            crate::ScalarValue::Bool(true) => Instruction::constant_true(type_id, id),
            crate::ScalarValue::Bool(false) => Instruction::constant_false(type_id, id),
        };

        instruction.to_words(&mut self.logical_layout.declarations);
        Ok(())
    }

    fn write_constant_composite(
        &mut self,
        id: Word,
        ty: Handle<crate::Type>,
        components: &[Handle<crate::Constant>],
        types: &Arena<crate::Type>,
    ) -> Result<(), Error> {
        let mut constituent_ids = Vec::with_capacity(components.len());
        for constituent in components.iter() {
            let constituent_id = self.constant_ids[constituent.index()];
            constituent_ids.push(constituent_id);
        }

        let type_id = self.get_type_id(types, LookupType::Handle(ty))?;
        Instruction::constant_composite(type_id, id, constituent_ids.as_slice())
            .to_words(&mut self.logical_layout.declarations);
        Ok(())
    }

    fn write_varying(
        &mut self,
        ir_module: &crate::Module,
        class: spirv::StorageClass,
        debug_name: Option<&str>,
        ty: Handle<crate::Type>,
        binding: &crate::Binding,
    ) -> Result<Word, Error> {
        let id = self.id_gen.next();
        let pointer_type_id = self.get_pointer_id(&ir_module.types, ty, class)?;
        Instruction::variable(pointer_type_id, id, class, None)
            .to_words(&mut self.logical_layout.declarations);

        if self.flags.contains(WriterFlags::DEBUG) {
            if let Some(name) = debug_name {
                self.debugs.push(Instruction::name(id, name));
            }
        }

        use spirv::{BuiltIn, Decoration};

        match *binding {
            crate::Binding::Location { location, interpolation, sampling } => {
                self.decorate(id, Decoration::Location, &[location]);

                match interpolation {
                    // Perspective-correct interpolation is the default in SPIR-V.
                    None | Some(crate::Interpolation::Perspective) => (),
                    Some(crate::Interpolation::Flat) => {
                        self.decorate(id, Decoration::Flat, &[]);
                    }
                    Some(crate::Interpolation::Linear) => {
                        self.decorate(id, Decoration::NoPerspective, &[]);
                    }
                }

                match sampling {
                    // Center sampling is the default in SPIR-V.
                    None | Some(crate::Sampling::Center) => (),
                    Some(crate::Sampling::Centroid) => {
                        self.decorate(id, Decoration::Centroid, &[]);
                    }
                    Some(crate::Sampling::Sample) => {
                        self.decorate(id, Decoration::Sample, &[]);
                    }
                }
            }
            crate::Binding::BuiltIn(built_in) => {
                use crate::BuiltIn as Bi;
                let built_in = match built_in {
                    Bi::Position => {
                        if class == spirv::StorageClass::Output {
                            BuiltIn::Position
                        } else {
                            BuiltIn::FragCoord
                        }
                    }
                    // vertex
                    Bi::BaseInstance => BuiltIn::BaseInstance,
                    Bi::BaseVertex => BuiltIn::BaseVertex,
                    Bi::ClipDistance => BuiltIn::ClipDistance,
                    Bi::InstanceIndex => BuiltIn::InstanceIndex,
                    Bi::PointSize => BuiltIn::PointSize,
                    Bi::VertexIndex => BuiltIn::VertexIndex,
                    // fragment
                    Bi::FragDepth => BuiltIn::FragDepth,
                    Bi::FrontFacing => BuiltIn::FrontFacing,
                    Bi::SampleIndex => BuiltIn::SampleId,
                    Bi::SampleMask => BuiltIn::SampleMask,
                    // compute
                    Bi::GlobalInvocationId => BuiltIn::GlobalInvocationId,
                    Bi::LocalInvocationId => BuiltIn::LocalInvocationId,
                    Bi::LocalInvocationIndex => BuiltIn::LocalInvocationIndex,
                    Bi::WorkGroupId => BuiltIn::WorkgroupId,
                    Bi::WorkGroupSize => BuiltIn::WorkgroupSize,
                };

                self.decorate(id, Decoration::BuiltIn, &[built_in as u32]);
            }
        }

        Ok(id)
    }

    fn write_global_variable(
        &mut self,
        ir_module: &crate::Module,
        global_variable: &crate::GlobalVariable,
    ) -> Result<(Instruction, Word, spirv::StorageClass), Error> {
        let id = self.id_gen.next();

        let class = map_storage_class(global_variable.class);
        self.check(class.required_capabilities())?;

        let init_word = global_variable
            .init
            .map(|constant| self.constant_ids[constant.index()]);
        let pointer_type_id = self.get_pointer_id(&ir_module.types, global_variable.ty, class)?;
        let instruction = Instruction::variable(pointer_type_id, id, class, init_word);

        if self.flags.contains(WriterFlags::DEBUG) {
            if let Some(ref name) = global_variable.name {
                self.debugs.push(Instruction::name(id, name));
            }
        }

        use spirv::Decoration;

        let access_decoration = match global_variable.storage_access {
            crate::StorageAccess::LOAD => Some(Decoration::NonWritable),
            crate::StorageAccess::STORE => Some(Decoration::NonReadable),
            _ => None,
        };
        if let Some(decoration) = access_decoration {
            self.decorate(id, decoration, &[]);
        }

        if let Some(ref res_binding) = global_variable.binding {
            self.decorate(id, Decoration::DescriptorSet, &[res_binding.group]);
            self.decorate(id, Decoration::Binding, &[res_binding.binding]);
        }

        // TODO Initializer is optional and not (yet) included in the IR
        Ok((instruction, id, class))
    }

    fn get_function_type(&mut self, lookup_function_type: LookupFunctionType) -> Word {
        match self
            .lookup_function_type
            .entry(lookup_function_type.clone())
        {
            Entry::Occupied(e) => *e.get(),
            _ => {
                let id = self.id_gen.next();
                let instruction = Instruction::type_function(
                    id,
                    lookup_function_type.return_type_id,
                    &lookup_function_type.parameter_type_ids,
                );
                instruction.to_words(&mut self.logical_layout.declarations);
                self.lookup_function_type.insert(lookup_function_type, id);
                id
            }
        }
    }

    fn write_texture_coordinates(
        &mut self,
        ir_module: &crate::Module,
        fun_info: &FunctionInfo,
        coordinates: Handle<crate::Expression>,
        array_index: Option<Handle<crate::Expression>>,
        block: &mut Block,
    ) -> Result<Word, Error> {
        let coordinate_id = self.cached[coordinates];

        Ok(if let Some(array_index) = array_index {
            let coordinate_scalar_type_id = self.get_type_id(
                &ir_module.types,
                LookupType::Local(LocalType::Value {
                    vector_size: None,
                    kind: crate::ScalarKind::Float,
                    width: 4,
                    pointer_class: None,
                }),
            )?;

            let mut constituent_ids = [0u32; 4];
            let size = match *fun_info[coordinates].ty.inner_with(&ir_module.types) {
                crate::TypeInner::Scalar { .. } => {
                    constituent_ids[0] = coordinate_id;
                    crate::VectorSize::Bi
                }
                crate::TypeInner::Vector { size, .. } => {
                    for i in 0..size as u32 {
                        let id = self.id_gen.next();
                        constituent_ids[i as usize] = id;
                        block.body.push(Instruction::composite_extract(
                            coordinate_scalar_type_id,
                            id,
                            coordinate_id,
                            &[i],
                        ));
                    }
                    match size {
                        crate::VectorSize::Bi => crate::VectorSize::Tri,
                        crate::VectorSize::Tri => crate::VectorSize::Quad,
                        crate::VectorSize::Quad => {
                            unimplemented!("Unable to extend the vec4 coordinate")
                        }
                    }
                }
                ref other => unimplemented!("wrong coordinate type {:?}", other),
            };

            let array_index_f32_id = self.id_gen.next();
            constituent_ids[size as usize - 1] = array_index_f32_id;

            let array_index_u32_id = self.cached[array_index];
            let cast_instruction = Instruction::unary(
                spirv::Op::ConvertUToF,
                coordinate_scalar_type_id,
                array_index_f32_id,
                array_index_u32_id,
            );
            block.body.push(cast_instruction);

            let extended_coordinate_type_id = self.get_type_id(
                &ir_module.types,
                LookupType::Local(LocalType::Value {
                    vector_size: Some(size),
                    kind: crate::ScalarKind::Float,
                    width: 4,
                    pointer_class: None,
                }),
            )?;

            let id = self.id_gen.next();
            block.body.push(Instruction::composite_construct(
                extended_coordinate_type_id,
                id,
                &constituent_ids[..size as usize],
            ));
            id
        } else {
            coordinate_id
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn promote_access_expression_to_variable(
        &mut self,
        ir_types: &Arena<crate::Type>,
        result_type_id: Word,
        container_id: Word,
        container_resolution: &TypeResolution,
        index_id: Word,
        element_ty: Handle<crate::Type>,
        block: &mut Block,
    ) -> Result<(Word, LocalVariable), Error> {
        let container_type_id = self.get_expression_type_id(ir_types, container_resolution)?;
        let pointer_type_id = self.id_gen.next();
        Instruction::type_pointer(
            pointer_type_id,
            spirv::StorageClass::Function,
            container_type_id,
        )
        .to_words(&mut self.logical_layout.declarations);

        let variable = {
            let id = self.id_gen.next();
            LocalVariable {
                id,
                instruction: Instruction::variable(
                    pointer_type_id,
                    id,
                    spirv::StorageClass::Function,
                    None,
                ),
            }
        };
        block
            .body
            .push(Instruction::store(variable.id, container_id, None));

        let element_pointer_id = self.id_gen.next();
        let element_pointer_type_id =
            self.get_pointer_id(ir_types, element_ty, spirv::StorageClass::Function)?;
        block.body.push(Instruction::access_chain(
            element_pointer_type_id,
            element_pointer_id,
            variable.id,
            &[index_id],
        ));
        let id = self.id_gen.next();
        block.body.push(Instruction::load(
            result_type_id,
            id,
            element_pointer_id,
            None,
        ));

        Ok((id, variable))
    }

    /// Cache an expression for a value.
    fn cache_expression_value(
        &mut self,
        ir_module: &crate::Module,
        ir_function: &crate::Function,
        fun_info: &FunctionInfo,
        expr_handle: Handle<crate::Expression>,
        block: &mut Block,
        function: &mut Function,
    ) -> Result<(), Error> {
        let result_type_id =
            self.get_expression_type_id(&ir_module.types, &fun_info[expr_handle].ty)?;

        let id = match ir_function.expressions[expr_handle] {
            crate::Expression::Access { base, index } => {
                let base_is_var = match ir_function.expressions[base] {
                    crate::Expression::GlobalVariable(_) | crate::Expression::LocalVariable(_) => {
                        true
                    }
                    _ => self.cached.ids[base.index()] == 0,
                };
                if base_is_var {
                    0
                } else {
                    let index_id = self.cached[index];
                    let base_id = self.cached[base];
                    match *fun_info[base].ty.inner_with(&ir_module.types) {
                        crate::TypeInner::Vector { .. } => {
                            let id = self.id_gen.next();
                            block.body.push(Instruction::vector_extract_dynamic(
                                result_type_id,
                                id,
                                base_id,
                                index_id,
                            ));
                            id
                        }
                        crate::TypeInner::Array {
                            base: ty_element, ..
                        } => {
                            let (id, variable) = self.promote_access_expression_to_variable(
                                &ir_module.types,
                                result_type_id,
                                base_id,
                                &fun_info[base].ty,
                                index_id,
                                ty_element,
                                block,
                            )?;
                            function.internal_variables.push(variable);
                            id
                        }
                        ref other => {
                            log::error!("Unable to access {:?}", other);
                            return Err(Error::FeatureNotImplemented("access for type"));
                        }
                    }
                }
            }
            crate::Expression::AccessIndex { base, index } => {
                let base_is_var = match ir_function.expressions[base] {
                    crate::Expression::GlobalVariable(_) | crate::Expression::LocalVariable(_) => {
                        true
                    }
                    _ => self.cached.ids[base.index()] == 0,
                };
                if base_is_var {
                    0
                } else {
                    match *fun_info[base].ty.inner_with(&ir_module.types) {
                        crate::TypeInner::Vector { .. }
                        | crate::TypeInner::Matrix { .. }
                        | crate::TypeInner::Array { .. }
                        | crate::TypeInner::Struct { .. } => {
                            let id = self.id_gen.next();
                            let base_id = self.cached[base];
                            block.body.push(Instruction::composite_extract(
                                result_type_id,
                                id,
                                base_id,
                                &[index],
                            ));
                            id
                        }
                        ref other => {
                            log::error!("Unable to access index of {:?}", other);
                            return Err(Error::FeatureNotImplemented("access index for type"));
                        }
                    }
                }
            }
            crate::Expression::GlobalVariable(handle) => self.global_variables[handle.index()].id,
            crate::Expression::Constant(handle) => self.constant_ids[handle.index()],
            crate::Expression::Splat { size, value } => {
                let value_id = self.cached[value];
                self.temp_list.clear();
                self.temp_list.resize(size as usize, value_id);

                let id = self.id_gen.next();
                block.body.push(Instruction::composite_construct(
                    result_type_id,
                    id,
                    &self.temp_list,
                ));
                id
            }
            crate::Expression::Compose {
                ty: _,
                ref components,
            } => {
                self.temp_list.clear();
                for &component in components {
                    self.temp_list.push(self.cached[component]);
                }

                let id = self.id_gen.next();
                block.body.push(Instruction::composite_construct(
                    result_type_id,
                    id,
                    &self.temp_list,
                ));
                id
            }
            crate::Expression::Unary { op, expr } => {
                let id = self.id_gen.next();
                let expr_id = self.cached[expr];
                let expr_ty_inner = fun_info[expr].ty.inner_with(&ir_module.types);

                let spirv_op = match op {
                    crate::UnaryOperator::Negate => match expr_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Float) => spirv::Op::FNegate,
                        Some(crate::ScalarKind::Sint) => spirv::Op::SNegate,
                        Some(crate::ScalarKind::Bool) => spirv::Op::LogicalNot,
                        Some(crate::ScalarKind::Uint) | None => {
                            log::error!("Unable to negate {:?}", expr_ty_inner);
                            return Err(Error::FeatureNotImplemented("negation"));
                        }
                    },
                    crate::UnaryOperator::Not => spirv::Op::Not,
                };

                block
                    .body
                    .push(Instruction::unary(spirv_op, result_type_id, id, expr_id));
                id
            }
            crate::Expression::Binary { op, left, right } => {
                let id = self.id_gen.next();
                let left_id = self.cached[left];
                let right_id = self.cached[right];

                let left_ty_inner = fun_info[left].ty.inner_with(&ir_module.types);
                let right_ty_inner = fun_info[right].ty.inner_with(&ir_module.types);

                let left_dimension = get_dimension(left_ty_inner);
                let right_dimension = get_dimension(right_ty_inner);

                let mut preserve_order = true;

                let spirv_op = match op {
                    crate::BinaryOperator::Add => match *left_ty_inner {
                        crate::TypeInner::Scalar { kind, .. }
                        | crate::TypeInner::Vector { kind, .. } => match kind {
                            crate::ScalarKind::Float => spirv::Op::FAdd,
                            _ => spirv::Op::IAdd,
                        },
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Subtract => match *left_ty_inner {
                        crate::TypeInner::Scalar { kind, .. }
                        | crate::TypeInner::Vector { kind, .. } => match kind {
                            crate::ScalarKind::Float => spirv::Op::FSub,
                            _ => spirv::Op::ISub,
                        },
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Multiply => match (left_dimension, right_dimension) {
                        (Dimension::Scalar, Dimension::Vector { .. }) => {
                            preserve_order = false;
                            spirv::Op::VectorTimesScalar
                        }
                        (Dimension::Vector, Dimension::Scalar { .. }) => {
                            spirv::Op::VectorTimesScalar
                        }
                        (Dimension::Vector, Dimension::Matrix) => spirv::Op::VectorTimesMatrix,
                        (Dimension::Matrix, Dimension::Scalar { .. }) => {
                            spirv::Op::MatrixTimesScalar
                        }
                        (Dimension::Matrix, Dimension::Vector) => spirv::Op::MatrixTimesVector,
                        (Dimension::Matrix, Dimension::Matrix) => spirv::Op::MatrixTimesMatrix,
                        (Dimension::Vector, Dimension::Vector)
                        | (Dimension::Scalar, Dimension::Scalar)
                            if left_ty_inner.scalar_kind() == Some(crate::ScalarKind::Float) =>
                        {
                            spirv::Op::FMul
                        }
                        (Dimension::Vector, Dimension::Vector)
                        | (Dimension::Scalar, Dimension::Scalar) => spirv::Op::IMul,
                        other => unimplemented!("Mul {:?}", other),
                    },
                    crate::BinaryOperator::Divide => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SDiv,
                        Some(crate::ScalarKind::Uint) => spirv::Op::UDiv,
                        Some(crate::ScalarKind::Float) => spirv::Op::FDiv,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Modulo => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SMod,
                        Some(crate::ScalarKind::Uint) => spirv::Op::UMod,
                        Some(crate::ScalarKind::Float) => spirv::Op::FMod,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Equal => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) | Some(crate::ScalarKind::Uint) => {
                            spirv::Op::IEqual
                        }
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdEqual,
                        Some(crate::ScalarKind::Bool) => spirv::Op::LogicalEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::NotEqual => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) | Some(crate::ScalarKind::Uint) => {
                            spirv::Op::INotEqual
                        }
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdNotEqual,
                        Some(crate::ScalarKind::Bool) => spirv::Op::LogicalNotEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Less => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SLessThan,
                        Some(crate::ScalarKind::Uint) => spirv::Op::ULessThan,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdLessThan,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::LessEqual => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SLessThanEqual,
                        Some(crate::ScalarKind::Uint) => spirv::Op::ULessThanEqual,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdLessThanEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::Greater => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SGreaterThan,
                        Some(crate::ScalarKind::Uint) => spirv::Op::UGreaterThan,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdGreaterThan,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::GreaterEqual => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::SGreaterThanEqual,
                        Some(crate::ScalarKind::Uint) => spirv::Op::UGreaterThanEqual,
                        Some(crate::ScalarKind::Float) => spirv::Op::FOrdGreaterThanEqual,
                        _ => unimplemented!(),
                    },
                    crate::BinaryOperator::And => spirv::Op::BitwiseAnd,
                    crate::BinaryOperator::ExclusiveOr => spirv::Op::BitwiseXor,
                    crate::BinaryOperator::InclusiveOr => spirv::Op::BitwiseOr,
                    crate::BinaryOperator::LogicalAnd => spirv::Op::LogicalAnd,
                    crate::BinaryOperator::LogicalOr => spirv::Op::LogicalOr,
                    crate::BinaryOperator::ShiftLeft => spirv::Op::ShiftLeftLogical,
                    crate::BinaryOperator::ShiftRight => match left_ty_inner.scalar_kind() {
                        Some(crate::ScalarKind::Sint) => spirv::Op::ShiftRightArithmetic,
                        Some(crate::ScalarKind::Uint) => spirv::Op::ShiftRightLogical,
                        _ => unimplemented!(),
                    },
                };

                block.body.push(Instruction::binary(
                    spirv_op,
                    result_type_id,
                    id,
                    if preserve_order { left_id } else { right_id },
                    if preserve_order { right_id } else { left_id },
                ));
                id
            }
            crate::Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
            } => {
                use crate::MathFunction as Mf;
                enum MathOp {
                    Ext(spirv::GLOp),
                    Custom(Instruction),
                }

                let arg0_id = self.cached[arg];
                let arg_scalar_kind = fun_info[arg].ty.inner_with(&ir_module.types).scalar_kind();
                let arg1_id = match arg1 {
                    Some(handle) => self.cached[handle],
                    None => 0,
                };
                let arg2_id = match arg2 {
                    Some(handle) => self.cached[handle],
                    None => 0,
                };

                let id = self.id_gen.next();
                let math_op = match fun {
                    // comparison
                    Mf::Abs => {
                        match arg_scalar_kind {
                            Some(crate::ScalarKind::Float) => MathOp::Ext(spirv::GLOp::FAbs),
                            Some(crate::ScalarKind::Sint) => MathOp::Ext(spirv::GLOp::SAbs),
                            Some(crate::ScalarKind::Uint) => {
                                MathOp::Custom(Instruction::unary(
                                    spirv::Op::CopyObject, // do nothing
                                    result_type_id,
                                    id,
                                    arg0_id,
                                ))
                            }
                            other => unimplemented!("Unexpected abs({:?})", other),
                        }
                    }
                    Mf::Min => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FMin,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SMin,
                        Some(crate::ScalarKind::Uint) => spirv::GLOp::UMin,
                        other => unimplemented!("Unexpected min({:?})", other),
                    }),
                    Mf::Max => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FMax,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SMax,
                        Some(crate::ScalarKind::Uint) => spirv::GLOp::UMax,
                        other => unimplemented!("Unexpected max({:?})", other),
                    }),
                    Mf::Clamp => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FClamp,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SClamp,
                        Some(crate::ScalarKind::Uint) => spirv::GLOp::UClamp,
                        other => unimplemented!("Unexpected max({:?})", other),
                    }),
                    // trigonometry
                    Mf::Sin => MathOp::Ext(spirv::GLOp::Sin),
                    Mf::Sinh => MathOp::Ext(spirv::GLOp::Sinh),
                    Mf::Asin => MathOp::Ext(spirv::GLOp::Asin),
                    Mf::Cos => MathOp::Ext(spirv::GLOp::Cos),
                    Mf::Cosh => MathOp::Ext(spirv::GLOp::Cosh),
                    Mf::Acos => MathOp::Ext(spirv::GLOp::Acos),
                    Mf::Tan => MathOp::Ext(spirv::GLOp::Tan),
                    Mf::Tanh => MathOp::Ext(spirv::GLOp::Tanh),
                    Mf::Atan => MathOp::Ext(spirv::GLOp::Atan),
                    Mf::Atan2 => MathOp::Ext(spirv::GLOp::Atan2),
                    // decomposition
                    Mf::Ceil => MathOp::Ext(spirv::GLOp::Ceil),
                    Mf::Round => MathOp::Ext(spirv::GLOp::Round),
                    Mf::Floor => MathOp::Ext(spirv::GLOp::Floor),
                    Mf::Fract => MathOp::Ext(spirv::GLOp::Fract),
                    Mf::Trunc => MathOp::Ext(spirv::GLOp::Trunc),
                    Mf::Modf => MathOp::Ext(spirv::GLOp::Modf),
                    Mf::Frexp => MathOp::Ext(spirv::GLOp::Frexp),
                    Mf::Ldexp => MathOp::Ext(spirv::GLOp::Ldexp),
                    // geometry
                    Mf::Dot => MathOp::Custom(Instruction::binary(
                        spirv::Op::Dot,
                        result_type_id,
                        id,
                        arg0_id,
                        arg1_id,
                    )),
                    Mf::Cross => MathOp::Ext(spirv::GLOp::Cross),
                    Mf::Distance => MathOp::Ext(spirv::GLOp::Distance),
                    Mf::Length => MathOp::Ext(spirv::GLOp::Length),
                    Mf::Normalize => MathOp::Ext(spirv::GLOp::Normalize),
                    Mf::FaceForward => MathOp::Ext(spirv::GLOp::FaceForward),
                    Mf::Reflect => MathOp::Ext(spirv::GLOp::Reflect),
                    // exponent
                    Mf::Exp => MathOp::Ext(spirv::GLOp::Exp),
                    Mf::Exp2 => MathOp::Ext(spirv::GLOp::Exp2),
                    Mf::Log => MathOp::Ext(spirv::GLOp::Log),
                    Mf::Log2 => MathOp::Ext(spirv::GLOp::Log2),
                    Mf::Pow => MathOp::Ext(spirv::GLOp::Pow),
                    // computational
                    Mf::Sign => MathOp::Ext(match arg_scalar_kind {
                        Some(crate::ScalarKind::Float) => spirv::GLOp::FSign,
                        Some(crate::ScalarKind::Sint) => spirv::GLOp::SSign,
                        other => unimplemented!("Unexpected sign({:?})", other),
                    }),
                    Mf::Fma => MathOp::Ext(spirv::GLOp::Fma),
                    Mf::Mix => MathOp::Ext(spirv::GLOp::FMix),
                    Mf::Step => MathOp::Ext(spirv::GLOp::Step),
                    Mf::SmoothStep => MathOp::Ext(spirv::GLOp::SmoothStep),
                    Mf::Sqrt => MathOp::Ext(spirv::GLOp::Sqrt),
                    Mf::InverseSqrt => MathOp::Ext(spirv::GLOp::InverseSqrt),
                    Mf::Inverse => MathOp::Ext(spirv::GLOp::MatrixInverse),
                    Mf::Transpose => MathOp::Custom(Instruction::unary(
                        spirv::Op::Transpose,
                        result_type_id,
                        id,
                        arg0_id,
                    )),
                    Mf::Determinant => MathOp::Ext(spirv::GLOp::Determinant),
                    Mf::Outer | Mf::ReverseBits | Mf::CountOneBits => {
                        log::error!("unimplemented math function {:?}", fun);
                        return Err(Error::FeatureNotImplemented("math function"));
                    }
                };

                block.body.push(match math_op {
                    MathOp::Ext(op) => Instruction::ext_inst(
                        self.gl450_ext_inst_id,
                        op,
                        result_type_id,
                        id,
                        &[arg0_id, arg1_id, arg2_id][..fun.argument_count()],
                    ),
                    MathOp::Custom(inst) => inst,
                });
                id
            }
            crate::Expression::LocalVariable(variable) => function.variables[&variable].id,
            crate::Expression::Load { pointer } => {
                let (pointer_id, _) = self.write_expression_pointer(
                    ir_module,
                    ir_function,
                    fun_info,
                    pointer,
                    block,
                    function,
                )?;

                let id = self.id_gen.next();
                block
                    .body
                    .push(Instruction::load(result_type_id, id, pointer_id, None));
                id
            }
            crate::Expression::FunctionArgument(index) => match function.entry_point_context {
                Some(ref context) => context.argument_ids[index as usize],
                None => function.parameters[index as usize].result_id.unwrap(),
            },
            crate::Expression::Call(_function) => self.lookup_function_call[&expr_handle],
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                let expr_id = self.cached[expr];
                let expr_kind = fun_info[expr]
                    .ty
                    .inner_with(&ir_module.types)
                    .scalar_kind()
                    .unwrap();

                let op = match (expr_kind, kind) {
                    _ if !convert => spirv::Op::Bitcast,
                    (crate::ScalarKind::Float, crate::ScalarKind::Uint) => spirv::Op::ConvertFToU,
                    (crate::ScalarKind::Float, crate::ScalarKind::Sint) => spirv::Op::ConvertFToS,
                    (crate::ScalarKind::Sint, crate::ScalarKind::Float) => spirv::Op::ConvertSToF,
                    (crate::ScalarKind::Uint, crate::ScalarKind::Float) => spirv::Op::ConvertUToF,
                    // We assume it's either an identity cast, or int-uint.
                    _ => spirv::Op::Bitcast,
                };

                let id = self.id_gen.next();
                let instruction = Instruction::unary(op, result_type_id, id, expr_id);
                block.body.push(instruction);
                id
            }
            crate::Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                index,
            } => {
                let image_id = self.get_expression_global(ir_function, image);
                let coordinate_id = self.write_texture_coordinates(
                    ir_module,
                    fun_info,
                    coordinate,
                    array_index,
                    block,
                )?;

                let id = self.id_gen.next();

                let image_ty = fun_info[image].ty.inner_with(&ir_module.types);
                let mut instruction = match *image_ty {
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Storage { .. },
                        ..
                    } => Instruction::image_read(result_type_id, id, image_id, coordinate_id),
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Depth,
                        ..
                    } => {
                        // Vulkan doesn't know about our `Depth` class, and it returns `vec4<f32>`,
                        // so we need to grab the first component out of it.
                        let load_result_type_id = self.get_type_id(
                            &ir_module.types,
                            LookupType::Local(LocalType::Value {
                                vector_size: Some(crate::VectorSize::Quad),
                                kind: crate::ScalarKind::Float,
                                width: 4,
                                pointer_class: None,
                            }),
                        )?;
                        Instruction::image_fetch(load_result_type_id, id, image_id, coordinate_id)
                    }
                    _ => Instruction::image_fetch(result_type_id, id, image_id, coordinate_id),
                };

                if let Some(index) = index {
                    let index_id = self.cached[index];
                    let image_ops = match *fun_info[image].ty.inner_with(&ir_module.types) {
                        crate::TypeInner::Image {
                            class: crate::ImageClass::Sampled { multi: true, .. },
                            ..
                        } => spirv::ImageOperands::SAMPLE,
                        _ => spirv::ImageOperands::LOD,
                    };
                    instruction.add_operand(image_ops.bits());
                    instruction.add_operand(index_id);
                }

                if instruction.type_id != Some(result_type_id) {
                    let sub_id = self.id_gen.next();
                    let index_id = self.get_index_constant(0, &ir_module.types)?;
                    let sub_instruction =
                        Instruction::vector_extract_dynamic(result_type_id, sub_id, id, index_id);
                    block.body.push(sub_instruction);
                    sub_id
                } else {
                    id
                }
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                use super::instructions::SampleLod;
                // image
                let image_id = self.get_expression_global(ir_function, image);
                let image_type = fun_info[image].ty.handle().unwrap();
                // Vulkan doesn't know about our `Depth` class, and it returns `vec4<f32>`,
                // so we need to grab the first component out of it.
                let needs_sub_access = match ir_module.types[image_type].inner {
                    crate::TypeInner::Image {
                        class: crate::ImageClass::Depth,
                        ..
                    } => depth_ref.is_none(),
                    _ => false,
                };
                let sample_result_type_id = if needs_sub_access {
                    self.get_type_id(
                        &ir_module.types,
                        LookupType::Local(LocalType::Value {
                            vector_size: Some(crate::VectorSize::Quad),
                            kind: crate::ScalarKind::Float,
                            width: 4,
                            pointer_class: None,
                        }),
                    )?
                } else {
                    result_type_id
                };

                // OpTypeSampledImage
                let sampled_image_type_id = self.get_type_id(
                    &ir_module.types,
                    LookupType::Local(LocalType::SampledImage { image_type }),
                )?;

                let sampler_id = self.get_expression_global(ir_function, sampler);
                let coordinate_id = self.write_texture_coordinates(
                    ir_module,
                    fun_info,
                    coordinate,
                    array_index,
                    block,
                )?;

                let sampled_image_id = self.id_gen.next();
                block.body.push(Instruction::sampled_image(
                    sampled_image_type_id,
                    sampled_image_id,
                    image_id,
                    sampler_id,
                ));
                let id = self.id_gen.next();

                let depth_id = depth_ref.map(|handle| self.cached[handle]);

                let mut main_instruction = match level {
                    crate::SampleLevel::Zero => {
                        let mut inst = Instruction::image_sample(
                            sample_result_type_id,
                            id,
                            SampleLod::Explicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        //TODO: cache this!
                        let zero_id = self.id_gen.next();
                        self.write_constant_scalar(
                            zero_id,
                            &crate::ScalarValue::Float(0.0),
                            4,
                            None,
                            &ir_module.types,
                        )?;
                        inst.add_operand(spirv::ImageOperands::LOD.bits());
                        inst.add_operand(zero_id);

                        inst
                    }
                    crate::SampleLevel::Auto => Instruction::image_sample(
                        sample_result_type_id,
                        id,
                        SampleLod::Implicit,
                        sampled_image_id,
                        coordinate_id,
                        depth_id,
                    ),
                    crate::SampleLevel::Exact(lod_handle) => {
                        let mut inst = Instruction::image_sample(
                            sample_result_type_id,
                            id,
                            SampleLod::Explicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        let lod_id = self.cached[lod_handle];
                        inst.add_operand(spirv::ImageOperands::LOD.bits());
                        inst.add_operand(lod_id);

                        inst
                    }
                    crate::SampleLevel::Bias(bias_handle) => {
                        let mut inst = Instruction::image_sample(
                            sample_result_type_id,
                            id,
                            SampleLod::Implicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        let bias_id = self.cached[bias_handle];
                        inst.add_operand(spirv::ImageOperands::BIAS.bits());
                        inst.add_operand(bias_id);

                        inst
                    }
                    crate::SampleLevel::Gradient { x, y } => {
                        let mut inst = Instruction::image_sample(
                            sample_result_type_id,
                            id,
                            SampleLod::Explicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        let x_id = self.cached[x];
                        let y_id = self.cached[y];
                        inst.add_operand(spirv::ImageOperands::GRAD.bits());
                        inst.add_operand(x_id);
                        inst.add_operand(y_id);

                        inst
                    }
                };

                if let Some(offset_const) = offset {
                    let offset_id = self.constant_ids[offset_const.index()];
                    main_instruction.add_operand(spirv::ImageOperands::CONST_OFFSET.bits());
                    main_instruction.add_operand(offset_id);
                }

                block.body.push(main_instruction);

                if needs_sub_access {
                    let sub_id = self.id_gen.next();
                    let index_id = self.get_index_constant(0, &ir_module.types)?;
                    let sub_instruction =
                        Instruction::vector_extract_dynamic(result_type_id, sub_id, id, index_id);
                    block.body.push(sub_instruction);
                    sub_id
                } else {
                    id
                }
            }
            crate::Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let id = self.id_gen.next();
                let condition_id = self.cached[condition];
                let accept_id = self.cached[accept];
                let reject_id = self.cached[reject];

                let instruction =
                    Instruction::select(result_type_id, id, condition_id, accept_id, reject_id);
                block.body.push(instruction);
                id
            }
            crate::Expression::Derivative { axis, expr } => {
                use crate::DerivativeAxis;

                let id = self.id_gen.next();
                let expr_id = self.cached[expr];
                block.body.push(match axis {
                    DerivativeAxis::X => Instruction::derive_x(result_type_id, id, expr_id),
                    DerivativeAxis::Y => Instruction::derive_y(result_type_id, id, expr_id),
                    DerivativeAxis::Width => Instruction::derive_width(result_type_id, id, expr_id),
                });
                id
            }
            crate::Expression::ImageQuery { .. }
            | crate::Expression::Relational { .. }
            | crate::Expression::ArrayLength(_) => {
                log::error!("unimplemented {:?}", ir_function.expressions[expr_handle]);
                return Err(Error::FeatureNotImplemented("expression"));
            }
        };

        self.cached[expr_handle] = id;
        Ok(())
    }

    /// Write a left-hand-side expression, returning an `id` of the pointer.
    fn write_expression_pointer<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        ir_function: &crate::Function,
        fun_info: &FunctionInfo,
        mut expr_handle: Handle<crate::Expression>,
        block: &mut Block,
        function: &mut Function,
    ) -> Result<(Word, spirv::StorageClass), Error> {
        let result_lookup_ty = match fun_info[expr_handle].ty {
            TypeResolution::Handle(ty_handle) => LookupType::Handle(ty_handle),
            TypeResolution::Value(ref inner) => {
                LookupType::Local(self.physical_layout.make_local(inner).unwrap())
            }
        };
        let result_type_id = self.get_type_id(&ir_module.types, result_lookup_ty)?;

        self.temp_list.clear();
        let (root_id, class) = loop {
            expr_handle = match ir_function.expressions[expr_handle] {
                crate::Expression::Access { base, index } => {
                    let index_id = self.cached[index];
                    self.temp_list.push(index_id);
                    base
                }
                crate::Expression::AccessIndex { base, index } => {
                    let const_id = self.get_index_constant(index, &ir_module.types)?;
                    self.temp_list.push(const_id);
                    base
                }
                crate::Expression::GlobalVariable(handle) => {
                    let gv = &self.global_variables[handle.index()];
                    break (gv.id, gv.class);
                }
                crate::Expression::LocalVariable(variable) => {
                    let local_var = &function.variables[&variable];
                    break (local_var.id, spirv::StorageClass::Function);
                }
                ref other => unimplemented!("Unexpected pointer expression {:?}", other),
            }
        };

        let id = if self.temp_list.is_empty() {
            root_id
        } else {
            self.temp_list.reverse();
            let id = self.id_gen.next();
            block.body.push(Instruction::access_chain(
                result_type_id,
                id,
                root_id,
                &self.temp_list,
            ));
            id
        };
        Ok((id, class))
    }

    fn get_expression_global(
        &self,
        ir_function: &crate::Function,
        expr_handle: Handle<crate::Expression>,
    ) -> Word {
        match ir_function.expressions[expr_handle] {
            crate::Expression::GlobalVariable(handle) => {
                let id = self.global_variables[handle.index()].handle_id;
                if id == 0 {
                    unreachable!("Global variable {:?} doesn't have a handle ID", handle);
                }
                id
            }
            ref other => unreachable!("Unexpected global expression {:?}", other),
        }
    }

    fn write_entry_point_return(
        &mut self,
        value_id: Word,
        ir_result: &crate::FunctionResult,
        type_arena: &Arena<crate::Type>,
        result_members: &[ResultMember],
        body: &mut Vec<Instruction>,
    ) -> Result<(), Error> {
        for (index, res_member) in result_members.iter().enumerate() {
            let member_value_id = match ir_result.binding {
                Some(_) => value_id,
                None => {
                    let member_value_id = self.id_gen.next();
                    body.push(Instruction::composite_extract(
                        res_member.type_id,
                        member_value_id,
                        value_id,
                        &[index as u32],
                    ));
                    member_value_id
                }
            };

            body.push(Instruction::store(res_member.id, member_value_id, None));

            // Flip Y coordinate to adjust for coordinate space difference
            // between SPIR-V and our IR.
            if self.flags.contains(WriterFlags::ADJUST_COORDINATE_SPACE)
                && res_member.built_in == Some(crate::BuiltIn::Position)
            {
                let access_id = self.id_gen.next();
                let float_ptr_type_id = self.get_type_id(
                    type_arena,
                    LookupType::Local(LocalType::Value {
                        vector_size: None,
                        kind: crate::ScalarKind::Float,
                        width: 4,
                        pointer_class: Some(spirv::StorageClass::Output),
                    }),
                )?;
                let index_y_id = self.get_index_constant(1, type_arena)?;
                body.push(Instruction::access_chain(
                    float_ptr_type_id,
                    access_id,
                    res_member.id,
                    &[index_y_id],
                ));

                let load_id = self.id_gen.next();
                let float_type_id = self.get_type_id(
                    type_arena,
                    LookupType::Local(LocalType::Value {
                        vector_size: None,
                        kind: crate::ScalarKind::Float,
                        width: 4,
                        pointer_class: None,
                    }),
                )?;
                body.push(Instruction::load(float_type_id, load_id, access_id, None));

                let neg_id = self.id_gen.next();
                body.push(Instruction::unary(
                    spirv::Op::FNegate,
                    float_type_id,
                    neg_id,
                    load_id,
                ));
                body.push(Instruction::store(access_id, neg_id, None));
            }
        }
        Ok(())
    }

    //TODO: put most of these into a `BlockContext` structure!
    #[allow(clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        label_id: Word,
        statements: &[crate::Statement],
        ir_module: &crate::Module,
        ir_function: &crate::Function,
        fun_info: &FunctionInfo,
        function: &mut Function,
        exit_id: Option<Word>,
        loop_context: LoopContext,
    ) -> Result<(), Error> {
        let mut block = Block::new(label_id);

        for statement in statements {
            if block.termination.is_some() {
                unimplemented!("No statements are expected after block termination");
            }
            match *statement {
                crate::Statement::Emit(ref range) => {
                    for handle in range.clone() {
                        self.cache_expression_value(
                            ir_module,
                            ir_function,
                            fun_info,
                            handle,
                            &mut block,
                            function,
                        )?;
                    }
                }
                crate::Statement::Block(ref block_statements) => {
                    let scope_id = self.id_gen.next();
                    function.consume(block, Instruction::branch(scope_id));

                    let merge_id = self.id_gen.next();
                    self.write_block(
                        scope_id,
                        block_statements,
                        ir_module,
                        ir_function,
                        fun_info,
                        function,
                        Some(merge_id),
                        loop_context,
                    )?;

                    block = Block::new(merge_id);
                }
                crate::Statement::If {
                    condition,
                    ref accept,
                    ref reject,
                } => {
                    let condition_id = self.cached[condition];

                    let merge_id = self.id_gen.next();
                    block.body.push(Instruction::selection_merge(
                        merge_id,
                        spirv::SelectionControl::NONE,
                    ));

                    let accept_id = if accept.is_empty() {
                        None
                    } else {
                        Some(self.id_gen.next())
                    };
                    let reject_id = if reject.is_empty() {
                        None
                    } else {
                        Some(self.id_gen.next())
                    };

                    function.consume(
                        block,
                        Instruction::branch_conditional(
                            condition_id,
                            accept_id.unwrap_or(merge_id),
                            reject_id.unwrap_or(merge_id),
                        ),
                    );

                    if let Some(block_id) = accept_id {
                        self.write_block(
                            block_id,
                            accept,
                            ir_module,
                            ir_function,
                            fun_info,
                            function,
                            Some(merge_id),
                            loop_context,
                        )?;
                    }
                    if let Some(block_id) = reject_id {
                        self.write_block(
                            block_id,
                            reject,
                            ir_module,
                            ir_function,
                            fun_info,
                            function,
                            Some(merge_id),
                            loop_context,
                        )?;
                    }

                    block = Block::new(merge_id);
                }
                crate::Statement::Switch {
                    selector,
                    ref cases,
                    ref default,
                } => {
                    let selector_id = self.cached[selector];

                    let merge_id = self.id_gen.next();
                    block.body.push(Instruction::selection_merge(
                        merge_id,
                        spirv::SelectionControl::NONE,
                    ));

                    let default_id = self.id_gen.next();
                    let raw_cases = cases
                        .iter()
                        .map(|c| super::instructions::Case {
                            value: c.value as Word,
                            label_id: self.id_gen.next(),
                        })
                        .collect::<Vec<_>>();

                    function.consume(
                        block,
                        Instruction::switch(selector_id, default_id, &raw_cases),
                    );

                    for (i, (case, raw_case)) in cases.iter().zip(raw_cases.iter()).enumerate() {
                        let case_finish_id = if case.fall_through {
                            match raw_cases.get(i + 1) {
                                Some(rc) => rc.label_id,
                                None => default_id,
                            }
                        } else {
                            merge_id
                        };
                        self.write_block(
                            raw_case.label_id,
                            &case.body,
                            ir_module,
                            ir_function,
                            fun_info,
                            function,
                            Some(case_finish_id),
                            LoopContext::default(),
                        )?;
                    }

                    self.write_block(
                        default_id,
                        default,
                        ir_module,
                        ir_function,
                        fun_info,
                        function,
                        Some(merge_id),
                        LoopContext::default(),
                    )?;

                    block = Block::new(merge_id);
                }
                crate::Statement::Loop {
                    ref body,
                    ref continuing,
                } => {
                    let preamble_id = self.id_gen.next();
                    function.consume(block, Instruction::branch(preamble_id));

                    let merge_id = self.id_gen.next();
                    let body_id = self.id_gen.next();
                    let continuing_id = self.id_gen.next();

                    // SPIR-V requires the continuing to the `OpLoopMerge`,
                    // so we have to start a new block with it.
                    block = Block::new(preamble_id);
                    block.body.push(Instruction::loop_merge(
                        merge_id,
                        continuing_id,
                        spirv::SelectionControl::NONE,
                    ));
                    function.consume(block, Instruction::branch(body_id));

                    self.write_block(
                        body_id,
                        body,
                        ir_module,
                        ir_function,
                        fun_info,
                        function,
                        Some(continuing_id),
                        LoopContext {
                            continuing_id: Some(continuing_id),
                            break_id: Some(merge_id),
                        },
                    )?;

                    self.write_block(
                        continuing_id,
                        continuing,
                        ir_module,
                        ir_function,
                        fun_info,
                        function,
                        Some(preamble_id),
                        LoopContext {
                            continuing_id: None,
                            break_id: Some(merge_id),
                        },
                    )?;

                    block = Block::new(merge_id);
                }
                crate::Statement::Break => {
                    block.termination = Some(Instruction::branch(loop_context.break_id.unwrap()));
                }
                crate::Statement::Continue => {
                    block.termination =
                        Some(Instruction::branch(loop_context.continuing_id.unwrap()));
                }
                crate::Statement::Return { value: Some(value) } => {
                    let value_id = self.cached[value];
                    let instruction = match function.entry_point_context {
                        // If this is an entry point, and we need to return anything,
                        // let's instead store the output variables and return `void`.
                        Some(ref context) => {
                            self.write_entry_point_return(
                                value_id,
                                ir_function.result.as_ref().unwrap(),
                                &ir_module.types,
                                &context.results,
                                &mut block.body,
                            )?;
                            Instruction::return_void()
                        }
                        None => Instruction::return_value(value_id),
                    };
                    block.termination = Some(instruction);
                }
                crate::Statement::Return { value: None } => {
                    block.termination = Some(Instruction::return_void());
                }
                crate::Statement::Kill => {
                    block.termination = Some(Instruction::kill());
                }
                crate::Statement::Store { pointer, value } => {
                    let (pointer_id, _) = self.write_expression_pointer(
                        ir_module,
                        ir_function,
                        fun_info,
                        pointer,
                        &mut block,
                        function,
                    )?;
                    let value_id = self.cached[value];

                    block
                        .body
                        .push(Instruction::store(pointer_id, value_id, None));
                }
                crate::Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    let image_id = self.get_expression_global(ir_function, image);
                    let coordinate_id = self.write_texture_coordinates(
                        ir_module,
                        fun_info,
                        coordinate,
                        array_index,
                        &mut block,
                    )?;
                    let value_id = self.cached[value];

                    block
                        .body
                        .push(Instruction::image_write(image_id, coordinate_id, value_id));
                }
                crate::Statement::Call {
                    function: local_function,
                    ref arguments,
                    result,
                } => {
                    let id = self.id_gen.next();
                    self.temp_list.clear();
                    for &argument in arguments {
                        self.temp_list.push(self.cached[argument]);
                    }

                    let type_id = match result {
                        Some(expr) => {
                            self.cached[expr] = id;
                            self.lookup_function_call.insert(expr, id);
                            let ty_handle = ir_module.functions[local_function]
                                .result
                                .as_ref()
                                .unwrap()
                                .ty;
                            self.get_type_id(&ir_module.types, LookupType::Handle(ty_handle))?
                        }
                        None => self.void_type,
                    };

                    block.body.push(Instruction::function_call(
                        type_id,
                        id,
                        self.lookup_function[&local_function],
                        &self.temp_list,
                    ));
                }
            }
        }

        if block.termination.is_none() {
            block.termination = Some(match exit_id {
                Some(id) => Instruction::branch(id),
                // This can happen if the last branch had all the paths
                // leading out of the graph (i.e. returning).
                // So it doesn't matter what we do here, but it has to be valid.
                None => Instruction::branch(label_id),
            });
        }

        function.blocks.push(block);
        Ok(())
    }

    fn write_physical_layout(&mut self) {
        self.physical_layout.bound = self.id_gen.0 + 1;
    }

    fn write_logical_layout(
        &mut self,
        ir_module: &crate::Module,
        mod_info: &ModuleInfo,
    ) -> Result<(), Error> {
        let has_storage_buffers = ir_module
            .global_variables
            .iter()
            .any(|(_, var)| var.class == crate::StorageClass::Storage);
        if self.physical_layout.version < 0x10300 && has_storage_buffers {
            // enable the storage buffer class on < SPV-1.3
            Instruction::extension("SPV_KHR_storage_buffer_storage_class")
                .to_words(&mut self.logical_layout.extensions);
        }
        Instruction::type_void(self.void_type).to_words(&mut self.logical_layout.declarations);
        Instruction::ext_inst_import(self.gl450_ext_inst_id, "GLSL.std.450")
            .to_words(&mut self.logical_layout.ext_inst_imports);

        if self.flags.contains(WriterFlags::DEBUG) {
            self.debugs
                .push(Instruction::source(spirv::SourceLanguage::GLSL, 450));
        }

        self.constant_ids.clear();
        self.constant_ids.resize(ir_module.constants.len(), 0);
        // first, output all the scalar constants
        for (handle, constant) in ir_module.constants.iter() {
            match constant.inner {
                crate::ConstantInner::Composite { .. } => continue,
                crate::ConstantInner::Scalar { width, ref value } => {
                    let id = self.id_gen.next();
                    self.constant_ids[handle.index()] = id;
                    self.write_constant_scalar(
                        id,
                        value,
                        width,
                        constant.name.as_ref(),
                        &ir_module.types,
                    )?;
                }
            }
        }

        // then all types, some of them may rely on constants and struct type set
        for (handle, _) in ir_module.types.iter() {
            self.write_type_declaration_arena(&ir_module.types, handle)?;
        }

        // the all the composite constants, they rely on types
        for (handle, constant) in ir_module.constants.iter() {
            match constant.inner {
                crate::ConstantInner::Scalar { .. } => continue,
                crate::ConstantInner::Composite { ty, ref components } => {
                    let id = self.id_gen.next();
                    self.constant_ids[handle.index()] = id;
                    if self.flags.contains(WriterFlags::DEBUG) {
                        if let Some(ref name) = constant.name {
                            self.debugs.push(Instruction::name(id, name));
                        }
                    }
                    self.write_constant_composite(id, ty, components, &ir_module.types)?;
                }
            }
        }
        debug_assert_eq!(self.constant_ids.iter().position(|&id| id == 0), None);

        // now write all globals
        self.global_variables.clear();
        for (_, var) in ir_module.global_variables.iter() {
            let (instruction, id, class) = self.write_global_variable(ir_module, var)?;
            instruction.to_words(&mut self.logical_layout.declarations);
            self.global_variables.push(GlobalVariable {
                id,
                handle_id: 0,
                class,
            });
        }

        // all functions
        for (handle, ir_function) in ir_module.functions.iter() {
            let info = &mod_info[handle];
            let id = self.write_function(ir_function, info, ir_module, None)?;
            self.lookup_function.insert(handle, id);
        }

        // and entry points
        for (ep_index, ir_ep) in ir_module.entry_points.iter().enumerate() {
            let info = mod_info.get_entry_point(ep_index);
            let ep_instruction = self.write_entry_point(ir_ep, info, ir_module)?;
            ep_instruction.to_words(&mut self.logical_layout.entry_points);
        }

        for capability in self.capabilities.iter() {
            Instruction::capability(*capability).to_words(&mut self.logical_layout.capabilities);
        }

        let addressing_model = spirv::AddressingModel::Logical;
        let memory_model = spirv::MemoryModel::GLSL450;
        self.check(addressing_model.required_capabilities())?;
        self.check(memory_model.required_capabilities())?;

        Instruction::memory_model(addressing_model, memory_model)
            .to_words(&mut self.logical_layout.memory_model);

        if self.flags.contains(WriterFlags::DEBUG) {
            for debug in self.debugs.iter() {
                debug.to_words(&mut self.logical_layout.debugs);
            }
        }

        for annotation in self.annotations.iter() {
            annotation.to_words(&mut self.logical_layout.annotations);
        }

        Ok(())
    }

    pub fn write(
        &mut self,
        ir_module: &crate::Module,
        info: &ModuleInfo,
        words: &mut Vec<Word>,
    ) -> Result<(), Error> {
        self.lookup_function.clear();
        self.lookup_function_type.clear();
        self.lookup_function_call.clear();

        self.write_logical_layout(ir_module, info)?;
        self.write_physical_layout();

        self.physical_layout.in_words(words);
        self.logical_layout.in_words(words);
        Ok(())
    }
}

#[test]
fn test_write_physical_layout() {
    let mut writer = Writer::new(&Options::default()).unwrap();
    assert_eq!(writer.physical_layout.bound, 0);
    writer.write_physical_layout();
    assert_eq!(writer.physical_layout.bound, 3);
}
