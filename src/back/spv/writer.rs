/*! Standard Portable Intermediate Representation (SPIR-V) backend !*/
use super::{Instruction, LogicalLayout, PhysicalLayout, WriterFlags};
use crate::proc::{Layouter, ResolveContext, ResolveError, Typifier};
use spirv::Word;
use std::collections::hash_map::Entry;
use thiserror::Error;

const BITS_PER_BYTE: crate::Bytes = 8;

#[derive(Clone, Debug, Error)]
pub enum Error {
    #[error("one of the required capabilities {0:?} is missing")]
    MissingCapabilities(Vec<spirv::Capability>),
    #[error("unimplemented {0:}")]
    FeatureNotImplemented(&'static str),
    #[error(transparent)]
    Resolve(#[from] ResolveError),
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

enum RawExpression {
    Value(Word),
    Pointer(Word, spirv::StorageClass),
}

#[derive(Default)]
struct Function {
    signature: Option<Instruction>,
    parameters: Vec<Instruction>,
    variables: crate::FastHashMap<crate::Handle<crate::LocalVariable>, LocalVariable>,
    blocks: Vec<Block>,
}

impl Function {
    fn to_words(&self, sink: &mut impl Extend<Word>) {
        self.signature.as_ref().unwrap().to_words(sink);
        for instruction in self.parameters.iter() {
            instruction.to_words(sink);
        }
        for (index, block) in self.blocks.iter().enumerate() {
            super::instructions::instruction_label(block.label_id).to_words(sink);
            if index == 0 {
                for local_var in self.variables.values() {
                    local_var.instruction.to_words(sink);
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
    Scalar {
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    Vector {
        size: crate::VectorSize,
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    Matrix {
        columns: crate::VectorSize,
        rows: crate::VectorSize,
        width: crate::Bytes,
    },
    Pointer {
        base: crate::Handle<crate::Type>,
        class: crate::StorageClass,
    },
    SampledImage {
        image_type: crate::Handle<crate::Type>,
    },
}

impl LocalType {
    fn from_inner(inner: &crate::TypeInner) -> Option<Self> {
        Some(match *inner {
            crate::TypeInner::Scalar { kind, width } => LocalType::Scalar { kind, width },
            crate::TypeInner::Vector { size, kind, width } => {
                LocalType::Vector { size, kind, width }
            }
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => LocalType::Matrix {
                columns,
                rows,
                width,
            },
            crate::TypeInner::Pointer { base, class } => LocalType::Pointer { base, class },
            _ => return None,
        })
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LookupType {
    Handle(crate::Handle<crate::Type>),
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

pub struct Writer {
    physical_layout: PhysicalLayout,
    logical_layout: LogicalLayout,
    id_count: u32,
    capabilities: crate::FastHashSet<spirv::Capability>,
    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    flags: WriterFlags,
    void_type: u32,
    lookup_type: crate::FastHashMap<LookupType, Word>,
    lookup_function: crate::FastHashMap<crate::Handle<crate::Function>, Word>,
    lookup_function_type: crate::FastHashMap<LookupFunctionType, Word>,
    lookup_constant: crate::FastHashMap<crate::Handle<crate::Constant>, Word>,
    lookup_global_variable:
        crate::FastHashMap<crate::Handle<crate::GlobalVariable>, (Word, spirv::StorageClass)>,
    // TODO: this is a type property that depends on the global variable that uses it
    // so it may require us to duplicate the type!
    struct_type_handles: crate::FastHashMap<crate::Handle<crate::Type>, crate::StorageAccess>,
    gl450_ext_inst_id: Word,
    layouter: Layouter,
    typifier: Typifier,
}

// type alias, for success return of write_expression
type ExpressionId = Word;
type PointerExpressionId = (Word, spirv::StorageClass);

impl Writer {
    pub fn new(
        header: &crate::Header,
        flags: WriterFlags,
        capabilities: crate::FastHashSet<spirv::Capability>,
    ) -> Self {
        let gl450_ext_inst_id = 1;
        let void_type = 2;
        Writer {
            physical_layout: PhysicalLayout::new(header),
            logical_layout: LogicalLayout::default(),
            id_count: 2, // see `gl450_ext_inst_id` and `void_type`
            capabilities,
            debugs: vec![],
            annotations: vec![],
            flags,
            void_type,
            lookup_type: crate::FastHashMap::default(),
            lookup_function: crate::FastHashMap::default(),
            lookup_function_type: crate::FastHashMap::default(),
            lookup_constant: crate::FastHashMap::default(),
            lookup_global_variable: crate::FastHashMap::default(),
            struct_type_handles: crate::FastHashMap::default(),
            gl450_ext_inst_id,
            layouter: Layouter::default(),
            typifier: Typifier::new(),
        }
    }

    fn generate_id(&mut self) -> Word {
        self.id_count += 1;
        self.id_count
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
        arena: &crate::Arena<crate::Type>,
        lookup_ty: LookupType,
    ) -> Result<Word, Error> {
        if let Entry::Occupied(e) = self.lookup_type.entry(lookup_ty) {
            Ok(*e.get())
        } else {
            match lookup_ty {
                LookupType::Handle(handle) => match arena[handle].inner {
                    crate::TypeInner::Scalar { kind, width } => self
                        .get_type_id(arena, LookupType::Local(LocalType::Scalar { kind, width })),
                    crate::TypeInner::Vector { size, kind, width } => self.get_type_id(
                        arena,
                        LookupType::Local(LocalType::Vector { size, kind, width }),
                    ),
                    crate::TypeInner::Matrix {
                        columns,
                        rows,
                        width,
                    } => self.get_type_id(
                        arena,
                        LookupType::Local(LocalType::Matrix {
                            columns,
                            rows,
                            width,
                        }),
                    ),
                    _ => self.write_type_declaration_arena(arena, handle),
                },
                LookupType::Local(local_ty) => self.write_type_declaration_local(arena, local_ty),
            }
        }
    }

    fn get_global_variable_id(
        &mut self,
        ir_module: &crate::Module,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> Result<(Word, spirv::StorageClass), Error> {
        Ok(match self.lookup_global_variable.entry(handle) {
            Entry::Occupied(e) => *e.get(),
            //Note: this intentionally frees `self` from borrowing
            Entry::Vacant(_) => {
                let (instruction, id, class) = self.write_global_variable(ir_module, handle)?;
                instruction.to_words(&mut self.logical_layout.declarations);
                (id, class)
            }
        })
    }

    fn get_pointer_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
        class: crate::StorageClass,
    ) -> Result<Word, Error> {
        let ty_id = self.get_type_id(arena, LookupType::Handle(handle))?;
        if let crate::TypeInner::Pointer { .. } = arena[handle].inner {
            return Ok(ty_id);
        }
        Ok(
            match self
                .lookup_type
                .entry(LookupType::Local(LocalType::Pointer {
                    base: handle,
                    class,
                })) {
                Entry::Occupied(e) => *e.get(),
                _ => {
                    let storage_class = self.parse_to_spirv_storage_class(class);
                    let id = self.generate_id();
                    let instruction =
                        super::instructions::instruction_type_pointer(id, storage_class, ty_id);
                    instruction.to_words(&mut self.logical_layout.declarations);
                    self.lookup_type.insert(
                        LookupType::Local(LocalType::Pointer {
                            base: handle,
                            class,
                        }),
                        id,
                    );
                    id
                }
            },
        )
    }

    fn create_pointer_type(&mut self, type_id: spirv::Word, class: spirv::StorageClass) -> Word {
        let id = self.generate_id();
        let instruction = super::instructions::instruction_type_pointer(id, class, type_id);
        instruction.to_words(&mut self.logical_layout.declarations);
        id
    }

    fn create_constant(&mut self, type_id: Word, value: &[Word]) -> Word {
        let id = self.generate_id();
        let instruction = super::instructions::instruction_constant(type_id, id, value);
        instruction.to_words(&mut self.logical_layout.declarations);
        id
    }

    fn write_function(
        &mut self,
        ir_function: &crate::Function,
        ir_module: &crate::Module,
    ) -> Result<Word, Error> {
        let mut function = Function::default();

        self.typifier.resolve_all(
            &ir_function.expressions,
            &ir_module.types,
            &ResolveContext {
                constants: &ir_module.constants,
                global_vars: &ir_module.global_variables,
                local_vars: &ir_function.local_variables,
                functions: &ir_module.functions,
                arguments: &ir_function.arguments,
            },
        )?;

        for (handle, variable) in ir_function.local_variables.iter() {
            let id = self.generate_id();

            if self.flags.contains(WriterFlags::DEBUG) {
                if let Some(ref name) = variable.name {
                    self.debugs
                        .push(super::instructions::instruction_name(id, name));
                }
            }

            let init_word = variable
                .init
                .map(|constant| self.lookup_constant[&constant]);
            let pointer_type_id =
                self.get_pointer_id(&ir_module.types, variable.ty, crate::StorageClass::Function)?;
            let instruction = super::instructions::instruction_variable(
                pointer_type_id,
                id,
                spirv::StorageClass::Function,
                init_word,
            );
            function
                .variables
                .insert(handle, LocalVariable { id, instruction });
        }

        let return_type_id = match ir_function.return_type {
            Some(handle) => self.get_type_id(&ir_module.types, LookupType::Handle(handle))?,
            None => self.void_type,
        };
        let mut parameter_type_ids = Vec::with_capacity(ir_function.arguments.len());

        for argument in ir_function.arguments.iter() {
            let id = self.generate_id();
            let parameter_type_id =
                self.get_type_id(&ir_module.types, LookupType::Handle(argument.ty))?;
            parameter_type_ids.push(parameter_type_id);
            function
                .parameters
                .push(super::instructions::instruction_function_parameter(
                    parameter_type_id,
                    id,
                ));
        }

        let lookup_function_type = LookupFunctionType {
            return_type_id,
            parameter_type_ids,
        };

        let function_id = self.generate_id();
        let function_type = self.get_function_type(lookup_function_type);
        function.signature = Some(super::instructions::instruction_function(
            return_type_id,
            function_id,
            spirv::FunctionControl::empty(),
            function_type,
        ));

        let main_id = self.generate_id();
        self.write_block(
            main_id,
            &ir_function.body,
            ir_module,
            ir_function,
            &mut function,
            None,
            LoopContext::default(),
        )?;

        function.to_words(&mut self.logical_layout.function_definitions);
        super::instructions::instruction_function_end()
            .to_words(&mut self.logical_layout.function_definitions);

        Ok(function_id)
    }

    // TODO Move to instructions module
    fn write_entry_point(
        &mut self,
        entry_point: &crate::EntryPoint,
        stage: crate::ShaderStage,
        name: &str,
        ir_module: &crate::Module,
    ) -> Result<Instruction, Error> {
        let function_id = self.write_function(&entry_point.function, ir_module)?;

        let mut interface_ids = vec![];
        for ((handle, var), &usage) in ir_module
            .global_variables
            .iter()
            .zip(&entry_point.function.global_usage)
        {
            let is_io = match var.class {
                crate::StorageClass::Input | crate::StorageClass::Output => !usage.is_empty(),
                _ => false,
            };
            if is_io {
                let (id, _) = self.get_global_variable_id(ir_module, handle)?;
                interface_ids.push(id);
            }
        }

        let exec_model = match stage {
            crate::ShaderStage::Vertex => spirv::ExecutionModel::Vertex,
            crate::ShaderStage::Fragment => {
                let execution_mode = spirv::ExecutionMode::OriginUpperLeft;
                self.check(execution_mode.required_capabilities())?;
                super::instructions::instruction_execution_mode(function_id, execution_mode, &[])
                    .to_words(&mut self.logical_layout.execution_modes);
                spirv::ExecutionModel::Fragment
            }
            crate::ShaderStage::Compute => {
                let execution_mode = spirv::ExecutionMode::LocalSize;
                self.check(execution_mode.required_capabilities())?;
                super::instructions::instruction_execution_mode(
                    function_id,
                    execution_mode,
                    &entry_point.workgroup_size,
                )
                .to_words(&mut self.logical_layout.execution_modes);
                spirv::ExecutionModel::GLCompute
            }
        };
        self.check(exec_model.required_capabilities())?;

        if self.flags.contains(WriterFlags::DEBUG) {
            self.debugs
                .push(super::instructions::instruction_name(function_id, name));
        }

        Ok(super::instructions::instruction_entry_point(
            exec_model,
            function_id,
            name,
            interface_ids.as_slice(),
        ))
    }

    fn write_scalar(&self, id: Word, kind: crate::ScalarKind, width: crate::Bytes) -> Instruction {
        let bits = (width * BITS_PER_BYTE) as u32;
        match kind {
            crate::ScalarKind::Sint => super::instructions::instruction_type_int(
                id,
                bits,
                super::instructions::Signedness::Signed,
            ),
            crate::ScalarKind::Uint => super::instructions::instruction_type_int(
                id,
                bits,
                super::instructions::Signedness::Unsigned,
            ),
            crate::ScalarKind::Float => super::instructions::instruction_type_float(id, bits),
            crate::ScalarKind::Bool => super::instructions::instruction_type_bool(id),
        }
    }

    fn parse_to_spirv_storage_class(&self, class: crate::StorageClass) -> spirv::StorageClass {
        match class {
            crate::StorageClass::Handle => spirv::StorageClass::UniformConstant,
            crate::StorageClass::Function => spirv::StorageClass::Function,
            crate::StorageClass::Input => spirv::StorageClass::Input,
            crate::StorageClass::Output => spirv::StorageClass::Output,
            crate::StorageClass::Private => spirv::StorageClass::Private,
            crate::StorageClass::Storage if self.physical_layout.supports_storage_buffers() => {
                spirv::StorageClass::StorageBuffer
            }
            crate::StorageClass::Storage | crate::StorageClass::Uniform => {
                spirv::StorageClass::Uniform
            }
            crate::StorageClass::WorkGroup => spirv::StorageClass::Workgroup,
            crate::StorageClass::PushConstant => spirv::StorageClass::PushConstant,
        }
    }

    fn write_type_declaration_local(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        local_ty: LocalType,
    ) -> Result<Word, Error> {
        let id = self.generate_id();
        let instruction = match local_ty {
            LocalType::Scalar { kind, width } => self.write_scalar(id, kind, width),
            LocalType::Vector { size, kind, width } => {
                let scalar_id =
                    self.get_type_id(arena, LookupType::Local(LocalType::Scalar { kind, width }))?;
                super::instructions::instruction_type_vector(id, scalar_id, size)
            }
            LocalType::Matrix {
                columns,
                rows,
                width,
            } => {
                let vector_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Vector {
                        size: rows,
                        kind: crate::ScalarKind::Float,
                        width,
                    }),
                )?;
                super::instructions::instruction_type_matrix(id, vector_id, columns)
            }
            LocalType::Pointer { .. } => {
                return Err(Error::FeatureNotImplemented("pointer declaration"))
            }
            LocalType::SampledImage { image_type } => {
                let image_type_id = self.get_type_id(arena, LookupType::Handle(image_type))?;
                super::instructions::instruction_type_sampled_image(id, image_type_id)
            }
        };

        self.lookup_type.insert(LookupType::Local(local_ty), id);
        instruction.to_words(&mut self.logical_layout.declarations);
        Ok(id)
    }

    fn write_type_declaration_arena(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
    ) -> Result<Word, Error> {
        let ty = &arena[handle];
        let id = self.generate_id();

        if self.flags.contains(WriterFlags::DEBUG) {
            if let Some(ref name) = ty.name {
                self.debugs
                    .push(super::instructions::instruction_name(id, name));
            }
        }

        let instruction = match ty.inner {
            crate::TypeInner::Scalar { kind, width } => {
                self.lookup_type
                    .insert(LookupType::Local(LocalType::Scalar { kind, width }), id);
                self.write_scalar(id, kind, width)
            }
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_id =
                    self.get_type_id(arena, LookupType::Local(LocalType::Scalar { kind, width }))?;
                self.lookup_type.insert(
                    LookupType::Local(LocalType::Vector { size, kind, width }),
                    id,
                );
                super::instructions::instruction_type_vector(id, scalar_id, size)
            }
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => {
                let vector_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Vector {
                        size: columns,
                        kind: crate::ScalarKind::Float,
                        width,
                    }),
                )?;
                self.lookup_type.insert(
                    LookupType::Local(LocalType::Matrix {
                        columns,
                        rows,
                        width,
                    }),
                    id,
                );
                super::instructions::instruction_type_matrix(id, vector_id, columns)
            }
            crate::TypeInner::Image {
                dim,
                arrayed,
                class,
            } => {
                let width = 4;
                let local_type = match class {
                    crate::ImageClass::Sampled { kind, multi: _ } => {
                        LocalType::Scalar { kind, width }
                    }
                    crate::ImageClass::Depth => LocalType::Scalar {
                        kind: crate::ScalarKind::Float,
                        width,
                    },
                    crate::ImageClass::Storage(format) => LocalType::Scalar {
                        kind: format.into(),
                        width,
                    },
                };
                let type_id = self.get_type_id(arena, LookupType::Local(local_type))?;
                let dim = map_dim(dim);
                self.check(dim.required_capabilities())?;
                super::instructions::instruction_type_image(id, type_id, dim, arrayed, class)
            }
            crate::TypeInner::Sampler { comparison: _ } => {
                super::instructions::instruction_type_sampler(id)
            }
            crate::TypeInner::Array { base, size, stride } => {
                if let Some(array_stride) = stride {
                    self.annotations
                        .push(super::instructions::instruction_decorate(
                            id,
                            spirv::Decoration::ArrayStride,
                            &[array_stride.get()],
                        ));
                }

                let type_id = self.get_type_id(arena, LookupType::Handle(base))?;
                match size {
                    crate::ArraySize::Constant(const_handle) => {
                        let length_id = self.lookup_constant[&const_handle];
                        super::instructions::instruction_type_array(id, type_id, length_id)
                    }
                    crate::ArraySize::Dynamic => {
                        super::instructions::instruction_type_runtime_array(id, type_id)
                    }
                }
            }
            crate::TypeInner::Struct {
                block: true,
                ref members,
            } => {
                if let Some(&access) = self.struct_type_handles.get(&handle) {
                    let decoration = if access.is_empty() {
                        spirv::Decoration::Block
                    } else {
                        spirv::Decoration::BufferBlock
                    };
                    self.annotations
                        .push(super::instructions::instruction_decorate(
                            id,
                            decoration,
                            &[],
                        ));
                }

                let mut current_offset = 0;
                let mut member_ids = Vec::with_capacity(members.len());
                for (index, member) in members.iter().enumerate() {
                    let layout = self.layouter.resolve(member.ty);
                    current_offset += layout.pad(current_offset);
                    self.annotations
                        .push(super::instructions::instruction_member_decorate(
                            id,
                            index as u32,
                            spirv::Decoration::Offset,
                            &[current_offset],
                        ));
                    current_offset += match member.span {
                        Some(span) => span.get(),
                        None => layout.size,
                    };

                    if self.flags.contains(WriterFlags::DEBUG) {
                        if let Some(ref name) = member.name {
                            self.debugs
                                .push(super::instructions::instruction_member_name(
                                    id,
                                    index as u32,
                                    name,
                                ));
                        }
                    }

                    if let crate::TypeInner::Matrix {
                        columns,
                        rows: _,
                        width,
                    } = arena[member.ty].inner
                    {
                        let byte_stride = match columns {
                            crate::VectorSize::Bi => 2 * width,
                            crate::VectorSize::Tri | crate::VectorSize::Quad => 4 * width,
                        };
                        self.annotations
                            .push(super::instructions::instruction_member_decorate(
                                id,
                                index as u32,
                                spirv::Decoration::ColMajor,
                                &[],
                            ));
                        self.annotations
                            .push(super::instructions::instruction_member_decorate(
                                id,
                                index as u32,
                                spirv::Decoration::MatrixStride,
                                &[byte_stride as u32],
                            ));
                    }

                    let member_id = self.get_type_id(arena, LookupType::Handle(member.ty))?;
                    member_ids.push(member_id);
                }
                super::instructions::instruction_type_struct(id, member_ids.as_slice())
            }
            crate::TypeInner::Struct {
                block: false,
                ref members,
            } => {
                let mut member_ids = Vec::with_capacity(members.len());
                for member in members {
                    let member_id = self.get_type_id(arena, LookupType::Handle(member.ty))?;
                    member_ids.push(member_id);
                }
                super::instructions::instruction_type_struct(id, member_ids.as_slice())
            }
            crate::TypeInner::Pointer { base, class } => {
                let type_id = self.get_type_id(arena, LookupType::Handle(base))?;
                self.lookup_type
                    .insert(LookupType::Local(LocalType::Pointer { base, class }), id);
                super::instructions::instruction_type_pointer(
                    id,
                    self.parse_to_spirv_storage_class(class),
                    type_id,
                )
            }
        };

        self.lookup_type.insert(LookupType::Handle(handle), id);
        instruction.to_words(&mut self.logical_layout.declarations);
        Ok(id)
    }

    fn write_constant_type(
        &mut self,
        id: Word,
        inner: &crate::ConstantInner,
        types: &crate::Arena<crate::Type>,
    ) -> Result<(), Error> {
        let instruction = match *inner {
            crate::ConstantInner::Scalar { width, ref value } => {
                let type_id = self.get_type_id(
                    types,
                    LookupType::Local(LocalType::Scalar {
                        kind: value.scalar_kind(),
                        width,
                    }),
                )?;
                let (solo, pair);
                match *value {
                    crate::ScalarValue::Sint(val) => {
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
                        super::instructions::instruction_constant(type_id, id, words)
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
                        super::instructions::instruction_constant(type_id, id, words)
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
                        super::instructions::instruction_constant(type_id, id, words)
                    }
                    crate::ScalarValue::Bool(true) => {
                        super::instructions::instruction_constant_true(type_id, id)
                    }
                    crate::ScalarValue::Bool(false) => {
                        super::instructions::instruction_constant_false(type_id, id)
                    }
                }
            }
            crate::ConstantInner::Composite { ty, ref components } => {
                let mut constituent_ids = Vec::with_capacity(components.len());
                for constituent in components.iter() {
                    let constituent_id = self.lookup_constant[constituent];
                    constituent_ids.push(constituent_id);
                }

                let type_id = self.get_type_id(types, LookupType::Handle(ty))?;
                super::instructions::instruction_constant_composite(
                    type_id,
                    id,
                    constituent_ids.as_slice(),
                )
            }
        };

        instruction.to_words(&mut self.logical_layout.declarations);
        Ok(())
    }

    fn write_global_variable(
        &mut self,
        ir_module: &crate::Module,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> Result<(Instruction, Word, spirv::StorageClass), Error> {
        let global_variable = &ir_module.global_variables[handle];
        let id = self.generate_id();

        let class = self.parse_to_spirv_storage_class(global_variable.class);
        self.check(class.required_capabilities())?;

        let init_word = global_variable
            .init
            .map(|constant| self.lookup_constant[&constant]);
        let pointer_type_id =
            self.get_pointer_id(&ir_module.types, global_variable.ty, global_variable.class)?;
        let instruction =
            super::instructions::instruction_variable(pointer_type_id, id, class, init_word);

        if self.flags.contains(WriterFlags::DEBUG) {
            if let Some(ref name) = global_variable.name {
                self.debugs
                    .push(super::instructions::instruction_name(id, name));
            }
        }

        let access_decoration = match global_variable.storage_access {
            crate::StorageAccess::LOAD => Some(spirv::Decoration::NonWritable),
            crate::StorageAccess::STORE => Some(spirv::Decoration::NonReadable),
            _ => None,
        };
        if let Some(decoration) = access_decoration {
            self.annotations
                .push(super::instructions::instruction_decorate(
                    id,
                    decoration,
                    &[],
                ));
        }

        if let Some(interpolation) = global_variable.interpolation {
            let decoration = match interpolation {
                crate::Interpolation::Linear => Some(spirv::Decoration::NoPerspective),
                crate::Interpolation::Flat => Some(spirv::Decoration::Flat),
                crate::Interpolation::Patch => Some(spirv::Decoration::Patch),
                crate::Interpolation::Centroid => Some(spirv::Decoration::Centroid),
                crate::Interpolation::Sample => Some(spirv::Decoration::Sample),
                crate::Interpolation::Perspective => None,
            };
            if let Some(decoration) = decoration {
                self.annotations
                    .push(super::instructions::instruction_decorate(
                        id,
                        decoration,
                        &[],
                    ));
            }
        }

        match global_variable.binding {
            Some(crate::Binding::Location(location)) => {
                self.annotations
                    .push(super::instructions::instruction_decorate(
                        id,
                        spirv::Decoration::Location,
                        &[location],
                    ));
            }
            Some(crate::Binding::Resource { group, binding }) => {
                self.annotations
                    .push(super::instructions::instruction_decorate(
                        id,
                        spirv::Decoration::DescriptorSet,
                        &[group],
                    ));
                self.annotations
                    .push(super::instructions::instruction_decorate(
                        id,
                        spirv::Decoration::Binding,
                        &[binding],
                    ));
            }
            Some(crate::Binding::BuiltIn(built_in)) => {
                use crate::BuiltIn as Bi;
                let built_in = match built_in {
                    Bi::BaseInstance => spirv::BuiltIn::BaseInstance,
                    Bi::BaseVertex => spirv::BuiltIn::BaseVertex,
                    Bi::ClipDistance => spirv::BuiltIn::ClipDistance,
                    Bi::InstanceIndex => spirv::BuiltIn::InstanceIndex,
                    Bi::PointSize => spirv::BuiltIn::PointSize,
                    Bi::Position => spirv::BuiltIn::Position,
                    Bi::VertexIndex => spirv::BuiltIn::VertexIndex,
                    // fragment
                    Bi::FragCoord => spirv::BuiltIn::FragCoord,
                    Bi::FragDepth => spirv::BuiltIn::FragDepth,
                    Bi::FrontFacing => spirv::BuiltIn::FrontFacing,
                    Bi::SampleIndex => spirv::BuiltIn::SampleId,
                    Bi::SampleMaskIn => spirv::BuiltIn::SampleMask,
                    Bi::SampleMaskOut => spirv::BuiltIn::SampleMask,
                    // compute
                    Bi::GlobalInvocationId => spirv::BuiltIn::GlobalInvocationId,
                    Bi::LocalInvocationId => spirv::BuiltIn::LocalInvocationId,
                    Bi::LocalInvocationIndex => spirv::BuiltIn::LocalInvocationIndex,
                    Bi::WorkGroupId => spirv::BuiltIn::WorkgroupId,
                    Bi::WorkGroupSize => spirv::BuiltIn::WorkgroupSize,
                };

                self.annotations
                    .push(super::instructions::instruction_decorate(
                        id,
                        spirv::Decoration::BuiltIn,
                        &[built_in as u32],
                    ));
            }
            None => {}
        }

        // TODO Initializer is optional and not (yet) included in the IR

        self.lookup_global_variable.insert(handle, (id, class));
        Ok((instruction, id, class))
    }

    fn get_function_type(&mut self, lookup_function_type: LookupFunctionType) -> Word {
        match self
            .lookup_function_type
            .entry(lookup_function_type.clone())
        {
            Entry::Occupied(e) => *e.get(),
            _ => {
                let id = self.generate_id();
                let instruction = super::instructions::instruction_type_function(
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

    fn write_composite_construct(
        &mut self,
        base_type_id: Word,
        constituent_ids: &[Word],
        block: &mut Block,
    ) -> Word {
        let id = self.generate_id();
        block
            .body
            .push(super::instructions::instruction_composite_construct(
                base_type_id,
                id,
                constituent_ids,
            ));
        id
    }

    /// Write an expression and return a value ID.
    fn write_expression<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        ir_function: &crate::Function,
        handle: crate::Handle<crate::Expression>,
        block: &mut Block,
        function: &mut Function,
    ) -> Result<ExpressionId, Error> {
        let (raw_expression, result_type_id) =
            self.write_expression_raw(ir_module, ir_function, handle, block, function)?;
        Ok(match raw_expression {
            RawExpression::Value(id) => id,
            RawExpression::Pointer(id, _) => {
                let load_id = self.generate_id();
                block.body.push(super::instructions::instruction_load(
                    result_type_id,
                    load_id,
                    id,
                    None,
                ));
                load_id
            }
        })
    }

    /// Write an expression and return a pointer ID to the result.
    fn write_expression_pointer<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        ir_function: &crate::Function,
        handle: crate::Handle<crate::Expression>,
        block: &mut Block,
        function: &mut Function,
    ) -> Result<PointerExpressionId, Error> {
        let (raw_expression, _) =
            self.write_expression_raw(ir_module, ir_function, handle, block, function)?;
        Ok(match raw_expression {
            RawExpression::Value(_id) => unimplemented!(
                "Expression {:?} is not a pointer",
                ir_function.expressions[handle]
            ),
            RawExpression::Pointer(id, class) => (id, class),
        })
    }

    /// Write an expression, and the result may be either a pointer, or a value.
    fn write_expression_raw<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        ir_function: &crate::Function,
        expr_handle: crate::Handle<crate::Expression>,
        block: &mut Block,
        function: &mut Function,
    ) -> Result<(RawExpression, ExpressionId), Error> {
        let result_lookup_ty = match self.typifier.get_handle(expr_handle) {
            Ok(ty_handle) => LookupType::Handle(ty_handle),
            Err(inner) => LookupType::Local(LocalType::from_inner(inner).unwrap()),
        };
        let result_type_id = self.get_type_id(&ir_module.types, result_lookup_ty)?;

        let raw_expr = match ir_function.expressions[expr_handle] {
            crate::Expression::Access { base, index } => {
                let id = self.generate_id();
                let (raw_base_expression, _) =
                    self.write_expression_raw(ir_module, ir_function, base, block, function)?;
                let index_id =
                    self.write_expression(ir_module, ir_function, index, block, function)?;
                let base_inner = self.typifier.get(base, &ir_module.types);

                match raw_base_expression {
                    RawExpression::Value(base_id) => {
                        if let crate::TypeInner::Array { .. } = *base_inner {
                            return Err(Error::FeatureNotImplemented(
                                "accessing index of a value array",
                            ));
                        }

                        block
                            .body
                            .push(super::instructions::instruction_vector_extract_dynamic(
                                result_type_id,
                                id,
                                base_id,
                                index_id,
                            ));

                        RawExpression::Value(id)
                    }
                    RawExpression::Pointer(base_id, class) => {
                        let pointer_type_id = self.create_pointer_type(result_type_id, class);

                        block
                            .body
                            .push(super::instructions::instruction_access_chain(
                                pointer_type_id,
                                id,
                                base_id,
                                &[index_id],
                            ));

                        RawExpression::Pointer(id, class)
                    }
                }
            }
            crate::Expression::AccessIndex { base, index } => {
                let id = self.generate_id();
                let (raw_base_expression, _) =
                    self.write_expression_raw(ir_module, ir_function, base, block, function)?;

                match raw_base_expression {
                    RawExpression::Value(base_id) => {
                        block
                            .body
                            .push(super::instructions::instruction_composite_extract(
                                result_type_id,
                                id,
                                base_id,
                                &[index],
                            ));

                        RawExpression::Value(id)
                    }
                    RawExpression::Pointer(base_id, class) => {
                        let pointer_type_id = self.create_pointer_type(result_type_id, class);

                        let const_ty_id = self.get_type_id(
                            &ir_module.types,
                            LookupType::Local(LocalType::Scalar {
                                kind: crate::ScalarKind::Sint,
                                width: 4,
                            }),
                        )?;
                        let const_id = self.create_constant(const_ty_id, &[index]);

                        block
                            .body
                            .push(super::instructions::instruction_access_chain(
                                pointer_type_id,
                                id,
                                base_id,
                                &[const_id],
                            ));

                        RawExpression::Pointer(id, class)
                    }
                }
            }
            crate::Expression::GlobalVariable(handle) => {
                let (id, class) = self.get_global_variable_id(&ir_module, handle)?;
                RawExpression::Pointer(id, class)
            }
            crate::Expression::Constant(handle) => {
                let id = self.lookup_constant[&handle];
                RawExpression::Value(id)
            }
            crate::Expression::Compose { ty, ref components } => {
                let base_type_id = self.get_type_id(&ir_module.types, LookupType::Handle(ty))?;

                let mut constituent_ids = Vec::with_capacity(components.len());
                for component in components {
                    let component_id = self.write_expression(
                        ir_module,
                        &ir_function,
                        *component,
                        block,
                        function,
                    )?;
                    constituent_ids.push(component_id);
                }

                let id = self.write_composite_construct(base_type_id, &constituent_ids, block);
                RawExpression::Value(id)
            }
            crate::Expression::Unary { op, expr } => {
                let id = self.generate_id();
                let expr_id =
                    self.write_expression(ir_module, ir_function, expr, block, function)?;
                let expr_ty_inner = self.typifier.get(expr, &ir_module.types);

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

                block.body.push(super::instructions::instruction_unary(
                    spirv_op,
                    result_type_id,
                    id,
                    expr_id,
                ));
                RawExpression::Value(id)
            }
            crate::Expression::Binary { op, left, right } => {
                let id = self.generate_id();
                let left_id =
                    self.write_expression(ir_module, ir_function, left, block, function)?;
                let right_id =
                    self.write_expression(ir_module, ir_function, right, block, function)?;

                let left_ty_inner = self.typifier.get(left, &ir_module.types);
                let right_ty_inner = self.typifier.get(right, &ir_module.types);

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

                block.body.push(super::instructions::instruction_binary(
                    spirv_op,
                    result_type_id,
                    id,
                    if preserve_order { left_id } else { right_id },
                    if preserve_order { right_id } else { left_id },
                ));
                RawExpression::Value(id)
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
                    Custom(super::Instruction),
                }

                let arg0_id =
                    self.write_expression(ir_module, ir_function, arg, block, function)?;
                let arg_scalar_kind = self.typifier.get(arg, &ir_module.types).scalar_kind();
                let arg1_id = match arg1 {
                    Some(id) => {
                        self.write_expression(ir_module, ir_function, id, block, function)?
                    }
                    None => 0,
                };
                let arg2_id = match arg2 {
                    Some(id) => {
                        self.write_expression(ir_module, ir_function, id, block, function)?
                    }
                    None => 0,
                };

                let id = self.generate_id();
                let math_op = match fun {
                    // comparison
                    Mf::Abs => {
                        match arg_scalar_kind {
                            Some(crate::ScalarKind::Float) => MathOp::Ext(spirv::GLOp::FAbs),
                            Some(crate::ScalarKind::Sint) => MathOp::Ext(spirv::GLOp::SAbs),
                            Some(crate::ScalarKind::Uint) => {
                                MathOp::Custom(super::instructions::instruction_unary(
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
                    Mf::Dot => MathOp::Custom(super::instructions::instruction_binary(
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
                    Mf::Transpose => MathOp::Custom(super::instructions::instruction_unary(
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
                    MathOp::Ext(op) => super::instructions::instruction_ext_inst(
                        self.gl450_ext_inst_id,
                        op,
                        result_type_id,
                        id,
                        &[arg0_id, arg1_id, arg2_id][..fun.argument_count()],
                    ),
                    MathOp::Custom(inst) => inst,
                });
                RawExpression::Value(id)
            }
            crate::Expression::LocalVariable(variable) => {
                let local_var = &function.variables[&variable];
                RawExpression::Pointer(local_var.id, spirv::StorageClass::Function)
            }
            crate::Expression::FunctionArgument(index) => {
                let id = function.parameters[index as usize].result_id.unwrap();
                RawExpression::Value(id)
            }
            crate::Expression::Call {
                function: local_function,
                ref arguments,
            } => {
                let id = self.generate_id();
                //TODO: avoid heap allocation
                let mut argument_ids = vec![];

                for argument in arguments {
                    let arg_id =
                        self.write_expression(ir_module, ir_function, *argument, block, function)?;
                    argument_ids.push(arg_id);
                }

                block
                    .body
                    .push(super::instructions::instruction_function_call(
                        result_type_id,
                        id,
                        *self.lookup_function.get(&local_function).unwrap(),
                        argument_ids.as_slice(),
                    ));

                RawExpression::Value(id)
            }
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                if !convert {
                    return Err(Error::FeatureNotImplemented("bitcast"));
                }

                let expr_id =
                    self.write_expression(ir_module, ir_function, expr, block, function)?;
                let expr_kind = self
                    .typifier
                    .get(expr, &ir_module.types)
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

                let id = self.generate_id();
                let instruction =
                    super::instructions::instruction_unary(op, result_type_id, id, expr_id);
                block.body.push(instruction);

                RawExpression::Value(id)
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
                let image_id =
                    self.write_expression(ir_module, ir_function, image, block, function)?;
                let image_type = self.typifier.get_handle(image).unwrap();

                // OpTypeSampledImage
                let sampled_image_type_id = self.get_type_id(
                    &ir_module.types,
                    LookupType::Local(LocalType::SampledImage { image_type }),
                )?;

                // sampler
                let sampler_id =
                    self.write_expression(ir_module, ir_function, sampler, block, function)?;

                // coordinate
                let mut coordinate_id =
                    self.write_expression(ir_module, ir_function, coordinate, block, function)?;

                if let Some(array_index) = array_index {
                    let coordinate_scalar_type_id = self.get_type_id(
                        &ir_module.types,
                        LookupType::Local(LocalType::Scalar {
                            kind: crate::ScalarKind::Float,
                            width: 4,
                        }),
                    )?;

                    let mut constituent_ids = [0u32; 4];
                    let size = match *self.typifier.get(coordinate, &ir_module.types) {
                        crate::TypeInner::Scalar { .. } => {
                            constituent_ids[0] = coordinate_id;
                            crate::VectorSize::Bi
                        }
                        crate::TypeInner::Vector { size, .. } => {
                            for i in 0..size as u32 {
                                let id = self.generate_id();
                                constituent_ids[i as usize] = id;
                                block.body.push(
                                    super::instructions::instruction_composite_extract(
                                        coordinate_scalar_type_id,
                                        id,
                                        coordinate_id,
                                        &[i],
                                    ),
                                );
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

                    let array_index_f32_id = self.generate_id();
                    constituent_ids[size as usize - 1] = array_index_f32_id;

                    let array_index_u32_id = self.write_expression(
                        ir_module,
                        ir_function,
                        array_index,
                        block,
                        function,
                    )?;
                    let cast_instruction = super::instructions::instruction_unary(
                        spirv::Op::ConvertUToF,
                        coordinate_scalar_type_id,
                        array_index_f32_id,
                        array_index_u32_id,
                    );
                    block.body.push(cast_instruction);

                    let extended_coordinate_type_id = self.get_type_id(
                        &ir_module.types,
                        LookupType::Local(LocalType::Vector {
                            size,
                            kind: crate::ScalarKind::Float,
                            width: 4,
                        }),
                    )?;

                    coordinate_id = self.write_composite_construct(
                        extended_coordinate_type_id,
                        &constituent_ids[..size as usize],
                        block,
                    );
                }

                let sampled_image_id = self.generate_id();
                block
                    .body
                    .push(super::instructions::instruction_sampled_image(
                        sampled_image_type_id,
                        sampled_image_id,
                        image_id,
                        sampler_id,
                    ));
                let id = self.generate_id();

                let depth_id = match depth_ref {
                    Some(handle) => {
                        let expr_id =
                            self.write_expression(ir_module, ir_function, handle, block, function)?;
                        Some(expr_id)
                    }
                    None => None,
                };

                let mut main_instruction = match level {
                    crate::SampleLevel::Zero => {
                        let mut inst = super::instructions::instruction_image_sample(
                            result_type_id,
                            id,
                            SampleLod::Explicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        //TODO: cache this!
                        let zero_id = self.generate_id();
                        let zero_inner = crate::ConstantInner::Scalar {
                            width: 4,
                            value: crate::ScalarValue::Float(0.0),
                        };
                        self.write_constant_type(zero_id, &zero_inner, &ir_module.types)?;
                        inst.add_operand(spirv::ImageOperands::LOD.bits());
                        inst.add_operand(zero_id);

                        inst
                    }
                    crate::SampleLevel::Auto => super::instructions::instruction_image_sample(
                        result_type_id,
                        id,
                        SampleLod::Implicit,
                        sampled_image_id,
                        coordinate_id,
                        depth_id,
                    ),
                    crate::SampleLevel::Exact(lod_handle) => {
                        let mut inst = super::instructions::instruction_image_sample(
                            result_type_id,
                            id,
                            SampleLod::Explicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        let lod_id = self.write_expression(
                            ir_module,
                            ir_function,
                            lod_handle,
                            block,
                            function,
                        )?;
                        inst.add_operand(spirv::ImageOperands::LOD.bits());
                        inst.add_operand(lod_id);

                        inst
                    }
                    crate::SampleLevel::Bias(bias_handle) => {
                        let mut inst = super::instructions::instruction_image_sample(
                            result_type_id,
                            id,
                            SampleLod::Implicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        let bias_id = self.write_expression(
                            ir_module,
                            ir_function,
                            bias_handle,
                            block,
                            function,
                        )?;
                        inst.add_operand(spirv::ImageOperands::BIAS.bits());
                        inst.add_operand(bias_id);

                        inst
                    }
                    crate::SampleLevel::Gradient { x, y } => {
                        let mut inst = super::instructions::instruction_image_sample(
                            result_type_id,
                            id,
                            SampleLod::Explicit,
                            sampled_image_id,
                            coordinate_id,
                            depth_id,
                        );

                        let x_id =
                            self.write_expression(ir_module, ir_function, x, block, function)?;
                        let y_id =
                            self.write_expression(ir_module, ir_function, y, block, function)?;
                        inst.add_operand(spirv::ImageOperands::GRAD.bits());
                        inst.add_operand(x_id);
                        inst.add_operand(y_id);

                        inst
                    }
                };

                if let Some(offset_const) = offset {
                    let offset_id = self.lookup_constant[&offset_const];
                    main_instruction.add_operand(spirv::ImageOperands::CONST_OFFSET.bits());
                    main_instruction.add_operand(offset_id);
                }

                block.body.push(main_instruction);
                RawExpression::Value(id)
            }
            ref other => {
                log::error!("unimplemented {:?}", other);
                return Err(Error::FeatureNotImplemented("expression"));
            }
        };

        Ok((raw_expr, result_type_id))
    }

    #[allow(clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        label_id: Word,
        statements: &[crate::Statement],
        ir_module: &crate::Module,
        ir_function: &crate::Function,
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
                crate::Statement::Block(ref block_statements) => {
                    let scope_id = self.generate_id();
                    function.consume(block, super::instructions::instruction_branch(scope_id));

                    let merge_id = self.generate_id();
                    self.write_block(
                        scope_id,
                        block_statements,
                        ir_module,
                        ir_function,
                        function,
                        Some(merge_id),
                        loop_context,
                    )?;

                    block = Block::new(merge_id);
                }
                crate::Statement::If {
                    ref condition,
                    ref accept,
                    ref reject,
                } => {
                    let condition_id = self.write_expression(
                        ir_module,
                        ir_function,
                        *condition,
                        &mut block,
                        function,
                    )?;

                    let merge_id = self.generate_id();
                    block
                        .body
                        .push(super::instructions::instruction_selection_merge(
                            merge_id,
                            spirv::SelectionControl::NONE,
                        ));

                    let accept_id = self.generate_id();
                    let reject_id = self.generate_id();
                    function.consume(
                        block,
                        super::instructions::instruction_branch_conditional(
                            condition_id,
                            accept_id,
                            reject_id,
                        ),
                    );

                    self.write_block(
                        accept_id,
                        accept,
                        ir_module,
                        ir_function,
                        function,
                        Some(merge_id),
                        loop_context,
                    )?;
                    self.write_block(
                        reject_id,
                        reject,
                        ir_module,
                        ir_function,
                        function,
                        Some(merge_id),
                        loop_context,
                    )?;

                    block = Block::new(merge_id);
                }
                crate::Statement::Switch { .. } => {
                    log::error!("unimplemented Switch");
                    return Err(Error::FeatureNotImplemented("switch"));
                }
                crate::Statement::Loop {
                    ref body,
                    ref continuing,
                } => {
                    let preamble_id = self.generate_id();
                    function.consume(block, super::instructions::instruction_branch(preamble_id));

                    let merge_id = self.generate_id();
                    let body_id = self.generate_id();
                    let continuing_id = self.generate_id();

                    // SPIR-V requires the continuing to the `OpLoopMerge`,
                    // so we have to start a new block with it.
                    block = Block::new(preamble_id);
                    block.body.push(super::instructions::instruction_loop_merge(
                        merge_id,
                        continuing_id,
                        spirv::SelectionControl::NONE,
                    ));
                    function.consume(block, super::instructions::instruction_branch(body_id));

                    self.write_block(
                        body_id,
                        body,
                        ir_module,
                        ir_function,
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
                    block.termination = Some(super::instructions::instruction_branch(
                        loop_context.break_id.unwrap(),
                    ));
                }
                crate::Statement::Continue => {
                    block.termination = Some(super::instructions::instruction_branch(
                        loop_context.continuing_id.unwrap(),
                    ));
                }
                crate::Statement::Return { value: Some(value) } => {
                    let id =
                        self.write_expression(ir_module, ir_function, value, &mut block, function)?;
                    block.termination = Some(super::instructions::instruction_return_value(id));
                }
                crate::Statement::Return { value: None } => {
                    block.termination = Some(super::instructions::instruction_return());
                }
                crate::Statement::Kill => {
                    block.termination = Some(super::instructions::instruction_kill());
                }
                crate::Statement::Store { pointer, value } => {
                    let (pointer_id, _) = self.write_expression_pointer(
                        ir_module,
                        ir_function,
                        pointer,
                        &mut block,
                        function,
                    )?;
                    let value_id =
                        self.write_expression(ir_module, ir_function, value, &mut block, function)?;

                    block.body.push(super::instructions::instruction_store(
                        pointer_id, value_id, None,
                    ));
                }
                crate::Statement::Call {
                    function: local_function,
                    ref arguments,
                } => {
                    let id = self.generate_id();
                    //TODO: avoid heap allocation
                    let mut argument_ids = vec![];

                    for argument in arguments {
                        let arg_id = self.write_expression(
                            ir_module,
                            ir_function,
                            *argument,
                            &mut block,
                            function,
                        )?;
                        argument_ids.push(arg_id);
                    }

                    block
                        .body
                        .push(super::instructions::instruction_function_call(
                            self.void_type,
                            id,
                            *self.lookup_function.get(&local_function).unwrap(),
                            argument_ids.as_slice(),
                        ));
                }
            }
        }

        if block.termination.is_none() {
            block.termination = Some(match exit_id {
                Some(id) => super::instructions::instruction_branch(id),
                None => super::instructions::instruction_return(),
            });
        }

        function.blocks.push(block);
        Ok(())
    }

    fn write_physical_layout(&mut self) {
        self.physical_layout.bound = self.id_count + 1;
    }

    fn write_logical_layout(&mut self, ir_module: &crate::Module) -> Result<(), Error> {
        super::instructions::instruction_type_void(self.void_type)
            .to_words(&mut self.logical_layout.declarations);
        super::instructions::instruction_ext_inst_import(self.gl450_ext_inst_id, "GLSL.std.450")
            .to_words(&mut self.logical_layout.ext_inst_imports);

        if self.flags.contains(WriterFlags::DEBUG) {
            self.debugs.push(super::instructions::instruction_source(
                spirv::SourceLanguage::GLSL,
                450,
            ));
        }

        for (handle, constant) in ir_module.constants.iter() {
            let id = self.generate_id();
            self.lookup_constant.insert(handle, id);
            self.write_constant_type(id, &constant.inner, &ir_module.types)?;
        }

        for (_, var) in ir_module.global_variables.iter() {
            if let crate::TypeInner::Struct { .. } = ir_module.types[var.ty].inner {
                self.struct_type_handles.insert(var.ty, var.storage_access);
            }
        }

        for (handle, ir_function) in ir_module.functions.iter() {
            let id = self.write_function(ir_function, ir_module)?;
            self.lookup_function.insert(handle, id);
        }

        for (&(stage, ref name), ir_ep) in ir_module.entry_points.iter() {
            let ep_instruction = self.write_entry_point(ir_ep, stage, name, ir_module)?;
            ep_instruction.to_words(&mut self.logical_layout.entry_points);
        }

        for capability in self.capabilities.iter() {
            super::instructions::instruction_capability(*capability)
                .to_words(&mut self.logical_layout.capabilities);
        }

        let addressing_model = spirv::AddressingModel::Logical;
        let memory_model = spirv::MemoryModel::GLSL450;
        self.check(addressing_model.required_capabilities())?;
        self.check(memory_model.required_capabilities())?;

        super::instructions::instruction_memory_model(addressing_model, memory_model)
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

    pub fn write(&mut self, ir_module: &crate::Module, words: &mut Vec<Word>) -> Result<(), Error> {
        self.layouter
            .initialize(&ir_module.types, &ir_module.constants);

        self.write_logical_layout(ir_module)?;
        self.write_physical_layout();

        self.physical_layout.in_words(words);
        self.logical_layout.in_words(words);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        back::spv::{Writer, WriterFlags},
        Header,
    };

    #[test]
    fn test_writer_generate_id() {
        let mut writer = create_writer();

        assert_eq!(writer.id_count, 2);
        writer.generate_id();
        assert_eq!(writer.id_count, 3);
    }

    #[test]
    fn test_write_physical_layout() {
        let mut writer = create_writer();
        assert_eq!(writer.physical_layout.bound, 0);
        writer.write_physical_layout();
        assert_eq!(writer.physical_layout.bound, 3);
    }

    fn create_writer() -> Writer {
        let header = Header {
            generator: 0,
            version: (1, 0, 0),
        };
        Writer::new(&header, WriterFlags::NONE, Default::default())
    }
}
