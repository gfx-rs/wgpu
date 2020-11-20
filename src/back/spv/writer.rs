/*! Standard Portable Intermediate Representation (SPIR-V) backend !*/
use super::{Instruction, LogicalLayout, PhysicalLayout, WriterFlags};
use spirv::Word;
use std::{collections::hash_map::Entry, ops};
use thiserror::Error;

const BITS_PER_BYTE: crate::Bytes = 8;

#[derive(Clone, Debug, Error)]
pub enum Error {
    #[error("can't find local variable: {0:?}")]
    UnknownLocalVariable(crate::LocalVariable),
    #[error("bad image class for op: {0:?}")]
    BadImageClass(crate::ImageClass),
    #[error("not an image")]
    NotImage,
    #[error("empty value")]
    FeatureNotImplemented(),
}

struct Block {
    label: Option<Instruction>,
    body: Vec<Instruction>,
    termination: Option<Instruction>,
}

impl Block {
    pub fn new() -> Self {
        Block {
            label: None,
            body: vec![],
            termination: None,
        }
    }
}

struct LocalVariable {
    id: Word,
    name: Option<String>,
    instruction: Instruction,
}

struct Function {
    signature: Option<Instruction>,
    parameters: Vec<Instruction>,
    variables: Vec<LocalVariable>,
    blocks: Vec<Block>,
}

impl Function {
    pub fn new() -> Self {
        Function {
            signature: None,
            parameters: vec![],
            variables: vec![],
            blocks: vec![],
        }
    }

    fn to_words(&self, sink: &mut impl Extend<Word>) {
        self.signature.as_ref().unwrap().to_words(sink);
        for instruction in self.parameters.iter() {
            instruction.to_words(sink);
        }
        for (index, block) in self.blocks.iter().enumerate() {
            block.label.as_ref().unwrap().to_words(sink);
            if index == 0 {
                for local_var in self.variables.iter() {
                    local_var.instruction.to_words(sink);
                }
            }
            for instruction in block.body.iter() {
                instruction.to_words(sink);
            }
            block.termination.as_ref().unwrap().to_words(sink);
        }
    }
}

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LocalType {
    Void,
    Scalar {
        kind: crate::ScalarKind,
        width: crate::Bytes,
    },
    Vector {
        size: crate::VectorSize,
        kind: crate::ScalarKind,
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

#[derive(Debug, PartialEq, Hash, Eq, Copy, Clone)]
enum LookupType {
    Handle(crate::Handle<crate::Type>),
    Local(LocalType),
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

enum MaybeOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> ops::Deref for MaybeOwned<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match *self {
            MaybeOwned::Owned(ref value) => value,
            MaybeOwned::Borrowed(reference) => reference,
        }
    }
}

enum Dimension {
    Scalar,
    Vector,
    Matrix,
}

fn get_dimension(ty_inner: &crate::TypeInner) -> Dimension {
    match *ty_inner {
        crate::TypeInner::Scalar { .. } => Dimension::Scalar,
        crate::TypeInner::Vector { .. } => Dimension::Vector,
        crate::TypeInner::Matrix { .. } => Dimension::Matrix,
        _ => unreachable!(),
    }
}

pub struct Writer {
    physical_layout: PhysicalLayout,
    logical_layout: LogicalLayout,
    id_count: u32,
    capabilities: crate::FastHashSet<spirv::Capability>,
    debugs: Vec<Instruction>,
    annotations: Vec<Instruction>,
    writer_flags: WriterFlags,
    void_type: Option<u32>,
    lookup_type: crate::FastHashMap<LookupType, Word>,
    lookup_function: crate::FastHashMap<crate::Handle<crate::Function>, Word>,
    lookup_function_type: crate::FastHashMap<LookupFunctionType, Word>,
    lookup_constant: crate::FastHashMap<crate::Handle<crate::Constant>, Word>,
    lookup_global_variable: crate::FastHashMap<crate::Handle<crate::GlobalVariable>, Word>,
}

// type alias, for success return of write_expression
type WriteExpressionOutput = (Word, LookupType);

impl Writer {
    pub fn new(header: &crate::Header, writer_flags: WriterFlags) -> Self {
        Writer {
            physical_layout: PhysicalLayout::new(header),
            logical_layout: LogicalLayout::default(),
            id_count: 0,
            capabilities: crate::FastHashSet::default(),
            debugs: vec![],
            annotations: vec![],
            writer_flags,
            void_type: None,
            lookup_type: crate::FastHashMap::default(),
            lookup_function: crate::FastHashMap::default(),
            lookup_function_type: crate::FastHashMap::default(),
            lookup_constant: crate::FastHashMap::default(),
            lookup_global_variable: crate::FastHashMap::default(),
        }
    }

    fn generate_id(&mut self) -> Word {
        self.id_count += 1;
        self.id_count
    }

    fn try_add_capabilities(&mut self, capabilities: &[spirv::Capability]) {
        for capability in capabilities.iter() {
            self.capabilities.insert(*capability);
        }
    }

    fn get_type_id(&mut self, arena: &crate::Arena<crate::Type>, lookup_ty: LookupType) -> Word {
        if let Entry::Occupied(e) = self.lookup_type.entry(lookup_ty) {
            *e.get()
        } else {
            match lookup_ty {
                LookupType::Handle(handle) => match arena[handle].inner {
                    crate::TypeInner::Scalar { kind, width } => self
                        .get_type_id(arena, LookupType::Local(LocalType::Scalar { kind, width })),
                    _ => self.write_type_declaration_arena(arena, handle),
                },
                LookupType::Local(local_ty) => self.write_type_declaration_local(arena, local_ty),
            }
        }
    }

    fn get_constant_id(
        &mut self,
        handle: crate::Handle<crate::Constant>,
        ir_module: &crate::Module,
    ) -> Word {
        match self.lookup_constant.entry(handle) {
            Entry::Occupied(e) => *e.get(),
            _ => {
                let (instruction, id) = self.write_constant_type(handle, ir_module);
                instruction.to_words(&mut self.logical_layout.declarations);
                id
            }
        }
    }

    fn get_global_variable_id(
        &mut self,
        ir_module: &crate::Module,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> Word {
        match self.lookup_global_variable.entry(handle) {
            Entry::Occupied(e) => *e.get(),
            _ => {
                let (instruction, id) = self.write_global_variable(ir_module, handle);
                instruction.to_words(&mut self.logical_layout.declarations);
                id
            }
        }
    }

    fn get_function_return_type(
        &mut self,
        ty: Option<crate::Handle<crate::Type>>,
        arena: &crate::Arena<crate::Type>,
    ) -> Word {
        match ty {
            Some(handle) => self.get_type_id(arena, LookupType::Handle(handle)),
            None => match self.void_type {
                Some(id) => id,
                None => {
                    let id = self.generate_id();
                    self.void_type = Some(id);
                    super::instructions::instruction_type_void(id)
                        .to_words(&mut self.logical_layout.declarations);
                    id
                }
            },
        }
    }

    fn get_pointer_id(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
        class: crate::StorageClass,
    ) -> Word {
        let ty = &arena[handle];
        let ty_id = self.get_type_id(arena, LookupType::Handle(handle));
        match ty.inner {
            crate::TypeInner::Pointer { .. } => ty_id,
            _ => {
                match self
                    .lookup_type
                    .entry(LookupType::Local(LocalType::Pointer {
                        base: handle,
                        class,
                    })) {
                    Entry::Occupied(e) => *e.get(),
                    _ => {
                        let id =
                            self.create_pointer(ty_id, self.parse_to_spirv_storage_class(class));
                        self.lookup_type.insert(
                            LookupType::Local(LocalType::Pointer {
                                base: handle,
                                class,
                            }),
                            id,
                        );
                        id
                    }
                }
            }
        }
    }

    fn create_pointer(&mut self, ty_id: Word, class: spirv::StorageClass) -> Word {
        let id = self.generate_id();
        let instruction = super::instructions::instruction_type_pointer(id, class, ty_id);
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
    ) -> spirv::Word {
        let mut function = Function::new();

        for (_, variable) in ir_function.local_variables.iter() {
            let id = self.generate_id();

            let init_word = variable
                .init
                .map(|constant| self.get_constant_id(constant, ir_module));

            let pointer_id =
                self.get_pointer_id(&ir_module.types, variable.ty, crate::StorageClass::Function);
            function.variables.push(LocalVariable {
                id,
                name: variable.name.clone(),
                instruction: super::instructions::instruction_variable(
                    pointer_id,
                    id,
                    spirv::StorageClass::Function,
                    init_word,
                ),
            });
        }

        let return_type_id =
            self.get_function_return_type(ir_function.return_type, &ir_module.types);
        let mut parameter_type_ids = Vec::with_capacity(ir_function.arguments.len());

        let mut function_parameter_pointer_ids = vec![];

        for argument in ir_function.arguments.iter() {
            let id = self.generate_id();
            let pointer_id =
                self.get_pointer_id(&ir_module.types, argument.ty, crate::StorageClass::Function);

            function_parameter_pointer_ids.push(pointer_id);
            parameter_type_ids
                .push(self.get_type_id(&ir_module.types, LookupType::Handle(argument.ty)));
            function
                .parameters
                .push(super::instructions::instruction_function_parameter(
                    pointer_id, id,
                ));
        }

        let lookup_function_type = LookupFunctionType {
            return_type_id,
            parameter_type_ids,
        };

        let function_id = self.generate_id();
        let function_type =
            self.get_function_type(lookup_function_type, function_parameter_pointer_ids);
        function.signature = Some(super::instructions::instruction_function(
            return_type_id,
            function_id,
            spirv::FunctionControl::empty(),
            function_type,
        ));

        self.write_block(&ir_function.body, ir_module, ir_function, &mut function);

        function.to_words(&mut self.logical_layout.function_definitions);
        super::instructions::instruction_function_end()
            .to_words(&mut self.logical_layout.function_definitions);

        function_id
    }

    // TODO Move to instructions module
    fn write_entry_point(
        &mut self,
        entry_point: &crate::EntryPoint,
        stage: crate::ShaderStage,
        name: &str,
        ir_module: &crate::Module,
    ) -> Instruction {
        let function_id = self.write_function(&entry_point.function, ir_module);

        let exec_model = match stage {
            crate::ShaderStage::Vertex => spirv::ExecutionModel::Vertex,
            crate::ShaderStage::Fragment { .. } => spirv::ExecutionModel::Fragment,
            crate::ShaderStage::Compute { .. } => spirv::ExecutionModel::GLCompute,
        };

        let mut interface_ids = vec![];
        for ((handle, _), &usage) in ir_module
            .global_variables
            .iter()
            .filter(|&(_, var)| {
                var.class == crate::StorageClass::Input || var.class == crate::StorageClass::Output
            })
            .zip(&entry_point.function.global_usage)
        {
            if usage.contains(crate::GlobalUse::STORE) || usage.contains(crate::GlobalUse::LOAD) {
                let id = self.get_global_variable_id(ir_module, handle);
                interface_ids.push(id);
            }
        }

        self.try_add_capabilities(exec_model.required_capabilities());
        match stage {
            crate::ShaderStage::Vertex => {}
            crate::ShaderStage::Fragment => {
                let execution_mode = spirv::ExecutionMode::OriginUpperLeft;
                self.try_add_capabilities(execution_mode.required_capabilities());
                super::instructions::instruction_execution_mode(function_id, execution_mode)
                    .to_words(&mut self.logical_layout.execution_modes);
            }
            crate::ShaderStage::Compute => {}
        }

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            self.debugs
                .push(super::instructions::instruction_name(function_id, name));
        }

        super::instructions::instruction_entry_point(
            exec_model,
            function_id,
            name,
            interface_ids.as_slice(),
        )
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
    ) -> Word {
        let id = self.generate_id();
        let instruction = match local_ty {
            LocalType::Void => unreachable!(),
            LocalType::Scalar { kind, width } => self.write_scalar(id, kind, width),
            LocalType::Vector { size, kind, width } => {
                let scalar_id =
                    self.get_type_id(arena, LookupType::Local(LocalType::Scalar { kind, width }));
                super::instructions::instruction_type_vector(id, scalar_id, size)
            }
            LocalType::Pointer { .. } => unimplemented!(),
            LocalType::SampledImage { image_type } => {
                let image_type_id = self.get_type_id(arena, LookupType::Handle(image_type));
                super::instructions::instruction_type_sampled_image(id, image_type_id)
            }
        };

        self.lookup_type.insert(LookupType::Local(local_ty), id);
        instruction.to_words(&mut self.logical_layout.declarations);
        id
    }

    fn write_type_declaration_arena(
        &mut self,
        arena: &crate::Arena<crate::Type>,
        handle: crate::Handle<crate::Type>,
    ) -> Word {
        let ty = &arena[handle];
        let id = self.generate_id();

        let instruction = match ty.inner {
            crate::TypeInner::Scalar { kind, width } => {
                self.lookup_type
                    .insert(LookupType::Local(LocalType::Scalar { kind, width }), id);
                self.write_scalar(id, kind, width)
            }
            crate::TypeInner::Vector { size, kind, width } => {
                let scalar_id =
                    self.get_type_id(arena, LookupType::Local(LocalType::Scalar { kind, width }));
                self.lookup_type.insert(
                    LookupType::Local(LocalType::Vector { size, kind, width }),
                    id,
                );
                super::instructions::instruction_type_vector(id, scalar_id, size)
            }
            crate::TypeInner::Matrix {
                columns,
                rows: _,
                width,
            } => {
                let vector_id = self.get_type_id(
                    arena,
                    LookupType::Local(LocalType::Vector {
                        size: columns,
                        kind: crate::ScalarKind::Float,
                        width,
                    }),
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
                let type_id = self.get_type_id(arena, LookupType::Local(local_type));
                let dim = map_dim(dim);
                self.try_add_capabilities(dim.required_capabilities());
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

                let type_id = self.get_type_id(arena, LookupType::Handle(base));
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
            crate::TypeInner::Struct { ref members } => {
                let mut member_ids = Vec::with_capacity(members.len());
                for member in members {
                    let member_id = self.get_type_id(arena, LookupType::Handle(member.ty));
                    member_ids.push(member_id);
                }
                super::instructions::instruction_type_struct(id, member_ids.as_slice())
            }
            crate::TypeInner::Pointer { base, class } => {
                let type_id = self.get_type_id(arena, LookupType::Handle(base));
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
        id
    }

    fn write_constant_type(
        &mut self,
        handle: crate::Handle<crate::Constant>,
        ir_module: &crate::Module,
    ) -> (Instruction, Word) {
        let id = self.generate_id();
        self.lookup_constant.insert(handle, id);
        let constant = &ir_module.constants[handle];
        let arena = &ir_module.types;

        match constant.inner {
            crate::ConstantInner::Sint(val) => {
                let ty = &ir_module.types[constant.ty];
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        4 => super::instructions::instruction_constant(type_id, id, &[val as u32]),
                        8 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            super::instructions::instruction_constant(type_id, id, &[low, high])
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };
                (instruction, id)
            }
            crate::ConstantInner::Uint(val) => {
                let ty = &ir_module.types[constant.ty];
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        4 => super::instructions::instruction_constant(type_id, id, &[val as u32]),
                        8 => {
                            let (low, high) = ((val >> 32) as u32, val as u32);
                            super::instructions::instruction_constant(type_id, id, &[low, high])
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };

                (instruction, id)
            }
            crate::ConstantInner::Float(val) => {
                let ty = &ir_module.types[constant.ty];
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = match ty.inner {
                    crate::TypeInner::Scalar { kind: _, width } => match width {
                        4 => super::instructions::instruction_constant(
                            type_id,
                            id,
                            &[(val as f32).to_bits()],
                        ),
                        8 => {
                            let bits = f64::to_bits(val);
                            let (low, high) = ((bits >> 32) as u32, bits as u32);
                            super::instructions::instruction_constant(type_id, id, &[low, high])
                        }
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };
                (instruction, id)
            }
            crate::ConstantInner::Bool(val) => {
                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));

                let instruction = if val {
                    super::instructions::instruction_constant_true(type_id, id)
                } else {
                    super::instructions::instruction_constant_false(type_id, id)
                };

                (instruction, id)
            }
            crate::ConstantInner::Composite(ref constituents) => {
                let mut constituent_ids = Vec::with_capacity(constituents.len());
                for constituent in constituents.iter() {
                    let constituent_id = self.get_constant_id(*constituent, &ir_module);
                    constituent_ids.push(constituent_id);
                }

                let type_id = self.get_type_id(arena, LookupType::Handle(constant.ty));
                let instruction = super::instructions::instruction_constant_composite(
                    type_id,
                    id,
                    constituent_ids.as_slice(),
                );
                (instruction, id)
            }
        }
    }

    fn write_global_variable(
        &mut self,
        ir_module: &crate::Module,
        handle: crate::Handle<crate::GlobalVariable>,
    ) -> (Instruction, Word) {
        let global_variable = &ir_module.global_variables[handle];
        let id = self.generate_id();

        let class = self.parse_to_spirv_storage_class(global_variable.class);
        self.try_add_capabilities(class.required_capabilities());

        let init_word = global_variable
            .init
            .map(|constant| self.get_constant_id(constant, ir_module));
        let pointer_id =
            self.get_pointer_id(&ir_module.types, global_variable.ty, global_variable.class);
        let instruction =
            super::instructions::instruction_variable(pointer_id, id, class, init_word);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            if let Some(ref name) = global_variable.name {
                self.debugs
                    .push(super::instructions::instruction_name(id, name.as_str()));
            }
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

        match *global_variable.binding.as_ref().unwrap() {
            crate::Binding::Location(location) => {
                self.annotations
                    .push(super::instructions::instruction_decorate(
                        id,
                        spirv::Decoration::Location,
                        &[location],
                    ));
            }
            crate::Binding::Resource { group, binding } => {
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
            crate::Binding::BuiltIn(built_in) => {
                let built_in = match built_in {
                    crate::BuiltIn::BaseInstance => spirv::BuiltIn::BaseInstance,
                    crate::BuiltIn::BaseVertex => spirv::BuiltIn::BaseVertex,
                    crate::BuiltIn::ClipDistance => spirv::BuiltIn::ClipDistance,
                    crate::BuiltIn::InstanceIndex => spirv::BuiltIn::InstanceIndex,
                    crate::BuiltIn::Position => spirv::BuiltIn::Position,
                    crate::BuiltIn::VertexIndex => spirv::BuiltIn::VertexIndex,
                    crate::BuiltIn::PointSize => spirv::BuiltIn::PointSize,
                    crate::BuiltIn::FragCoord => spirv::BuiltIn::FragCoord,
                    crate::BuiltIn::FrontFacing => spirv::BuiltIn::FrontFacing,
                    crate::BuiltIn::SampleIndex => spirv::BuiltIn::SampleId,
                    crate::BuiltIn::FragDepth => spirv::BuiltIn::FragDepth,
                    crate::BuiltIn::GlobalInvocationId => spirv::BuiltIn::GlobalInvocationId,
                    crate::BuiltIn::LocalInvocationId => spirv::BuiltIn::LocalInvocationId,
                    crate::BuiltIn::LocalInvocationIndex => spirv::BuiltIn::LocalInvocationIndex,
                    crate::BuiltIn::WorkGroupId => spirv::BuiltIn::WorkgroupId,
                };

                self.annotations
                    .push(super::instructions::instruction_decorate(
                        id,
                        spirv::Decoration::BuiltIn,
                        &[built_in as u32],
                    ));
            }
        }

        // TODO Initializer is optional and not (yet) included in the IR

        self.lookup_global_variable.insert(handle, id);
        (instruction, id)
    }

    fn get_function_type(
        &mut self,
        lookup_function_type: LookupFunctionType,
        parameter_pointer_ids: Vec<Word>,
    ) -> Word {
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
                    parameter_pointer_ids.as_slice(),
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

    fn get_type_inner<'a>(
        &self,
        ty_arena: &'a crate::Arena<crate::Type>,
        lookup_ty: LookupType,
    ) -> MaybeOwned<'a, crate::TypeInner> {
        match lookup_ty {
            LookupType::Handle(handle) => MaybeOwned::Borrowed(&ty_arena[handle].inner),
            LookupType::Local(local_ty) => match local_ty {
                LocalType::Scalar { kind, width } => {
                    MaybeOwned::Owned(crate::TypeInner::Scalar { kind, width })
                }
                LocalType::Vector { size, kind, width } => {
                    MaybeOwned::Owned(crate::TypeInner::Vector { size, kind, width })
                }
                LocalType::Pointer { base, class } => {
                    MaybeOwned::Owned(crate::TypeInner::Pointer { base, class })
                }
                _ => unreachable!(),
            },
        }
    }

    fn write_expression<'a>(
        &mut self,
        ir_module: &'a crate::Module,
        ir_function: &crate::Function,
        expression: &crate::Expression,
        block: &mut Block,
        function: &mut Function,
    ) -> Result<WriteExpressionOutput, Error> {
        match *expression {
            crate::Expression::Access { base, index } => {
                let id = self.generate_id();

                let (base_id, base_lookup_ty) = self.write_expression(
                    ir_module,
                    ir_function,
                    &ir_function.expressions[base],
                    block,
                    function,
                )?;
                let (index_id, _) = self.write_expression(
                    ir_module,
                    ir_function,
                    &ir_function.expressions[index],
                    block,
                    function,
                )?;

                let base_ty_inner = self.get_type_inner(&ir_module.types, base_lookup_ty);

                let (pointer_id, type_id, lookup_ty) = match *base_ty_inner {
                    crate::TypeInner::Vector { kind, width, .. } => {
                        let scalar_id = self.get_type_id(
                            &ir_module.types,
                            LookupType::Local(LocalType::Scalar { kind, width }),
                        );
                        (
                            self.create_pointer(scalar_id, spirv::StorageClass::Function),
                            scalar_id,
                            LookupType::Local(LocalType::Scalar { kind, width }),
                        )
                    }
                    _ => unimplemented!(),
                };

                block
                    .body
                    .push(super::instructions::instruction_access_chain(
                        pointer_id,
                        id,
                        base_id,
                        &[index_id],
                    ));

                let load_id = self.generate_id();
                block.body.push(super::instructions::instruction_load(
                    type_id, load_id, id, None,
                ));

                Ok((load_id, lookup_ty))
            }
            crate::Expression::AccessIndex { base, index } => {
                let id = self.generate_id();
                let (base_id, base_lookup_ty) = self
                    .write_expression(
                        ir_module,
                        ir_function,
                        &ir_function.expressions[base],
                        block,
                        function,
                    )
                    .unwrap();

                let base_ty_inner = self.get_type_inner(&ir_module.types, base_lookup_ty);

                let (pointer_id, type_id, lookup_ty) = match *base_ty_inner {
                    crate::TypeInner::Vector { kind, width, .. } => {
                        let scalar_id = self.get_type_id(
                            &ir_module.types,
                            LookupType::Local(LocalType::Scalar { kind, width }),
                        );
                        (
                            self.create_pointer(scalar_id, spirv::StorageClass::Function),
                            scalar_id,
                            LookupType::Local(LocalType::Scalar { kind, width }),
                        )
                    }
                    crate::TypeInner::Struct { ref members } => {
                        let member = &members[index as usize];
                        let type_id =
                            self.get_type_id(&ir_module.types, LookupType::Handle(member.ty));
                        (
                            self.create_pointer(type_id, spirv::StorageClass::Uniform),
                            type_id,
                            LookupType::Handle(member.ty),
                        )
                    }
                    _ => unimplemented!(),
                };

                let const_ty_id = self.get_type_id(
                    &ir_module.types,
                    LookupType::Local(LocalType::Scalar {
                        kind: crate::ScalarKind::Sint,
                        width: 4,
                    }),
                );
                let const_id = self.create_constant(const_ty_id, &[index]);

                block
                    .body
                    .push(super::instructions::instruction_access_chain(
                        pointer_id,
                        id,
                        base_id,
                        &[const_id],
                    ));

                let load_id = self.generate_id();
                block.body.push(super::instructions::instruction_load(
                    type_id, load_id, id, None,
                ));

                Ok((load_id, lookup_ty))
            }
            crate::Expression::GlobalVariable(handle) => {
                let var = &ir_module.global_variables[handle];
                let id = self.get_global_variable_id(&ir_module, handle);

                Ok((id, LookupType::Handle(var.ty)))
            }
            crate::Expression::Constant(handle) => {
                let var = &ir_module.constants[handle];
                let id = self.get_constant_id(handle, ir_module);
                Ok((id, LookupType::Handle(var.ty)))
            }
            crate::Expression::Compose { ty, ref components } => {
                let base_type_id = self.get_type_id(&ir_module.types, LookupType::Handle(ty));

                let mut constituent_ids = Vec::with_capacity(components.len());
                for component in components {
                    let expression = &ir_function.expressions[*component];
                    let (component_id, component_local_ty) = self.write_expression(
                        ir_module,
                        &ir_function,
                        expression,
                        block,
                        function,
                    )?;

                    let component_id = match expression {
                        crate::Expression::LocalVariable(_)
                        | crate::Expression::GlobalVariable(_) => {
                            let load_id = self.generate_id();
                            block.body.push(super::instructions::instruction_load(
                                self.get_type_id(&ir_module.types, component_local_ty),
                                load_id,
                                component_id,
                                None,
                            ));
                            load_id
                        }
                        _ => component_id,
                    };

                    constituent_ids.push(component_id);
                }
                let constituent_ids_slice = constituent_ids.as_slice();

                let id = match ir_module.types[ty].inner {
                    crate::TypeInner::Vector { .. } => {
                        self.write_composite_construct(base_type_id, constituent_ids_slice, block)
                    }
                    crate::TypeInner::Matrix {
                        rows,
                        columns,
                        width,
                    } => {
                        let vector_type_id = self.get_type_id(
                            &ir_module.types,
                            LookupType::Local(LocalType::Vector {
                                width,
                                kind: crate::ScalarKind::Float,
                                size: columns,
                            }),
                        );

                        let capacity = match rows {
                            crate::VectorSize::Bi => 2,
                            crate::VectorSize::Tri => 3,
                            crate::VectorSize::Quad => 4,
                        };

                        let mut vector_ids = Vec::with_capacity(capacity);

                        for _ in 0..capacity {
                            let vector_id = self.write_composite_construct(
                                vector_type_id,
                                constituent_ids_slice,
                                block,
                            );
                            vector_ids.push(vector_id);
                        }

                        self.write_composite_construct(base_type_id, vector_ids.as_slice(), block)
                    }
                    _ => unreachable!(),
                };

                Ok((id, LookupType::Handle(ty)))
            }
            crate::Expression::Binary { op, left, right } => {
                let id = self.generate_id();
                let left_expression = &ir_function.expressions[left];
                let right_expression = &ir_function.expressions[right];
                let (left_id, left_lookup_ty) = self.write_expression(
                    ir_module,
                    ir_function,
                    left_expression,
                    block,
                    function,
                )?;
                let (right_id, right_lookup_ty) = self.write_expression(
                    ir_module,
                    ir_function,
                    right_expression,
                    block,
                    function,
                )?;

                let left_lookup_ty = left_lookup_ty;
                let right_lookup_ty = right_lookup_ty;

                let left_ty_inner = self.get_type_inner(&ir_module.types, left_lookup_ty);
                let right_ty_inner = self.get_type_inner(&ir_module.types, right_lookup_ty);

                let left_result_type_id = self.get_type_id(&ir_module.types, left_lookup_ty);

                let right_result_type_id = self.get_type_id(&ir_module.types, right_lookup_ty);

                let left_id = match *left_expression {
                    crate::Expression::LocalVariable(_) | crate::Expression::GlobalVariable(_) => {
                        let load_id = self.generate_id();
                        block.body.push(super::instructions::instruction_load(
                            left_result_type_id,
                            load_id,
                            left_id,
                            None,
                        ));
                        load_id
                    }
                    _ => left_id,
                };

                let right_id = match *right_expression {
                    crate::Expression::LocalVariable(..)
                    | crate::Expression::GlobalVariable(..) => {
                        let load_id = self.generate_id();
                        block.body.push(super::instructions::instruction_load(
                            right_result_type_id,
                            load_id,
                            right_id,
                            None,
                        ));
                        load_id
                    }
                    _ => right_id,
                };

                let left_dimension = get_dimension(&left_ty_inner);
                let right_dimension = get_dimension(&right_ty_inner);

                let (instruction, lookup_ty) = match op {
                    crate::BinaryOperator::Multiply => match (left_dimension, right_dimension) {
                        (Dimension::Vector, Dimension::Scalar { .. }) => (
                            super::instructions::instruction_vector_times_scalar(
                                left_result_type_id,
                                id,
                                left_id,
                                right_id,
                            ),
                            left_lookup_ty,
                        ),
                        (Dimension::Vector, Dimension::Matrix) => (
                            super::instructions::instruction_vector_times_matrix(
                                left_result_type_id,
                                id,
                                left_id,
                                right_id,
                            ),
                            left_lookup_ty,
                        ),
                        (Dimension::Matrix, Dimension::Scalar { .. }) => (
                            super::instructions::instruction_matrix_times_scalar(
                                left_result_type_id,
                                id,
                                left_id,
                                right_id,
                            ),
                            left_lookup_ty,
                        ),
                        (Dimension::Matrix, Dimension::Vector) => (
                            super::instructions::instruction_matrix_times_vector(
                                right_result_type_id,
                                id,
                                left_id,
                                right_id,
                            ),
                            right_lookup_ty,
                        ),
                        (Dimension::Matrix, Dimension::Matrix) => (
                            super::instructions::instruction_matrix_times_matrix(
                                left_result_type_id,
                                id,
                                left_id,
                                right_id,
                            ),
                            left_lookup_ty,
                        ),
                        (Dimension::Vector, Dimension::Vector)
                        | (Dimension::Scalar, Dimension::Scalar)
                            if left_ty_inner.scalar_kind() == Some(crate::ScalarKind::Float) =>
                        {
                            (
                                super::instructions::instruction_f_mul(
                                    left_result_type_id,
                                    id,
                                    left_id,
                                    right_id,
                                ),
                                left_lookup_ty,
                            )
                        }
                        (Dimension::Vector, Dimension::Vector)
                        | (Dimension::Scalar, Dimension::Scalar) => (
                            super::instructions::instruction_i_mul(
                                left_result_type_id,
                                id,
                                left_id,
                                right_id,
                            ),
                            left_lookup_ty,
                        ),
                        _ => unreachable!(),
                    },
                    crate::BinaryOperator::Subtract => match *left_ty_inner {
                        crate::TypeInner::Scalar { kind, .. } => match kind {
                            crate::ScalarKind::Sint | crate::ScalarKind::Uint => (
                                super::instructions::instruction_i_sub(
                                    left_result_type_id,
                                    id,
                                    left_id,
                                    right_id,
                                ),
                                left_lookup_ty,
                            ),
                            crate::ScalarKind::Float => (
                                super::instructions::instruction_f_sub(
                                    left_result_type_id,
                                    id,
                                    left_id,
                                    right_id,
                                ),
                                left_lookup_ty,
                            ),
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    },
                    crate::BinaryOperator::And => (
                        super::instructions::instruction_bitwise_and(
                            left_result_type_id,
                            id,
                            left_id,
                            right_id,
                        ),
                        left_lookup_ty,
                    ),
                    _ => unimplemented!("{:?}", op),
                };

                block.body.push(instruction);
                Ok((id, lookup_ty))
            }
            crate::Expression::LocalVariable(variable) => {
                let var = &ir_function.local_variables[variable];
                function
                    .variables
                    .iter()
                    .find(|&v| v.name.as_ref().unwrap() == var.name.as_ref().unwrap())
                    .map(|local_var| (local_var.id, LookupType::Handle(var.ty)))
                    .ok_or_else(|| Error::UnknownLocalVariable(var.clone()))
            }
            crate::Expression::FunctionArgument(index) => {
                let handle = ir_function.arguments[index as usize].ty;
                let type_id = self.get_type_id(&ir_module.types, LookupType::Handle(handle));
                let load_id = self.generate_id();

                block.body.push(super::instructions::instruction_load(
                    type_id,
                    load_id,
                    function.parameters[index as usize].result_id.unwrap(),
                    None,
                ));
                Ok((load_id, LookupType::Handle(handle)))
            }
            crate::Expression::Call {
                ref origin,
                ref arguments,
            } => match *origin {
                crate::FunctionOrigin::Local(local_function) => {
                    let origin_function = &ir_module.functions[local_function];
                    let id = self.generate_id();
                    let mut argument_ids = vec![];

                    for argument in arguments {
                        let expression = &ir_function.expressions[*argument];
                        let (id, lookup_ty) = self.write_expression(
                            ir_module,
                            ir_function,
                            expression,
                            block,
                            function,
                        )?;

                        // Create variable - OpVariable
                        // Store value to variable - OpStore
                        // Use id of variable

                        let handle = match lookup_ty {
                            LookupType::Handle(handle) => handle,
                            LookupType::Local(_) => unreachable!(),
                        };

                        let pointer_id = self.get_pointer_id(
                            &ir_module.types,
                            handle,
                            crate::StorageClass::Function,
                        );

                        let variable_id = self.generate_id();
                        function.variables.push(LocalVariable {
                            id: variable_id,
                            name: None,
                            instruction: super::instructions::instruction_variable(
                                pointer_id,
                                variable_id,
                                spirv::StorageClass::Function,
                                None,
                            ),
                        });
                        block.body.push(super::instructions::instruction_store(
                            variable_id,
                            id,
                            None,
                        ));
                        argument_ids.push(variable_id);
                    }

                    let return_type_id = self
                        .get_function_return_type(origin_function.return_type, &ir_module.types);

                    block
                        .body
                        .push(super::instructions::instruction_function_call(
                            return_type_id,
                            id,
                            *self.lookup_function.get(&local_function).unwrap(),
                            argument_ids.as_slice(),
                        ));

                    let result_type = match origin_function.return_type {
                        Some(ty_handle) => LookupType::Handle(ty_handle),
                        None => LookupType::Local(LocalType::Void),
                    };
                    Ok((id, result_type))
                }
                _ => unimplemented!("{:?}", origin),
            },
            crate::Expression::As {
                expr,
                kind,
                convert,
            } => {
                if !convert {
                    return Err(Error::FeatureNotImplemented());
                }

                let (expr_id, expr_type) = self.write_expression(
                    ir_module,
                    ir_function,
                    &ir_function.expressions[expr],
                    block,
                    function,
                )?;

                let expr_type_inner = self.get_type_inner(&ir_module.types, expr_type);

                let (expr_kind, local_type) = match *expr_type_inner {
                    crate::TypeInner::Scalar {
                        kind: expr_kind,
                        width,
                    } => (expr_kind, LocalType::Scalar { kind, width }),
                    crate::TypeInner::Vector {
                        size,
                        kind: expr_kind,
                        width,
                    } => (expr_kind, LocalType::Vector { size, kind, width }),
                    _ => unreachable!(),
                };

                let lookup_type = LookupType::Local(local_type);
                let op = match (expr_kind, kind) {
                    _ if !convert => spirv::Op::Bitcast,
                    (crate::ScalarKind::Float, crate::ScalarKind::Uint) => spirv::Op::ConvertFToU,
                    (crate::ScalarKind::Float, crate::ScalarKind::Sint) => spirv::Op::ConvertFToS,
                    (crate::ScalarKind::Sint, crate::ScalarKind::Float) => spirv::Op::ConvertSToF,
                    (crate::ScalarKind::Uint, crate::ScalarKind::Float) => spirv::Op::ConvertUToF,
                    // We assume it's either an identity cast, or int-uint.
                    // In both cases no SPIR-V instructions need to be generated.
                    _ => {
                        let id = match ir_function.expressions[expr] {
                            crate::Expression::LocalVariable(_)
                            | crate::Expression::GlobalVariable(_) => {
                                let load_id = self.generate_id();
                                let kind_type_id = self.get_type_id(&ir_module.types, expr_type);
                                block.body.push(super::instructions::instruction_load(
                                    kind_type_id,
                                    load_id,
                                    expr_id,
                                    None,
                                ));
                                load_id
                            }
                            _ => expr_id,
                        };
                        return Ok((id, lookup_type));
                    }
                };

                let id = self.generate_id();
                let kind_type_id = self.get_type_id(&ir_module.types, lookup_type);
                let instruction =
                    super::instructions::instruction_unary(op, kind_type_id, id, expr_id);
                block.body.push(instruction);

                Ok((id, lookup_type))
            }
            crate::Expression::ImageSample {
                image,
                sampler,
                coordinate,
                level: _,
                depth_ref: _,
            } => {
                // image
                let image_expression = &ir_function.expressions[image];
                let (image_id, image_lookup_ty) = self.write_expression(
                    ir_module,
                    ir_function,
                    image_expression,
                    block,
                    function,
                )?;

                let image_result_type_id = self.get_type_id(&ir_module.types, image_lookup_ty);
                let image_id = match *image_expression {
                    crate::Expression::LocalVariable(_) | crate::Expression::GlobalVariable(_) => {
                        let load_id = self.generate_id();
                        block.body.push(super::instructions::instruction_load(
                            image_result_type_id,
                            load_id,
                            image_id,
                            None,
                        ));
                        load_id
                    }
                    _ => image_id,
                };

                let image_ty = match image_lookup_ty {
                    LookupType::Handle(handle) => handle,
                    LookupType::Local(_) => unreachable!(),
                };

                // OpTypeSampledImage
                let sampled_image_type_id = self.get_type_id(
                    &ir_module.types,
                    LookupType::Local(LocalType::SampledImage {
                        image_type: image_ty,
                    }),
                );

                // sampler
                let sampler_expression = &ir_function.expressions[sampler];
                let (sampler_id, sampler_lookup_ty) = self.write_expression(
                    ir_module,
                    ir_function,
                    sampler_expression,
                    block,
                    function,
                )?;

                let sampler_result_type_id = self.get_type_id(&ir_module.types, sampler_lookup_ty);
                let sampler_id = match *sampler_expression {
                    crate::Expression::LocalVariable(_) | crate::Expression::GlobalVariable(_) => {
                        let load_id = self.generate_id();
                        block.body.push(super::instructions::instruction_load(
                            sampler_result_type_id,
                            load_id,
                            sampler_id,
                            None,
                        ));
                        load_id
                    }
                    _ => sampler_id,
                };

                // coordinate
                let coordinate_expression = &ir_function.expressions[coordinate];
                let (coordinate_id, coordinate_lookup_ty) = self.write_expression(
                    ir_module,
                    ir_function,
                    coordinate_expression,
                    block,
                    function,
                )?;

                let coordinate_result_type_id =
                    self.get_type_id(&ir_module.types, coordinate_lookup_ty);
                let coordinate_id = match *coordinate_expression {
                    crate::Expression::LocalVariable(_) | crate::Expression::GlobalVariable(_) => {
                        let load_id = self.generate_id();
                        block.body.push(super::instructions::instruction_load(
                            coordinate_result_type_id,
                            load_id,
                            coordinate_id,
                            None,
                        ));
                        load_id
                    }
                    _ => coordinate_id,
                };

                // component kind
                let image_type = &ir_module.types[image_ty];
                let image_sample_result_type =
                    if let crate::TypeInner::Image { class, .. } = image_type.inner {
                        let width = 4;
                        LookupType::Local(match class {
                            crate::ImageClass::Sampled { kind, multi: _ } => LocalType::Vector {
                                kind,
                                width,
                                size: crate::VectorSize::Quad,
                            },
                            crate::ImageClass::Depth => LocalType::Scalar {
                                kind: crate::ScalarKind::Float,
                                width,
                            },
                            _ => return Err(Error::BadImageClass(class)),
                        })
                    } else {
                        return Err(Error::NotImage);
                    };

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
                let image_sample_result_type_id =
                    self.get_type_id(&ir_module.types, image_sample_result_type);
                block
                    .body
                    .push(super::instructions::instruction_image_sample_implicit_lod(
                        image_sample_result_type_id,
                        id,
                        sampled_image_id,
                        coordinate_id,
                    ));
                Ok((id, image_sample_result_type))
            }
            _ => unimplemented!("{:?}", expression),
        }
    }

    fn write_block(
        &mut self,
        statements: &[crate::Statement],
        ir_module: &crate::Module,
        ir_function: &crate::Function,
        function: &mut Function,
    ) -> spirv::Word {
        let mut block = Block::new();
        let id = self.generate_id();
        block.label = Some(super::instructions::instruction_label(id));

        for statement in statements {
            match *statement {
                crate::Statement::Block(ref ir_block) => {
                    if !ir_block.is_empty() {
                        //TODO: link the block with `OpBranch`
                        self.write_block(ir_block, ir_module, ir_function, function);
                    }
                }
                crate::Statement::Return { value } => {
                    block.termination = Some(match ir_function.return_type {
                        Some(_) => {
                            let expression = &ir_function.expressions[value.unwrap()];
                            let (id, lookup_ty) = self
                                .write_expression(
                                    ir_module,
                                    ir_function,
                                    expression,
                                    &mut block,
                                    function,
                                )
                                .unwrap();

                            let id = match *expression {
                                crate::Expression::LocalVariable(_)
                                | crate::Expression::GlobalVariable(_) => {
                                    let load_id = self.generate_id();
                                    let value_ty_id = self.get_type_id(&ir_module.types, lookup_ty);
                                    block.body.push(super::instructions::instruction_load(
                                        value_ty_id,
                                        load_id,
                                        id,
                                        None,
                                    ));
                                    load_id
                                }

                                _ => id,
                            };
                            super::instructions::instruction_return_value(id)
                        }
                        None => super::instructions::instruction_return(),
                    });
                }
                crate::Statement::Store { pointer, value } => {
                    let pointer_expression = &ir_function.expressions[pointer];
                    let value_expression = &ir_function.expressions[value];
                    let (pointer_id, _) = self
                        .write_expression(
                            ir_module,
                            ir_function,
                            pointer_expression,
                            &mut block,
                            function,
                        )
                        .unwrap();
                    let (value_id, value_lookup_ty) = self
                        .write_expression(
                            ir_module,
                            ir_function,
                            value_expression,
                            &mut block,
                            function,
                        )
                        .unwrap();

                    let value_id = match value_expression {
                        crate::Expression::LocalVariable(_)
                        | crate::Expression::GlobalVariable(_) => {
                            let load_id = self.generate_id();
                            let value_ty_id = self.get_type_id(&ir_module.types, value_lookup_ty);
                            block.body.push(super::instructions::instruction_load(
                                value_ty_id,
                                load_id,
                                value_id,
                                None,
                            ));
                            load_id
                        }
                        _ => value_id,
                    };

                    block.body.push(super::instructions::instruction_store(
                        pointer_id, value_id, None,
                    ));
                }
                _ => unimplemented!("{:?}", statement),
            }
        }

        function.blocks.push(block);
        id
    }

    fn write_physical_layout(&mut self) {
        self.physical_layout.bound = self.id_count + 1;
    }

    fn write_logical_layout(&mut self, ir_module: &crate::Module) {
        let id = self.generate_id();
        super::instructions::instruction_ext_inst_import(id, "GLSL.std.450")
            .to_words(&mut self.logical_layout.ext_inst_imports);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            self.debugs.push(super::instructions::instruction_source(
                spirv::SourceLanguage::GLSL,
                450,
            ));
        }

        for (handle, ir_function) in ir_module.functions.iter() {
            let id = self.write_function(ir_function, ir_module);
            self.lookup_function.insert(handle, id);
        }

        for (&(stage, ref name), ir_ep) in ir_module.entry_points.iter() {
            let entry_point_instruction = self.write_entry_point(ir_ep, stage, name, ir_module);
            entry_point_instruction.to_words(&mut self.logical_layout.entry_points);
        }

        for capability in self.capabilities.iter() {
            super::instructions::instruction_capability(*capability)
                .to_words(&mut self.logical_layout.capabilities);
        }

        let addressing_model = spirv::AddressingModel::Logical;
        let memory_model = spirv::MemoryModel::GLSL450;
        self.try_add_capabilities(addressing_model.required_capabilities());
        self.try_add_capabilities(memory_model.required_capabilities());

        super::instructions::instruction_memory_model(addressing_model, memory_model)
            .to_words(&mut self.logical_layout.memory_model);

        if self.writer_flags.contains(WriterFlags::DEBUG) {
            for debug in self.debugs.iter() {
                debug.to_words(&mut self.logical_layout.debugs);
            }
        }

        for annotation in self.annotations.iter() {
            annotation.to_words(&mut self.logical_layout.annotations);
        }
    }

    pub fn write(&mut self, ir_module: &crate::Module) -> Vec<Word> {
        let mut words: Vec<Word> = vec![];

        self.write_logical_layout(ir_module);
        self.write_physical_layout();

        self.physical_layout.in_words(&mut words);
        self.logical_layout.in_words(&mut words);
        words
    }
}

#[cfg(test)]
mod tests {
    use crate::back::spv::{Writer, WriterFlags};
    use crate::Header;

    #[test]
    fn test_writer_generate_id() {
        let mut writer = create_writer();

        assert_eq!(writer.id_count, 0);
        writer.generate_id();
        assert_eq!(writer.id_count, 1);
    }

    #[test]
    fn test_try_add_capabilities() {
        let mut writer = create_writer();

        assert_eq!(writer.capabilities.len(), 0);
        writer.try_add_capabilities(&[spirv::Capability::Shader]);
        assert_eq!(writer.capabilities.len(), 1);

        writer.try_add_capabilities(&[spirv::Capability::Shader]);
        assert_eq!(writer.capabilities.len(), 1);
    }

    #[test]
    fn test_write_physical_layout() {
        let mut writer = create_writer();
        assert_eq!(writer.physical_layout.bound, 0);
        writer.write_physical_layout();
        assert_eq!(writer.physical_layout.bound, 1);
    }

    fn create_writer() -> Writer {
        let header = Header {
            generator: 0,
            version: (1, 0, 0),
        };
        Writer::new(&header, WriterFlags::NONE)
    }
}
