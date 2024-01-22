use super::ModuleState;
use crate::arena::Handle;
use codespan_reporting::diagnostic::Diagnostic;
use codespan_reporting::files::SimpleFile;
use codespan_reporting::term;
use termcolor::{NoColor, WriteColor};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid header")]
    InvalidHeader,
    #[error("invalid word count")]
    InvalidWordCount,
    #[error("unknown instruction {0}")]
    UnknownInstruction(u16),
    #[error("unknown capability %{0}")]
    UnknownCapability(spirv::Word),
    #[error("unsupported instruction {1:?} at {0:?}")]
    UnsupportedInstruction(ModuleState, spirv::Op),
    #[error("unsupported capability {0:?}")]
    UnsupportedCapability(spirv::Capability),
    #[error("unsupported extension {0}")]
    UnsupportedExtension(String),
    #[error("unsupported extension set {0}")]
    UnsupportedExtSet(String),
    #[error("unsupported extension instantiation set %{0}")]
    UnsupportedExtInstSet(spirv::Word),
    #[error("unsupported extension instantiation %{0}")]
    UnsupportedExtInst(spirv::Word),
    #[error("unsupported type {0:?}")]
    UnsupportedType(Handle<crate::Type>),
    #[error("unsupported execution model %{0}")]
    UnsupportedExecutionModel(spirv::Word),
    #[error("unsupported execution mode %{0}")]
    UnsupportedExecutionMode(spirv::Word),
    #[error("unsupported storage class %{0}")]
    UnsupportedStorageClass(spirv::Word),
    #[error("unsupported image dimension %{0}")]
    UnsupportedImageDim(spirv::Word),
    #[error("unsupported image format %{0}")]
    UnsupportedImageFormat(spirv::Word),
    #[error("unsupported builtin %{0}")]
    UnsupportedBuiltIn(spirv::Word),
    #[error("unsupported control flow %{0}")]
    UnsupportedControlFlow(spirv::Word),
    #[error("unsupported binary operator %{0}")]
    UnsupportedBinaryOperator(spirv::Word),
    #[error("Naga supports OpTypeRuntimeArray in the StorageBuffer storage class only")]
    UnsupportedRuntimeArrayStorageClass,
    #[error("unsupported matrix stride {stride} for a {columns}x{rows} matrix with scalar width={width}")]
    UnsupportedMatrixStride {
        stride: u32,
        columns: u8,
        rows: u8,
        width: u8,
    },
    #[error("unknown binary operator {0:?}")]
    UnknownBinaryOperator(spirv::Op),
    #[error("unknown relational function {0:?}")]
    UnknownRelationalFunction(spirv::Op),
    #[error("invalid parameter {0:?}")]
    InvalidParameter(spirv::Op),
    #[error("invalid operand count {1} for {0:?}")]
    InvalidOperandCount(spirv::Op, u16),
    #[error("invalid operand")]
    InvalidOperand,
    #[error("invalid id %{0}")]
    InvalidId(spirv::Word),
    #[error("invalid decoration %{0}")]
    InvalidDecoration(spirv::Word),
    #[error("invalid type width %{0}")]
    InvalidTypeWidth(spirv::Word),
    #[error("invalid sign %{0}")]
    InvalidSign(spirv::Word),
    #[error("invalid inner type %{0}")]
    InvalidInnerType(spirv::Word),
    #[error("invalid vector size %{0}")]
    InvalidVectorSize(spirv::Word),
    #[error("invalid access type %{0}")]
    InvalidAccessType(spirv::Word),
    #[error("invalid access {0:?}")]
    InvalidAccess(crate::Expression),
    #[error("invalid access index %{0}")]
    InvalidAccessIndex(spirv::Word),
    #[error("invalid index type %{0}")]
    InvalidIndexType(spirv::Word),
    #[error("invalid binding %{0}")]
    InvalidBinding(spirv::Word),
    #[error("invalid global var {0:?}")]
    InvalidGlobalVar(crate::Expression),
    #[error("invalid image/sampler expression {0:?}")]
    InvalidImageExpression(crate::Expression),
    #[error("invalid image base type {0:?}")]
    InvalidImageBaseType(Handle<crate::Type>),
    #[error("invalid image {0:?}")]
    InvalidImage(Handle<crate::Type>),
    #[error("invalid as type {0:?}")]
    InvalidAsType(Handle<crate::Type>),
    #[error("invalid vector type {0:?}")]
    InvalidVectorType(Handle<crate::Type>),
    #[error("inconsistent comparison sampling {0:?}")]
    InconsistentComparisonSampling(Handle<crate::GlobalVariable>),
    #[error("wrong function result type %{0}")]
    WrongFunctionResultType(spirv::Word),
    #[error("wrong function argument type %{0}")]
    WrongFunctionArgumentType(spirv::Word),
    #[error("missing decoration {0:?}")]
    MissingDecoration(spirv::Decoration),
    #[error("bad string")]
    BadString,
    #[error("incomplete data")]
    IncompleteData,
    #[error("invalid terminator")]
    InvalidTerminator,
    #[error("invalid edge classification")]
    InvalidEdgeClassification,
    #[error("cycle detected in the CFG during traversal at {0}")]
    ControlFlowGraphCycle(crate::front::spv::BlockId),
    #[error("recursive function call %{0}")]
    FunctionCallCycle(spirv::Word),
    #[error("invalid array size {0:?}")]
    InvalidArraySize(Handle<crate::Constant>),
    #[error("invalid barrier scope %{0}")]
    InvalidBarrierScope(spirv::Word),
    #[error("invalid barrier memory semantics %{0}")]
    InvalidBarrierMemorySemantics(spirv::Word),
    #[error(
        "arrays of images / samplers are supported only through bindings for \
         now (i.e. you can't create an array of images or samplers that doesn't \
         come from a binding)"
    )]
    NonBindingArrayOfImageOrSamplers,
}

impl Error {
    pub fn emit_to_writer(&self, writer: &mut impl WriteColor, source: &str) {
        self.emit_to_writer_with_path(writer, source, "glsl");
    }

    pub fn emit_to_writer_with_path(&self, writer: &mut impl WriteColor, source: &str, path: &str) {
        let path = path.to_string();
        let files = SimpleFile::new(path, source);
        let config = codespan_reporting::term::Config::default();
        let diagnostic = Diagnostic::error().with_message(format!("{self:?}"));

        term::emit(writer, &config, &files, &diagnostic).expect("cannot write error");
    }

    pub fn emit_to_string(&self, source: &str) -> String {
        let mut writer = NoColor::new(Vec::new());
        self.emit_to_writer(&mut writer, source);
        String::from_utf8(writer.into_inner()).unwrap()
    }
}
