use wgt::WasmNotSendSync;

// TODO: docs
pub trait DynResource: WasmNotSendSync + std::fmt::Debug + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

// TODO: actually use this one
pub trait DynBuffer: DynResource {}
