use std::sync::Arc;

pub mod command;
pub mod device;

type Global = core::hub::Global<parking_lot::Mutex<core::hub::IdentityManager>>;

lazy_static::lazy_static! {
    static ref GLOBAL: Arc<Global> = Arc::new(Global::new("wgpu"));
}
