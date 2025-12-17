use crate::arena::Handle;

#[derive(Clone, Debug, thiserror::Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum ZeroValueError {
    #[error("ZeroValue construction of dynamically-sized array is not allowed")]
    DynamicallySizedArray,
}

pub fn validate_zero_value(
    self_ty_handle: Handle<crate::Type>,
    gctx: crate::proc::GlobalCtx,
) -> Result<(), ZeroValueError> {
    use crate::TypeInner as Ti;
    match gctx.types[self_ty_handle].inner {
        Ti::Array {
            base: _,
            size: crate::ArraySize::Dynamic,
            stride: _,
        } => {
            log::error!("Constructing zero value of dynamically-sized array");
            Err(ZeroValueError::DynamicallySizedArray)
        }
        _ => Ok(()),
    }
}
