use crate::*;

/// Handle to a query set.
///
/// It can be created with [`Device::create_query_set`].
///
/// Corresponds to [WebGPU `GPUQuerySet`](https://gpuweb.github.io/gpuweb/#queryset).
#[derive(Debug)]
pub struct QuerySet {
    pub(crate) inner: dispatch::DispatchQuerySet,
}
#[cfg(send_sync)]
#[cfg(send_sync)]
static_assertions::assert_impl_all!(QuerySet: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(QuerySet => .inner);

/// Describes a [`QuerySet`].
///
/// For use with [`Device::create_query_set`].
///
/// Corresponds to [WebGPU `GPUQuerySetDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuquerysetdescriptor).
pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(QuerySetDescriptor<'_>: Send, Sync);
