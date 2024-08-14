use std::{sync::Arc, thread};

use crate::context::ObjectId;
use crate::*;

/// Handle to a query set.
///
/// It can be created with [`Device::create_query_set`].
///
/// Corresponds to [WebGPU `GPUQuerySet`](https://gpuweb.github.io/gpuweb/#queryset).
#[derive(Debug)]
pub struct QuerySet {
    pub(crate) context: Arc<C>,
    pub(crate) id: ObjectId,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
#[cfg(send_sync)]
static_assertions::assert_impl_all!(QuerySet: Send, Sync);

impl QuerySet {
    /// Returns a globally-unique identifier for this `QuerySet`.
    ///
    /// Calling this method multiple times on the same object will always return the same value.
    /// The returned value is guaranteed to be different for all resources created from the same `Instance`.
    pub fn global_id(&self) -> Id<Self> {
        Id::new(self.id)
    }
}

impl Drop for QuerySet {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.query_set_drop(&self.id, self.data.as_ref());
        }
    }
}

/// Describes a [`QuerySet`].
///
/// For use with [`Device::create_query_set`].
///
/// Corresponds to [WebGPU `GPUQuerySetDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuquerysetdescriptor).
pub type QuerySetDescriptor<'a> = wgt::QuerySetDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(QuerySetDescriptor<'_>: Send, Sync);
