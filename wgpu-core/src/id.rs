use core::{borrow::Borrow, mem::ManuallyDrop, ops::Deref, ptr::NonNull};
use crate::{Epoch, Index};
use std::{cmp::Ordering, fmt, marker::PhantomData, num::NonZeroU64, sync::Arc};
use wgt::Backend;

const BACKEND_BITS: usize = 3;
pub const BACKEND_MASK: usize = (1 << BACKEND_BITS) - 1;
const EPOCH_MASK: u32 = (1 << (32 - BACKEND_BITS)) - 1;
pub type Dummy = hal::api::Empty;

#[repr(transparent)]
#[cfg_attr(feature = "trace", derive(serde::Serialize), serde(into = "SerialId"))]
#[cfg_attr(
    feature = "replay",
    derive(serde::Deserialize),
    serde(from = "SerialId")
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "trace")),
    derive(serde::Serialize)
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "replay")),
    derive(serde::Deserialize)
)]
pub struct Id<T>(NonZeroU64, PhantomData<T>);

// This type represents Id in a more readable (and editable) way.
#[allow(dead_code)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
enum SerialId {
    // The only variant forces RON to not ignore "Id"
    Id(Index, Epoch, Backend),
}
#[cfg(feature = "trace")]
impl<T> From<Id<T>> for SerialId {
    fn from(id: Id<T>) -> Self {
        let (index, epoch, backend) = id.unzip();
        Self::Id(index, epoch, backend)
    }
}
#[cfg(feature = "replay")]
impl<T> From<SerialId> for Id<T> {
    fn from(id: SerialId) -> Self {
        match id {
            SerialId::Id(index, epoch, backend) => TypedId::zip(index, epoch, backend),
        }
    }
}

impl<T> Id<T> {
    #[cfg(test)]
    pub(crate) fn dummy() -> Valid<Self> {
        Valid(Id(NonZeroU64::new(1).unwrap(), PhantomData))
    }

    pub fn backend(self) -> Backend {
        match self.0.get() >> (64 - BACKEND_BITS) as u8 {
            0 => Backend::Empty,
            1 => Backend::Vulkan,
            2 => Backend::Metal,
            3 => Backend::Dx12,
            4 => Backend::Dx11,
            5 => Backend::Gl,
            _ => unreachable!(),
        }
    }
}

impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.unzip().fmt(formatter)
    }
}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

/// An internal ID that has been checked to point to
/// a valid object in the storages.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[cfg_attr(feature = "trace", derive(serde::Serialize))]
#[cfg_attr(feature = "replay", derive(serde::Deserialize))]
pub(crate) struct Valid<I>(pub I);

#[repr(transparent)]
/* #[cfg_attr(feature = "trace", derive(serde_state::Serialize))]
#[cfg_attr(
    feature = "replay",
    derive(serde_state::Deserialize),
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "trace")),
    derive(serde_state::Serialize)
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "replay")),
    derive(serde_state::Deserialize)
)] */
pub struct Id2<T: AllBackends> {
    ptr: NonNull<T>,
}

/// FIXME: Make sure all versions with all backends are Send + Sync.
unsafe impl<T: AllBackends + Send> Send for Id2<T> {}
unsafe impl<T: AllBackends + Sync> Sync for Id2<T> {}

impl<T: AllBackends> PartialEq for Id2<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T: AllBackends> Eq for Id2<T> {}

impl<T: AllBackends> std::hash::Hash for Id2<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

impl<T: AllBackends> fmt::Debug for Id2<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.ptr.fmt(formatter)
    }
}

impl<T: AllBackends> Clone for Id2<T> {
    #[inline]
    fn clone(&self) -> Id2<T> {
        unsafe {
            // Pretend we are cloning an Arc.
            // Safety: Id2<T> can only be constructed from Arc<T> plus some optional backend bits,
            // so we can perform raw pointer Arc operations on the raw pointer once the backend
            // bits are removed; additionally, by making sure we increment the reference count
            // here, we ensure that the pointer we return lives until Id2<T> is dropped.
            let ptr = self.ptr;
            let raw_ptr : *const T = ((ptr.as_ptr() as usize) & !BACKEND_MASK) as _;
            Arc::increment_strong_count(raw_ptr);
            Id2 { ptr }
        }
    }
}

impl<T: AllBackends> Drop for Id2<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // We can learn our backend and proceed to drop.
            //
            // Safety: constructing an Id2 guarantees that either a new backend was chosen (in the
            // BACKEND_MASK bits), or we can just drop T as Arc<T>.  ptr is never accessed after
            // the Id2 containing it is dropped.
            let ptr = ((self.ptr.as_ptr() as usize) & !BACKEND_MASK) as *const T;
            match self.backend() {
                // If there are no mask bits, this is the original backend.
                Backend::Empty => Arc::decrement_strong_count(ptr),
                // Otherwise, we can cast the masked pointer to the appropriate type.
                #[cfg(vulkan)]
                Backend::Vulkan => Arc::decrement_strong_count(ptr as *const <T as CastBackend<hal::api::Vulkan/*, Self*/>>::Output),
                #[cfg(metal)]
                Backend::Metal => Arc::decrement_strong_count(ptr as *const <T as CastBackend<hal::api::Metal/*, Self*/>>::Output),
                #[cfg(dx12)]
                Backend::Dx12 => Arc::decrement_strong_count(ptr as *const <T as CastBackend<hal::api::Dx12/*, Self*/>>::Output),
                #[cfg(dx11)]
                Backend::Dx11 => Arc::decrement_strong_count(ptr as *const <T as CastBackend<hal::api::Dx11/*, Self*/>>::Output),
                #[cfg(gl)]
                Backend::Gl => Arc::decrement_strong_count(ptr as *const <T as CastBackend<hal::api::Gles/*, Self*/>>::Output),
                _ => unreachable!(),
            }
        }
    }
}

impl<T> Drop for ValidId2<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // Safety: ValidId2<T> is the same as Arc<T>, so we can just decrement the strong
            // count.
            Arc::decrement_strong_count(self.ptr.as_ptr())
        }
    }
}

#[cfg(feature = "trace")]
impl<'a, T: AllBackends> From<&'a Id2<T>> for Id2<T> {
    #[inline]
    fn from(id: &'a Id2<T>) -> Self {
        id.clone()
    }
}

#[cfg(feature = "trace")]
impl<'a, T: AllBackends> From<&'a Id2<T>> for usize {
    #[inline]
    fn from(id: &'a Id2<T>) -> Self {
        *id.borrow()
    }
}

#[cfg(feature = "replay")]
#[derive(Debug)]
pub struct IdMap<K, A: hal::Api, F: AllResources<A>> {
    cache: crate::FastHashMap<K, Cached<A, F>>,
}

#[cfg(feature = "replay")]
impl<K, A: hal::Api, F: AllResources<A>> Default for IdMap<K, A, F> {
    fn default() -> Self {
        Self { cache: Default::default() }
    }
}

#[cfg(feature = "replay")]
pub type IdCache2 = IdMap<usize, Dummy, IdCon>;

#[cfg(feature = "replay")]
impl<K: Eq + core::hash::Hash, A: hal::Api, F: AllResources<A>> IdMap<K, A, F> {
    /// NOTE: Returns the old id if present.
    pub fn create<T: AnyBackend<Backend=A>>(
        &mut self,
        key: K,
        value: <F as Hkt<T>>::Output,
    ) -> Option<Cached<A, F>>
        where F: Hkt<T>
    {
        self.cache.insert(key, T::upcast(value))
    }

    /// NOTE: Panics if key is not found.
    pub fn destroy(&mut self, key: &K) -> Cached<A, F>
    {
        self.cache.remove(key).unwrap()
    }

    /// NOTE: Returns the old id if present.
    pub fn remove_resource<T: AnyBackend<Backend=A>>
        (&mut self, key: &K) -> Option</*T::Id*/<F as Hkt<T>>::Output>
        where F: Hkt<T>
    {
        self.cache.remove(key).and_then(T::downcast)
    }

    /// NOTE: Returns the old id if present.
    pub fn insert(&mut self, key: K, value: Cached<A, F>) -> Option<Cached<A, F>>
    {
        self.cache.insert(key, value)
    }

    pub fn get<T: AnyBackend<Backend=A>>(&self, key: &K) -> Option<&/*T::Id*/<F as Hkt<T>>::Output>
        where F: Hkt<T>
    {
        self.cache.get(key).and_then(T::downcast_ref)
    }
}

#[cfg(feature = "replay")]
impl<'a, B: crate::hub::HalApi, F: AllResources<T::Backend>, T: AllBackends + CastBackend<B> + 'a>
    core::convert::TryFrom<(&'a IdMap<usize, T::Backend, F>, &'a usize)> for IdGuard<'a, B, T>
    where F: Hkt<T, Output=Id2<T>>
{
    type Error = &'static str;

    fn try_from((cache, ptr): (&'a IdMap<usize, T::Backend, F>, &usize)) -> Result<Self, Self::Error> {
        // TODO: Better errors.
        Ok(expect_backend(cache.get::</*_, */T>(ptr).ok_or("invalid typed resource id")?))
    }
}

#[cfg(feature = "replay")]
impl<'a, B: crate::hub::HalApi, F: AllResources<T::Backend>, T: AllBackends + CastBackend<B> + 'a>
    core::convert::TryFrom<(&'a IdMap<usize, T::Backend, F>, usize)> for IdGuard<'a, B, T>
    where F: Hkt<T, Output=Id2<T>>
{
    type Error = &'static str;

    fn try_from((cache, ptr): (&'a IdMap<usize, T::Backend, F>, usize)) -> Result<Self, Self::Error> {
        // TODO: Better errors.
        Ok(expect_backend(cache.get::</*_, */T>(&ptr).ok_or("invalid typed resource id")?))
    }
}

#[cfg(feature = "replay")]
impl<'a, F: AllResources<T::Backend>, T: AllBackends>
    core::convert::TryFrom<(&'a IdMap<usize, T::Backend, F>, usize)> for &'a Id2<T>
    where F: Hkt<T, Output=Id2<T>>
{
    type Error = &'static str;
 
    fn try_from((cache, ptr): (&'a IdMap<usize, T::Backend, F>, usize)) -> Result<Self, Self::Error> {
        // TODO: Better errors.
        cache.get::</*_, */T>(&ptr).ok_or("invalid typed resource id")
    }
}

#[cfg(feature = "replay")]
impl<'a, F: AllResources<T::Backend>, T: AllBackends>
    core::convert::TryFrom<(&'a IdMap<usize, T::Backend, F>, usize)> for &'a BoxId2<T>
    where F: Hkt<T, Output=BoxId2<T>>
{
    type Error = &'static str;

    fn try_from((cache, ptr): (&'a IdMap<usize, T::Backend, F>, usize)) -> Result<Self, Self::Error> {
        // TODO: Better errors.
        cache.get::</*_, */T>(&ptr).ok_or("invalid typed resource id")
    }
}

#[cfg(feature = "replay")]
impl<'a, B: crate::hub::HalApi, F: AllResources<T::Backend>, T: AllBackends + CastBackend<B> + 'a>
    core::convert::TryFrom<(&'a IdMap<usize, T::Backend, F>, usize)> for BoxIdGuard<'a, B, T>
    where F: Hkt<T, Output=BoxId2<T>>,
{
    type Error = &'static str;

    fn try_from((cache, ptr): (&'a IdMap<usize, T::Backend, F>, usize)) -> Result<Self, Self::Error> {
        // TODO: Better errors.
        Ok(expect_backend_box(cache.get::</*_, */T>(&ptr).ok_or("invalid typed resource id")?))
    }
}

/* #[cfg(feature = "replay")]
impl<'a, A: hal::Api, B: crate::hub::HalApi, F: AllResources<A>, T: AllBackends + CastBackend<B> + 'a>
    core::convert::TryFrom<(&'a IdMap<usize, A, F>, usize)> for ValidId2<T>
    where F: Hkt<T, Output=Id2<T>>,
{
    type Error = &'static str;

    fn try_from((cache, ptr): (&'a IdMap<usize, A, F>, usize)) -> Result<Self, Self::Error> {
        // TODO: Better errors.
        Ok(expect_backend_owned(Id2::clone(cache.get::</*_, */U>(ptr).ok_or("invalid typed resource id")?)))
        // Ok(expect_backend_owned(cache.get::</*_, */T>(&ptr).ok_or("invalid typed resource id")?))
    }
} */

#[cfg(any(feature = "trace", feature = "serde"))]
impl<T: AllBackends> serde::Serialize for Id2<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let raw_ptr = self.ptr.as_ptr() as usize;
        raw_ptr.serialize(serializer)
    }
}

/* #[cfg(any(feature = "trace", feature = "serde"))]
impl<T: AllBackends> serde_state::SerializeState<core::cell::RefCell<IdCache>> for Id2<T> {
    fn serialize_state<S>(&self, serializer: S, seed: &core::cell::RefCell<IdCache>) -> Result<S::Ok, S::Error>
    where
        S: serde_state::ser::Serializer,
    {
        use serde_state::Serialize;
        let raw_ptr = self.ptr.as_ptr() as usize;
        seed.borrow_mut().cache.get_or_insert_with(&raw_ptr, |_| T::upcast(self.clone()));
        raw_ptr.serialize(serializer)
    }
} */

/* impl SerializeState<IdCache> for Id2 {
    fn serialize_state<S>(&self, serializer: S, seed: &Cell<i32>) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        seed.set(seed.get() + 1);
        self.serialize(serializer)
    }
}

impl<'de, S> DeserializeState<'de, S> for Id2 where S: BorrowMut<i32> {

    fn deserialize_state<D>(seed: &mut S, deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        *seed.borrow_mut() += 1;
        Self::deserialize(deserializer)
    }
} */

/* #[cfg(any(feature = "replay", feature = "serde"))]
impl<'a, 'de/*, S*/, T> serde_state::DeserializeState<'de, /*S*/&'a IdCache> for &'a Id2<T>/* where S: Borrow<IdCache> + 'a*/
    where
        T: AllBackends,
{
    fn deserialize_state<D>(seed: &mut /*S*/&'a IdCache, deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_state::de::Deserializer<'de>,
    {
        let ptr: usize = <usize as serde_state::de::Deserialize>::deserialize(deserializer)?;
        // TODO: Better errors.
        if let Some(id) = seed./*borrow().*/cache.get(&ptr).and_then(T::downcast_ref) {
            Ok(id)
        } else {
            Err(<D::Error as serde_state::de::Error>::invalid_value(serde_state::de::Unexpected::Unsigned(ptr as u64),  &"a valid typed resource id"))
        }
    }
}

#[cfg(any(feature = "replay", feature = "serde"))]
impl<'de/*, S*/, T> serde_state::DeserializeState<'de, /*S*/IdCache> for Id2<T>/* where S: Borrow<IdCache> + 'a*/
    where
        T: AllBackends,
{
    fn deserialize_state<D>(seed: &mut /*S*/IdCache, deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_state::de::Deserializer<'de>,
    {
        let ptr: usize = <usize as serde_state::de::Deserialize>::deserialize(deserializer)?;
        // TODO: Better errors.
        if let Some(id) = seed./*borrow().*/cache.take(&ptr).and_then(T::downcast) {
            Ok(id)
        } else {
            Err(<D::Error as serde_state::de::Error>::invalid_value(serde_state::de::Unexpected::Unsigned(ptr as u64),  &"a valid typed resource id"))
        }
    }
} */

impl<T: AllBackends> Borrow<usize> for Id2<T> {
    #[inline]
    fn borrow(&self) -> &usize {
        // Safety: &*const T -> &*const U should always be safe
        // when *const T -> *const U is safe.
        //
        // TODO: Verify this is also true of &*const T -> U
        // when *const T -> U is safe (since we're using usize here).
        unsafe {
            core::mem::transmute(&self.ptr)
        }
    }
}

impl<T: AllBackends> Id2<T> {
    #[inline]
    pub fn backend(&self) -> Backend {
        match (self.ptr.as_ptr() as usize) & BACKEND_MASK/*>> (64 - BACKEND_BITS) as u8*/ {
            0 => Backend::Empty,
            1 => Backend::Vulkan,
            2 => Backend::Metal,
            3 => Backend::Dx12,
            4 => Backend::Dx11,
            5 => Backend::Gl,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn upcast_backend<B: crate::hub::HalApi>(id: ValidId2<<T as CastBackend<B/*, Self*/>>::Output>) -> Self
        where T: CastBackend<B/*, Self*/>
    {
        unsafe {
            // Make sure not to run the ValidId2 destructor directly.
            let id = ManuallyDrop::new(id);
            // Safety: the only legal way to construct an Id2 is by taking a ValidId2 pointer of
            // type T::Output and casting it to Id2<T> with the same pointer masked with its
            // backend bits.
            //
            // B::VARIANT is unique for each implementation of the HalApi trait, so the the
            // pointer tag we store here is a bijection.
            let bits = backend_mask(B::VARIANT);
            // NOTE: bits can only be from backends that are part of AllBackends, which is enforced
            // by the constraint on T.  This means no dummy backends and no backends not supported
            // by the platform.  Additionally, ValidId2<T> has a pointer to data aligned to at least
            // (1 << BACKEND_BITS).  Thus we can just mask the bits with no additional pattern
            // match.
            //
            // NOTE: The NonNull constraint is upheld trivially because the original pointer was not
            // null, and | can never make a non-null pointer null (since Rust guarantees null is 0).
            Id2 { ptr: NonNull::new_unchecked((id.ptr.as_ptr() as usize | bits) as *mut _) }
        }
    }

    #[inline]
    #[cfg(feature="trace")]
    /// NOTE: Used on object creation when tracing.  The trace lock is owned by the device, so we
    /// need to have the device open after the object is created (since we don't know its address
    /// until that point).  But we can only take the trace lock on a ValidId, so we need to be able
    /// to figure out what the pointer component is going to be *before* we actually perform the
    /// upcast.  In theory this may hurt performance slightly while tracing since we calculate the
    /// id twice; we could resolve this with enough unsafe code, but it probably doesn't matter.
    ///
    /// Note that we repeat the logic in `upcast_backend` above with an explanation of safety
    /// requirements.
    pub(crate) fn as_usize<B: crate::hub::HalApi>(id: &ValidId2<<T as CastBackend<B/*, Self*/>>::Output>) -> usize
        where T: CastBackend<B/*, Self*/>
    {
        let bits = backend_mask(B::VARIANT);
        id.ptr.as_ptr() as usize | bits
    }

    #[inline]
    /// Used to help with implementing bind group deduplication outside of wgpu-core.
    ///
    /// NOTE: &mut self would ideally be used to verify that there are no other references to the
    /// Arc.  Unfortunately, HashMap iteration doesn't (yet?) provide a method with mutable access
    /// to the keys, even though this would probably be safe in the context of the raw entry API,
    /// so we take &self instead; it is the user's responsibility to make sure there are no
    /// references to the key in other threads if the count is 1 (for example, this can be
    /// enforced by iterating over the HashMap mutably).
    pub fn is_unique(&self) -> bool {
        unsafe {
            // Safety: Id2<T> can only be constructed from Arc<T> plus some optional backend bits,
            // so we can perform raw pointer Arc operations on the raw pointer once the backend
            // bits are removed.  Moreover, the borrow here cannot outlive the function, and
            // the original reference to self outlives the function lifetime.
            let ptr = self.ptr;
            let raw_ptr : *const T = ((ptr.as_ptr() as usize) & !BACKEND_MASK) as _;
            // NOTE: ManuallyDrop is required for safety here in order to keep the Arc from getting
            // deallocated.
            let arc = &ManuallyDrop::new(Arc::from_raw(raw_ptr));
            // NOTE: Currently, we take advantage of the fact that we don't actually have any weak
            // pointers to ValidId2 instances.  However, this may change in the future, at which
            // point we'll have to either update this function (and eat the performance loss), or
            // make it more specific to apply only to BindGroupLayouts (which likely won't have
            // weak pointers).  If it doesn't change in the future, we should consider switching to
            // a no-weak-pointer version of Arc, since our use of Arc is fully encapsulated anyway.
            let strong_count = Arc::strong_count(&*arc);
            strong_count == 1
        }
    }
}

#[repr(transparent)]
/* #[cfg_attr(feature = "trace", derive(serde_state::Serialize))]
#[cfg_attr(
    feature = "replay",
    derive(serde_state::Deserialize),
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "trace")),
    derive(serde_state::Serialize)
)]
#[cfg_attr(
    all(feature = "serde", not(feature = "replay")),
    derive(serde_state::Deserialize)
)] */
pub struct BoxId2<T: AllBackends> {
    ptr: NonNull<T>,
}

/// FIXME: Make sure all versions with all backends are Send + Sync.
unsafe impl<T: AllBackends + Send> Send for BoxId2<T> {}
unsafe impl<T: AllBackends + Sync> Sync for BoxId2<T> {}

impl<T: AllBackends> PartialEq for BoxId2<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T: AllBackends> Eq for BoxId2<T> {}

impl<T: AllBackends> std::hash::Hash for BoxId2<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

impl<T: AllBackends> fmt::Debug for BoxId2<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.ptr.fmt(formatter)
    }
}

impl<T: AllBackends> Drop for BoxId2<T> {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            // We can learn our backend and proceed to drop.
            //
            // Safety: constructing a BoxId2 guarantees that either a new backend was chosen (in the
            // BACKEND_MASK bits), or we can just drop T as Box<T>.  ptr is never accessed after
            // the BoxId2 containing it is dropped.
            let ptr = ((self.ptr.as_ptr() as usize) & !BACKEND_MASK) as *mut T;
            match self.backend() {
                // If there are no mask bits, this is the original backend.
                Backend::Empty => { Box::from_raw(ptr); },
                // Otherwise, we can cast the masked pointer to the appropriate type.
                #[cfg(vulkan)]
                Backend::Vulkan => { Box::from_raw(ptr as *mut <T as CastBackend<hal::api::Vulkan/*, Self*/>>::Output); },
                #[cfg(metal)]
                Backend::Metal => { Box::from_raw(ptr as *mut <T as CastBackend<hal::api::Metal/*, Self*/>>::Output); },
                #[cfg(dx12)]
                Backend::Dx12 => { Box::from_raw(ptr as *mut <T as CastBackend<hal::api::Dx12/*, Self*/>>::Output); },
                #[cfg(dx11)]
                Backend::Dx11 => { Box::from_raw(ptr as *mut <T as CastBackend<hal::api::Dx12/*, Self*/>>::Output); },
                #[cfg(gl)]
                Backend::Gl => { Box::from_raw(ptr as *mut <T as CastBackend<hal::api::Gles/*, Self*/>>::Output); },
                _ => unreachable!(),
            }
        }
    }
}

/* #[cfg(any(feature = "trace", feature = "serde"))]
impl<T: AllBackends> serde_state::SerializeState<core::cell::RefCell<IdCache>> for BoxId2<T> {
    fn serialize_state<S>(&self, serializer: S, seed: &core::cell::RefCell<IdCache>) -> Result<S::Ok, S::Error>
    where
        S: serde_state::ser::Serializer,
    {
        use serde_state::Serialize;
        let raw_ptr = self.ptr.as_ptr() as usize;
        seed.borrow_mut().cache.get_or_insert_with(&raw_ptr, |_| T::upcast(self.clone()));
        raw_ptr.serialize(serializer)
    }
} */

/* impl SerializeState<IdCache> for BoxId2 {
    fn serialize_state<S>(&self, serializer: S, seed: &Cell<i32>) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        seed.set(seed.get() + 1);
        self.serialize(serializer)
    }
}

impl<'de, S> DeserializeState<'de, S> for BoxId2 where S: BorrowMut<i32> {

    fn deserialize_state<D>(seed: &mut S, deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        *seed.borrow_mut() += 1;
        Self::deserialize(deserializer)
    }
} */

/* #[cfg(any(feature = "replay", feature = "serde"))]
impl<'a, 'de/*, S*/, T> serde_state::DeserializeState<'de, /*S*/&'a IdCache> for &'a BoxId2<T>/* where S: Borrow<IdCache> + 'a*/
    where
        T: AllBackends,
{
    fn deserialize_state<D>(seed: &mut /*S*/&'a IdCache, deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_state::de::Deserializer<'de>,
    {
        let ptr: usize = <usize as serde_state::de::Deserialize>::deserialize(deserializer)?;
        // TODO: Better errors.
        if let Some(id) = seed./*borrow().*/cache.get(&ptr).and_then(T::downcast_ref) {
            Ok(id)
        } else {
            Err(<D::Error as serde_state::de::Error>::invalid_value(serde_state::de::Unexpected::Unsigned(ptr as u64),  &"a valid typed resource id"))
        }
    }
}

#[cfg(any(feature = "replay", feature = "serde"))]
impl<'de/*, S*/, T> serde_state::DeserializeState<'de, /*S*/IdCache> for BoxId2<T>/* where S: Borrow<IdCache> + 'a*/
    where
        T: AllBackends,
{
    fn deserialize_state<D>(seed: &mut /*S*/IdCache, deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_state::de::Deserializer<'de>,
    {
        let ptr: usize = <usize as serde_state::de::Deserialize>::deserialize(deserializer)?;
        // TODO: Better errors.
        if let Some(id) = seed./*borrow().*/cache.take(&ptr).and_then(T::downcast) {
            Ok(id)
        } else {
            Err(<D::Error as serde_state::de::Error>::invalid_value(serde_state::de::Unexpected::Unsigned(ptr as u64),  &"a valid typed resource id"))
        }
    }
} */

impl<T: AllBackends> Borrow<usize> for BoxId2<T> {
    #[inline]
    fn borrow(&self) -> &usize {
        // Safety: &*const T -> &*const U should always be safe
        // when *const T -> *const U is safe.
        //
        // TODO: Verify this is also true of &*const T -> U
        // when *const T -> U is safe (since we're using usize here).
        unsafe {
            core::mem::transmute(&self.ptr)
        }
    }
}

impl<T: AllBackends> BoxId2<T> {
    #[inline]
    pub fn backend(&self) -> Backend {
        match (self.ptr.as_ptr() as usize) & BACKEND_MASK/*>> (64 - BACKEND_BITS) as u8*/ {
            0 => Backend::Empty,
            1 => Backend::Vulkan,
            2 => Backend::Metal,
            3 => Backend::Dx12,
            4 => Backend::Dx11,
            5 => Backend::Gl,
            _ => unreachable!(),
        }
    }

    #[inline]
    /// Constructs a new tagged pointer from Box<T>, making sure alignment is
    /// accurate.
    ///
    /// NOTE: Safe to construct for arbitrary T. It's just the conversion to/from
    /// BoxId2<T> that relies on safety properties of backends.
    pub fn upcast_backend<B: crate::hub::HalApi>(id: Box<<T as CastBackend<B/*, Self*/>>::Output>) -> Self
        where T: CastBackend<B/*, Self*/>
    {
        unsafe {
            // NOTE: This should hopefully vanish in optimized builds.
            assert!(core::mem::align_of::<T>() & BACKEND_MASK == 0);

            // Safety: the only legal way to construct a BoxId2 is by taking a Box pointer of
            // type T::Output and casting it to BoxId2<T> with the same pointer masked with its
            // backend bits.
            //
            // B::VARIANT is unique for each implementation of the HalApi trait, so the the
            // pointer tag we store here is a bijection.
            let bits = backend_mask(B::VARIANT);
            // NOTE: bits can only be from backends that are part of AllBackends, which is enforced
            // by the constraint on T.  This means no dummy backends and no backends not supported
            // by the platform.  Additionally, the Box<T> has a pointer to data aligned to at least
            // (1 << BACKEND_BITS).  Thus we can just mask the bits with no additional pattern
            // match.
            //
            // NOTE: The NonNull constraint is upheld trivially because the original pointer was not
            // null, and | can never make a non-null pointer null (since Rust guarantees null is 0).
            BoxId2 { ptr: NonNull::new_unchecked((Box::into_raw(id) as usize | bits) as *mut _) }
        }
    }
}

impl<T: AllBackends> BoxId2<T> {
    #[inline]
    #[cfg(feature="trace")]
    // False positive, cannot cast from &T to usize directly.
    #[allow(trivial_casts)]
    /// NOTE: Used on object creation when tracing.  The trace lock is owned by the device, so we
    /// need to have the device open after the object is created (since we don't know its address
    /// until that point).  But we can only take the trace lock on a Box<_>, so we need to be able
    /// to figure out what the pointer component is going to be *before* we actually perform the
    /// upcast.  In theory this may hurt performance slightly while tracing since we calculate the
    /// id twice; we could resolve this with enough unsafe code, but it probably doesn't matter.
    ///
    /// Note that we repeat the logic in `upcast_backend` above with an explanation of safety
    /// requirements.
    pub(crate) fn as_usize<B: crate::hub::HalApi>(id: &Box<<T as CastBackend<B/*, Self*/>>::Output>) -> usize
        where T: CastBackend<B/*, Self*/>
    {
        let bits = backend_mask(B::VARIANT);
        &**id as *const _ as usize | bits
    }
}

/// Id2 with no masked backend bits.
pub struct ValidId2<T> {
    ptr: NonNull<T>,
}

unsafe impl<T: Send> Send for ValidId2<T> {}
unsafe impl<T: Sync> Sync for ValidId2<T> {}

impl<T> PartialEq for ValidId2<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T> Eq for ValidId2<T> {}

impl<T> std::hash::Hash for ValidId2<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

impl<T> fmt::Debug for ValidId2<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        self.ptr.fmt(formatter)
    }
}

// Exactly the same as `&'a T`, just with different implementations of serde
// to account for the fact that it should have come from a BoxId<T>.
#[repr(transparent)]
pub struct BoxIdGuard<'a, A, T: CastBackend<A>>(&'a <T as CastBackend<A>>::Output);
// pub struct BoxIdGuard<'a, A, T>(&'a /*<IdGuardCon<'a> as Hkt<<T as CastBackend<A>>::Output>>::Output*/T, PhantomData<A>);
/*    where
        T: CastBackend<A>,
        IdGuardCon<'a>: Hkt<<T as CastBackend<A>>::Output, Output=Self>,
;*/

impl<'a, A, T: CastBackend<A>> Clone for BoxIdGuard<'a, A, T> {
    #[inline]
    fn clone(&self) -> Self { *self }
}

impl<'a, A, T: CastBackend<A>> Copy for BoxIdGuard<'a, A, T> {}

// False positive, cast affects formatting.
#[allow(trivial_casts)]
impl<'a, A, T: CastBackend<A>> fmt::Debug for BoxIdGuard<'a, A, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        (self.0 as *const T::Output).fmt(formatter)
    }
}

impl<'a, A, T: CastBackend<A>> Deref for BoxIdGuard<'a, A, T> {
    type Target = <T as CastBackend<A>>::Output;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, A, T: CastBackend<A>> PartialEq for BoxIdGuard<'a, A, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl<'a, A, T: CastBackend<A>> Eq for BoxIdGuard<'a, A, T> {}

// False positive, cast affects hash function.
#[allow(trivial_casts)]
impl<'a, A, T: CastBackend<A>> std::hash::Hash for BoxIdGuard<'a, A, T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const T::Output).hash(state);
    }
}

#[cfg(feature = "trace")]
impl<'a, T: AllBackends> From<&'a BoxId2<T>> for usize {
    #[inline]
    fn from(id: &'a BoxId2<T>) -> Self {
        *id.borrow()
        // BoxId2::<A>::as_usize(id.borrow())
    }
}

#[cfg(feature = "trace")]
impl<'a, A: crate::hub::HalApi, T: AllBackends + CastBackend<A>>
    From<&'_ BoxIdGuard<'a, A, T>> for usize {
    #[inline]
    fn from(id: &'_ BoxIdGuard<'a, A, T>) -> Self {
        BoxId2::<T>::as_usize::<A>(id.borrow())
    }
}

impl<'a, A, T: CastBackend<A>> Borrow<Box<T::Output>> for BoxIdGuard<'a, A, T> {
    #[inline]
    fn borrow(&self) -> &Box<T::Output> {
        unsafe {
            // Safety: &BoxIdGuard<'a, A, T> is the same as &Box<T::Output>,
            // because the only difference between BoxIdGuard and Box
            // is that one is an owning pointer and BoxIdGuard is guaranteed
            // aligned; since that ownership only applies to owned Box, not
            // shared, and we're going from BoxIdGuard to Box rather than
            // the other way around, there's no difference here.
            core::mem::transmute(self)
        }
    }
}

// Basically, represents a backend-validated reference to an existing Arc.
// Maintains a lifetime preventing the prior Arc from going away, but still
// retains a direct pointer to the underlying value; thanks to the lifetime
// ensuring the original Arc is still alive, IdGuard does not need to
// decrement reference count on drop and is freely copyable, but has an
// into_owned implementation that produces the original Arc.
//
// NOTE: Since this can only be obtained from an Arc from an Id2, it follows
// that the pointer respects alignment etc. of Id2 pointers.
#[repr(transparent)]
pub struct IdGuard<'a, A, T: CastBackend<A>>(&'a <T as CastBackend<A>>::Output);
// pub struct IdGuard<'a, T>(&'a T);

impl<'a, A, T: CastBackend<A>> Clone for IdGuard<'a, A, T> {
    #[inline]
    fn clone(&self) -> Self { *self }
}

impl<'a, A, T: CastBackend<A>> Copy for IdGuard<'a, A, T> {}

// False positive, cast affects formatting.
#[allow(trivial_casts)]
impl<'a, A, T: CastBackend<A>> fmt::Debug for IdGuard<'a, A, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        (self.0 as *const T::Output).fmt(formatter)
    }
}

impl<'a, A, T: CastBackend<A>> Deref for IdGuard<'a, A, T> {
    type Target = T::Output;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a, A, T: CastBackend<A>> PartialEq for IdGuard<'a, A, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(self.0, other.0)
    }
}
impl<'a, A, T: CastBackend<A>> Eq for IdGuard<'a, A, T> {}

// False positive, cast affects hash function.
#[allow(trivial_casts)]
impl<'a, A, T: CastBackend<A>> std::hash::Hash for IdGuard<'a, A, T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.0 as *const T::Output).hash(state);
    }
}

impl<'a, A, T: CastBackend<A>> Borrow<ValidId2<T::Output>> for IdGuard<'a, A, T> {
    #[inline]
    fn borrow(&self) -> &ValidId2<T::Output> {
        unsafe {
            // Safety: &IdGuard<'a, A, T> is the same as &ValidId2<T::Output>,
            // because the only difference between IdGuard and ValidId2
            // is that one owns part of the refcount; since that
            // ownership only applies to owned ValidId2, not shared,
            // there's no difference here.
            core::mem::transmute(self)
        }
    }
}

impl<'a, A, T: CastBackend<A>> IdGuard<'a, A, T> {
    #[inline]
    pub fn as_ref(self) -> &'a T::Output {
        self.0
    }

    #[inline]
    pub fn to_owned(self) -> ValidId2<T::Output> {
        unsafe {
            // Safety: always constructed from &'a Id2<T> with zeroed
            // BACKEND_BITS, which is always cast from Arc<T::Output>.  Since
            // the original Arc is still alive, it's also safe to
            // increment the strong count here.
            let ptr = self.0;
            Arc::increment_strong_count(ptr);
            ValidId2 { ptr: NonNull::from(ptr) }
        }
    }
}

impl<'a, A, T: CastBackend<A>> From<IdGuard<'a, A, T>> for ValidId2<T::Output> {
    #[inline]
    fn from(id: IdGuard<'a, A, T>) -> Self {
        id.to_owned()
    }
}

#[cfg(feature = "trace")]
impl<'a, A: crate::hub::HalApi, T: AllBackends + CastBackend<A>> From<IdGuard<'a, A, T>> for usize {
    #[inline]
    fn from(id: IdGuard<'a, A, T>) -> Self {
        Id2::<T>::as_usize::<A>(id.borrow())
    }
}

impl<T> Deref for ValidId2<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe {
            // Safety: equivalent to Arc<T>, so safety follows from safety of Arc::deref
            // and Arc::into_raw
            self.ptr.as_ref()
        }
    }
}

impl<T> Clone for ValidId2<T> {
    #[inline]
    fn clone(&self) -> Self {
        unsafe {
            // Safety: equivalent to Arc<T>, so safety follows from safety of
            // Arc::increment_strong_count on pointers constructed with
            // Arc::into_raw
            Arc::increment_strong_count(self.ptr.as_ptr());
            ValidId2 { ptr: self.ptr }
        }
    }
}

impl<'a, A, T: CastBackend<A>> From<&'a ValidId2<T::Output>> for IdGuard<'a, A, T> {
    #[inline]
    fn from(id: &'a ValidId2<T::Output>) -> Self {
        id.borrow()
    }
}

impl<T> ValidId2<T> {
    #[inline]
    /// Constructs a new valid id from Arc<T>, making sure alignment is
    /// accurate.
    ///
    /// NOTE: Safe to construct for arbitrary T. It's just the conversion to/from
    /// Id2<T> that relies on safety properties of backends.
    #[inline]
    pub(crate) fn new(ptr: Arc<T>) -> Self {
        // NOTE: This should hopefully vanish in optimized builds.
        assert!(core::mem::align_of::<T>() & BACKEND_MASK == 0);
        unsafe {
            // Safety: We can call Arc::deref to get &T, with the same
            // address as the data in the Arc.  Rust promises any reference is
            // non-null, so so must be the raw pointer, satisfying the requirements
            // of NonNull::new_unchecked.
            ValidId2 { ptr: NonNull::new_unchecked(Arc::into_raw(ptr) as *mut _) }
        }
    }

    #[inline]
    pub fn borrow<'a, A, U: CastBackend<A, Output=T>>(&'a self) -> IdGuard<'a, A, U> {
        // Safety: since ValidId2 is equivalent to Arc<T>, and its deref method returns a pointer
        // to the inner arc data, this constructs a valid IdGuard.
        IdGuard(self.deref())
    }

    /* #[inline]
    /// NOTE: Exposing Arc::get_mut like this (rather than just `is_unique`) means we can't rely
    /// on data being immutable once observed (without something like Pin).
    pub fn get_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        unsafe {
            // Safety: since we were originally constructed from `Arc<T>`, we can just call `get_mut`
            // on that.  `get_mut()` will then appear not to outlive the function, since it was
            // derived from the local value `Arc<T>` rather than directly from the mutable pointer;
            // however, we know that in reality it's just a mutable version of self.ptr that's been
            // checked for uniqueness so it is safe to transmute the lifetime to 'a.
            core::mem::transmute(Arc::get_mut(&mut Arc::from_raw(self.ptr)))
        }
    } */
}

#[inline]
/// NOTE: Must match Id2::backend.
fn backend_mask(b: Backend) -> usize {
    match b {
        Backend::Empty => 0,
        Backend::Vulkan => 1,
        Backend::Metal => 2,
        Backend::Dx12 => 3,
        Backend::Dx11 => 4,
        Backend::Gl => 5,
        Backend::BrowserWebGpu => panic!("Should not use masked ids in WebGPU"),
    }
}

/* impl<T> Id2<T> where T: CacheBackend<>
    pub fn cast_expect<B>(self) -> Id2<T::Output> where T: CastBackend<B> {
    }
} */

pub trait CastBackend<B> {
    type Output;
}

/// Safety: id.backend() == B::VARIANT must be true when the method is called.
pub unsafe fn expect_backend_unchecked<'a, T: AllBackends + CastBackend<B/*, Id2<T>*/>, B: crate::hub::HalApi>
    (id: &'a Id2<T>) -> IdGuard<'a, /*<T as CastBackend<B/*, Id2<T>*/>>::Output*/B, T>
{
    // Safety: ids can only be constructed initially without backend bits set (since
    // ArcInner is asserted to be 8-byte-aligned), so any backend bits set must have been
    // set by upcast_backend.  Since backend bits set by upcast_backend are in bijection with
    // types that implement HalApi through their VARIANT associated type, the `backend`
    // function reads back the original VARIANT, and this matches the expected B::VARIANT,
    // we know that downcasting to backend B is safe (as it just inverts the operation of the
    // upcast).
    IdGuard(&*(((id.ptr.as_ptr() as usize) & !BACKEND_MASK) as *const _))
}

#[inline]
pub fn expect_backend<'a, T: AllBackends + CastBackend<B/*, Id2<T>*/>, B: crate::hub::HalApi>
    (id: &'a Id2<T>) -> IdGuard<'a, /*<T as CastBackend<B/*, Id2<T>*/>>::Output*/B, T>
{
    unsafe {
        assert_eq!(id.backend(), B::VARIANT, "Backend differed from expected backend.");
        // Safety: above line dynamically checks the precondition for expect_backend_unchecked.
        expect_backend_unchecked::<T,B>(id)
    }
}

/// Safety: id.backend() == B::VARIANT must be true when the method is called.
pub unsafe fn expect_backend_box_unchecked<'a, T: AllBackends + CastBackend<B/*, BoxId2<T>*/>, B: crate::hub::HalApi>
    (id: &'a BoxId2<T>) -> /*&'a <T as CastBackend<B/*, BoxId2<T>*/>>::Output*/BoxIdGuard<'a, B, T>
{
    // Safety: ids can only be constructed initially without backend bits set (since
    // T::Output is asserted to be 8-byte-aligned), so any backend bits set must have been
    // set by upcast_backend.  Since backend bits set by upcast_backend are in bijection with
    // types that implement HalApi through their VARIANT associated type, the `backend`
    // function reads back the original VARIANT, and this matches the expected B::VARIANT,
    // we know that downcasting to backend B is safe (as it just inverts the operation of the
    // upcast).
    BoxIdGuard(&*(((id.ptr.as_ptr() as usize) & !BACKEND_MASK) as *const _))
}

/// Safety: id.backend() == B::VARIANT must be true when the method is called.
pub unsafe fn expect_backend_box_owned_unchecked<T: AllBackends + CastBackend<B/*, BoxId2<T>*/>, B: crate::hub::HalApi>
    (id: BoxId2<T>) -> Box<<T as CastBackend<B/*, BoxId2<T>*/>>::Output>
{
    // Make sure not to run the BoxId2 destructor directly.
    let id = ManuallyDrop::new(id);
    // Safety: ids can only be constructed initially without backend bits set (since
    // T::Output is asserted to be 8-byte-aligned), so any backend bits set must have been
    // set by upcast_backend.  Since backend bits set by upcast_backend are in bijection with
    // types that implement HalApi through their VARIANT associated type, the `backend`
    // function reads back the original VARIANT, and this matches the expected B::VARIANT,
    // we know that downcasting to backend B is safe (as it just inverts the operation of the
    // upcast).
    Box::from_raw(((id.ptr.as_ptr() as usize) & !BACKEND_MASK) as *mut _)
}

#[inline]
pub fn expect_backend_box<'a, T: AllBackends + CastBackend<B/*, BoxId2<T>*/>, B: crate::hub::HalApi>
    (id: &'a BoxId2<T>) -> BoxIdGuard<'a, /*<T as CastBackend<B/*, BoxId2<T>*/>>::Output*/B, T>
{
    unsafe {
        assert_eq!(id.backend(), B::VARIANT, "Backend differed from expected backend.");
        // Safety: above line dynamically checks the precondition for expect_backend_unchecked.
        expect_backend_box_unchecked::<T,B>(id)
    }
}

#[inline]
pub fn expect_backend_box_owned<T: AllBackends + CastBackend<B/*, BoxId2<T>*/>, B: crate::hub::HalApi>
    (id: BoxId2<T>) -> Box<<T as CastBackend<B/*, BoxId2<T>*/>>::Output>
{
    unsafe {
        assert_eq!(id.backend(), B::VARIANT, "Backend differed from expected backend.");
        // Safety: above line dynamically checks the precondition for expect_backend_unchecked.
        expect_backend_box_owned_unchecked::<T,B>(id)
    }
}

/// Safety: id.backend() == B::VARIANT must be true when the method is called.
#[inline]
pub unsafe fn expect_backend_owned_unchecked<T: AllBackends + CastBackend<B/*, Id2<T>*/>, B: crate::hub::HalApi>
    (id: Id2<T>) -> ValidId2<<T as CastBackend<B/*, Id2<T>*/>>::Output>
{
    // Make sure not to run the Id2 destructor directly.
    let id = ManuallyDrop::new(id);
    // Safety: ids can only be constructed initially without backend bits set (since
    // ArcInner is asserted to be 8-byte-aligned), so any backend bits set must have been
    // set by upcast_backend.  Since backend bits set by upcast_backend are in bijection with
    // types that implement HalApi through their VARIANT associated type, the `backend`
    // function reads back the original VARIANT, and this matches the expected B::VARIANT,
    // we know that downcasting to backend B is safe (as it just inverts the operation of the
    // upcast).
    //
    // NOTE: The NonNull constraint is upheld because the original pointer was not null and has
    // the expected alignment; Rust guarantees null is 0, so to satisfy the alignment
    // requirements its value must have been at least 1 << BACKEND_BITS.  Thus, the intersection
    // with !BACKEND_MASK is still non-0, i.e. non-null.
    ValidId2 { ptr: NonNull::new_unchecked((id.ptr.as_ptr() as usize  & !BACKEND_MASK) as *mut _) }
}

#[inline]
pub fn expect_backend_owned<T: AllBackends + CastBackend<B/*, Id2<T>*/>, B: crate::hub::HalApi>
    (id: Id2<T>) -> ValidId2<<T as CastBackend<B/*, Id2<T>*/>>::Output>
{
    unsafe {
        assert_eq!(id.backend(), B::VARIANT, "Backend differed from expected backend.");
        // Safety: above line dynamically checks the precondition for expect_backend_owned_unchecked.
        expect_backend_owned_unchecked::<T,B>(id)
    }
}

pub trait Hkt<T> {
    /// NOTE: Requiring Debug is a hack, it could be replaced with a more specific requirement
    /// where needed.
    type Output : fmt::Debug;
}

impl<'a, T, F: Hkt<T>> Hkt<T> for &'a F where <F as Hkt<T>>::Output: 'a {
    type Output = &'a <F as Hkt<T>>::Output;
}

#[cfg(feature="trace")]
pub trait BorrowHkt<A, T, U: CastBackend<A>, Owned: Hkt<U::Output>> : Hkt<T> {
    fn borrow(borrowed: &<Owned as Hkt<U::Output>>::Output) -> <Self as Hkt<T>>::Output;
}

#[cfg(feature="trace")]
impl<T, A: crate::hub::HalApi, U: AllBackends + CastBackend<A>>
    BorrowHkt<A, T, U, ValidId2Con> for UsizeCon
    where
        ValidId2Con: Hkt<<U as CastBackend<A>>::Output, Output=ValidId2<<U as CastBackend<A>>::Output>>,
{
    #[inline]
    fn borrow(borrowed: &ValidId2<<U as CastBackend<A>>::Output>) -> <Self as Hkt<<U as CastBackend<A>>::Output>>::Output {
        Id2::<U>::as_usize::<A>(borrowed)
    }
}

#[derive(Debug)]
pub struct NullCon<T>(PhantomData<T>);

pub type UsizeCon = NullCon<usize>;
pub type UnitCon = NullCon<()>;

impl<C: fmt::Debug, T> Hkt<T> for NullCon<C> {
    type Output = C;
}

#[derive(Debug)]
pub struct IdCon;

#[derive(Debug)]
pub struct ValidId2Con {}

#[derive(Debug)]
pub struct IdGuardCon<'a> { data: PhantomData<&'a ()> }

macro_rules! impl_cast_backend {
    ($( $tycon:ident )::*, $( $idcon:ident )::*, $( $idvalidcon:ident )::*, $( $idguardcon:ident )::*, $variant:ident) => {
        impl AllBackends for $( $tycon )::*<Dummy> {}

        impl<A: hal::Api> AnyBackend for $( $tycon )::*<A> {
            type Backend = A;

            #[inline] fn upcast<F: AllResources<A> + Hkt<Self>>
                (id: <F as Hkt<Self>>::Output) -> Cached<A, F> {
                Cached::$variant(id)
            }
            #[inline] fn downcast<F: AllResources<A> + Hkt<Self>>(cached: Cached<A, F>) ->
                Option<<F as Hkt<Self>>::Output> {
                if let Cached::$variant(id) = cached { Some(id) } else { None }
            }
            #[inline] fn downcast_ref<F: AllResources<A> + Hkt<Self>>(cached: &Cached<A, F>) ->
                Option<&<F as Hkt<Self>>::Output> {
                if let Cached::$variant(id) = cached { Some(id) } else { None }
            }
        }

        impl Hkt<$( $tycon )::* <Dummy>> for IdCon {
            type Output = $( $idcon )::* <$( $tycon )::* <Dummy>>;
        }

        impl<'a, A: crate::hub::HalApi + 'a> Hkt<$( $tycon )::* <A>> for IdGuardCon<'a> {
            type Output = $( $idguardcon )::* <'a, A, $( $tycon )::* <Dummy>>;
        }

        impl<A: hal::Api> Hkt<$( $tycon )::* <A>> for ValidId2Con {
            type Output = $( $idvalidcon )::* <$( $tycon )::* <A>>;
        }

        impl<B: crate::hub::HalApi> CastBackend<B> for $( $tycon )::*<Dummy> {
            type Output = $( $tycon )::* <B>;
        }
    };
}

/// FIXME: Box can't be used currently due to fmt::Debug requirements, so we may need something
/// like ValidBoxId2 if we want the owned version to work properly (which would be nice to remove
/// boilerplate).  For now we just use a dummy for these cases.
type ValidBoxId2<T> = /*Box*/PhantomData<T>;

impl_cast_backend!(/*Box*/ crate::instance::Adapter, BoxId2, ValidBoxId2, BoxIdGuard, Adapter);
// Device
impl_cast_backend!(/*Arc*/ crate::device::Device, Id2, ValidId2, IdGuard, Device);
// Resource
impl_cast_backend!(/*Arc*/ crate::resource::Buffer, Id2, ValidId2, IdGuard, Buffer);
impl_cast_backend!(/*Arc*/ crate::resource::TextureView, Id2, ValidId2, IdGuard, TextureView);
impl_cast_backend!(/*Arc*/ crate::resource::Texture, Id2, ValidId2, IdGuard, Texture);
impl_cast_backend!(/*Arc*/ crate::resource::Sampler, Id2, ValidId2, IdGuard, Sampler);
// Binding model
impl_cast_backend!(/*Arc*/ crate::binding_model::BindGroupLayout, Id2, ValidId2, IdGuard, BindGroupLayout);
impl_cast_backend!(/*Arc*/ crate::binding_model::PipelineLayout, Id2, ValidId2, IdGuard, PipelineLayout);
impl_cast_backend!(/*Arc*/ crate::binding_model::BindGroup, Id2, ValidId2, IdGuard, BindGroup);
// Pipeline
impl_cast_backend!(/*Box*/ crate::pipeline::ShaderModule, BoxId2, ValidBoxId2, BoxIdGuard, ShaderModule);
impl_cast_backend!(/*Arc*/ crate::pipeline::RenderPipeline, Id2, ValidId2, IdGuard, RenderPipeline);
impl_cast_backend!(/*Arc*/ crate::pipeline::ComputePipeline, Id2, ValidId2, IdGuard, ComputePipeline);
// Command
impl_cast_backend!(/*Arc*/ crate::command::CommandBuffer, Id2, ValidId2, IdGuard, CommandBuffer);
impl_cast_backend!(/*Arc*/ crate::command::RenderBundle, Id2, ValidId2, IdGuard, RenderBundle);
impl_cast_backend!(/*Arc*/ crate::resource::QuerySet, Id2, ValidId2, IdGuard, QuerySet);
// Presentation
// impl_cast_backend!(/*Arc*/ crate::present::Presentation, Id2, ValidId2, IdGuard, Presentation);

/* /// NOTE: Used in macros to avoid unsafe.  These are pretty much just rank-2 closures.
pub trait GfxSelectOnce {
    type Output<B>;
    type Caster<T> where T : CastBackend<>;

    /// cast_id casts Id2<T> Id2<T::Output> for B::Backend.
    fn call_once<B: hal::Backend>(self, id: Id2<T::Output>, cast_id: fn <> -> Id2<G::Output>) -> Self::Output<B> where T: CastBackend<B>;
}

impl<T: AllBackends> T {
    pub fn gfx_select<F: GfxSelectOnce<T>>(f: F, id: Id2<T>) -> F::Output {
        match id.backend() {
        }
    }
        match $id.backend() {
            #[cfg(all(not(target_arch = "wasm32"), not(target_os = "ios"), not(target_os = "macos")))]
            wgt::Backend::Vulkan => $global.$method::<$crate::hal::api::Vulkan>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), any(target_os = "ios", target_os = "macos")))]
            wgt::Backend::Metal => $global.$method::<$crate::hal::api::Metal>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), windows))]
            wgt::Backend::Dx12 => $global.$method::<$crate::hal::api::Dx12>( $($param),* ),
            #[cfg(all(not(target_arch = "wasm32"), windows))]
            wgt::Backend::Dx11 => $global.$method::<$crate::hal::api::Dx11>( $($param),* ),
            #[cfg(any(target_arch = "wasm32", all(unix, not(any(target_os = "ios", target_os = "macos")))))]
            wgt::Backend::Gl => $global.$method::<$crate::hal::api::Gles>( $($param),+ ),
            other => panic!("Unexpected backend {:?}", other),
        }
    #[inline] fn downcast(id: T::Output) -> Arc<T> {

    }
} */

pub trait AllResources<B: hal::Api> :
    Hkt<crate::instance::Adapter<B>> +
    // Hkt<crate::instance::Surface> +
    Hkt<crate::device::Device<B>> +
    Hkt<crate::resource::Buffer<B>> +
    Hkt<crate::resource::TextureView<B>> +
    Hkt<crate::resource::Texture<B>> +
    Hkt<crate::resource::Sampler<B>> +
    Hkt<crate::binding_model::BindGroupLayout<B>> +
    Hkt<crate::binding_model::PipelineLayout<B>> +
    Hkt<crate::binding_model::BindGroup<B>> +
    Hkt<crate::pipeline::ShaderModule<B>> +
    Hkt<crate::pipeline::RenderPipeline<B>> +
    Hkt<crate::pipeline::ComputePipeline<B>> +
    Hkt<crate::command::CommandBuffer<B>> +
    // Hkt<crate::command::RenderPass> +
    // Hkt<crate::command::ComputePass> +
    // Hkt<crate::command::RenderBundleEncoder> +
    Hkt<crate::command::RenderBundle<B>> +
    Hkt<crate::resource::QuerySet<B>>/* +
    Hkt<crate::present::Presentation<B>>*/
{
    /// The owned version of this resource.
    type Owned: AllResources<B>;
}

impl<C: fmt::Debug> AllResources<Dummy> for NullCon<C> {
    type Owned = NullCon<C>;
}
impl AllResources<Dummy> for IdCon {
    type Owned = IdCon;
}
impl<'a, B: hal::Api + 'a, F: AllResources<B>> AllResources<B> for &'a F {
    type Owned = F;
}
impl<'a, B: crate::hub::HalApi + 'a> AllResources<B> for IdGuardCon<'a> {
    type Owned = ValidId2Con;
}
impl<B: hal::Api> AllResources<B> for ValidId2Con {
    type Owned = ValidId2Con;
}

pub type AdapterId2 = BoxId2<crate::instance::Adapter<Dummy>>;
pub type SurfaceId2 = Id2<crate::instance::Surface>;
// Device
pub type DeviceId2 = Id2<crate::device::Device<Dummy>>;
pub type QueueId2 = DeviceId2;
// Resource
pub type BufferId2 = Id2<crate::resource::Buffer<Dummy>>;
pub type TextureViewId2 = Id2<crate::resource::TextureView<Dummy>>;
pub type TextureId2 = Id2<crate::resource::Texture<Dummy>>;
pub type SamplerId2 = Id2<crate::resource::Sampler<Dummy>>;
// Binding model
pub type BindGroupLayoutId2 = Id2<crate::binding_model::BindGroupLayout<Dummy>>;
pub type PipelineLayoutId2 = Id2<crate::binding_model::PipelineLayout<Dummy>>;
pub type BindGroupId2 = Id2<crate::binding_model::BindGroup<Dummy>>;
// Pipeline
pub type ShaderModuleId2 = BoxId2<crate::pipeline::ShaderModule<Dummy>>;
pub type RenderPipelineId2 = Id2<crate::pipeline::RenderPipeline<Dummy>>;
pub type ComputePipelineId2 = Id2<crate::pipeline::ComputePipeline<Dummy>>;
// Command
pub type CommandEncoderId2 = CommandBufferId2;
pub type CommandBufferId2 = Id2<crate::command::CommandBuffer<Dummy>>;
pub type RenderPassEncoderId2<'a> = *mut crate::command::RenderPass<'a>;
pub type ComputePassEncoderId2<'a> = *mut crate::command::ComputePass<'a>;
pub type RenderBundleEncoderId2<'a> = *mut crate::command::RenderBundleEncoder<'a>;
pub type RenderBundleId2 = Id2<crate::command::RenderBundle<Dummy>>;
pub type QuerySetId2 = Id2<crate::resource::QuerySet<Dummy>>;
// Presentation
// pub type PresentationId2 = Id2<crate::present::Presentation<Dummy>>;

#[cfg(vulkan)] pub trait VulkanBackend : CastBackend<hal::api::Vulkan> {}
#[cfg(vulkan)] impl<T: CastBackend<hal::api::Vulkan>> VulkanBackend for T {}
#[cfg(not(vulkan))] pub trait VulkanBackend {}
#[cfg(not(vulkan))] impl<T> VulkanBackend for T {}

#[cfg(metal)] pub trait MetalBackend : CastBackend<hal::api::Metal> {}
#[cfg(metal)] impl<T: CastBackend<hal::api::Metal>> MetalBackend for T {}
#[cfg(not(metal))] pub trait MetalBackend {}
#[cfg(not(metal))] impl<T> MetalBackend for T {}

#[cfg(dx11)] pub trait Dx11Backend : CastBackend<hal::api::Dx11> {}
#[cfg(dx11)] impl<T: CastBackend<hal::api::Dx11>> Dx11Backend for T {}
#[cfg(not(dx11))] pub trait Dx11Backend {}
#[cfg(not(dx11))] impl<T> Dx11Backend for T {}

#[cfg(dx12)] pub trait Dx12Backend : CastBackend<hal::api::Dx12> {}
#[cfg(dx12)] impl<T: CastBackend<hal::api::Dx12>> Dx12Backend for T {}
#[cfg(not(dx12))] pub trait Dx12Backend {}
#[cfg(not(dx12))] impl<T> Dx12Backend for T {}

#[cfg(gl)] pub trait GlBackend : CastBackend<hal::api::Gles> {}
#[cfg(gl)] impl<T: CastBackend<hal::api::Gles>> GlBackend for T {}
#[cfg(not(gl))] pub trait GlBackend {}
#[cfg(not(gl))] impl<T> GlBackend for T {}

/// Implemented for any resource type with an associated backend (including the dummy backend).
pub trait AnyBackend : Sized {
    /// The associated backend for this resource.
    type Backend: hal::Api;

    fn upcast<F: AllResources<Self::Backend> + Hkt<Self>>(id: /*Self::Id*/<F as Hkt<Self>>::Output) -> Cached<Self::Backend, F>
        /*where Self: AllBackends*/;
    fn downcast<F: AllResources<Self::Backend> + Hkt<Self>>(cached: Cached<Self::Backend, F>) ->
        Option</*Self::Id*/<F as Hkt<Self>>::Output>
        /*where Self: AllBackends*/;
    fn downcast_ref<F: AllResources<Self::Backend> + Hkt<Self>>(cached: &Cached<Self::Backend, F>) ->
        Option<&/*Self::Id*/<F as Hkt<Self>>::Output>
        /*where Self: AllBackends*/;
}


pub trait AllBackends :
    AnyBackend<Backend=Dummy> +
    VulkanBackend + MetalBackend + Dx11Backend + Dx12Backend + GlBackend
{}

/* impl<T> AllBackends for T
    where T:
        VulkanBackend<Id> +
        MetalBackend<Id> +
        Dx11Backend<Id> +
        Dx12Backend<Id> +
        GlBackend<Id>,
{} */

/* pub enum Device<T: AllBackends> {
    Vulkan(<T as CastBackend<hal::api::Vulkan>>::Output),
    Metal(<T as CastBackend<hal::api::Metal>>::Output),
    Dx12(<T as CastBackend<hal::api::Dx12>>::Output),
    Dx11(<T as CastBackend<hal::api::Dx11>>::Output),
    Gl(<T as CastBackend<hal::api::Gles>>::Output),
} */

#[derive(Debug)]
pub enum Cached<A: hal::Api, F: AllResources<A>> {
    // Empty,
    Adapter(/*AdapterId2*/<F as Hkt<crate::instance::Adapter<A>>>::Output),
    // Surface(/*SurfaceId2*/<F as Hkt<crate::instance::Surface>>::Output),
    Device(/*DeviceId2*/<F as Hkt<crate::device::Device<A>>>::Output),
    Buffer(/*BufferId2*/<F as Hkt<crate::resource::Buffer<A>>>::Output),
    TextureView(/*TextureViewId2*/<F as Hkt<crate::resource::TextureView<A>>>::Output),
    Texture(/*TextureId2*/<F as Hkt<crate::resource::Texture<A>>>::Output),
    Sampler(/*SamplerId2*/<F as Hkt<crate::resource::Sampler<A>>>::Output),
    BindGroupLayout(/*BindGroupLayoutId2*/<F as Hkt<crate::binding_model::BindGroupLayout<A>>>::Output),
    PipelineLayout(/*PipelineLayoutId2*/<F as Hkt<crate::binding_model::PipelineLayout<A>>>::Output),
    BindGroup(/*BindGroupId2*/<F as Hkt<crate::binding_model::BindGroup<A>>>::Output),
    ShaderModule(/*ShaderModuleId2*/<F as Hkt<crate::pipeline::ShaderModule<A>>>::Output),
    RenderPipeline(/*RenderPipelineId2*/<F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output),
    ComputePipeline(/*ComputePipelineId2*/<F as Hkt<crate::pipeline::ComputePipeline<A>>>::Output),
    CommandBuffer(/*CommandBufferId2*/<F as Hkt<crate::command::CommandBuffer<A>>>::Output),
    // RenderPass(<F as Hkt<crate::command::RenderPass>>::Output),
    // ComputePass(<F as Hkt<crate::command::ComputePass>>::Output),
    // RenderBundleEncoder(<F as Hkt<crate::command::RenderBundleEncoder>>::Output),
    RenderBundle(/*RenderBundleId2*/<F as Hkt<crate::command::RenderBundle<A>>>::Output),
    QuerySet(/*QuerySetId2*/<F as Hkt<crate::resource::QuerySet<A>>>::Output),
    // Presentation(<F as Hkt<crate::present::Presentation<A>>>::Output),
}

impl<A: hal::Api, F: AllResources<A>> PartialEq for Cached<A, F>
    where Self: Borrow<usize>,
{
    fn eq(&self, other: &Self) -> bool {
        // NOTE: non-Empty is semantically non-null, so there should be no equality conflict there.
        let this: &usize = self.borrow();
        let other: &usize = other.borrow();
        this == other
    }
}

impl<A: hal::Api, F: AllResources<A>> std::hash::Hash for Cached<A, F> where Self: Borrow<usize> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let this: &usize = self.borrow();
        this.hash(state);
    }
}

impl<A: hal::Api, F: AllResources<A>> Eq for Cached<A, F> where Self: Borrow<usize> {}

impl<A: hal::Api, F: AllResources<A>> Borrow<usize> for Cached<A, F>
    where
        <F as Hkt<crate::instance::Adapter<A>>>::Output: Borrow<usize>,
        // <F as Hkt<crate::instance::Surface>>::Output: Borrow<usize>,
        <F as Hkt<crate::device::Device<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::resource::Buffer<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::resource::TextureView<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::resource::Texture<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::resource::Sampler<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::binding_model::BindGroupLayout<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::binding_model::PipelineLayout<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::binding_model::BindGroup<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::pipeline::ShaderModule<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::pipeline::RenderPipeline<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::pipeline::ComputePipeline<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::command::CommandBuffer<A>>>::Output: Borrow<usize>,
        // <F as Hkt<crate::command::RenderPass>>::Output: Borrow<usize>,
        // <F as Hkt<crate::command::ComputePass>>::Output: Borrow<usize>,
        // <F as Hkt<crate::command::RenderBundleEncoder>>::Output: Borrow<usize>,
        <F as Hkt<crate::command::RenderBundle<A>>>::Output: Borrow<usize>,
        <F as Hkt<crate::resource::QuerySet<A>>>::Output: Borrow<usize>,
        // <F as Hkt<crate::present::Presentation<A>>>::Output: Borrow<usize>,
{
    fn borrow(&self) -> &usize {
        match self {
            // // Empty cached slots all compare equal to 0, which is not a legal address in Rust
            // // since it is null (NOTE: due to potential differences between cast and as,
            // // revisit and make sure this logic works as expected).
            // Self::Empty => &0,
            Self::Adapter(id) => id.borrow(),
            // Self::Surface(id) => Self::Surface(id).borrow(),
            Self::Device(id) => id.borrow(),
            Self::Buffer(id) => id.borrow(),
            Self::TextureView(id) => id.borrow(),
            Self::Texture(id) => id.borrow(),
            Self::Sampler(id) => id.borrow(),
            Self::BindGroupLayout(id) => id.borrow(),
            Self::PipelineLayout(id) => id.borrow(),
            Self::BindGroup(id) => id.borrow(),
            Self::ShaderModule(id) => id.borrow(),
            Self::RenderPipeline(id) => id.borrow(),
            Self::ComputePipeline(id) => id.borrow(),
            Self::CommandBuffer(id) => id.borrow(),
            Self::RenderBundle(id) => id.borrow(),
            Self::QuerySet(id) => id.borrow(),
            // Self::Presentation(id) => id.borrow(),
        }
    }
}

/* impl<A: hal::Api, F: AllResources<A>> Default for Cached<A, F> {
    fn default() -> Self {
        Self::Empty
    }
} */

/* pub struct IdCache {
    cache: crate::FastHashSet<Cached>,
} */

pub trait TypedId {
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self;
    fn unzip(self) -> (Index, Epoch, Backend);
}

impl<T> TypedId for Id<T> {
    fn zip(index: Index, epoch: Epoch, backend: Backend) -> Self {
        assert_eq!(0, epoch >> (32 - BACKEND_BITS));
        let v = index as u64 | ((epoch as u64) << 32) | ((backend as u64) << (64 - BACKEND_BITS));
        Id(NonZeroU64::new(v).unwrap(), PhantomData)
    }

    fn unzip(self) -> (Index, Epoch, Backend) {
        (
            self.0.get() as u32,
            (self.0.get() >> 32) as u32 & EPOCH_MASK,
            self.backend(),
        )
    }
}

pub type AdapterId = /*Id<crate::instance::Adapter<Dummy>>*/AdapterId2;
pub type SurfaceId = Id<crate::instance::Surface>;
// Device
pub type DeviceId = Id2<crate::device::Device<Dummy>>;
pub type QueueId = DeviceId;
// Resource
pub type BufferId = Id<crate::resource::Buffer<Dummy>>;
pub type TextureViewId = Id<crate::resource::TextureView<Dummy>>;
pub type TextureId = Id<crate::resource::Texture<Dummy>>;
pub type SamplerId = /*Id<crate::resource::Sampler<Dummy>>*/SamplerId2;
// Binding model
pub type BindGroupLayoutId = /*Id<crate::binding_model::BindGroupLayout<Dummy>>*/BindGroupLayoutId2;
pub type PipelineLayoutId = /*Id<crate::binding_model::PipelineLayout<Dummy>>*/PipelineLayoutId2;
pub type BindGroupId = /*Id<crate::binding_model::BindGroup<Dummy>>*/BindGroupId2;
// Pipeline
pub type ShaderModuleId = /*Id<crate::pipeline::ShaderModule<Dummy>>*/ShaderModuleId2;
pub type RenderPipelineId = /*Id<crate::pipeline::RenderPipeline<Dummy>>*/RenderPipelineId2;
pub type ComputePipelineId = /*Id<crate::pipeline::ComputePipeline<Dummy>>*/ComputePipelineId2;
// Command
pub type CommandEncoderId = CommandBufferId;
pub type CommandBufferId = Id<crate::command::CommandBuffer<Dummy>>;
pub type RenderPassEncoderId<'a> = *mut crate::command::RenderPass<'a>;
pub type ComputePassEncoderId<'a> = *mut crate::command::ComputePass<'a>;
pub type RenderBundleEncoderId<'a> = *mut crate::command::RenderBundleEncoder<'a>;
pub type RenderBundleId = /*Id<crate::command::RenderBundle<Dummy>>*/RenderBundleId2;
pub type QuerySetId = /*Id<crate::resource::QuerySet<Dummy>>*/QuerySetId2;
// Presentation
// pub type PresentationId = PresentationId2;

#[test]
fn test_id_backend() {
    for &b in &[
        Backend::Empty,
        Backend::Vulkan,
        Backend::Metal,
        Backend::Dx12,
        Backend::Dx11,
        Backend::Gl,
    ] {
        let id: Id<()> = Id::zip(1, 0, b);
        assert_eq!(id.backend(), b);
    }
}
