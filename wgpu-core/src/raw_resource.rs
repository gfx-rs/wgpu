//! Wrappers for resources so they automatically destroy themselves of drop.
//! Core resources sometimes need to create other resources (an extra buffer, etc.).
//! If creating one of these resources fails, it could leak the others if just
//! a `?` is used, but to prevent this leak messy code is needed. By implementing
//! special raw resources that wrap hal resources during creation, but on drop
//! destroy the resource.

use core::mem::ManuallyDrop;

use crate::device::Device;

macro_rules! resource {
    ($($resource_name:ident => $hal_resource:ty { destroy: $destroy_call:ident child of $parent:ty })*) => {
        $(
            pub(crate) struct $resource_name<'a> {
                raw: core::mem::ManuallyDrop<alloc::boxed::Box<$hal_resource>>,
                parent: &'a $parent,
            }

            impl Drop for $resource_name<'_> {
                fn drop(&mut self) {
                    // # Safety: we are in the drop impl and won't use raw anymore
                    let raw = unsafe { core::mem::ManuallyDrop::take(&mut self.raw) };
                    unsafe { self.parent.raw().$destroy_call(raw) }
                }
            }

            impl core::ops::Deref for $resource_name<'_> {
                type Target = $hal_resource;

                fn deref(&self) -> &Self::Target {
                    self.raw.as_ref()
                }
            }

            impl core::convert::AsRef<$hal_resource> for $resource_name<'_> {
                fn as_ref(&self) -> &$hal_resource {
                    self.raw.as_ref()
                }
            }

            impl $resource_name<'_> {
                /// Unwraps this raw structure, returning the inner value.
                /// This means that the value will not be destroyed on drop.
                pub(crate) fn unwrap_raw(self) -> alloc::boxed::Box<$hal_resource> {
                    // We need to get out the raw value. We can't destructure this as it
                    // implements drop (which we want to skip here). Therefore, we
                    // read the raw field. Pointers don't need to be destroyed.
                    let mut wrapped = core::mem::ManuallyDrop::new(self);
                    // # Safety:
                    // We will never use the other version of this, and it will not be
                    // used in the drop because `wrapped` is in a manually drop.
                    let raw = unsafe { core::mem::ManuallyDrop::take(&mut wrapped.raw) };

                    return raw
                }
            }
        )*
    }
}

macro_rules! device_resource {
    ($($resource_name:ident => $hal_resource:ty { destroy: $destroy_call:ident })*) => {
        resource!{
            $(
                $resource_name => $hal_resource {
                    destroy: $destroy_call
                    child of $crate::device::Device
                }
            )*
        }
    };
}

device_resource!(
    RawBuffer => dyn hal::DynBuffer {
        destroy: destroy_buffer
    } 
);

impl Device {
    /// Creates a raw buffer that automatically destroys itself on drop.
    /// 
    /// # Safety
    /// Same as [`hal::Device::create_buffer`] plus
    /// - The buffer must not be used in any resources
    ///   that hold a reference it (e.g. bind groups)
    ///    - Note: this likely indicates a resource leak
    pub(crate) unsafe fn create_raw_buffer<'a>(&'a self, desc: &hal::BufferDescriptor) -> Result<(RawBuffer<'a>, u64), hal::DeviceError> {
        let (raw, size) = unsafe { self.raw().create_buffer(desc)? };

        Ok((RawBuffer {
            raw: ManuallyDrop::new(raw),
            parent: self,
        }, size))
    }
}