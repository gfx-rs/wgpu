//! Wrappers for resources so they automatically destroy themselves of drop.
//! Core resources sometimes need to create other resources (an extra buffer, etc.).
//! If creating one of these resources fails, it could leak the others if just
//! a `?` is used, but to prevent this leak messy code is needed. By implementing
//! special raw resources that wrap hal resources during creation, but on drop
//! destroy the resource.

use std::boxed::Box;

use scopeguard::{guard, ScopeGuard};

use crate::device::Device;

/// Defines the type for a resource that will be destroyed on drop
macro_rules! scoped_resource {
    ($hal_out:ty, $lifetime:lifetime) => {
        ScopeGuard<$hal_out, impl FnOnce($hal_out) + $lifetime>
    }
}

macro_rules! create_raw_docs {
    ($fn_create:ident) => {
        concat!(
            "Creates a raw resource that automatically destroys itself on drop.\
        \n\
        \n # Safety\
        \n Same as [`hal::Device::",
            stringify!($fn_create),
            "`] plus\
        \n - The buffer must not be used in any resources\
        \n   that hold a reference it (e.g. bind groups)\
        \n    - Note: this likely indicates a resource leak"
        )
    };
}

macro_rules! create_raw_multi_args {
    ($($fn_name:ident($($arg:ident:$arg_ty:ty,)*) -> Result<$hal_out:ty, $err_ty:ty> { wraps $fn_create:ident, destroy $fn_destroy:ident })*) => {
        impl Device {
            $(
            #[doc = create_raw_docs!($fn_create)]
            pub(crate) unsafe fn $fn_name<'a>(
                &'a self,
                $($arg:$arg_ty,)*
            ) -> Result<scoped_resource!($hal_out, 'a), $err_ty> {
                let raw = unsafe { self.raw().$fn_create($($arg,)*)? };

                return Ok(scopeguard::guard(raw, |r| unsafe {self.raw().$fn_destroy(r)}));
            })*
        }
    };
}

/// [`create_raw_multi_args`], but with only a descriptor argument.
macro_rules! create_raw {
    ($($fn_name:ident($desc:ty) -> $hal_out:ty { wraps $fn_create:ident, destroy $fn_destroy:ident })*) => {
        create_raw_multi_args! {
            $($fn_name(desc: &$desc,) -> Result<$hal_out, hal::DeviceError> { wraps $fn_create, destroy $fn_destroy })*
        }
    };
}

create_raw! {
    create_raw_texture(hal::TextureDescriptor) -> Box<dyn hal::DynTexture> {
        wraps create_texture,
        destroy destroy_texture
    }
    create_raw_acceleration_structure(hal::AccelerationStructureDescriptor) -> Box<dyn hal::DynAccelerationStructure> {
        wraps create_acceleration_structure,
        destroy destroy_acceleration_structure
    }
    create_raw_bind_group_layout(hal::BindGroupLayoutDescriptor) -> Box<dyn hal::DynBindGroupLayout> {
        wraps create_bind_group_layout,
        destroy destroy_bind_group_layout
    }
    create_raw_pipeline_layout(hal::PipelineLayoutDescriptor<dyn hal::DynBindGroupLayout>) -> Box<dyn hal::DynPipelineLayout> {
        wraps create_pipeline_layout,
        destroy destroy_pipeline_layout
    }
}

create_raw_multi_args! {
    create_raw_texture_view(
        texture: &dyn hal::DynTexture,
        desc: &hal::TextureViewDescriptor,
    ) -> Result<Box<dyn hal::DynTextureView>, hal::DeviceError> {
        wraps create_texture_view,
        destroy destroy_texture_view
    }
    create_raw_shader_module(
        desc: &hal::ShaderModuleDescriptor,
        shader: hal::ShaderInput,
    ) -> Result<Box<dyn hal::DynShaderModule>, hal::ShaderError> {
        wraps create_shader_module,
        destroy destroy_shader_module
    }
}

impl Device {
    #[doc = create_raw_docs!(create_buffer)]
    pub(crate) unsafe fn create_raw_buffer<'a>(
        &'a self,
        desc: &hal::BufferDescriptor,
    ) -> Result<(scoped_resource!(Box<dyn hal::DynBuffer>, 'a), u64), hal::DeviceError> {
        let (raw, size) = unsafe { self.raw().create_buffer(desc)? };

        Ok((
            guard(raw, |raw| unsafe { self.raw().destroy_buffer(raw) }),
            size,
        ))
    }
}
