/// FFI-friendly analogue of [`std::option::Option`].
///
/// In some cases, Rust's standard `Option` type is FFI-friendly, in that `T` and `Option<T>` map
/// to the same C++ type. For example, both `&U` and `Option<&U>` can be represented as `U *` in
/// C++.
///
/// For other types, `Option<T>` may not be FFI-safe. For such cases, this type is a `repr(u8)`
/// analog to the standard
///
/// See also: <https://doc.rust-lang.org/nomicon/ffi.html#the-nullable-pointer-optimization>
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FfiOption<T> {
    Some(T),
    None,
}

impl<T> FfiOption<T> {
    pub fn to_std(self) -> std::option::Option<T> {
        match self {
            Self::Some(value) => Some(value),
            Self::None => None,
        }
    }

    pub fn as_ref(&self) -> std::option::Option<&T> {
        match *self {
            Self::Some(ref value) => Some(value),
            Self::None => None,
        }
    }
}

#[macro_export]
macro_rules! assert_ffi_safe {
    ($ty:ty) => {
        const _: () = {
            #[deny(improper_ctypes_definitions)]
            #[export_name = concat!("_compile_check_ffi_export_", stringify!($ty))]
            pub extern "C" fn _compile_check_ffi_export(_x: $ty) {}
        };
    };
}
