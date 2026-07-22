/// Macro for use with [`macro_rules_attribute::derive`] to derive a `const fn default()`
/// in addition to `impl Default`.
///
/// Generics are not supported.
///
/// When using this macro on an enum, the default variant must be marked with `#[custom(default)]`,
/// rather than `#[default]` as the built-in derive macro does. This is a limitation of
/// [`macro_rules_attribute::derive`].
macro_rules! ConstDefault {
    // Simplify by dropping attributes and visibility from an `enum`.
    (
        $(#[$attr:meta])+
        $vis:vis enum $enum_name:ident {
            $($body:tt)*
        }
    ) => {
        $crate::macros::ConstDefault!(enum $enum_name { $($body)* });
    };

    // Simplify by dropping attributes and visibility from a `struct`.
    (
        $(#[$attr:meta])+
        $vis:vis struct $struct_name:ident $($rest:tt)*
    ) => {
        $crate::macros::ConstDefault!(struct $struct_name $($rest)*);
    };

    // Base case: match an `enum` with `#[default]` on the first variant.
    (
        enum $enum_name:ident {
            #[custom(default)]
            $(#[$other_variant_attr:meta])*
            $variant_name:ident $(= $discriminant:expr)?,
            $( $rest_of_body:tt )*
        }
    ) => {
        impl $enum_name {
            /// This function is identical to [`Default::default()`] except that it is a `const fn`.
            pub const fn default() -> Self {
                Self::$variant_name
            }
        }

        impl $crate::macros::ConstDefaultHelper for $enum_name {
            const DEFAULT: Self = Self::$variant_name;
        }

        impl ::core::default::Default for $enum_name {
            fn default() -> Self {
                Self::$variant_name
            }
        }
    };

    // Recursive case: remove an irrelevant attribute from the first variant.
    (
        enum $enum_name:ident {
            #[$attr_that_is_not_default:meta]
            $( $rest_of_body:tt )*
        }
    ) => {
        $crate::macros::ConstDefault!(enum $enum_name { $( $rest_of_body )* });
    };


    // Recursive cases: remove the first variant of an enum (which may or may not have fields),
    // because we have checked it is not the default.
    (
        enum $enum_name:ident {
            $variant_name:ident $(= $discriminant:expr)?,
            $( $rest_of_body:tt )*
        }
    ) => {
        $crate::macros::ConstDefault!(enum $enum_name { $( $rest_of_body )* });
    };
    (
        enum $enum_name:ident {
            $variant_name:ident ( $( $field_tokens:tt )* ),
            $( $rest_of_body:tt )*
        }
    ) => {
        $crate::macros::ConstDefault!(enum $enum_name { $( $rest_of_body )* });
    };
    (
        enum $enum_name:ident {
            $variant_name:ident { $( $field_tokens:tt )* },
            $( $rest_of_body:tt )*
        }
    ) => {
        $crate::macros::ConstDefault!(enum $enum_name { $( $rest_of_body )* });
    };

    // Struct (without generics)
    (
        $(#[$struct_attr:meta])*
        $vis:vis struct $struct_name:ident {
            $(
                $(#[$field_attr:meta])*
                $field_vis:vis $field_name:ident : $field_type:ty,
            )*
        }
    ) => {
        impl $struct_name {
            /// This function is identical to [`Default::default()`] except that it is a `const fn`.
            pub const fn default() -> Self {
                <Self as $crate::macros::ConstDefaultHelper>::DEFAULT
            }
        }

        impl $crate::macros::ConstDefaultHelper for $struct_name {
            const DEFAULT: Self = Self {
                $(
                    $field_name: <$field_type as $crate::macros::ConstDefaultHelper>::DEFAULT,
                )*
            };
        }

        impl ::core::default::Default for $struct_name {
            fn default() -> Self {
                <Self as $crate::macros::ConstDefaultHelper>::DEFAULT
            }
        }
    };


    // Tuple struct
    (
        $(#[$struct_attr:meta])*
        $vis:vis struct $struct_name:ident(
            $(
                $(#[$field_attr:meta])*
                $field_vis:vis $field_type:ty
            ),* $(,)?
        );
    ) => {
        impl $struct_name {
            /// This function is identical to [`Default::default()`] except that it is a `const fn`.
            pub const fn default() -> Self {
                <Self as $crate::macros::ConstDefaultHelper>::DEFAULT
            }
        }

        impl $crate::macros::ConstDefaultHelper for $struct_name {
            const DEFAULT: Self = Self(
                $(
                    <$field_type as $crate::macros::ConstDefaultHelper>::DEFAULT,
                )*
            );
        }

        impl ::core::default::Default for $struct_name {
            fn default() -> Self {
                <Self as $crate::macros::ConstDefaultHelper>::DEFAULT
            }
        }
    };
}
pub(crate) use ConstDefault;

/// Default value of a type, as a `const` item.
///
/// This trait is not public and is used solely to help [`ConstDefault`] provide *inherent*
/// `const fn default()`s.
///
/// Specifically, it is implemented both for `wgpu-types` types and `std` types,
/// so that the macro [`ConstDefault`] can call it without needing to understand the type.
pub(crate) trait ConstDefaultHelper {
    const DEFAULT: Self;
}

impl ConstDefaultHelper for bool {
    const DEFAULT: Self = false;
}
impl ConstDefaultHelper for f32 {
    const DEFAULT: Self = 0.;
}
impl ConstDefaultHelper for f64 {
    const DEFAULT: Self = 0.;
}
impl ConstDefaultHelper for i32 {
    const DEFAULT: Self = 0;
}
impl ConstDefaultHelper for i64 {
    const DEFAULT: Self = 0;
}
impl ConstDefaultHelper for u32 {
    const DEFAULT: Self = 0;
}
impl ConstDefaultHelper for u64 {
    const DEFAULT: Self = 0;
}

impl<T> ConstDefaultHelper for Option<T> {
    const DEFAULT: Self = None;
}

impl<T: ConstDefaultHelper, const N: usize> ConstDefaultHelper for [T; N] {
    const DEFAULT: Self = [<T as ConstDefaultHelper>::DEFAULT; N];
}
