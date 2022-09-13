// TODO: Do not merge without docs
#![allow(missing_docs)]

#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {
        assert!( $( $arg )* )
    }
}

#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {
        assert_eq!( $( $arg )* )
    }
}

#[cfg(feature = "strict_asserts")]
#[macro_export]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {
        assert_ne!( $( $arg )* )
    }
}

#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert {
    ( $( $arg:tt )* ) => {
        debug_assert!( $( $arg )* )
    };
}

#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert_eq {
    ( $( $arg:tt )* ) => {
        debug_assert_eq!( $( $arg )* )
    };
}

#[cfg(not(feature = "strict_asserts"))]
#[macro_export]
macro_rules! strict_assert_ne {
    ( $( $arg:tt )* ) => {
        debug_assert_ne!( $( $arg )* )
    };
}
