mod core_tests {
    // Contains all the helper routines which facilitate this testing framework.
    mod common;

    // All files containing tests
    mod instance;
    mod device;
    mod vertex_indexes;
}

pub use core_tests::*;
