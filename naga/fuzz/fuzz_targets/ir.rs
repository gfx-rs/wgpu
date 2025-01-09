#![cfg_attr(enable_fuzzing, no_main)]

#[cfg(enable_fuzzing)]
mod fuzz {
    use libfuzzer_sys::fuzz_target;

    fuzz_target!(|module: naga::Module| {
        use naga::valid as v;
        // Check if the module validates without errors.
        //TODO: may also fuzz the flags and capabilities
        let mut validator =
            v::Validator::new(v::ValidationFlags::all(), v::Capabilities::default());
        let _result = validator.validate(&module);
    });
}

#[cfg(not(enable_fuzzing))]
fn main() {}
