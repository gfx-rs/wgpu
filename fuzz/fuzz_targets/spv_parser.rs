#![no_main]
use libfuzzer_sys::fuzz_target;
use naga::front::spv::Parser;

fuzz_target!(|data: Vec<u32>| {
    // Ensure the parser can handle potentially malformed data without crashing.
    let _result = Parser::new(data.into_iter()).parse();
});
