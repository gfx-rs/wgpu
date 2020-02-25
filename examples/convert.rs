extern crate env_logger;
extern crate naga;

use std::{env, fs};

fn main() {
    env_logger::init();

    let args = env::args().collect::<Vec<_>>();
    let input = fs::read(&args[1]).unwrap();

    let module = naga::front::spirv::parse_u8_slice(&input).unwrap();
    //println!("{:?}", module);

    let options = naga::back::msl::Options {};
    let msl = naga::back::msl::write_string(&module, &options).unwrap();
    println!("{}", msl);
}

