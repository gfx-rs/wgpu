extern crate env_logger;
extern crate javelin;

use std::{env, fs};

fn main() {
    env_logger::init();

    let args = env::args().collect::<Vec<_>>();
    let input = fs::read(&args[1]).unwrap();

    let module = javelin::front::spirv::parse_u8_slice(&input).unwrap();
    //println!("{:?}", module);

    let options = javelin::back::msl::Options {};
    let msl = module.to_msl(&options).unwrap();
    println!("{}", msl);
}

