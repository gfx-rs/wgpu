extern crate env_logger;
extern crate javelin;

use std::{env, fs};

fn main() {
    env_logger::init();

    let args = env::args().collect::<Vec<_>>();
    let input = fs::read(&args[1]).unwrap();

    let mut transpiler = javelin::Transpiler::new();
    let module = transpiler.load(&input).unwrap();

    let options = javelin::msl::Options {};
    let msl = module.to_msl(&options).unwrap();
    println!("{}", msl);
}

