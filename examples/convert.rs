use std::{env, fs};

fn main() {
    env_logger::init();

    let args = env::args().collect::<Vec<_>>();
    let input = fs::read(&args[1]).unwrap();

    let module = naga::front::spirv::parse_u8_slice(&input).unwrap();
    //println!("{:?}", module);

    let mut binding_map = naga::back::msl::BindingMap::default();
    binding_map.insert(
        naga::back::msl::BindSource { set: 0, binding: 0 },
        naga::back::msl::BindTarget { buffer: None, texture: None, sampler: None },
    );
    let options = naga::back::msl::Options {
        binding_map: &binding_map,
    };
    let msl = naga::back::msl::write_string(&module, options).unwrap();
    println!("{}", msl);
}
