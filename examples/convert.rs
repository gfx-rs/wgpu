use std::{env, fs};

fn main() {
    env_logger::init();

    let args = env::args().collect::<Vec<_>>();

    let module = if args.len() <= 1 {
        println!("Call with <input> <output>");
        return
    } else if args[1].ends_with(".spv") {
        let input = fs::read(&args[1]).unwrap();
        naga::front::spirv::parse_u8_slice(&input).unwrap()
    } else if args[1].ends_with(".wgsl") {
        let input = fs::read_to_string(&args[1]).unwrap();
        naga::front::wgsl::parse_str(&input).unwrap()
    } else {
        panic!("Unknown input: {:?}", args[1]);
    };

    if args.len() <= 2 {
        println!("{:?}", module);
        return
    } else if args[2].ends_with(".msl") {
        use naga::back::msl;
        let mut binding_map = msl::BindingMap::default();
        binding_map.insert(
            msl::BindSource { set: 0, binding: 0 },
            msl::BindTarget { buffer: None, texture: Some(1), sampler: None },
        );
        binding_map.insert(
            msl::BindSource { set: 0, binding: 1 },
            msl::BindTarget { buffer: None, texture: None, sampler: Some(1) },
        );
        let options = msl::Options {
            binding_map: &binding_map,
        };
        let msl = msl::write_string(&module, options).unwrap();
        fs::write(&args[2], msl).unwrap();
    } else {
        panic!("Unknown output: {:?}", args[2]);
    }
}
