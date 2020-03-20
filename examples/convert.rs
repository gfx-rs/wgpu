use serde::{Serialize, Deserialize};
use std::{env, fs};

#[derive(Hash, PartialEq, Eq, Serialize, Deserialize)]
struct BindSource {
    set: u32,
    binding: u32,
}

#[derive(Serialize, Deserialize)]
struct BindTarget {
    buffer: Option<u8>,
    texture: Option<u8>,
    sampler: Option<u8>,
    mutable: bool,
}

#[derive(Default, Serialize, Deserialize)]
struct Parameters {
    metal_bindings: naga::FastHashMap<BindSource, BindTarget>,
}

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
        println!("{:#?}", module);
        return;
    }

    let param_path = std::path::PathBuf::from(&args[1])
        .with_extension("ron");
    let params = match fs::read_to_string(param_path) {
        Ok(string) => ron::de::from_str(&string).unwrap(),
        Err(_) => Parameters::default(),
    };

    if args[2].ends_with(".metal") {
        use naga::back::msl;
        let mut binding_map = msl::BindingMap::default();
        for (key, value) in params.metal_bindings {
            binding_map.insert(
                msl::BindSource {
                    set: key.set,
                    binding: key.binding,
                },
                msl::BindTarget {
                    buffer: value.buffer,
                    texture: value.texture,
                    sampler: value.sampler,
                    mutable: value.mutable,
                },
            );
        }
        let options = msl::Options {
            binding_map: &binding_map,
        };
        let msl = msl::write_string(&module, options).unwrap();
        fs::write(&args[2], msl).unwrap();
    } else {
        panic!("Unknown output: {:?}", args[2]);
    }
}
