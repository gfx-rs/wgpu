use serde::{Deserialize, Serialize};
use std::{env, fs, path::Path};

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

    if args.len() <= 1 {
        println!("Call with <input> <output>");
        return;
    }
    #[cfg(any(feature = "glsl", feature = "glsl-new"))]
    let prefer_glsl_new =
        !cfg!(feature = "glsl") || env::var("PREFER_GLSL_NEW").unwrap_or_default() == "1";
    let module = match Path::new(&args[1])
        .extension()
        .expect("Input has no extension?")
        .to_str()
        .unwrap()
    {
        #[cfg(feature = "spirv")]
        "spv" => {
            let input = fs::read(&args[1]).unwrap();
            naga::front::spv::parse_u8_slice(&input).unwrap()
        }
        "wgsl" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            naga::front::wgsl::parse_str(&input).unwrap()
        }
        #[cfg(any(feature = "glsl", feature = "glsl-new"))]
        "vert" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            let mut module: Option<naga::Module> = None;
            if prefer_glsl_new {
                #[cfg(feature = "glsl-new")]
                {
                    module = Some(
                        naga::front::glsl_new::parse_str(
                            &input,
                            "main".to_string(),
                            naga::ShaderStage::Vertex,
                        )
                        .unwrap(),
                    )
                }
            }
            if module.is_none() {
                #[cfg(feature = "glsl")]
                {
                    module = Some(
                        naga::front::glsl::parse_str(
                            &input,
                            "main".to_string(),
                            naga::ShaderStage::Vertex,
                        )
                        .unwrap(),
                    )
                }
            }
            module.unwrap()
        }
        #[cfg(feature = "glsl")]
        "frag" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            naga::front::glsl::parse_str(&input, "main".to_string(), naga::ShaderStage::Fragment)
                .unwrap()
        }
        #[cfg(feature = "glsl")]
        "comp" => {
            let input = fs::read_to_string(&args[1]).unwrap();
            naga::front::glsl::parse_str(&input, "main".to_string(), naga::ShaderStage::Compute)
                .unwrap()
        }
        other => panic!("Unknown input extension: {}", other),
    };

    if args.len() <= 2 {
        println!("{:#?}", module);
        return;
    }

    let param_path = std::path::PathBuf::from(&args[1]).with_extension("ron");
    let params = match fs::read_to_string(param_path) {
        Ok(string) => ron::de::from_str(&string).unwrap(),
        Err(_) => Parameters::default(),
    };

    match Path::new(&args[2])
        .extension()
        .expect("Output has no extension?")
        .to_str()
        .unwrap()
    {
        "metal" => {
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
        }
        #[cfg(feature = "spirv")]
        "spv" => {
            use naga::back::spv;

            let debug_flag = args.get(3).map_or(spv::WriterFlags::DEBUG, |arg| {
                if arg.parse().unwrap() {
                    spv::WriterFlags::DEBUG
                } else {
                    spv::WriterFlags::NONE
                }
            });

            let spv = spv::Writer::new(&module.header, debug_flag).write(&module);

            let bytes = spv
                .iter()
                .fold(Vec::with_capacity(spv.len() * 4), |mut v, w| {
                    v.extend_from_slice(&w.to_le_bytes());
                    v
                });

            fs::write(&args[2], bytes.as_slice()).unwrap();
        }
        #[cfg(feature = "glsl-out")]
        "vert" | "frag" => {
            use naga::back::glsl;

            let mut file = fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&args[2])
                .unwrap();

            glsl::write(&module, &mut file).unwrap();
        }
        other => {
            panic!("Unknown output extension: {}", other);
        }
    }
}
