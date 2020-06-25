#[cfg(feature = "glsl")]
use spirv::ExecutionModel;

fn load_test_data(name: &str) -> String {
    let path = format!("{}/test-data/{}", env!("CARGO_MANIFEST_DIR"), name);
    std::fs::read_to_string(path).unwrap()
}

fn load_wgsl(name: &str) -> naga::Module {
    let input = load_test_data(name);
    naga::front::wgsl::parse_str(&input).unwrap()
}

#[cfg(feature = "glsl")]
fn load_glsl(name: &str, entry: &str, exec: ExecutionModel) -> naga::Module {
    let input = load_test_data(name);
    naga::front::glsl::parse_str(&input, entry.to_owned(), exec).unwrap()
}

#[test]
fn convert_quad() {
    let module = load_wgsl("quad.wgsl");
    naga::proc::Validator::new().validate(&module).unwrap();
    {
        use naga::back::msl;
        let mut binding_map = msl::BindingMap::default();
        binding_map.insert(
            msl::BindSource { set: 0, binding: 0 },
            msl::BindTarget {
                buffer: None,
                texture: Some(1),
                sampler: None,
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource { set: 0, binding: 1 },
            msl::BindTarget {
                buffer: None,
                texture: None,
                sampler: Some(1),
                mutable: false,
            },
        );
        let options = msl::Options {
            binding_map: &binding_map,
        };
        msl::write_string(&module, options).unwrap();
    }
}

#[test]
fn convert_boids() {
    let module = load_wgsl("boids.wgsl");
    naga::proc::Validator::new().validate(&module).unwrap();
    {
        use naga::back::msl;
        let mut binding_map = msl::BindingMap::default();
        binding_map.insert(
            msl::BindSource { set: 0, binding: 0 },
            msl::BindTarget {
                buffer: Some(0),
                texture: None,
                sampler: None,
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource { set: 0, binding: 1 },
            msl::BindTarget {
                buffer: Some(1),
                texture: None,
                sampler: Some(1),
                mutable: false,
            },
        );
        binding_map.insert(
            msl::BindSource { set: 0, binding: 2 },
            msl::BindTarget {
                buffer: Some(2),
                texture: None,
                sampler: Some(1),
                mutable: false,
            },
        );
        let options = msl::Options {
            binding_map: &binding_map,
        };
        msl::write_string(&module, options).unwrap();
    }
}

#[cfg(feature = "glsl")]
#[test]
#[ignore]
fn convert_phong_lighting() {
    let module = load_glsl("glsl_phong_lighting.frag", "main", ExecutionModel::Fragment);
    naga::proc::Validator::new().validate(&module).unwrap();

    let header = naga::Header {
        version: (1, 0, 0),
        generator: 1234,
    };
    let writer_flags = naga::back::spv::WriterFlags::empty();
    let mut w = naga::back::spv::Writer::new(&header, writer_flags);
    w.write(&module);
}
