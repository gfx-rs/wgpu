use std::{fs, path::Path};

fn test_rosetta(dir_name: &str) {
    let dir_path = Path::new("test-data").join(dir_name);
    #[cfg_attr(not(feature = "serde"), allow(unused))]
    let expected = fs::read_to_string(dir_path.join("module.ron")).unwrap();

    #[cfg(feature = "deserialize")]
    {
        use naga::back::*;
        let module = ron::de::from_str::<naga::Module>(&expected).unwrap();
        naga::proc::Validator::new().validate(&module).unwrap();

        #[cfg(feature = "spv-out")]
        {
            let spv = spv::Writer::new(&module.header, spv::WriterFlags::NONE).write(&module);
            assert!(spv.len() > 0);
        }
    }

    #[cfg(feature = "serialize")]
    {
        use naga::front::*;

        fn check(name: &str, module: &naga::Module, expected: &str) {
            let output = ron::ser::to_string_pretty(&module, Default::default()).unwrap();
            if output != expected {
                let changeset = difference::Changeset::new(expected, &output, "");
                panic!("'{}' failed with diff:\n{}", name, changeset);
            }
        }

        #[cfg(feature = "wgsl-in")]
        {
            let input = fs::read_to_string(dir_path.join("x.wgsl")).unwrap();
            let module = wgsl::parse_str(&input).unwrap();
            check("wgsl", &module, &expected);
        }

        #[cfg(feature = "glsl-in")]
        {
            if let Ok(input) = fs::read_to_string(dir_path.join("x.vert")) {
                let module = glsl::parse_str(
                    &input,
                    "main",
                    naga::ShaderStage::Vertex,
                    Default::default(),
                )
                .unwrap();
                check("vert", &module, &expected);
            }
            if let Ok(input) = fs::read_to_string(dir_path.join("x.frag")) {
                let module = glsl::parse_str(
                    &input,
                    "main",
                    naga::ShaderStage::Fragment,
                    Default::default(),
                )
                .unwrap();
                check("frag", &module, &expected);
            }
            if let Ok(input) = fs::read_to_string(dir_path.join("x.comp")) {
                let module = glsl::parse_str(
                    &input,
                    "main",
                    naga::ShaderStage::Compute,
                    Default::default(),
                )
                .unwrap();
                check("comp", &module, &expected);
            }
        }
    }
}

#[test]
fn simple() {
    test_rosetta("simple");
}
