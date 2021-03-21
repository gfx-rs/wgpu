//TODO: consider converting this to snapshots?

#[cfg(feature = "glsl-in")]
fn _check_glsl(name: &str) {
    let path = std::path::PathBuf::from("tests/cases").join(name);
    let input = std::fs::read_to_string(path).unwrap();
    let stage = if name.ends_with(".vert") {
        naga::ShaderStage::Vertex
    } else if name.ends_with(".frag") {
        naga::ShaderStage::Fragment
    } else if name.ends_with(".comp") {
        naga::ShaderStage::Compute
    } else {
        panic!("Unknown extension in {:?}", name)
    };

    let mut entry_points = naga::FastHashMap::default();
    entry_points.insert("main".to_string(), stage);
    match naga::front::glsl::parse_str(
        &input,
        &naga::front::glsl::Options {
            entry_points,
            defines: Default::default(),
        },
    ) {
        Ok(m) => {
            match naga::valid::Validator::new(naga::valid::ValidationFlags::all()).validate(&m) {
                Ok(_info) => (),
                //TODO: panic
                Err(e) => log::error!("Unable to validate {}: {:?}", name, e),
            }
        }
        Err(e) => panic!("Unable to parse {}: {:?}", name, e),
    };
}

#[cfg(feature = "glsl-in")]
#[test]
fn parse_glsl() {
    //check_glsl("glsl_constant_expression.vert"); //TODO
    //check_glsl("glsl_if_preprocessor.vert");
    //check_glsl("glsl_preprocessor_abuse.vert");
    //check_glsl("glsl_vertex_test_shader.vert"); //TODO
}
