fn convert_wgsl(name: &str) {
    let path = format!("{}/test-data/{}.wgsl", env!("CARGO_MANIFEST_DIR"), name);
    let input = std::fs::read_to_string(path).unwrap();
    let _module = naga::front::wgsl::parse_str(&input).unwrap();
}

#[test]
fn convert_quad() {
    convert_wgsl("quad");
}

#[test]
fn convert_boids() {
    convert_wgsl("boids");
}
