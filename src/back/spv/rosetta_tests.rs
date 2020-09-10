use std::{fs, path::Path};

const TEST_PATH: &str = "test-data";

fn rosetta_test(file_name: &str) {
    let test_dir = Path::new(TEST_PATH);
    let input = fs::read_to_string(test_dir.join(file_name)).unwrap();
    let module: crate::Module = ron::de::from_str(&input).unwrap();

    let spv = super::Writer::new(&module.header, super::WriterFlags::NONE).write(&module);
    assert!(spv.len() > 0, "spv.len() > 0");
}

#[test]
fn simple() {
    rosetta_test("simple/simple.expected.ron")
}
