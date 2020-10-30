use std::{fs, path::Path};

const TEST_PATH: &str = "test-data";

fn rosetta_test(file_name: &str) {
    if true {
        return; //TODO: fix this test
    }
    let file_path = Path::new(TEST_PATH).join(file_name);
    let input = fs::read(&file_path).unwrap();

    let module = super::parse_u8_slice(&input, &Default::default()).unwrap();
    let output = ron::ser::to_string_pretty(&module, Default::default()).unwrap();

    let expected = fs::read_to_string(file_path.with_extension("expected.ron")).unwrap();

    difference::assert_diff!(output.as_str(), expected.as_str(), "", 0);
}

#[test]
fn simple() {
    rosetta_test("simple/simple.spv")
}
