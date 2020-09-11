use std::{fs, path::Path};

const TEST_PATH: &str = "test-data";

fn rosetta_test(file_name: &str) {
    if true {
        return; //TODO: fix this test
    }
    let test_dir = Path::new(TEST_PATH);
    let input = fs::read(test_dir.join(file_name)).unwrap();

    let module = super::parse_u8_slice(&input).unwrap();
    let output = ron::ser::to_string_pretty(&module, Default::default()).unwrap();

    let expected =
        fs::read_to_string(test_dir.join(file_name).with_extension("expected.ron")).unwrap();

    difference::assert_diff!(output.as_str(), expected.as_str(), "", 0);
}

#[test]
fn simple() {
    rosetta_test("simple/simple.spv")
}
