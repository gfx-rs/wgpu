//! Regression test for https://github.com/gfx-rs/wgpu/issues/8082: a shader-module label
//! (or source) that contains embedded NUL characters caused Naga's SPIR-V backend to emit
//! invalid SPIR-V. SPIR-V literal strings are NUL-terminated, but Naga's string encoder
//! packed the raw UTF-8 bytes without any interior-NUL handling, so the emitted instruction
//! word count disagreed with where a parser stops the string.

use std::io::Write as _;
use std::process::{Command, Stdio};

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Emit SPIR-V for a trivial compute shader with debug info forced on, then run
/// the result through `spirv-val`. Panics if `spirv-val` rejects the module.
fn write_and_validate(file_name: &str, source_code: &str) {
    let module = naga::front::wgsl::parse_str("@compute @workgroup_size(1) fn main() {}").unwrap();

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    let info = validator.validate(&module).unwrap();

    let mut options = naga::back::spv::Options::default();
    // `Options::default()` only sets `DEBUG` under `debug_assertions`, so force it
    // on to guarantee the debug-info strings are emitted regardless of build mode.
    options.flags |= naga::back::spv::WriterFlags::DEBUG;
    options.debug_info = Some(naga::back::spv::DebugInfo {
        source_code,
        file_name,
        language: naga::back::spv::SourceLanguage::Unknown,
    });

    let pipeline_options = naga::back::spv::PipelineOptions {
        entry_point: "main".to_string(),
        shader_stage: naga::ShaderStage::Compute,
    };

    let words =
        naga::back::spv::write_vec(&module, &info, &options, Some(&pipeline_options)).unwrap();
    let bytes = words_to_bytes(&words);

    let mut child = Command::new("spirv-val")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect(
            "Failed to execute spirv-val. It can be installed \
            by installing the Vulkan SDK and adding it to your path.",
        );

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&bytes)
        .expect("failed to write SPIR-V module to spirv-val stdin");

    let output = child
        .wait_with_output()
        .expect("failed to wait for spirv-val");

    assert!(
        output.status.success(),
        "spirv-val rejected naga output:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Sanity check: a trivial shader with clean strings must validate, proving the
/// harness itself is sound.
#[cfg_attr(miri, ignore)]
#[test]
fn clean_label_validates() {
    write_and_validate("shader.wgsl", "@compute @workgroup_size(1) fn main() {}");
}

/// The exact label from the issue #8082 trace, exercising the `OpString`
/// (debug source file name) path.
#[cfg_attr(miri, ignore)]
#[test]
fn embedded_nul_in_file_name() {
    write_and_validate(
        "\0)@\u{1}\u{1}ompute\n@workgroup_size(1)\nfn main() {\n}\u{15}\0\0\u{1}\0\u{1}\u{1}\u{1})@compute\n@workgroup_size(1)\nfn main() {\n}\u{15}\0\0\u{1}\0",
        "@compute @workgroup_size(1) fn main() {}",
    );
}

/// An embedded NUL in the source code, exercising the `OpSource` path.
#[cfg_attr(miri, ignore)]
#[test]
fn embedded_nul_in_source_code() {
    write_and_validate("shader.wgsl", "before\0after");
}

/// A NUL that lands exactly on a 4-byte boundary, exercising the trailing-NUL
/// branch of `debug_str_bytes_to_words`.
#[cfg_attr(miri, ignore)]
#[test]
fn embedded_nul_at_word_boundary() {
    write_and_validate("abcd\0efgh", "@compute @workgroup_size(1) fn main() {}");
}
