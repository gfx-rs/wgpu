use std::process::Command;

/// Path to the built `naga` binary, provided by cargo to integration tests.
fn naga() -> Command {
    Command::new(env!("CARGO_BIN_EXE_naga"))
}

#[test]
fn force_loop_bounding_flag_applies_to_spv() {
    // A shader whose SPIR-V differs when loop bounding is off vs on would be ideal,
    // but at minimum assert the flag parses, is accepted, and compiles successfully.
    let dir = std::env::temp_dir().join("naga_cli_p3_flb");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();
    let out = dir.join("s.spv");
    let r = naga().arg(&src).arg(&out).arg("--force-loop-bounding").arg("false").output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    assert_eq!(&std::fs::read(&out).unwrap()[0..4], &[0x03, 0x02, 0x23, 0x07]);
}

#[test]
fn zero_init_flag_accepts_modes() {
    for mode in ["native", "polyfill", "none"] {
        let out = naga().arg("--help").output().unwrap(); // ensure binary exists
        assert!(out.status.success());
        // parse-only check via a compile:
        let dir = std::env::temp_dir().join("naga_cli_p3_zi");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("s.wgsl");
        std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
        let dst = dir.join(format!("s_{mode}.spv"));
        let r = naga().arg(&src).arg(&dst)
            .arg("--zero-initialize-workgroup-memory").arg(mode).output().unwrap();
        assert!(r.status.success(), "mode {mode} stderr: {}", String::from_utf8_lossy(&r.stderr));
    }
}

#[test]
fn config_nested_common_flat_json() {
    let dir = std::env::temp_dir().join("naga_cli_p3_cfg");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();
    let out = dir.join("s.spv");
    // serde(flatten) keeps common keys flat inside spv_out:
    let r = naga().arg(&src).arg(&out)
        .arg("--config-json").arg(r#"{"spv_out":{"force_loop_bounding":false}}"#)
        .output().unwrap();
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
}

#[test]
fn config_json_and_common_flag_are_exclusive() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config-json").arg("{}")
        .arg("--force-loop-bounding").arg("false")
        .output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
        "expected clap conflict error, got: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn config_json_matches_equivalent_flag() {
    let dir = std::env::temp_dir().join("naga_cli_p2_config");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();
    let out_flag = dir.join("flag.spv");
    let out_cfg = dir.join("cfg.spv");

    // Via flag:
    let a = naga().arg(&src).arg(&out_flag).arg("--spirv-version").arg("1.5").output().unwrap();
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));

    // Via config-json (same lang_version):
    let b = naga()
        .arg(&src).arg(&out_cfg)
        .arg("--config-json").arg(r#"{"spv_out":{"lang_version":[1,5]}}"#)
        .output().unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));

    assert_eq!(std::fs::read(&out_flag).unwrap(), std::fs::read(&out_cfg).unwrap());
}

#[test]
fn config_conflicts_with_option_flag() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config-json").arg("{}")
        .arg("--spirv-version").arg("1.5")
        .output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
        "expected clap conflict error, got: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn config_and_config_json_mutually_exclusive() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config").arg("x.json")
        .arg("--config-json").arg("{}")
        .output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
        "expected clap conflict error, got: {}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn config_composes_with_compact_flag() {
    let dir = std::env::temp_dir().join("naga_cli_p2_cfg_compact");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();
    let out = dir.join("o.spv");
    let r = naga()
        .arg(&src).arg(&out)
        .arg("--config-json").arg(r#"{"spv_out":{"lang_version":[1,3]}}"#)
        .arg("--compact")
        .output().unwrap();
    // Must NOT be a clap conflict error; compaction + config compose.
    assert!(r.status.success(), "stderr: {}", String::from_utf8_lossy(&r.stderr));
    assert!(!String::from_utf8_lossy(&r.stderr).contains("cannot be used with"));
}

#[test]
fn help_lists_all_options() {
    let out = naga().arg("--help").output().unwrap();
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).unwrap();
    for flag in [
        "--validate",
        "--index-bounds-check-policy",
        "--buffer-bounds-check-policy",
        "--image-load-bounds-check-policy",
        "--entry-point",
        "--profile",
        "--shader-model",
        "--spirv-version",
        "--shader-stage",
        "--input-kind",
        "--metal-version",
        "--keep-coordinate-space",
        "--stdin-file-path",
        "--compact",
        "--bulk-validate",
        "--override",
        "--defines",
        "--capabilities",
        "--task-limits",
        "--validate-mesh-output",
    ] {
        assert!(help.contains(flag), "help missing {flag}\n---\n{help}");
    }
}

#[test]
fn validates_wgsl_from_file() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_validate");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("ok.wgsl");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();

    let out = naga().arg(&src).output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}

#[test]
fn compiles_wgsl_to_spv() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_spv");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    let dst = dir.join("s.spv");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();

    let out = naga().arg(&src).arg(&dst).output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(&dst).unwrap();
    // SPIR-V magic number 0x07230203, little-endian.
    assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]);
}

#[test]
fn reads_wgsl_from_stdin() {
    use std::io::Write;
    let mut child = naga()
        .args(["--stdin-file-path", "in.wgsl"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}

#[test]
fn generates_debug_symbols_spv() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_debug_spv");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("debug.wgsl");
    let dst = dir.join("debug.spv");
    std::fs::write(&src, "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").unwrap();

    let out = naga().arg(&src).arg(&dst).arg("-g").output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(&dst).unwrap();
    // SPIR-V magic number 0x07230203, little-endian.
    assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07], "output is not valid SPIR-V");
}

#[test]
fn glsl_input_stage_from_flag() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_glsl");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("frag.glsl");
    std::fs::write(
        &src,
        "#version 450\nlayout(location=0) out vec4 c;\nvoid main() { c = vec4(1.0); }",
    )
    .unwrap();

    let out = naga()
        .args(["--input-kind", "glsl", "--shader-stage", "frag"])
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains("Validation successful"),
        "stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn prints_config_schema() {
    let out = naga().arg("--print-config-schema").output().unwrap();
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let schema = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&schema).unwrap();
    assert!(v.is_object());
    assert!(schema.contains("spv_out") || schema.contains("lang_version"), "schema: {schema}");
}

#[test]
fn help_matches_snapshot() {
    let expected = include_str!("snapshots/help.txt");
    let out = naga().arg("--help").output().unwrap();
    let actual = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "--help output changed. If intentional, regenerate:\n\
         cargo run -q -p naga-cli -- --help > naga-cli/tests/snapshots/help.txt"
    );
}
