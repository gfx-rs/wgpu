use std::process::Command;

/// Path to the built `naga` binary, provided by cargo to integration tests.
fn naga() -> Command {
    Command::new(env!("CARGO_BIN_EXE_naga"))
}

/// Resolve an example file relative to the naga-cli manifest directory.
fn example_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(name)
}

#[test]
fn force_loop_bounding_flag_applies_to_spv() {
    // A shader whose SPIR-V differs when loop bounding is off vs on would be ideal,
    // but at minimum assert the flag parses, is accepted, and compiles successfully.
    let dir = std::env::temp_dir().join("naga_cli_p3_flb");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(
        &src,
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    )
    .unwrap();
    let out = dir.join("s.spv");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .arg("--force-loop-bounding")
        .arg("false")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert_eq!(
        &std::fs::read(&out).unwrap()[0..4],
        &[0x03, 0x02, 0x23, 0x07]
    );
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
        let r = naga()
            .arg(&src)
            .arg(&dst)
            .arg("--zero-initialize-workgroup-memory")
            .arg(mode)
            .output()
            .unwrap();
        assert!(
            r.status.success(),
            "mode {mode} stderr: {}",
            String::from_utf8_lossy(&r.stderr)
        );
    }
}

#[test]
fn config_nested_common_flat_json() {
    let dir = std::env::temp_dir().join("naga_cli_p3_cfg");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(
        &src,
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    )
    .unwrap();
    let out = dir.join("s.spv");
    // serde(flatten) keeps common keys flat inside spv_out:
    let r = naga()
        .arg(&src)
        .arg(&out)
        .arg("--config-json")
        .arg(r#"{"spv_out":{"force_loop_bounding":false}}"#)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
}

#[test]
fn config_json_and_common_flag_are_exclusive() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config-json")
        .arg("{}")
        .arg("--force-loop-bounding")
        .arg("false")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
        "expected clap conflict error, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn config_json_matches_equivalent_flag() {
    let dir = std::env::temp_dir().join("naga_cli_p2_config");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(
        &src,
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    )
    .unwrap();
    let out_flag = dir.join("flag.spv");
    let out_cfg = dir.join("cfg.spv");

    // Via flag:
    let a = naga()
        .arg(&src)
        .arg(&out_flag)
        .arg("--spirv-version")
        .arg("1.5")
        .output()
        .unwrap();
    assert!(a.status.success(), "{}", String::from_utf8_lossy(&a.stderr));

    // Via config-json (same lang_version):
    let b = naga()
        .arg(&src)
        .arg(&out_cfg)
        .arg("--config-json")
        .arg(r#"{"spv_out":{"lang_version":[1,5]}}"#)
        .output()
        .unwrap();
    assert!(b.status.success(), "{}", String::from_utf8_lossy(&b.stderr));

    assert_eq!(
        std::fs::read(&out_flag).unwrap(),
        std::fs::read(&out_cfg).unwrap()
    );
}

#[test]
fn config_conflicts_with_option_flag() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config-json")
        .arg("{}")
        .arg("--spirv-version")
        .arg("1.5")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
        "expected clap conflict error, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn config_and_config_json_mutually_exclusive() {
    let out = naga()
        .arg("in.wgsl")
        .arg("--config")
        .arg("x.json")
        .arg("--config-json")
        .arg("{}")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("cannot be used with"),
        "expected clap conflict error, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn config_composes_with_compact_flag() {
    let dir = std::env::temp_dir().join("naga_cli_p2_cfg_compact");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(
        &src,
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    )
    .unwrap();
    let out = dir.join("o.spv");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .arg("--config-json")
        .arg(r#"{"spv_out":{"lang_version":[1,3]}}"#)
        .arg("--compact")
        .output()
        .unwrap();
    // Must NOT be a clap conflict error; compaction + config compose.
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
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
    std::fs::write(
        &src,
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    )
    .unwrap();

    let out = naga().arg(&src).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}

#[test]
fn compiles_wgsl_to_spv() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_spv");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    let dst = dir.join("s.spv");
    std::fs::write(
        &src,
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    )
    .unwrap();

    let out = naga().arg(&src).arg(&dst).output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
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
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}

#[test]
fn generates_debug_symbols_spv() {
    let dir = std::env::temp_dir().join("naga_cli_phase1_debug_spv");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("debug.wgsl");
    let dst = dir.join("debug.spv");
    std::fs::write(
        &src,
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    )
    .unwrap();

    let out = naga().arg(&src).arg(&dst).arg("-g").output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let bytes = std::fs::read(&dst).unwrap();
    // SPIR-V magic number 0x07230203, little-endian.
    assert_eq!(
        &bytes[0..4],
        &[0x03, 0x02, 0x23, 0x07],
        "output is not valid SPIR-V"
    );
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
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let schema = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(&schema).unwrap();
    assert!(v.is_object());
    assert!(
        schema.contains("spv_out") || schema.contains("lang_version"),
        "schema: {schema}"
    );
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

#[test]
fn json_format_valid_shader_reflection() {
    let dir = std::env::temp_dir().join("naga_cli_p4_ok");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = naga()
        .arg(&src)
        .args(["--format", "json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["success"], true);
    assert_eq!(v["reflection"]["entry_points"][0]["name"], "main");
    assert_eq!(v["diagnostics"].as_array().unwrap().len(), 0);
}

#[test]
fn json_format_parse_error() {
    let dir = std::env::temp_dir().join("naga_cli_p4_err");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("bad.wgsl");
    std::fs::write(&src, "fn f( { }").unwrap(); // parse error
    let out = naga()
        .arg(&src)
        .args(["--format", "json"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["success"], false);
    let diags = v["diagnostics"].as_array().unwrap();
    assert!(!diags.is_empty());
    assert_eq!(diags[0]["severity"], "error");
    assert!(!diags[0]["message"].as_str().unwrap().is_empty());
}

#[test]
fn text_format_unchanged_default() {
    // Without --format, behavior is text (validation success message on stdout).
    let dir = std::env::temp_dir().join("naga_cli_p4_text");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = naga().arg(&src).output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("Validation successful"));
}

#[test]
fn json_unknown_output_extension_keeps_stdout_pure() {
    let dir = std::env::temp_dir().join("naga_cli_p4_unkext");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("s.xyz"); // unknown output extension
    let r = naga()
        .arg(&src)
        .arg(&out)
        .args(["--format", "json"])
        .output()
        .unwrap();
    // stdout must be exactly one parseable JSON doc (the notice goes to stderr):
    let v: serde_json::Value =
        serde_json::from_slice(&r.stdout).expect("stdout must be a single JSON document");
    assert!(v.get("success").is_some());
}

/// True if a tool is on PATH (tests skip when absent).
fn tool_on_path(name: &str) -> bool {
    which::which(name).is_ok()
}

#[test]
fn spirv_val_hook_when_available() {
    if !tool_on_path("spirv-val") {
        eprintln!("skipping spirv_val_hook_when_available: spirv-val not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("naga_cli_p5_val");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("s.spv");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .arg("--spirv-val")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(out.exists());
}

#[test]
fn spirv_opt_hook_when_available() {
    if !tool_on_path("spirv-opt") {
        eprintln!("skipping spirv_opt_hook_when_available: spirv-opt not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("naga_cli_p5_opt");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("s.spv");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .arg("--spirv-opt")
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    // Still valid SPIR-V after optimization.
    assert_eq!(
        &std::fs::read(&out).unwrap()[0..4],
        &[0x03, 0x02, 0x23, 0x07]
    );
}

#[test]
fn dxc_hook_when_available() {
    if !tool_on_path("dxc") {
        eprintln!("skipping dxc_hook_when_available: dxc not on PATH");
        return;
    }
    let dir = std::env::temp_dir().join("naga_cli_p5_dxc");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("s.hlsl");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .args(["--dxc", "--shader-model", "60"])
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(dir.join("s.main.dxil").exists());
}

#[test]
fn spirv_val_without_spv_output_errors() {
    let dir = std::env::temp_dir().join("naga_cli_p5_noout");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    // No .spv output path → --spirv-val should error (not silently no-op).
    let r = naga().arg(&src).arg("--spirv-val").output().unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr)
        .to_lowercase()
        .contains("spir"));
}

#[test]
fn spirv_opt_without_spv_output_errors() {
    let dir = std::env::temp_dir().join("naga_cli_p5_noopt");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let r = naga().arg(&src).arg("--spirv-opt").output().unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr)
        .to_lowercase()
        .contains("spir"));
}

#[test]
fn dxc_without_hlsl_output_errors() {
    let dir = std::env::temp_dir().join("naga_cli_p5_nodxc");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let r = naga().arg(&src).arg("--dxc").output().unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr)
        .to_lowercase()
        .contains("hlsl"));
}

#[test]
fn bulk_validate_json_is_rejected() {
    let dir = std::env::temp_dir().join("naga_cli_p6_bulkjson");
    std::fs::create_dir_all(&dir).unwrap();
    let a = dir.join("a.wgsl");
    std::fs::write(&a, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = naga()
        .args(["--bulk-validate", "--format", "json"])
        .arg(&a)
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr)
            .to_lowercase()
            .contains("bulk"),
        "expected a clear bulk+json error, got: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn json_debug_symbols_warning_is_a_diagnostic() {
    // `-g` with non-human-readable (bincode) input triggers the advisory warning.
    // First produce a .bin module, then feed it back with -g in json mode.
    let dir = std::env::temp_dir().join("naga_cli_p6_gwarn");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let bin = dir.join("s.bin");
    let mk = naga().arg(&src).arg(&bin).output().unwrap();
    assert!(
        mk.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&mk.stderr)
    );

    let out = naga()
        .arg(&bin)
        .args(["--format", "json", "-g"])
        .output()
        .unwrap();
    // Producing a .bin re-import with -g: validation still succeeds → success true,
    // but a warning diagnostic about -g on non-human-readable input is present.
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let diags = v["diagnostics"].as_array().unwrap();
    assert!(
        diags.iter().any(|d| {
            d["severity"] == "warning"
                && d["message"]
                    .as_str()
                    .unwrap_or("")
                    .contains("--generate-debug-symbols")
        }),
        "expected a --generate-debug-symbols warning diagnostic in json mode, got: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn readme_config_file_example() {
    let dir = std::env::temp_dir().join("naga_cli_p6_readme_cfg");
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("tri.spv");
    let r = naga()
        .arg(example_path("triangle.wgsl"))
        .arg(&out)
        .arg("--config")
        .arg(example_path("options.json"))
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert_eq!(
        &std::fs::read(&out).unwrap()[0..4],
        &[0x03, 0x02, 0x23, 0x07]
    );
}

#[test]
fn readme_json_reflection_example() {
    let r = naga()
        .arg(example_path("triangle.wgsl"))
        .args(["--format", "json"])
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&r.stdout).unwrap();
    assert_eq!(v["success"], true);
    let names: Vec<&str> = v["reflection"]["entry_points"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert!(
        names.contains(&"vs_main") && names.contains(&"fs_main"),
        "names: {names:?}"
    );
    assert!(v["reflection"]["resources"]
        .as_array()
        .unwrap()
        .iter()
        .any(|res| res["name"] == "tint"));
}
