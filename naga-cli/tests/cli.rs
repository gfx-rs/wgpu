use std::process::Command;

/// Path to the built `naga` binary, provided by cargo to integration tests.
fn naga() -> Command {
    Command::new(env!("CARGO_BIN_EXE_naga"))
}

/// Write `contents` to `<temp_dir>/<dir>/<name>`, creating directories as needed.
fn write_tmp(dir: &str, name: &str, contents: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(dir);
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, contents).unwrap();
    p
}

/// A small WGSL module with a vertex + fragment entry and a bound resource,
/// broad enough to emit to every backend.
const TRIANGLE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> tint: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(i), 0.0, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return tint; }
"#;

/// A sample partial JSON config (mirrors the README example): only the specified
/// keys are applied, everything else uses defaults.
const SAMPLE_CONFIG_JSON: &str = r#"{
  "spv_out": { "lang_version": [1, 3], "force_loop_bounding": false },
  "msl": { "lang_version": [2, 0] }
}"#;

// ── stdout output (`-` + --output-kind) ──────────────────────────────────────

#[test]
fn stdout_output_wgsl() {
    let src = write_tmp("naga_cli_stdout_wgsl", "in.wgsl", TRIANGLE_WGSL);
    let r = naga()
        .arg(&src)
        .args(["-", "--output-kind", "wgsl"])
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(String::from_utf8_lossy(&r.stdout).contains("fn "));
}

#[test]
fn stdout_output_spv_magic() {
    let src = write_tmp("naga_cli_stdout_spv", "in.wgsl", TRIANGLE_WGSL);
    let r = naga()
        .arg(&src)
        .args(["-", "--output-kind", "spv"])
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert_eq!(&r.stdout[0..4], &[0x03, 0x02, 0x23, 0x07]);
}

#[test]
fn stdout_requires_output_kind() {
    let src = write_tmp("naga_cli_stdout_nokind", "in.wgsl", TRIANGLE_WGSL);
    let r = naga().arg(&src).arg("-").output().unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("output-kind"));
}

#[test]
fn stdout_rejects_tool_hooks() {
    let src = write_tmp("naga_cli_stdout_dxc", "in.wgsl", TRIANGLE_WGSL);
    let r = naga()
        .arg(&src)
        .args(["-", "--output-kind", "hlsl", "--dxc"])
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr)
        .to_lowercase()
        .contains("stdout"));
}

#[test]
fn stdout_rejects_json_format() {
    let src = write_tmp("naga_cli_stdout_json", "in.wgsl", TRIANGLE_WGSL);
    let r = naga()
        .arg(&src)
        .args(["-", "--output-kind", "spv", "--format", "json"])
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr)
        .to_lowercase()
        .contains("stdout"));
}

#[test]
fn output_language_matrix_from_wgsl() {
    let src = write_tmp("naga_cli_p7_out", "in.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap().to_path_buf();

    struct Case {
        file: &'static str,
        args: &'static [&'static str],
        check: fn(&[u8]) -> bool,
    }
    let cases: &[Case] = &[
        // IR debug dump: starts with "Module {", always contains "("
        Case {
            file: "out.txt",
            args: &[],
            check: |b| String::from_utf8_lossy(b).contains("Module"),
        },
        // Bincode IR: non-empty binary blob; round-trip fidelity is validated by cross_language_conversions.
        Case {
            file: "out.bin",
            args: &[],
            check: |b| b.len() > 16,
        },
        // MSL: header comment contains "metal"
        Case {
            file: "out.metal",
            args: &[],
            check: |b| {
                let s = String::from_utf8_lossy(b);
                s.contains("metal") || s.contains("[[")
            },
        },
        // SPIR-V binary (magic 0x07230203, little-endian)
        Case {
            file: "out.spv",
            args: &[],
            check: |b| b.len() >= 4 && b[0..4] == [0x03, 0x02, 0x23, 0x07],
        },
        // .spirv alias — same magic
        Case {
            file: "out.spirv",
            args: &[],
            check: |b| b.len() >= 4 && b[0..4] == [0x03, 0x02, 0x23, 0x07],
        },
        // Graphviz DOT
        Case {
            file: "out.dot",
            args: &[],
            check: |b| String::from_utf8_lossy(b).contains("digraph"),
        },
        // HLSL: must contain a cbuffer/register binding declaration
        Case {
            file: "out.hlsl",
            args: &[],
            check: |b| {
                let s = String::from_utf8_lossy(b);
                s.contains("register(") || s.contains("cbuffer")
            },
        },
        // WGSL round-trip: contains a function
        Case {
            file: "out.wgsl",
            args: &[],
            check: |b| String::from_utf8_lossy(b).contains("fn "),
        },
        // GLSL vertex (needs single entry point)
        Case {
            file: "out.vert",
            args: &["--entry-point", "vs_main"],
            check: |b| String::from_utf8_lossy(b).contains("#version"),
        },
        // GLSL fragment (needs single entry point)
        Case {
            file: "out.frag",
            args: &["--entry-point", "fs_main"],
            check: |b| String::from_utf8_lossy(b).contains("#version"),
        },
    ];
    for c in cases {
        let out = dir.join(c.file);
        let r = naga().arg(&src).arg(&out).args(c.args).output().unwrap();
        assert!(
            r.status.success(),
            "{}: {}",
            c.file,
            String::from_utf8_lossy(&r.stderr)
        );
        let bytes = std::fs::read(&out).unwrap();
        assert!(
            (c.check)(&bytes),
            "{} signature mismatch (len={})",
            c.file,
            bytes.len()
        );
    }
}

#[test]
fn compute_wgsl_to_comp_glsl() {
    // A compute-only shader → .comp (GLSL compute extension), assert #version header.
    let src = write_tmp(
        "naga_cli_p7_comp",
        "cs.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );
    let dir = src.parent().unwrap().to_path_buf();
    let out = dir.join("out.comp");
    let r = naga().arg(&src).arg(&out).output().unwrap();
    assert!(
        r.status.success(),
        "comp: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("#version"),
        "out.comp should contain #version, got: {text}"
    );
}

#[test]
fn input_language_matrix() {
    let dir = std::env::temp_dir().join("naga_cli_p7_in");
    std::fs::create_dir_all(&dir).unwrap();

    // 1. WGSL input — validate only (no output path).
    let wgsl = write_tmp(
        "naga_cli_p7_in",
        "a.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );
    assert!(naga().arg(&wgsl).output().unwrap().status.success());

    // 2. GLSL input (fragment stage via --input-kind/--shader-stage).
    let glsl = write_tmp(
        "naga_cli_p7_in",
        "a.frag",
        "#version 450\nlayout(location=0) out vec4 c;\nvoid main(){ c=vec4(1.0); }",
    );
    let r = naga()
        .args(["--input-kind", "glsl", "--shader-stage", "frag"])
        .arg(&glsl)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "glsl: {}",
        String::from_utf8_lossy(&r.stderr)
    );

    // 3. SPIR-V input — produce via wgsl→spv, then read back.
    let spv = dir.join("a.spv");
    assert!(naga()
        .arg(&wgsl)
        .arg(&spv)
        .output()
        .unwrap()
        .status
        .success());
    let r = naga().arg(&spv).output().unwrap();
    assert!(
        r.status.success(),
        "spv-in: {}",
        String::from_utf8_lossy(&r.stderr)
    );

    // 4. Bincode IR input — produce via wgsl→bin, then read back.
    let bin = dir.join("a.bin");
    assert!(naga()
        .arg(&wgsl)
        .arg(&bin)
        .output()
        .unwrap()
        .status
        .success());
    let r = naga().arg(&bin).output().unwrap();
    assert!(
        r.status.success(),
        "bin-in: {}",
        String::from_utf8_lossy(&r.stderr)
    );
}

#[test]
fn cross_language_conversions() {
    let dir = std::env::temp_dir().join("naga_cli_p7_cross");
    std::fs::create_dir_all(&dir).unwrap();
    let wgsl = write_tmp(
        "naga_cli_p7_cross",
        "a.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );

    // spv -> wgsl
    let spv = dir.join("a.spv");
    assert!(naga()
        .arg(&wgsl)
        .arg(&spv)
        .output()
        .unwrap()
        .status
        .success());
    let out_wgsl = dir.join("from_spv.wgsl");
    let r = naga().arg(&spv).arg(&out_wgsl).output().unwrap();
    assert!(
        r.status.success(),
        "spv->wgsl: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(
        std::fs::read_to_string(&out_wgsl).unwrap().contains("fn "),
        "spv->wgsl missing fn"
    );

    // bin -> txt
    let bin = dir.join("a.bin");
    assert!(naga()
        .arg(&wgsl)
        .arg(&bin)
        .output()
        .unwrap()
        .status
        .success());
    let out_txt = dir.join("from_bin.txt");
    let r = naga().arg(&bin).arg(&out_txt).output().unwrap();
    assert!(
        r.status.success(),
        "bin->txt: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(
        std::fs::read_to_string(&out_txt)
            .unwrap()
            .contains("Module"),
        "bin->txt missing Module"
    );

    // glsl -> spv
    let glsl = write_tmp(
        "naga_cli_p7_cross",
        "a.frag",
        "#version 450\nlayout(location=0) out vec4 c;\nvoid main(){ c=vec4(1.0); }",
    );
    let out_spv = dir.join("from_glsl.spv");
    let r = naga()
        .args(["--input-kind", "glsl", "--shader-stage", "frag"])
        .arg(&glsl)
        .arg(&out_spv)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "glsl->spv: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    let spv_bytes = std::fs::read(&out_spv).unwrap();
    assert_eq!(
        &spv_bytes[0..4],
        &[0x03, 0x02, 0x23, 0x07],
        "glsl->spv bad magic"
    );
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
fn config_conflicts_with_compact_flag() {
    // Processing flags (compact/-g/spirv-*/dxc) are config keys now, so passing them
    // as flags alongside --config is a hard conflict.
    let dir = std::env::temp_dir().join("naga_cli_cfg_compact_conflict");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("o.spv");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .arg("--config-json")
        .arg(r#"{"spv_out":{"lang_version":[1,3]}}"#)
        .arg("--compact")
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("cannot be used with"));
}

#[test]
fn config_conflicts_with_dxc_flag() {
    let dir = std::env::temp_dir().join("naga_cli_cfg_dxc_conflict");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("o.hlsl");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .args(["--config-json", "{}", "--dxc"])
        .output()
        .unwrap();
    assert!(!r.status.success());
    assert!(String::from_utf8_lossy(&r.stderr).contains("cannot be used with"));
}

#[test]
fn config_drives_compact() {
    // Setting compact via the config JSON (rather than the flag) still works.
    let dir = std::env::temp_dir().join("naga_cli_cfg_drives_compact");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("o.spv");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .args([
            "--config-json",
            r#"{"compact":true,"spv_out":{"lang_version":[1,3]}}"#,
        ])
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
fn before_compaction_composes_with_config() {
    // --before-compaction is I/O (an output path), so it still composes with --config.
    let dir = std::env::temp_dir().join("naga_cli_cfg_before_compaction");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("s.wgsl");
    std::fs::write(&src, "@compute @workgroup_size(1) fn main() {}").unwrap();
    let out = dir.join("o.spv");
    let pre = dir.join("pre.txt");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .args(["--config-json", "{}"])
        .arg("--before-compaction")
        .arg(&pre)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(pre.exists() && std::fs::metadata(&pre).unwrap().len() > 0);
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
        "--output-kind",
        "--metal-version",
        "--keep-coordinate-space",
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
    // `-` reads from stdin; --input-kind names the format (no extension to infer).
    let mut child = naga()
        .args(["-", "--input-kind", "wgsl"])
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
fn stdin_requires_input_kind() {
    // `-` with no --input-kind has no way to know the format.
    let out = naga().arg("-").output().unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("input-kind"));
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
fn config_schema_matches_snapshot() {
    let expected = include_str!("snapshots/config-schema.json");
    let out = naga().arg("--print-config-schema").output().unwrap();
    assert!(out.status.success());
    let actual = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        actual.trim_end(),
        expected.trim_end(),
        "--print-config-schema output changed (a naga Options field or schemars derive moved). \
         If intentional, regenerate:\n\
         cargo run -q -p naga-cli -- --print-config-schema > naga-cli/tests/snapshots/config-schema.json"
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

// ── Phase 7 Task 3: modes, bulk-validate, version, remaining option flags ─────

#[test]
fn version_flag_prints_version() {
    let out = naga().arg("--version").output().unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let expected_version = env!("CARGO_PKG_VERSION");
    assert!(
        stdout.contains(expected_version),
        "--version should contain '{}', got: {stdout}",
        expected_version
    );
}

/// A WGSL fragment shader with an invalid return type (f32 without @location) —
/// naga validation rejects it because entry-point return values must have bindings.
const INVALID_RETURN_WGSL: &str = "@fragment fn main() -> f32 { return 0.0; }";

#[test]
fn validate_flag_bitmask() {
    let src = write_tmp("naga_cli_p7t3_val", "invalid.wgsl", INVALID_RETURN_WGSL);

    // Default: validation is on → non-zero exit.
    let default = naga().arg(&src).output().unwrap();
    assert!(
        !default.status.success(),
        "invalid shader should fail validation by default; stdout: {}, stderr: {}",
        String::from_utf8_lossy(&default.stdout),
        String::from_utf8_lossy(&default.stderr)
    );

    // --validate 0 disables validation → exit 0 (shader is accepted without validating).
    let disabled = naga()
        .arg("--validate")
        .arg("0")
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        disabled.status.success(),
        "--validate 0 should disable validation and succeed; stderr: {}",
        String::from_utf8_lossy(&disabled.stderr)
    );

    // --validate <max-bits> keeps full validation on → non-zero exit again.
    // u32::MAX as a decimal string covers all ValidationFlags bits.
    let max_bits = u32::MAX.to_string();
    let all = naga()
        .arg("--validate")
        .arg(&max_bits)
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        !all.status.success(),
        "--validate {} (all bits) should still reject the invalid shader",
        max_bits
    );
}

#[test]
fn bulk_validate_success() {
    let a = write_tmp(
        "naga_cli_p7t3_bv_ok",
        "a.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );
    let b = write_tmp(
        "naga_cli_p7t3_bv_ok",
        "b.wgsl",
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
    );

    let r = naga()
        .arg("--bulk-validate")
        .arg(&a)
        .arg(&b)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "--bulk-validate with two valid shaders should succeed; stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
}

#[test]
fn bulk_validate_reports_invalid() {
    let valid = write_tmp(
        "naga_cli_p7t3_bv_err",
        "valid.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );
    // Deliberately malformed WGSL (parse error).
    let invalid = write_tmp("naga_cli_p7t3_bv_err", "invalid.wgsl", "fn f( { }");

    let r = naga()
        .arg("--bulk-validate")
        .arg(&valid)
        .arg(&invalid)
        .output()
        .unwrap();
    assert!(
        !r.status.success(),
        "--bulk-validate with one invalid shader should exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&r.stderr);
    assert!(
        stderr.contains("invalid.wgsl"),
        "--bulk-validate stderr should name the invalid file; got: {stderr}"
    );
}

#[test]
fn task_limits_accepted() {
    // codegen effect needs feature-specific shaders; this asserts the flag parses + plumbs to the backend option
    let src = write_tmp(
        "naga_cli_p7t3_tl",
        "cs.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );
    let dir = src.parent().unwrap();
    let out = dir.join("cs.spv");

    let r = naga()
        .arg("--task-limits")
        .arg("8,8")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "--task-limits 8,8 should be accepted; stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
}

#[test]
fn fake_missing_bindings_accepted() {
    // codegen effect needs feature-specific shaders; this asserts the flag parses + plumbs to the backend option
    let src = write_tmp(
        "naga_cli_p7t3_fmb",
        "cs.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );
    let dir = src.parent().unwrap();
    let out = dir.join("cs.spv");

    let r = naga()
        .arg("--fake-missing-bindings")
        .arg("false")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "--fake-missing-bindings false should be accepted; stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
}

#[test]
fn ray_query_initialization_tracking_accepted() {
    // codegen effect needs feature-specific shaders; this asserts the flag parses + plumbs to the backend option
    let src = write_tmp(
        "naga_cli_p7t3_rqit",
        "cs.wgsl",
        "@compute @workgroup_size(1) fn main() {}",
    );
    let dir = src.parent().unwrap();
    let out = dir.join("cs.spv");

    let r = naga()
        .arg("--ray-query-initialization-tracking")
        .arg("false")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "--ray-query-initialization-tracking false should be accepted; stderr: {}",
        String::from_utf8_lossy(&r.stderr)
    );
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
    let dir = std::env::temp_dir().join("naga_cli_readme_cfg");
    std::fs::create_dir_all(&dir).unwrap();
    let src = write_tmp("naga_cli_readme_cfg", "triangle.wgsl", TRIANGLE_WGSL);
    let cfg = write_tmp("naga_cli_readme_cfg", "options.json", SAMPLE_CONFIG_JSON);
    let out = dir.join("tri.spv");
    let r = naga()
        .arg(&src)
        .arg(&out)
        .arg("--config")
        .arg(&cfg)
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
    let src = write_tmp("naga_cli_readme_json", "triangle.wgsl", TRIANGLE_WGSL);
    let r = naga()
        .arg(&src)
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

// ── Phase 7 Task 2: per-flag translation-option coverage ──────────────────────

/// A WGSL shader with a pipeline-overridable constant used in output.
const OVERRIDE_WGSL: &str = r#"
override k: f32 = 1.0;
@fragment fn main() -> @location(0) vec4<f32> {
    return vec4<f32>(k, 0.0, 0.0, 1.0);
}
"#;

/// A WGSL fragment that uses `enable f16;` — capability-gated in naga.
/// Without the `f16` capability the WGSL parser refuses the `enable` directive.
const F16_WGSL: &str = r#"
enable f16;
@fragment fn main() -> @location(0) vec4<f16> { return vec4<f16>(1.0h, 0.0h, 0.0h, 1.0h); }
"#;

/// A GLSL fragment that branches on a preprocessor define.
const IFDEF_GLSL: &str = r#"
#version 450
layout(location=0) out vec4 color;
void main() {
#ifdef FOO
    color = vec4(1.0, 0.0, 0.0, 1.0);
#else
    color = vec4(0.0, 1.0, 0.0, 1.0);
#endif
}
"#;

/// A WGSL shader with a storage buffer array — exercises bounds-check policy flags.
const BOUNDS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> arr: array<f32>;
@fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let idx = u32(pos.x);
    return vec4<f32>(arr[idx], 0.0, 0.0, 1.0);
}
"#;

#[test]
fn override_value_applied() {
    // Effect observable in MSL: `constant float k = 1.0;` vs `constant float k = 5.0;`
    let src = write_tmp("naga_cli_p7t2_ovr", "override.wgsl", OVERRIDE_WGSL);
    let dir = src.parent().unwrap();
    let out_default = dir.join("no_override.metal");
    let out_overridden = dir.join("with_override.metal");

    let r1 = naga().arg(&src).arg(&out_default).output().unwrap();
    assert!(
        r1.status.success(),
        "no-override: {}",
        String::from_utf8_lossy(&r1.stderr)
    );

    let r2 = naga()
        .arg("--override")
        .arg("k=5.0")
        .arg(&src)
        .arg(&out_overridden)
        .output()
        .unwrap();
    assert!(
        r2.status.success(),
        "with-override: {}",
        String::from_utf8_lossy(&r2.stderr)
    );

    let default_msl = std::fs::read_to_string(&out_default).unwrap();
    let overridden_msl = std::fs::read_to_string(&out_overridden).unwrap();

    // The overridden constant value must change in MSL output.
    assert!(
        default_msl.contains("k = 1.0"),
        "expected k=1.0 in default MSL:\n{default_msl}"
    );
    assert!(
        overridden_msl.contains("k = 5.0"),
        "expected k=5.0 in overridden MSL:\n{overridden_msl}"
    );
    assert_ne!(
        default_msl, overridden_msl,
        "MSL outputs should differ after override"
    );
}

#[test]
fn capabilities_restrict_rejects() {
    // `enable f16;` requires the f16 capability; `--capabilities none` must reject it.
    let src = write_tmp("naga_cli_p7t2_caps", "f16.wgsl", F16_WGSL);

    let ok = naga().arg(&src).output().unwrap();
    assert!(
        ok.status.success(),
        "f16 with default caps should succeed: {}",
        String::from_utf8_lossy(&ok.stderr)
    );

    let fail = naga()
        .arg("--capabilities")
        .arg("none")
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        !fail.status.success(),
        "f16 with --capabilities none should fail"
    );
    let stderr = String::from_utf8_lossy(&fail.stderr);
    // naga reports "unsupported enable-extension" or similar
    assert!(
        stderr.contains("f16") || stderr.contains("extension") || stderr.contains("unsupported"),
        "expected capability error mentioning f16/extension, got: {stderr}"
    );
}

#[test]
fn keep_coordinate_space_changes_output() {
    // A vertex shader writing @builtin(position) — naga normally inserts a Y-flip.
    // --keep-coordinate-space suppresses it, so the SPIR-V binaries must differ.
    let src = write_tmp("naga_cli_p7t2_kcs", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();
    let out_normal = dir.join("normal.spv");
    let out_kept = dir.join("kept.spv");

    let r1 = naga()
        .arg("--entry-point")
        .arg("vs_main")
        .arg(&src)
        .arg(&out_normal)
        .output()
        .unwrap();
    assert!(
        r1.status.success(),
        "normal: {}",
        String::from_utf8_lossy(&r1.stderr)
    );

    let r2 = naga()
        .arg("--entry-point")
        .arg("vs_main")
        .arg("--keep-coordinate-space")
        .arg(&src)
        .arg(&out_kept)
        .output()
        .unwrap();
    assert!(
        r2.status.success(),
        "kept: {}",
        String::from_utf8_lossy(&r2.stderr)
    );

    let normal = std::fs::read(&out_normal).unwrap();
    let kept = std::fs::read(&out_kept).unwrap();
    assert_ne!(
        normal, kept,
        "--keep-coordinate-space should change SPIR-V (coordinate flip toggled)"
    );
}

#[test]
fn dot_cfg_only_changes_output() {
    // --dot-cfg-only produces a smaller DOT with only control-flow nodes, no expression nodes.
    let src = write_tmp("naga_cli_p7t2_dot", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();
    let out_full = dir.join("full.dot");
    let out_cfg = dir.join("cfg.dot");

    let r1 = naga().arg(&src).arg(&out_full).output().unwrap();
    assert!(
        r1.status.success(),
        "full dot: {}",
        String::from_utf8_lossy(&r1.stderr)
    );

    let r2 = naga()
        .arg("--dot-cfg-only")
        .arg(&src)
        .arg(&out_cfg)
        .output()
        .unwrap();
    assert!(
        r2.status.success(),
        "cfg dot: {}",
        String::from_utf8_lossy(&r2.stderr)
    );

    let full = std::fs::read_to_string(&out_full).unwrap();
    let cfg = std::fs::read_to_string(&out_cfg).unwrap();

    // CFG-only strips expression nodes (labels contain "Literal", "Load", etc.)
    assert!(
        full.contains("Literal") || full.contains("Load"),
        "full DOT should contain expression nodes"
    );
    assert!(
        !cfg.contains("Literal") && !cfg.contains("Load"),
        "cfg-only DOT must not contain expression nodes"
    );
    assert_ne!(full, cfg, "--dot-cfg-only should change DOT output");
}

#[test]
fn before_compaction_writes_file() {
    // --compact --before-compaction <file> must write the pre-compaction IR to <file>.
    let src = write_tmp("naga_cli_p7t2_bc", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();
    let pre = dir.join("pre.txt");
    let out = dir.join("out.spv");

    // --before-compaction implies --compact; no explicit --compact needed.
    let r = naga()
        .arg("--before-compaction")
        .arg(&pre)
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "before-compaction: {}",
        String::from_utf8_lossy(&r.stderr)
    );

    assert!(pre.exists(), "--before-compaction file was not written");
    let content = std::fs::read(&pre).unwrap();
    assert!(!content.is_empty(), "--before-compaction file is empty");
}

#[test]
fn unknown_entry_point_errors() {
    let src = write_tmp("naga_cli_p7t2_ep", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();
    let out = dir.join("bogus.spv");

    let r = naga()
        .arg("--entry-point")
        .arg("bogus")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(!r.status.success(), "--entry-point bogus should fail");
    let stderr = String::from_utf8_lossy(&r.stderr);
    assert!(
        stderr.contains("bogus") || stderr.contains("entry"),
        "expected error mentioning entry point, got: {stderr}"
    );
}

#[test]
fn entry_point_selects() {
    // A real entry point name succeeds.
    let src = write_tmp("naga_cli_p7t2_epsel", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();
    let out = dir.join("fs.spv");

    let r = naga()
        .arg("--entry-point")
        .arg("fs_main")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "entry-point fs_main: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    // Produced valid SPIR-V.
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(
        &bytes[0..4],
        &[0x03, 0x02, 0x23, 0x07],
        "expected SPIR-V magic"
    );
}

#[test]
fn profile_sets_glsl_version() {
    // --profile es300 → #version 300 es; --profile core330 → #version 330 core
    let src = write_tmp("naga_cli_p7t2_prof", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();

    let out_es = dir.join("es300.frag");
    let r1 = naga()
        .arg("--entry-point")
        .arg("fs_main")
        .arg("--profile")
        .arg("es300")
        .arg(&src)
        .arg(&out_es)
        .output()
        .unwrap();
    assert!(
        r1.status.success(),
        "es300: {}",
        String::from_utf8_lossy(&r1.stderr)
    );
    let es_text = std::fs::read_to_string(&out_es).unwrap();
    assert!(
        es_text.contains("#version 300 es"),
        "expected '#version 300 es' in output, got:\n{es_text}"
    );

    let out_core = dir.join("core330.frag");
    let r2 = naga()
        .arg("--entry-point")
        .arg("fs_main")
        .arg("--profile")
        .arg("core330")
        .arg(&src)
        .arg(&out_core)
        .output()
        .unwrap();
    assert!(
        r2.status.success(),
        "core330: {}",
        String::from_utf8_lossy(&r2.stderr)
    );
    let core_text = std::fs::read_to_string(&out_core).unwrap();
    assert!(
        core_text.contains("#version 330 core"),
        "expected '#version 330 core' in output, got:\n{core_text}"
    );
}

#[test]
fn metal_version_in_output() {
    // --metal-version 2.0 → MSL header comment reads "// language: metal2.0"
    let src = write_tmp("naga_cli_p7t2_mv", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();
    let out = dir.join("v20.metal");

    let r = naga()
        .arg("--metal-version")
        .arg("2.0")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "metal-version 2.0: {}",
        String::from_utf8_lossy(&r.stderr)
    );

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("metal2.0"),
        "expected 'metal2.0' in MSL header comment, got:\n{text}"
    );
}

#[test]
fn shader_model_affects_hlsl() {
    // --shader-model 60 is accepted and produces valid HLSL with register binding.
    let src = write_tmp("naga_cli_p7t2_sm", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap();
    let out = dir.join("sm60.hlsl");

    let r = naga()
        .arg("--shader-model")
        .arg("60")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "shader-model 60: {}",
        String::from_utf8_lossy(&r.stderr)
    );

    let text = std::fs::read_to_string(&out).unwrap();
    assert!(
        text.contains("register(") || text.contains("cbuffer"),
        "expected HLSL register/cbuffer declaration, got:\n{text}"
    );
}

#[test]
fn defines_affect_glsl() {
    // A GLSL fragment using #ifdef FOO: compiling with -D FOO=1 vs without must produce
    // different SPIR-V (different constant color values in the IR).
    let src = write_tmp("naga_cli_p7t2_def", "ifdef.frag", IFDEF_GLSL);
    let dir = src.parent().unwrap();
    let out_no_def = dir.join("no_def.spv");
    let out_with_def = dir.join("with_def.spv");

    let r1 = naga()
        .args(["--input-kind", "glsl", "--shader-stage", "frag"])
        .arg(&src)
        .arg(&out_no_def)
        .output()
        .unwrap();
    assert!(
        r1.status.success(),
        "without define: {}",
        String::from_utf8_lossy(&r1.stderr)
    );

    let r2 = naga()
        .args(["--input-kind", "glsl", "--shader-stage", "frag"])
        .arg("-D")
        .arg("FOO=1")
        .arg(&src)
        .arg(&out_with_def)
        .output()
        .unwrap();
    assert!(
        r2.status.success(),
        "with define: {}",
        String::from_utf8_lossy(&r2.stderr)
    );

    let no_def_spv = std::fs::read(&out_no_def).unwrap();
    let with_def_spv = std::fs::read(&out_with_def).unwrap();
    assert_ne!(
        no_def_spv, with_def_spv,
        "-D FOO=1 should change SPIR-V output (different branch taken in #ifdef)"
    );
}

#[test]
fn bounds_check_policies_accepted() {
    // All three policy flags parse and compile successfully (kebab-case values).
    let src = write_tmp("naga_cli_p7t2_bcp", "bounds.wgsl", BOUNDS_WGSL);
    let dir = src.parent().unwrap();
    let out = dir.join("bounds.spv");

    let r = naga()
        .arg("--index-bounds-check-policy")
        .arg("restrict")
        .arg("--buffer-bounds-check-policy")
        .arg("read-zero-skip-write")
        .arg("--image-load-bounds-check-policy")
        .arg("unchecked")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "all three bounds-check policies should be accepted: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    // Produced valid SPIR-V.
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(
        &bytes[0..4],
        &[0x03, 0x02, 0x23, 0x07],
        "expected SPIR-V magic"
    );
}

#[test]
fn input_kind_overrides_extension() {
    // WGSL content in a file named "weird.txt" with --input-kind wgsl → validates successfully.
    let src = write_tmp("naga_cli_p7t2_ik", "weird.txt", TRIANGLE_WGSL);

    let r = naga()
        .arg("--input-kind")
        .arg("wgsl")
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "--input-kind wgsl on .txt file should succeed: {}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert!(
        String::from_utf8_lossy(&r.stdout).contains("Validation successful"),
        "expected 'Validation successful', got: {}",
        String::from_utf8_lossy(&r.stdout)
    );
}

#[test]
fn block_ctx_dir_accepted() {
    // --block-ctx-dir <dir> on a SPIR-V input → success and dumps block-ctx files into dir.
    let spv_src = write_tmp("naga_cli_p7t2_bcd", "tri.wgsl", TRIANGLE_WGSL);
    let dir = spv_src.parent().unwrap();
    let spv = dir.join("tri.spv");

    // Produce a SPIR-V file first.
    let r_spv = naga().arg(&spv_src).arg(&spv).output().unwrap();
    assert!(
        r_spv.status.success(),
        "wgsl→spv: {}",
        String::from_utf8_lossy(&r_spv.stderr)
    );

    let bctx_dir = dir.join("bctx");
    std::fs::create_dir_all(&bctx_dir).unwrap();

    let r = naga()
        .arg("--block-ctx-dir")
        .arg(&bctx_dir)
        .arg(&spv)
        .output()
        .unwrap();
    assert!(
        r.status.success(),
        "--block-ctx-dir should succeed: {}",
        String::from_utf8_lossy(&r.stderr)
    );

    // naga dumps one file per entry point: block_ctx.<Stage>-<name>.txt
    let dumps: Vec<_> = std::fs::read_dir(&bctx_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with("block_ctx"))
        .collect();
    assert!(
        !dumps.is_empty(),
        "--block-ctx-dir should write block_ctx.*.txt files into {:?}",
        bctx_dir
    );
}

// ── Phase 7 Task 4: diagnostic converter coverage + reflection depth ──────────

#[test]
fn glsl_stage_mismatch_is_json_warning() {
    // TRIANGLE_WGSL has fs_main (Fragment). Writing to .vert implies Vertex stage.
    // With --entry-point fs_main, naga detects Fragment != Vertex and emits a warning.
    let src = write_tmp("naga_cli_p7t4_mismatch", "tri.wgsl", TRIANGLE_WGSL);
    let dir = src.parent().unwrap().to_path_buf();
    let out = dir.join("out.vert");

    let r = naga()
        .arg("--entry-point")
        .arg("fs_main")
        .arg("--format")
        .arg("json")
        .arg(&src)
        .arg(&out)
        .output()
        .unwrap();

    // The command may succeed (output is still written) or fail; either way the JSON
    // must carry the stage-mismatch warning in its diagnostics array.
    let stdout = String::from_utf8_lossy(&r.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("stdout must be JSON, got: {stdout}\nerr: {e}"));

    let diags = v["diagnostics"]
        .as_array()
        .unwrap_or_else(|| panic!("JSON must have a 'diagnostics' array; got: {v}"));

    assert!(
        diags.iter().any(|d| {
            d["severity"] == "warning"
                && d["message"]
                    .as_str()
                    .unwrap_or("")
                    .to_lowercase()
                    .contains("stage")
        }),
        "expected a 'stage' warning diagnostic from glsl stage mismatch; diagnostics: {diags:?}"
    );
}

/// WGSL with a pipeline-overridable constant, a uniform binding, and a storage binding.
/// All three reflection categories are exercised by one shader.
const RICH_REFLECTION_WGSL: &str = r#"
override scale: f32 = 1.0;
@group(0) @binding(0) var<uniform> u: vec4<f32>;
@group(0) @binding(1) var<storage, read> s: array<f32>;
@fragment fn main() -> @location(0) vec4<f32> {
    return u * scale + vec4<f32>(s[0], 0.0, 0.0, 0.0);
}
"#;

#[test]
fn json_reflection_reports_overrides_and_resources() {
    let src = write_tmp("naga_cli_p7t4_refl", "rich.wgsl", RICH_REFLECTION_WGSL);

    let r = naga()
        .arg(&src)
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

    // Override 'scale' must be listed.
    let overrides = v["reflection"]["overrides"].as_array().unwrap();
    assert!(
        overrides.iter().any(|o| o["name"] == "scale"),
        "expected 'scale' in reflection.overrides; got: {overrides:?}"
    );

    // Both bound resources (group=0, binding=0 and group=0, binding=1) must appear.
    let resources = v["reflection"]["resources"].as_array().unwrap();
    assert!(
        resources
            .iter()
            .any(|r| r["group"] == 0 && r["binding"] == 0),
        "expected resource at group=0,binding=0; got: {resources:?}"
    );
    assert!(
        resources
            .iter()
            .any(|r| r["group"] == 0 && r["binding"] == 1),
        "expected resource at group=0,binding=1; got: {resources:?}"
    );
}
