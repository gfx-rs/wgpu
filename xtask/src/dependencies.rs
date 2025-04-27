use std::process::Command;

use anyhow::Context;

pub struct Prerequisites {
    pub cargo_nextest: bool,
    pub taplo: bool,
    pub wasm_bindgen: bool,
    pub simple_http_server: bool,
    pub cargo_llvm_cov: bool,
    pub spirv_assembler: bool,
    pub vulkan_sdk: bool,
}

impl Prerequisites {
    pub const ALL: Self = Self {
        cargo_nextest: true,
        taplo: true,
        wasm_bindgen: true,
        simple_http_server: true,
        cargo_llvm_cov: true,
        spirv_assembler: true,
        vulkan_sdk: true,
    };

    pub const NONE: Self = Self {
        cargo_nextest: false,
        taplo: false,
        wasm_bindgen: false,
        simple_http_server: false,
        cargo_llvm_cov: false,
        spirv_assembler: false,
        vulkan_sdk: false,
    };
}

/// Sets the PATH environment variable to include the .binaries/bin directory
/// so that our installed pre-requisites are available in the PATH.
pub fn set_path() -> Result<(), anyhow::Error> {
    let path_separator = if cfg!(windows) { ";" } else { ":" };
    let old_path = std::env::var("PATH").context("Couldn't get PATH")?;
    unsafe {
        std::env::set_var(
            "PATH",
            format!(
                "{}/../.binaries/bin{path_separator}{}",
                env!("CARGO_MANIFEST_DIR"),
                old_path
            ),
        )
    };
    Ok(())
}

fn indented_utf8(text: &[u8]) -> anyhow::Result<String> {
    let text = std::str::from_utf8(text).context("Failed to convert to utf8")?;
    Ok(indent_string(text.trim()))
}

fn indent_string(text: &str) -> String {
    let mut indented = String::new();
    for line in text.lines() {
        indented.push_str(&format!("    {line}\n"));
    }
    indented
}

/// Runs a command silently, printing information about it if it fails.
fn print_on_failure(mut command: Command) -> anyhow::Result<()> {
    let command_string = format!("{command:?}");
    let output = command
        .output()
        .with_context(|| format!("Failed to run {command_string}"))?;

    if !output.status.success() {
        let stdout = indented_utf8(&output.stdout).context("stdout")?;
        let stderr = indented_utf8(&output.stderr).context("stderr")?;

        println!("\nCommand failed: {command_string}");
        println!("stdout:\n{stdout}");
        println!("stderr:\n{stderr}");

        return Err(anyhow::anyhow!("Command failed"));
    }

    Ok(())
}

fn run_version_check(
    shell: &xshell::Shell,
    command: &[&str],
    version_regex: Option<&str>,
) -> anyhow::Result<String> {
    let output = shell.cmd(command[0]).args(&command[1..]).quiet().output()?;
    let version = std::str::from_utf8(&output.stdout)
        .context("Failed to convert output to utf8")?
        .trim()
        .to_string();

    if let Some(regex) = version_regex {
        let regex = regex_lite::Regex::new(regex).unwrap();

        let Some(refined_version) = regex.find(&version) else {
            return Err(anyhow::anyhow!(
                "Failed to parse version from output: {version}"
            ));
        };

        Ok(refined_version.as_str().to_string())
    } else {
        Ok(version)
    }
}

pub fn setup_prerequisites(shell: &xshell::Shell, prereq: Prerequisites) -> anyhow::Result<()> {
    let binstall_output = run_version_check(shell, &["cargo", "binstall", "-V"], None);

    println!("Checking prerequisites...");

    let mut fail = false;

    if prereq.vulkan_sdk {
        let vulkan_sdk_path = shell.var("VULKAN_SDK");
        match vulkan_sdk_path {
            Ok(path) if !path.trim().is_empty() => {
                println!(" ✅  Found VULKAN_SDK set to {}", path)
            }
            _ => {
                println!(
                    " ❌  VULKAN_SDK not set, please install the Vulkan SDK from https://vulkan.lunarg.com/"
                );
                fail = true;
            }
        };
    }

    if prereq.spirv_assembler {
        let spirv_as_version = run_version_check(shell, &["spirv-as", "--version"], Some("v[^ ]*"));

        match spirv_as_version {
            Ok(version) => println!(" ✅  Found spirv-as {version}"),
            Err(_) => {
                println!(" ❌  spirv-as not found, please install the Vulkan SDK from https://vulkan.lunarg.com/");
                fail = true;
            }
        }
    }

    let command = &if let Ok(output) = binstall_output {
        println!(" ✅  Found cargo binstall {output}",);
        vec!["binstall", "--no-confirm"]
    } else {
        println!(
            " ⚠️   cargo binstall not found, using cargo install instead, this may take a while"
        );
        println!(
            "     Please install cargo binstall for faster installs: cargo install cargo-binstall"
        );
        vec!["install"]
    };

    if prereq.cargo_nextest {
        install_dep(command, "cargo-nextest@0.9.93")?;
    }

    if prereq.taplo {
        install_dep(command, "taplo-cli@0.9.3")?;
    }

    if prereq.cargo_llvm_cov {
        install_dep(command, "cargo-llvm-cov@0.6.16")?;
    }

    if prereq.wasm_bindgen {
        install_dep(command, "wasm-bindgen-cli@0.2.100")?;
    }

    if prereq.simple_http_server {
        install_dep(command, "simple-http-server@0.6.11")?;
    }

    if fail {
        log::error!("Some prerequisites are missing, please install them and try again.");
        return Err(anyhow::anyhow!("Missing prerequisites"));
    }

    Ok(())
}

fn install_dep(raw_command: &[&str], dep: &str) -> Result<(), anyhow::Error> {
    println!(" ⚒️   Installing {dep}...");
    let mut command = Command::new("cargo");
    command.args(raw_command);
    command.arg(dep);

    print_on_failure(command)?;
    println!(" ✅  Installed {dep}");
    Ok(())
}
