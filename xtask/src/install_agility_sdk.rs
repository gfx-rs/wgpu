use anyhow::Context;
use pico_args::Arguments;
use xshell::Shell;

/// Keep this in sync with the version used in CI
const AGILITY_SDK_VERSION: &str = "1.619.0";
const AGILITY_SDK_FEATURE_VERSION: u32 = 619;

pub struct AgilitySDKInfo {
    pub sdk_path: String,
    pub sdk_version: u32,
}

/// Run the install-agility-sdk subcommand
pub fn run_install_agility_sdk(shell: Shell, _args: Arguments) -> anyhow::Result<()> {
    if cfg!(not(target_os = "windows")) {
        anyhow::bail!("Agility SDK installation is only supported on Windows");
    }

    let info = install_agility_sdk(&shell)?;

    println!("WGPU_DX12_AGILITY_SDK_PATH={}", info.sdk_path);
    println!("WGPU_DX12_AGILITY_SDK_VERSION={}", info.sdk_version);

    Ok(())
}

/// Returns the Agility SDK NuGet package's architecture directory name
/// for the current target architecture.
fn sdk_arch_dir() -> &'static str {
    if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "x86") {
        "win32"
    } else {
        panic!("Unsupported target architecture for D3D12 Agility SDK")
    }
}

/// Install the D3D12 Agility SDK on Windows.
///
/// This downloads the Microsoft.Direct3D.D3D12 NuGet package and extracts
/// the native binaries into `target/agility-sdk/`.
pub fn install_agility_sdk(shell: &Shell) -> anyhow::Result<AgilitySDKInfo> {
    let sdk_dir = "target/agility-sdk";
    let version_file = format!("{sdk_dir}/version.txt");
    let arch = sdk_arch_dir();
    let bin_subdir = format!("build/native/bin/{arch}");

    // Check if the SDK is already installed with the correct version
    if let Ok(installed_version) = shell.read_file(&version_file) {
        if installed_version.trim() == AGILITY_SDK_VERSION {
            log::info!("Agility SDK {AGILITY_SDK_VERSION} already installed, skipping download");
            let sdk_path = shell.current_dir().join(sdk_dir).join(&bin_subdir);
            return Ok(AgilitySDKInfo {
                sdk_path: sdk_path.to_string_lossy().into_owned(),
                sdk_version: AGILITY_SDK_FEATURE_VERSION,
            });
        } else {
            log::info!(
                "Agility SDK version mismatch (installed: {}, required: {}), re-downloading",
                installed_version.trim(),
                AGILITY_SDK_VERSION
            );
        }
    }

    log::info!("Installing Agility SDK {AGILITY_SDK_VERSION}");

    let url = format!(
        "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/{AGILITY_SDK_VERSION}"
    );

    // Download the NuGet package
    log::info!("Downloading Agility SDK from {url}");
    shell
        .cmd("curl.exe")
        .args(["-L", "--retry", "5", &url, "-o", "target/agility-sdk.zip"])
        .run()
        .context("Failed to download Agility SDK package")?;

    // Create target directory
    shell
        .create_dir(sdk_dir)
        .context("Failed to create target/agility-sdk directory")?;

    // Extract the native binaries using tar (available on Windows 10+)
    log::info!("Extracting Agility SDK ({arch})");
    shell
        .cmd("tar")
        .args(["-xf", "target/agility-sdk.zip", "-C", sdk_dir, &bin_subdir])
        .run()
        .context("Failed to extract Agility SDK using tar")?;

    // Write version file to track installed version
    shell
        .write_file(&version_file, AGILITY_SDK_VERSION)
        .context("Failed to write version file")?;

    // Cleanup
    let _ = shell.remove_path("target/agility-sdk.zip");

    let sdk_path = shell.current_dir().join(sdk_dir).join(&bin_subdir);

    log::info!("Agility SDK installation complete");

    Ok(AgilitySDKInfo {
        sdk_path: sdk_path.to_string_lossy().into_owned(),
        sdk_version: AGILITY_SDK_FEATURE_VERSION,
    })
}
