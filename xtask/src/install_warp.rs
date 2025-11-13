use anyhow::Context;
use pico_args::Arguments;
use xshell::Shell;

/// Keep this in sync with .github/actions/install-warp/action.yml
const WARP_VERSION: &str = "1.0.16.1";

/// Run the install-warp subcommand
pub fn run_install_warp(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    if cfg!(not(target_os = "windows")) {
        anyhow::bail!("WARP installation is only supported on Windows");
    }

    // Accept either --target-dir or --profile, but not both
    let target_dir_arg = args
        .opt_value_from_str::<_, String>("--target-dir")
        .ok()
        .flatten();

    let profile_arg = args
        .opt_value_from_str::<_, String>("--profile")
        .ok()
        .flatten();

    let target_dir = match (target_dir_arg, profile_arg) {
        (Some(_), Some(_)) => {
            anyhow::bail!("Cannot specify both --target-dir and --profile");
        }
        (Some(dir), None) => dir,
        (None, Some(profile)) => format!("target/{profile}"),
        (None, None) => "target/debug".to_string(),
    };

    install_warp(&shell, &target_dir)?;

    Ok(())
}

/// Install WARP DLL on Windows for testing.
///
/// This downloads the Microsoft.Direct3D.WARP NuGet package and extracts
/// the d3d10warp.dll into the specified target directory and its deps subdirectory.
pub fn install_warp(shell: &Shell, target_dir: &str) -> anyhow::Result<()> {
    // Check if WARP is already installed with the correct version
    let version_file = format!("{target_dir}/warp.txt");
    if let Ok(installed_version) = shell.read_file(&version_file) {
        if installed_version.trim() == WARP_VERSION {
            log::info!("WARP {WARP_VERSION} already installed, skipping download");
            return Ok(());
        } else {
            log::info!(
                "WARP version mismatch (installed: {}, required: {}), re-downloading",
                installed_version.trim(),
                WARP_VERSION
            );
        }
    }

    log::info!("Installing WARP {WARP_VERSION}");

    let warp_url =
        format!("https://www.nuget.org/api/v2/package/Microsoft.Direct3D.WARP/{WARP_VERSION}");

    // Download WARP NuGet package
    log::info!("Downloading WARP from {warp_url}");
    shell
        .cmd("curl.exe")
        .args(["-L", "--retry", "5", &warp_url, "-o", "warp.zip"])
        .run()
        .context("Failed to download WARP package")?;

    // Create target/warp directory
    shell
        .create_dir("target/warp")
        .context("Failed to create target/warp directory")?;

    // Extract the DLL using tar (available on Windows 10+)
    log::info!("Extracting WARP DLL");
    shell
        .cmd("tar")
        .args([
            "-xf",
            "warp.zip",
            "-C",
            "target/warp",
            "build/native/bin/x64/d3d10warp.dll",
        ])
        .run()
        .context("Failed to extract WARP DLL using tar")?;

    // Copy the DLL to target directory and deps subdirectory
    let source = "target/warp/build/native/bin/x64/d3d10warp.dll";
    let target_deps = format!("{target_dir}/deps");
    let target_dirs = [target_dir, target_deps.as_str()];

    for dir in &target_dirs {
        // Create target directory if it doesn't exist
        shell
            .create_dir(dir)
            .with_context(|| format!("Failed to create target directory: {dir}"))?;

        let dest = format!("{dir}/d3d10warp.dll");

        log::info!("Copying WARP DLL to {dir}");
        shell
            .copy_file(source, &dest)
            .with_context(|| format!("Failed to copy WARP DLL to {dir}"))?;
    }

    // Write version file to track installed version (only at root)
    let version_file = format!("{target_dir}/warp.txt");
    shell
        .write_file(&version_file, WARP_VERSION)
        .context("Failed to write version file")?;

    // Cleanup temporary files
    let _ = shell.remove_path("warp.zip");
    let _ = shell.remove_path("target/warp");

    log::info!("WARP installation complete");

    Ok(())
}
