//! External tool hooks: spirv-val, spirv-opt, dxc (subprocesses on PATH).

use anyhow::{anyhow, bail, Context as _};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Copy, Default)]
pub struct Hooks {
    pub spirv_val: bool,
    pub spirv_opt: bool,
    pub dxc: bool,
}

impl Hooks {
    pub fn any(&self) -> bool {
        self.spirv_val || self.spirv_opt || self.dxc
    }
}

/// Map a shader stage + HLSL shader model to a DXC target profile, e.g. `cs_6_0`.
pub fn dxc_profile(stage: naga::ShaderStage, model: naga::back::hlsl::ShaderModel) -> String {
    let prefix = match stage {
        naga::ShaderStage::Vertex => "vs",
        naga::ShaderStage::Fragment => "ps",
        naga::ShaderStage::Compute => "cs",
        naga::ShaderStage::Mesh => "ms",
        naga::ShaderStage::Task => "as",
        // Ray-tracing stages compile under lib_*; fall back to lib for anything else.
        _ => "lib",
    };
    format!("{prefix}_{}", model.to_str())
}

/// Locate a tool on PATH or produce an actionable error.
fn find_tool(name: &str) -> anyhow::Result<std::path::PathBuf> {
    which::which(name).map_err(|_| {
        anyhow!("`{name}` was not found on PATH; install it or remove the corresponding flag")
    })
}

pub fn run_spirv_val(spv_path: &Path) -> anyhow::Result<()> {
    let tool = find_tool("spirv-val")?;
    let output = Command::new(&tool)
        .arg(spv_path)
        .output()
        .with_context(|| format!("failed to run spirv-val ({})", tool.display()))?;
    if !output.status.success() {
        bail!(
            "spirv-val failed for {}:\n{}",
            spv_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

pub fn run_spirv_opt(spv_path: &Path) -> anyhow::Result<()> {
    let tool = find_tool("spirv-opt")?;
    // Optimize in place: read from spv_path, write back to spv_path.
    let output = Command::new(&tool)
        .arg(spv_path)
        .arg("-O")
        .arg("-o")
        .arg(spv_path)
        .output()
        .with_context(|| format!("failed to run spirv-opt ({})", tool.display()))?;
    if !output.status.success() {
        bail!(
            "spirv-opt failed for {}:\n{}",
            spv_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

pub fn run_dxc(
    hlsl_path: &Path,
    entry_points: &[(String, naga::ShaderStage)],
    model: naga::back::hlsl::ShaderModel,
) -> anyhow::Result<()> {
    let tool = find_tool("dxc")?;
    if entry_points.is_empty() {
        bail!("--dxc: no entry points to compile in {}", hlsl_path.display());
    }
    for (name, stage) in entry_points {
        let profile = dxc_profile(*stage, model);
        // Output: <hlsl_path stem>.<entry>.dxil next to the HLSL file.
        let out = hlsl_path.with_extension(format!("{name}.dxil"));
        let output = Command::new(&tool)
            .arg(hlsl_path)
            .arg("-T")
            .arg(&profile)
            .arg("-E")
            .arg(name)
            .arg("-Fo")
            .arg(&out)
            .output()
            .with_context(|| format!("failed to run dxc ({})", tool.display()))?;
        if !output.status.success() {
            bail!(
                "dxc failed for entry `{name}` ({profile}) in {}:\n{}",
                hlsl_path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::dxc_profile;
    use naga::back::hlsl::ShaderModel;
    use naga::ShaderStage;

    #[test]
    fn profile_mapping() {
        assert_eq!(dxc_profile(ShaderStage::Vertex, ShaderModel::V6_0), "vs_6_0");
        assert_eq!(dxc_profile(ShaderStage::Fragment, ShaderModel::V6_2), "ps_6_2");
        assert_eq!(dxc_profile(ShaderStage::Compute, ShaderModel::V6_0), "cs_6_0");
        assert_eq!(dxc_profile(ShaderStage::Mesh, ShaderModel::V6_5), "ms_6_5");
        assert_eq!(dxc_profile(ShaderStage::Task, ShaderModel::V6_5), "as_6_5");
    }
}
