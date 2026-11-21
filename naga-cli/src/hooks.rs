//! External tool hooks: spirv-val, spirv-opt, dxc (subprocesses).

use anyhow::{anyhow, bail, Context as _};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Default)]
pub struct Hooks {
    pub spirv_val: bool,
    pub spirv_opt: bool,
    pub dxc: bool,
    pub spirv_val_path: Option<PathBuf>,
    pub spirv_opt_path: Option<PathBuf>,
    pub dxc_path: Option<PathBuf>,
}

impl Hooks {
    pub fn any(&self) -> bool {
        self.spirv_val || self.spirv_opt || self.dxc
    }

    /// Run the requested SPIR-V hooks on the SPIR-V file at `spv_path`.
    pub fn run_spirv(&self, spv_path: &Path) -> anyhow::Result<()> {
        if self.spirv_opt {
            run_spirv_opt(&find_tool("spirv-opt", &self.spirv_opt_path)?, spv_path)?;
        }
        if self.spirv_val {
            run_spirv_val(&find_tool("spirv-val", &self.spirv_val_path)?, spv_path)?;
        }
        Ok(())
    }

    pub fn dxc_tool(&self) -> anyhow::Result<PathBuf> {
        find_tool("dxc", &self.dxc_path)
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

/// Use the explicit tool path if given, else locate the tool on PATH.
fn find_tool(name: &str, explicit: &Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.clone());
    }
    which::which(name).map_err(|_| {
        anyhow!(
            "`{name}` was not found on PATH; install it, pass --{name}-path, or remove the \
             corresponding flag"
        )
    })
}

fn run_spirv_val(tool: &Path, spv_path: &Path) -> anyhow::Result<()> {
    let output = Command::new(tool)
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

fn run_spirv_opt(tool: &Path, spv_path: &Path) -> anyhow::Result<()> {
    // Optimize in place: read from spv_path, write back to spv_path.
    let output = Command::new(tool)
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

/// Compile one entry point of the HLSL file at `hlsl_path` to DXIL at `out`.
pub fn run_dxc(
    tool: &Path,
    hlsl_path: &Path,
    name: &str,
    stage: naga::ShaderStage,
    model: naga::back::hlsl::ShaderModel,
    out: &Path,
) -> anyhow::Result<()> {
    let profile = dxc_profile(stage, model);
    let output = Command::new(tool)
        .arg(hlsl_path)
        .arg("-T")
        .arg(&profile)
        .arg("-E")
        .arg(name)
        .arg("-Fo")
        .arg(out)
        .output()
        .with_context(|| format!("failed to run dxc ({})", tool.display()))?;
    if !output.status.success() {
        bail!(
            "dxc failed for entry `{name}` ({profile}) in {}:\n{}",
            hlsl_path.display(),
            String::from_utf8_lossy(&output.stderr)
        );
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
        assert_eq!(
            dxc_profile(ShaderStage::Vertex, ShaderModel::V6_0),
            "vs_6_0"
        );
        assert_eq!(
            dxc_profile(ShaderStage::Fragment, ShaderModel::V6_2),
            "ps_6_2"
        );
        assert_eq!(
            dxc_profile(ShaderStage::Compute, ShaderModel::V6_0),
            "cs_6_0"
        );
        assert_eq!(dxc_profile(ShaderStage::Mesh, ShaderModel::V6_5), "ms_6_5");
        assert_eq!(dxc_profile(ShaderStage::Task, ShaderModel::V6_5), "as_6_5");
    }
}
