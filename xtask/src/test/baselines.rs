use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use pico_args::Arguments;
use termcolor::Color;

use super::{
    analyze::ExpectationBaseline,
    report::{print_general, print_section_line},
};

const BASELINE_DIR_REL_PATH: &str = "target\\xtask";

pub(super) fn opt_value_alias(
    args: &mut Arguments,
    long: &'static str,
    short: &'static str,
    _label: &'static str,
) -> Result<Option<String>> {
    let long_value = args
        .opt_value_from_str::<_, String>(long)
        .with_context(|| format!("Invalid --{long} value"))?;
    let short_value = args
        .opt_value_from_str::<_, String>(short)
        .with_context(|| format!("Invalid -{short} value"))?;
    if long_value.is_some() && short_value.is_some() {
        anyhow::bail!("Specify only one of -{short} or --{long}.");
    }
    Ok(long_value.or(short_value))
}

fn baseline_dir() -> PathBuf {
    PathBuf::from(BASELINE_DIR_REL_PATH)
}

fn baseline_path(name: &str) -> Result<PathBuf> {
    // Validate baseline name contains no path separators or parent directory references.
    // This is intentionally strict to prevent path traversal issues.
    // Note: checks for both Unix (/) and Windows (\) separators for cross-platform safety.
    if name.is_empty()
        || name.contains('\\')
        || name.contains('/')
        || name.contains(':')
        || name.contains("..")
    {
        anyhow::bail!(
            "Baseline name `{name}` is invalid; use a simple name like `my-machine` (no path separators)."
        );
    }
    Ok(baseline_dir().join(format!("{name}.json")))
}

fn available_baseline_names() -> Result<Vec<String>> {
    let dir = baseline_dir();
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut names = Vec::new();
    for entry in fs::read_dir(&dir)
        .with_context(|| format!("Failed to read baseline directory {}", dir.display()))?
    {
        let entry = entry
            .with_context(|| format!("Failed to inspect baseline entry in {}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
            names.push(stem.to_owned());
        }
    }
    names.sort();
    Ok(names)
}

pub(super) fn print_baseline_list() -> Result<()> {
    let names = available_baseline_names()?;
    if names.is_empty() {
        print_general("No baselines found.");
        return Ok(());
    }
    print_section_line("Available baselines:", Color::Blue);
    for name in names {
        print_general(format!("  {name}"));
    }
    Ok(())
}

pub(super) fn clear_baselines_on_disk() -> Result<usize> {
    let dir = baseline_dir();
    if !dir.exists() {
        return Ok(0);
    }
    let mut removed = 0usize;
    for entry in fs::read_dir(&dir)
        .with_context(|| format!("Failed to read baseline directory {}", dir.display()))?
    {
        let entry = entry
            .with_context(|| format!("Failed to inspect baseline entry in {}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        fs::remove_file(&path)
            .with_context(|| format!("Failed to remove baseline file {}", path.display()))?;
        removed += 1;
    }
    Ok(removed)
}

pub(super) fn load_baseline(name: Option<&str>) -> Result<ExpectationBaseline> {
    let Some(name) = name else {
        return Ok(ExpectationBaseline::default());
    };
    let path = baseline_path(name)?;
    if !path.exists() {
        let available = available_baseline_names()?;
        if available.is_empty() {
            anyhow::bail!(
                "Baseline `{name}` not found in {}. No baselines exist yet. Save one with `--save-baseline <name>`.",
                baseline_dir().display()
            );
        }
        anyhow::bail!(
            "Baseline `{name}` not found in {}. Available baselines: {}",
            baseline_dir().display(),
            available.join(", ")
        );
    }

    let content = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read baseline file {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse baseline file {}", path.display()))
}

pub(super) fn save_baseline(name: &str, baseline: &ExpectationBaseline) -> Result<()> {
    let path = baseline_path(name)?;
    let dir = baseline_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create baseline directory {}", dir.display()))?;
    }

    let content =
        serde_json::to_string_pretty(baseline).context("Failed to serialize baseline JSON")?;
    fs::write(&path, content)
        .with_context(|| format!("Failed to write baseline file {}", path.display()))?;
    print_section_line(
        format!("Saved baseline `{name}` to {}", path.display()),
        Color::Green,
    );
    Ok(())
}
