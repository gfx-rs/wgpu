use anyhow::Context as _;

use crate::BenchmarkFile;

const FILE_PREFIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../target/bench/");
pub const PREVIOUS: &str = "previous";

pub(crate) fn get_comparison_file(baseline: Option<&str>) -> Option<BenchmarkFile> {
    let file_name = baseline.unwrap_or(PREVIOUS);
    let path = format!("{FILE_PREFIX}{file_name}.json");

    let file = std::fs::read_to_string(path).ok()?;
    let benchmark_file: BenchmarkFile = serde_json::from_str(&file).ok()?;
    Some(benchmark_file)
}

pub(crate) fn write_results_file(
    file_name: &str,
    output_file: &BenchmarkFile,
) -> anyhow::Result<()> {
    let path = format!("{FILE_PREFIX}{file_name}.json");
    let json = serde_json::to_string_pretty(output_file)?;
    std::fs::create_dir_all(FILE_PREFIX)
        .with_context(|| format!("Trying to create directory {FILE_PREFIX}"))?;
    std::fs::write(&path, json).with_context(|| format!("Trying to write file {path}"))?;
    Ok(())
}
