use std::{io, process::Command};

pub(crate) fn check_all_programs(programs: &[&str]) -> anyhow::Result<()> {
    let mut failed = Vec::new();
    for &program in programs {
        let mut cmd = Command::new(program);
        cmd.arg("--help");
        let output = cmd.output();
        match output {
            Ok(_output) => {
                log::info!("Checking for {program} in PATH: ✅");
            }
            Err(e) if matches!(e.kind(), io::ErrorKind::NotFound) => {
                log::error!("Checking for {program} in PATH: ❌");
                failed.push(program);
            }
            Err(e) => {
                log::error!("Checking for {program} in PATH: ❌");
                panic!("Unknown IO error: {:?}", e);
            }
        }
    }

    if !failed.is_empty() {
        log::error!(
            "Please install them with: cargo install {}",
            failed.join(" ")
        );
        anyhow::bail!("Missing programs in PATH");
    }

    Ok(())
}
