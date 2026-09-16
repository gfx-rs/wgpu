use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{bail, Context};
#[derive(Clone, Copy)]
pub(super) enum Mode {
    Check,
    Write,
}

// Recursively syncs `root/source` into `root/target`.
//
// In `Check` mode errors on any differences, in `Write` mode
// deletes extra files/directories, copies updated ones, and makes
// missing ones.
pub(super) fn sync_directory(
    root: &Path,
    source: &Path,
    target: &Path,
    mode: Mode,
    discrepancies: &mut Vec<String>,
) -> anyhow::Result<()> {
    let source_root = root.join(source);
    let target_root = root.join(target);
    let source_files = collect_files(&source_root).with_context(|| {
        format!(
            "could not read source-of-truth directory `{}`",
            source.display()
        )
    })?;
    let target_files = if target_root.try_exists()? {
        collect_files(&target_root)
            .with_context(|| format!("could not read generated directory `{}`", target.display()))?
    } else {
        BTreeSet::new()
    };

    for relative in source_files.difference(&target_files) {
        let source_file = source.join(relative);
        let target_file = target.join(relative);
        match mode {
            Mode::Check => discrepancies.push(format!(
                "generated file `{}` is missing; `{}` is the source of truth",
                target_file.display(),
                source_file.display()
            )),
            Mode::Write => copy_file(root, &source_file, &target_file)?,
        }
    }

    let mut extra_files = target_files
        .difference(&source_files)
        .map(|relative| target.join(relative))
        .collect::<Vec<_>>();
    let ignored = git_ignored_paths(root, &extra_files)?;
    extra_files.retain(|target_file| !ignored.contains(target_file));

    for target_file in extra_files {
        let source_file = source.join(target_file.strip_prefix(target)?);
        match mode {
            Mode::Check => discrepancies.push(format!(
                "generated file `{}` is extra; source-of-truth file `{}` does not exist",
                target_file.display(),
                source_file.display()
            )),
            Mode::Write => {
                eprintln!(
                    "Removing extra generated file `{}`; `{}` is the source of truth.",
                    target_file.display(),
                    source.display()
                );
                fs::remove_file(root.join(&target_file)).with_context(|| {
                    format!(
                        "could not remove extra generated file `{}`",
                        target_file.display()
                    )
                })?;
            }
        }
    }

    for relative in source_files.intersection(&target_files) {
        sync_file(
            root,
            &source.join(relative),
            &target.join(relative),
            mode,
            discrepancies,
        )?;
    }

    if matches!(mode, Mode::Write) && target_root.try_exists()? {
        remove_empty_directories(&target_root)?;
    }

    Ok(())
}

pub(super) fn sync_file(
    root: &Path,
    source: &Path,
    target: &Path,
    mode: Mode,
    discrepancies: &mut Vec<String>,
) -> anyhow::Result<()> {
    let source_path = root.join(source);
    let target_path = root.join(target);
    let source_contents = fs::read(&source_path)
        .with_context(|| format!("could not read source-of-truth file `{}`", source.display()))?;
    let target_contents = match fs::read(&target_path) {
        Ok(contents) => Some(contents),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("could not read generated file `{}`", target.display()));
        }
    };

    if target_contents.as_deref() == Some(source_contents.as_slice()) {
        return Ok(());
    }

    match mode {
        Mode::Check => {
            let problem = if target_contents.is_some() {
                "differs from"
            } else {
                "is missing; expected a copy of"
            };
            discrepancies.push(format!(
                "generated file `{}` {problem} source-of-truth file `{}`",
                target.display(),
                source.display()
            ));
            Ok(())
        }
        Mode::Write => copy_file(root, source, target),
    }
}

fn copy_file(root: &Path, source: &Path, target: &Path) -> anyhow::Result<()> {
    let source_path = root.join(source);
    let target_path = root.join(target);
    if let Some(parent) = target_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create directory `{}`", parent.display()))?;
    }
    eprintln!(
        "Copying source-of-truth file `{}` to generated file `{}`.",
        source.display(),
        target.display()
    );
    fs::copy(&source_path, &target_path).with_context(|| {
        format!(
            "could not copy source-of-truth file `{}` to generated file `{}`",
            source.display(),
            target.display()
        )
    })?;
    Ok(())
}

/// Returns the subset of `paths` that git ignores.
fn git_ignored_paths(root: &Path, paths: &[PathBuf]) -> anyhow::Result<BTreeSet<PathBuf>> {
    if paths.is_empty() {
        return Ok(BTreeSet::new());
    }

    let mut input = Vec::new();
    for path in paths {
        let path = path
            .to_str()
            .with_context(|| format!("path `{}` is not valid UTF-8", path.display()))?;
        input.extend_from_slice(path.as_bytes());
        input.push(0);
    }

    // The `-z` form is necessary because the default output quotes any path that
    // holds a backslash, which every relative path does on Windows.
    let mut child = Command::new("git")
        .args(["check-ignore", "--stdin", "-z"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("could not run `git check-ignore`")?;
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(&input)
        .context("could not write to `git check-ignore`")?;
    let output = child
        .wait_with_output()
        .context("could not read from `git check-ignore`")?;
    // `git check-ignore` exits 1 when it matches nothing, which is not an error.
    if !matches!(output.status.code(), Some(0 | 1)) {
        bail!(
            "`git check-ignore` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let stdout =
        String::from_utf8(output.stdout).context("`git check-ignore` wrote invalid UTF-8")?;
    Ok(stdout
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(PathBuf::from)
        .collect())
}

fn collect_files(root: &Path) -> anyhow::Result<BTreeSet<PathBuf>> {
    if !root.try_exists()? {
        bail!("directory `{}` does not exist", root.display());
    }

    let mut files = BTreeSet::new();
    collect_files_recursive(root, root, &mut files)?;
    Ok(files)
}

fn collect_files_recursive(
    root: &Path,
    directory: &Path,
    files: &mut BTreeSet<PathBuf>,
) -> anyhow::Result<()> {
    let entries = fs::read_dir(directory)
        .with_context(|| format!("could not read directory `{}`", directory.display()))?;
    for entry in entries {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_dir() {
            collect_files_recursive(root, &path, files)?;
        } else if file_type.is_file() {
            files.insert(path.strip_prefix(root)?.to_owned());
        } else {
            bail!("unsupported file entry `{}`", path.display());
        }
    }
    Ok(())
}

fn remove_empty_directories(root: &Path) -> anyhow::Result<bool> {
    let mut empty = true;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if !remove_empty_directories(&entry.path())? {
                empty = false;
            }
        } else {
            empty = false;
        }
    }

    if empty {
        fs::remove_dir(root)?;
    }
    Ok(empty)
}
