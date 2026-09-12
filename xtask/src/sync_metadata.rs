use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{bail, Context};
use pico_args::Arguments;
use xshell::Shell;

const LICENSE_FILES: &[&str] = &["LICENSE.APACHE", "LICENSE.MIT"];

#[derive(Clone, Copy)]
enum Mode {
    Check,
    Write,
}

pub(crate) fn run_sync_metadata(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    let mode = if args.contains("--check") {
        Mode::Check
    } else {
        Mode::Write
    };

    let unknown_args = args.finish();
    if !unknown_args.is_empty() {
        crate::bad_arguments!(
            "Unknown arguments to sync-metadata subcommand: {:?}",
            unknown_args
        );
    }

    let root = shell.current_dir();
    let license_targets = license_targets(&shell, &root)?;
    let discrepancies = sync_metadata(&root, &license_targets, mode)?;

    if discrepancies.is_empty() {
        match mode {
            Mode::Check => eprintln!("Repository metadata is synchronized."),
            Mode::Write => eprintln!("Repository metadata synchronized."),
        }
        return Ok(());
    }

    for discrepancy in discrepancies {
        eprintln!("error: {discrepancy}");
    }
    eprintln!("hint: edit the source-of-truth files, then run `cargo xtask sync-metadata`");
    bail!("repository metadata is not synchronized")
}

/// Sync all repository metadata.
fn sync_metadata(
    root: &Path,
    license_targets: &[PathBuf],
    mode: Mode,
) -> anyhow::Result<Vec<String>> {
    let mut discrepancies = Vec::new();

    // Sync AGENTS.md
    sync_file(
        root,
        Path::new("AGENTS.md"),
        Path::new("CLAUDE.md"),
        mode,
        &mut discrepancies,
    )?;
    // Sync .agents/
    sync_directory(
        root,
        Path::new(".agents"),
        Path::new(".claude"),
        mode,
        &mut discrepancies,
    )?;

    // Sync license files to each crate
    for target in license_targets {
        for license in LICENSE_FILES {
            sync_file(
                root,
                Path::new(license),
                &Path::new(target).join(license),
                mode,
                &mut discrepancies,
            )?;
        }
    }

    Ok(discrepancies)
}

/// Returns the list of all crates that need license files.
fn license_targets(shell: &Shell, root: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let output = shell
        .cmd("cargo")
        .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
        .read()
        .context("could not read Cargo metadata for license targets")?;
    let metadata = serde_json::from_str(&output).context("could not parse Cargo metadata")?;
    publishable_default_member_directories(&metadata, root)
}

/// Extracts the directories of all default members of the repository.
///
/// Grabs `$.workspace_default_members[*]` then uses that to index
/// into `$.packages[*]` and grab the `id`, `publish`, and `manifest_path`
/// fields.
fn publishable_default_member_directories(
    metadata: &serde_json::Value,
    root: &Path,
) -> anyhow::Result<Vec<PathBuf>> {
    // Both sides of the `strip_prefix` below must be canonical. Windows
    // canonicalization adds a `\\?\` prefix that `cargo metadata` does not use.
    let root = root
        .canonicalize()
        .with_context(|| format!("could not resolve the repository root `{}`", root.display()))?;

    let default_members = metadata
        .get("workspace_default_members")
        .and_then(serde_json::Value::as_array)
        .context("Cargo metadata has no `workspace_default_members` array")?;
    let default_member_ids = default_members
        .iter()
        .map(|member| {
            member
                .as_str()
                .context("Cargo metadata contains a non-string default member ID")
        })
        .collect::<anyhow::Result<BTreeSet<_>>>()?;
    let packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .context("Cargo metadata has no `packages` array")?;

    let mut matched_default_members = BTreeSet::new();
    let mut targets = BTreeSet::new();
    for package in packages {
        let id = package
            .get("id")
            .and_then(serde_json::Value::as_str)
            .context("Cargo metadata package has no string `id`")?;
        if !default_member_ids.contains(id) {
            continue;
        }
        matched_default_members.insert(id);

        let publishable = match package.get("publish") {
            None | Some(serde_json::Value::Null) => true,
            Some(serde_json::Value::Array(registries)) => !registries.is_empty(),
            Some(_) => bail!("Cargo metadata package `{id}` has an invalid `publish` value"),
        };
        if !publishable {
            continue;
        }

        let manifest_path = package
            .get("manifest_path")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("Cargo metadata package `{id}` has no manifest path"))?;
        let package_directory = Path::new(manifest_path)
            .parent()
            .with_context(|| format!("package manifest `{manifest_path}` has no parent directory"))?
            .canonicalize()
            .with_context(|| {
                format!("could not resolve the directory of default member package `{id}`")
            })?;
        let relative_directory = package_directory.strip_prefix(&root).with_context(|| {
            format!(
                "default member package `{id}` is outside the repository root `{}`",
                root.display()
            )
        })?;
        targets.insert(relative_directory.to_owned());
    }

    let missing_packages = default_member_ids
        .difference(&matched_default_members)
        .copied()
        .collect::<Vec<_>>();
    if !missing_packages.is_empty() {
        bail!(
            "Cargo metadata has no package entries for default members: {}",
            missing_packages.join(", ")
        );
    }

    Ok(targets.into_iter().collect())
}

// Recursively syncs `root/source` into `root/target`.
//
// In `Check` mode errors on any differences, in `Write` mode
// deletes extra files/directories, copies updated ones, and makes
// missing ones.
fn sync_directory(
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

fn sync_file(
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
///
/// `git check-ignore` exits 1 when it matches nothing, which is not an error.
/// The `-z` form is necessary because the default output quotes any path that
/// holds a backslash, which every relative path does on Windows.
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
    if !matches!(output.status.code(), Some(0 | 1)) {
        bail!(
            "`git check-ignore` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let stdout =
        String::from_utf8(output.stdout).context("`git check-ignore` wrote invalid UTF-8")?;
    Ok(stdout
        .split(' ')
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
            bail!("unsupported metadata entry `{}`", path.display());
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

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        process::{Command, Stdio},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::{publishable_default_member_directories, sync_directory, sync_file, Mode};

    static NEXT_TEMP_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    fn git(directory: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(directory)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "`git {}` failed", args.join(" "));
    }

    struct TempDirectory(PathBuf);

    impl TempDirectory {
        fn new() -> Self {
            let id = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("wgpu-sync-metadata-{}-{id}", std::process::id()));
            fs::create_dir(&path).unwrap();
            git(&path, &["init"]);
            // Keep the developer's global ignore rules out of the test.
            git(&path, &["config", "core.excludesFile", "no-such-file"]);
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn finds_publishable_default_member_directories() {
        let temporary = TempDirectory::new();
        for package in ["public", "custom", "private", "other"] {
            fs::create_dir(temporary.path().join(package)).unwrap();
        }
        let public_manifest = temporary.path().join("public/Cargo.toml");
        let custom_manifest = temporary.path().join("custom/Cargo.toml");
        let private_manifest = temporary.path().join("private/Cargo.toml");
        let other_manifest = temporary.path().join("other/Cargo.toml");
        let metadata = serde_json::json!({
            "workspace_default_members": ["public", "custom", "private"],
            "packages": [
                {
                    "id": "public",
                    "manifest_path": public_manifest,
                    "publish": null,
                },
                {
                    "id": "custom",
                    "manifest_path": custom_manifest,
                    "publish": ["custom-registry"],
                },
                {
                    "id": "private",
                    "manifest_path": private_manifest,
                    "publish": [],
                },
                {
                    "id": "other",
                    "manifest_path": other_manifest,
                    "publish": null,
                },
            ],
        });

        assert_eq!(
            publishable_default_member_directories(&metadata, temporary.path()).unwrap(),
            [PathBuf::from("custom"), PathBuf::from("public")]
        );
    }

    #[test]
    fn check_reports_missing_and_different_files() {
        let temporary = TempDirectory::new();
        fs::write(temporary.path().join("source"), "source").unwrap();

        let mut discrepancies = Vec::new();
        sync_file(
            temporary.path(),
            Path::new("source"),
            Path::new("target"),
            Mode::Check,
            &mut discrepancies,
        )
        .unwrap();
        assert_eq!(discrepancies.len(), 1);
        assert!(discrepancies[0].contains("is missing"));

        fs::write(temporary.path().join("target"), "target").unwrap();
        discrepancies.clear();
        sync_file(
            temporary.path(),
            Path::new("source"),
            Path::new("target"),
            Mode::Check,
            &mut discrepancies,
        )
        .unwrap();
        assert_eq!(discrepancies.len(), 1);
        assert!(discrepancies[0].contains("differs from"));
    }

    #[test]
    fn directory_check_reports_missing_extra_and_different_files() {
        let temporary = TempDirectory::new();
        fs::create_dir_all(temporary.path().join("source/nested")).unwrap();
        fs::create_dir_all(temporary.path().join("target/obsolete")).unwrap();
        fs::write(temporary.path().join("source/nested/missing"), "missing").unwrap();
        fs::write(temporary.path().join("source/different"), "source").unwrap();
        fs::write(temporary.path().join("target/different"), "target").unwrap();
        fs::write(temporary.path().join("target/obsolete/extra"), "extra").unwrap();

        let mut discrepancies = Vec::new();
        sync_directory(
            temporary.path(),
            Path::new("source"),
            Path::new("target"),
            Mode::Check,
            &mut discrepancies,
        )
        .unwrap();

        assert_eq!(discrepancies.len(), 3);
        assert!(discrepancies
            .iter()
            .any(|message| message.contains("missing")));
        assert!(discrepancies
            .iter()
            .any(|message| message.contains("extra")));
        assert!(discrepancies
            .iter()
            .any(|message| message.contains("differs from")));
    }

    #[test]
    fn directory_sync_keeps_git_ignored_files() {
        let temporary = TempDirectory::new();
        let root = temporary.path();
        fs::create_dir(root.join("source")).unwrap();
        fs::create_dir(root.join("generated")).unwrap();
        fs::write(
            root.join(".gitignore"),
            "generated/local
",
        )
        .unwrap();
        fs::write(root.join("source/shared"), "shared").unwrap();
        fs::write(root.join("generated/shared"), "shared").unwrap();
        fs::write(root.join("generated/local"), "local").unwrap();
        fs::write(root.join("generated/stale"), "stale").unwrap();

        let mut discrepancies = Vec::new();
        sync_directory(
            root,
            Path::new("source"),
            Path::new("generated"),
            Mode::Check,
            &mut discrepancies,
        )
        .unwrap();
        assert_eq!(discrepancies.len(), 1);
        assert!(discrepancies[0].contains("stale"));

        discrepancies.clear();
        sync_directory(
            root,
            Path::new("source"),
            Path::new("generated"),
            Mode::Write,
            &mut discrepancies,
        )
        .unwrap();
        assert!(discrepancies.is_empty());
        assert!(root.join("generated/local").exists());
        assert!(!root.join("generated/stale").exists());
    }

    #[test]
    fn directory_write_creates_updates_and_removes_files() {
        let temporary = TempDirectory::new();
        fs::create_dir_all(temporary.path().join("source/nested")).unwrap();
        fs::create_dir_all(temporary.path().join("target/obsolete")).unwrap();
        fs::write(temporary.path().join("source/nested/missing"), "missing").unwrap();
        fs::write(temporary.path().join("source/different"), "source").unwrap();
        fs::write(temporary.path().join("target/different"), "target").unwrap();
        fs::write(temporary.path().join("target/obsolete/extra"), "extra").unwrap();

        let mut discrepancies = Vec::new();
        sync_directory(
            temporary.path(),
            Path::new("source"),
            Path::new("target"),
            Mode::Write,
            &mut discrepancies,
        )
        .unwrap();

        assert!(discrepancies.is_empty());
        assert_eq!(
            fs::read_to_string(temporary.path().join("target/nested/missing")).unwrap(),
            "missing"
        );
        assert_eq!(
            fs::read_to_string(temporary.path().join("target/different")).unwrap(),
            "source"
        );
        assert!(!temporary.path().join("target/obsolete").exists());
    }
}
