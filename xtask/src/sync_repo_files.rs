use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context};
use pico_args::Arguments;
use xshell::Shell;

mod file_sync;

use file_sync::{sync_directory, sync_file, Mode};

const LICENSE_FILES: &[&str] = &["LICENSE.APACHE", "LICENSE.MIT"];

pub(crate) fn run_sync_repo_files(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    let mode = if args.contains("--check") {
        Mode::Check
    } else {
        Mode::Write
    };

    let unknown_args = args.finish();
    if !unknown_args.is_empty() {
        crate::bad_arguments!(
            "Unknown arguments to sync-repo-files subcommand: {:?}",
            unknown_args
        );
    }

    let root = shell.current_dir();
    let license_targets = license_targets(&shell, &root)?;
    let discrepancies = sync_repo_files(&root, &license_targets, mode)?;

    if discrepancies.is_empty() {
        match mode {
            Mode::Check => eprintln!("Repository files are synchronized."),
            Mode::Write => eprintln!("Repository files synchronized."),
        }
        return Ok(());
    }

    for discrepancy in discrepancies {
        eprintln!("error: {discrepancy}");
    }
    eprintln!("hint: edit the source-of-truth files, then run `cargo xtask sync-repo-files`");
    bail!("repository files are not synchronized")
}

/// Sync all repository files.
fn sync_repo_files(
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
                .join(format!("wgpu-sync-repo-files-{}-{id}", std::process::id()));
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
