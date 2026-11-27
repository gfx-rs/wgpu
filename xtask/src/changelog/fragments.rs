use std::{
    collections::{btree_map::Entry, BTreeMap, BTreeSet},
    fmt::Write as _,
    fs::{self, OpenOptions},
    io::{self, BufRead, Write as _},
};

use anyhow::{bail, ensure, Context, Result};
use pico_args::Arguments;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use xshell::Shell;

use super::find_odd_characters;

const DIRECTORY: &str = "changelog.d";
const CHANGELOG: &str = "CHANGELOG.md";
const UNRELEASED: &str = "\n## [Unreleased](changelog.d/)\n\n";
const REPOSITORY: &str = "gfx-rs/wgpu";

pub(super) const CATEGORIES: &str = "major, added, changes, bug-fixes, performance, documentation, dependency-updates, deno-webgpu, examples, testing";
pub(super) const AREAS: &str =
    "general, naga, validation, dx12, vulkan, metal, gles, webgpu, emscripten, hal";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Category {
    Major,
    Added,
    Changes,
    BugFixes,
    Performance,
    Documentation,
    DependencyUpdates,
    DenoWebgpu,
    Examples,
    Testing,
}

impl Category {
    fn heading(self) -> &'static str {
        match self {
            Self::Major => "Major changes",
            Self::Added => "Added/New Features",
            Self::Changes => "Changes",
            Self::BugFixes => "Bug Fixes",
            Self::Performance => "Performance",
            Self::Documentation => "Documentation",
            Self::DependencyUpdates => "Dependency Updates",
            Self::DenoWebgpu => "deno_webgpu",
            Self::Examples => "Examples",
            Self::Testing => "Testing/Internal",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Area {
    General,
    Naga,
    Validation,
    Dx12,
    Vulkan,
    Metal,
    Gles,
    Webgpu,
    Emscripten,
    Hal,
}

impl Area {
    fn heading(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::Naga => "naga",
            Self::Validation => "Validation",
            Self::Dx12 => "DX12",
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
            Self::Gles => "GLES",
            Self::Webgpu => "WebGPU",
            Self::Emscripten => "Emscripten",
            Self::Hal => "Hal",
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FrontMatter {
    category: Category,
    #[serde(skip_serializing_if = "Option::is_none")]
    area: Option<Area>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prs: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    authors: Option<Vec<String>>,
}

struct Fragment {
    name: String,
    metadata: FrontMatter,
    body: String,
    commit: Option<String>,
}

impl Fragment {
    fn read(shell: &Shell, name: String) -> Result<Self> {
        let path = shell.current_dir().join(DIRECTORY).join(&name);
        let read = || -> Result<_> {
            ensure!(
                path.symlink_metadata()?.file_type().is_file(),
                "expected a regular Markdown file"
            );
            let text = fs::read_to_string(&path)?;
            if let Some(character) = find_odd_characters(&text).first() {
                bail!("odd character {character}");
            }

            let text = text.replace("\r\n", "\n");
            let header_and_body = text
                .strip_prefix("+++\n")
                .context("expected TOML front matter starting with `+++`")?;
            let (header, body) = header_and_body
                .split_once("\n+++\n")
                .context("expected a closing `+++` on its own line")?;
            let metadata: FrontMatter = toml::from_str(header)?;
            let body = body.trim_matches('\n');
            ensure!(
                !body.trim().is_empty(),
                "write the entry after the closing `+++`"
            );
            if metadata.category == Category::Major {
                ensure!(
                    body.lines().next().is_some_and(|line| {
                        line.strip_prefix("#### ")
                            .is_some_and(|title| !title.trim().is_empty())
                    }),
                    "a major entry must start with a `####` heading"
                );
            } else {
                ensure!(metadata.area.is_some(), "missing `area`");
            }
            if let Some(prs) = &metadata.prs {
                ensure!(
                    !prs.is_empty() && !prs.contains(&0),
                    "`prs` must list positive PR numbers"
                );
            }
            if let Some(authors) = &metadata.authors {
                ensure!(
                    !authors.is_empty()
                        && authors.iter().all(|author| {
                            let name = author.strip_suffix("[bot]").unwrap_or(author);
                            !name.is_empty()
                                && name.bytes().all(|byte| {
                                    byte.is_ascii_alphanumeric() || b"-_".contains(&byte)
                                })
                        }),
                    "`authors` must list GitHub usernames without the `@` prefix"
                );
            }
            Ok(Self {
                name,
                metadata,
                body: body.to_owned(),
                commit: None,
            })
        };
        read().with_context(|| format!("invalid fragment {}", path.display()))
    }

    fn area(&self) -> Option<Area> {
        if self.metadata.category == Category::Major {
            None
        } else {
            self.metadata.area
        }
    }
}

pub(super) fn new(
    shell: &Shell,
    mut args: Arguments,
    mut input: impl BufRead,
    mut prompts: impl io::Write,
) -> Result<()> {
    let slug = args.opt_free_from_str()?;
    let category = args.opt_free_from_str()?;
    let area: Option<String> = args.opt_free_from_str()?;
    ensure!(
        args.finish().is_empty(),
        "expected new [<slug> [<category> [<area>]]]"
    );
    let directory = shell.current_dir().join(DIRECTORY);
    let name = parse_or_prompt(&mut input, &mut prompts, slug, "Slug", |slug| {
        let name = fragment_name(slug)?;
        ensure!(
            directory.join(&name).symlink_metadata().is_err(),
            "{DIRECTORY}/{name} already exists"
        );
        Ok(name)
    })?;
    let category: Category = parse_or_prompt(
        &mut input,
        &mut prompts,
        category,
        &format!("Category ({CATEGORIES})"),
        parse_choice,
    )?;
    let area = if category == Category::Major {
        ensure!(area.is_none(), "omit the area for a major change");
        None
    } else {
        Some(parse_or_prompt(
            &mut input,
            &mut prompts,
            area,
            &format!("Area ({AREAS}) [general]"),
            |value| parse_choice(if value.is_empty() { "general" } else { value }),
        )?)
    };
    let metadata = FrontMatter {
        category,
        area,
        prs: None,
        authors: None,
    };
    let header = toml::to_string(&metadata)?;
    let body = if category == Category::Major {
        "#### \n\n"
    } else {
        ""
    };
    fs::create_dir_all(&directory)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join(&name))
        .with_context(|| format!("could not create {DIRECTORY}/{name}"))?;
    write!(file, "+++\n{header}+++\n\n{body}")?;
    println!("{DIRECTORY}/{name}");
    Ok(())
}

fn parse_choice<T: DeserializeOwned>(value: &str) -> Result<T> {
    Ok(toml::Value::String(value.to_owned()).try_into()?)
}

fn parse_or_prompt<T>(
    input: &mut impl BufRead,
    prompts: &mut impl io::Write,
    value: Option<String>,
    prompt: &str,
    parse: impl Fn(&str) -> Result<T>,
) -> Result<T> {
    if let Some(value) = value {
        return parse(&value);
    }
    let mut line = String::new();
    loop {
        write!(prompts, "{prompt}: ")?;
        prompts.flush()?;
        line.clear();
        ensure!(
            input.read_line(&mut line)? != 0,
            "missing input for {prompt}"
        );
        match parse(line.trim()) {
            Ok(value) => return Ok(value),
            Err(error) => writeln!(prompts, "{error}")?,
        }
    }
}

fn fragment_name(slug: &str) -> Result<String> {
    ensure!(
        !slug.trim().is_empty() && !slug.contains(['/', '\\', ':']),
        "enter a slug without slashes or colons"
    );
    ensure!(!slug.eq_ignore_ascii_case("README"), "README is reserved");
    Ok(format!("{slug}.md"))
}

pub(super) fn preview(shell: &Shell, mut args: Arguments) -> Result<()> {
    let trunk: Option<String> = args.opt_value_from_str("--trunk")?;
    let slug: Option<String> = args.opt_free_from_str()?;
    ensure!(
        args.finish().is_empty(),
        "expected preview [<slug> | changelog.d/<slug>.md]"
    );
    let fragments = if let Some(slug) = slug {
        ensure!(trunk.is_none(), "--trunk is only used for a full preview");
        let slug = slug
            .strip_prefix("changelog.d/")
            .and_then(|path| path.strip_suffix(".md"))
            .unwrap_or(&slug);
        vec![Fragment::read(shell, fragment_name(slug)?)?]
    } else {
        ordered_fragments(shell, trunk.as_deref().unwrap_or("trunk"))?
    };

    let body = render(&fragments, |fragment| {
        let metadata = &fragment.metadata;
        Ok(credit(metadata.prs.as_deref(), metadata.authors.as_deref()))
    })?;
    print!("## Unreleased\n\n{body}");
    Ok(())
}

pub(super) fn check(shell: &Shell, mut args: Arguments) -> Result<()> {
    let allow_released_changes = args.contains("--allow-released-changes");
    let base: Option<String> = args.opt_free_from_str()?;
    ensure!(
        args.finish().is_empty(),
        "expected check [<base-ref>] [--allow-released-changes]"
    );
    read_fragments(shell)?;
    if let Some(character) = find_odd_characters(&shell.read_file(CHANGELOG)?).first() {
        bail!("odd character in {CHANGELOG} {character}");
    }
    if !allow_released_changes {
        let base = shell
            .cmd("git")
            .args([
                "merge-base",
                "--",
                base.as_deref().unwrap_or("trunk"),
                "HEAD",
            ])
            .read()
            .context("could not find the changelog merge base")?;
        let diff = shell
            .cmd("git")
            .args(["diff", "--name-only", &base, "--", CHANGELOG])
            .read()?;
        ensure!(
            diff.is_empty(),
            "changes to {CHANGELOG} require the 'changelog: released entry changed' label"
        );
    }
    Ok(())
}

fn read_fragments(shell: &Shell) -> Result<Vec<Fragment>> {
    let mut fragments = Vec::new();
    let directory = match fs::read_dir(shell.current_dir().join(DIRECTORY)) {
        Ok(directory) => directory,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(fragments),
        Err(error) => return Err(error.into()),
    };
    for path in directory {
        let path = path?.path();
        if path.extension().is_some_and(|ext| ext == "md") {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .context("fragment filename must be UTF-8")?;
            if !name.eq_ignore_ascii_case("README.md") {
                fragments.push(Fragment::read(shell, name.to_owned())?);
            }
        }
    }
    Ok(fragments)
}

fn ordered_fragments(shell: &Shell, trunk: &str) -> Result<Vec<Fragment>> {
    let mut fragments = read_fragments(shell)?;
    if fragments.is_empty() {
        return Ok(fragments);
    }

    let history = shell
        .cmd("git")
        .args([
            "log",
            "--first-parent",
            "--diff-merges=first-parent",
            "--reverse",
            "--no-renames",
            "--diff-filter=A",
            "--format=%x00%H",
            "--name-only",
            "-z",
            "--end-of-options",
            trunk,
            "--",
            "changelog.d/",
        ])
        .read()
        .with_context(|| {
            format!("could not read fragment history from {trunk}; use --trunk <ref>")
        })?;

    let names: BTreeSet<_> = fragments
        .iter()
        .map(|fragment| fragment.name.as_str())
        .collect();
    let mut additions = BTreeMap::new();
    for (index, record) in history.trim_matches('\0').split("\0\0").enumerate() {
        let Some((commit, paths)) = record.split_once("\0\n") else {
            continue;
        };
        for path in paths.split('\0') {
            if let Some(name) = path
                .strip_prefix("changelog.d/")
                .filter(|name| names.contains(name))
            {
                ensure!(
                    additions.insert(name, (index, commit)).is_none(),
                    "fragment {DIRECTORY}/{name} was added to trunk more than once"
                );
            }
        }
    }
    fragments.sort_by(|a, b| {
        let order = |fragment: &Fragment| {
            additions
                .get(fragment.name.as_str())
                .map_or(usize::MAX, |(index, _)| *index)
        };
        order(a).cmp(&order(b)).then_with(|| a.name.cmp(&b.name))
    });
    for fragment in &mut fragments {
        fragment.commit = additions
            .get(fragment.name.as_str())
            .map(|(_, commit)| (*commit).to_owned());
    }
    Ok(fragments)
}

fn credit(prs: Option<&[u64]>, authors: Option<&[String]>) -> String {
    let prs = prs.map_or_else(
        || "*PR link*".to_owned(),
        |prs| {
            prs.iter()
                .map(|pr| format!("[#{pr}](https://github.com/{REPOSITORY}/pull/{pr})"))
                .collect::<Vec<_>>()
                .join(", ")
        },
    );
    let authors = authors.map_or_else(
        || "*PR author*".to_owned(),
        |authors| {
            authors
                .iter()
                .map(|author| format!("@{author}"))
                .collect::<Vec<_>>()
                .join(", ")
        },
    );
    format!("By {authors} in {prs}.")
}

fn render(
    fragments: &[Fragment],
    mut credit_for: impl FnMut(&Fragment) -> Result<String>,
) -> Result<String> {
    let mut fragments: Vec<_> = fragments.iter().collect();
    fragments.sort_by_key(|fragment| (fragment.metadata.category, fragment.area()));

    let mut output = String::new();
    let mut category = None;
    let mut area = None;
    for fragment in fragments {
        let next_category = fragment.metadata.category;
        if category != Some(next_category) {
            writeln!(output, "### {}\n", next_category.heading()).unwrap();
            category = Some(next_category);
            area = None;
        }
        if let Some(next_area) = fragment.area().filter(|next| area != Some(*next)) {
            writeln!(output, "#### {}\n", next_area.heading()).unwrap();
            area = Some(next_area);
        }
        let credit = credit_for(fragment)
            .with_context(|| format!("could not attribute {DIRECTORY}/{}", fragment.name))?;
        if next_category == Category::Major {
            writeln!(output, "{}\n\n{credit}\n", fragment.body).unwrap();
        } else if let Some((first, rest)) = fragment.body.split_once('\n') {
            writeln!(output, "- {first}").unwrap();
            for line in rest.lines() {
                if line.is_empty() {
                    output.push('\n');
                } else {
                    writeln!(output, "  {line}").unwrap();
                }
            }
            writeln!(output, "\n  {credit}\n").unwrap();
        } else {
            writeln!(output, "- {} {credit}\n", fragment.body).unwrap();
        }
    }
    output.pop();
    Ok(output)
}

#[derive(Deserialize)]
struct Pull {
    number: u64,
    user: User,
}

#[derive(Deserialize)]
struct User {
    login: String,
}

fn lookup_pull(shell: &Shell, commit: &str) -> Result<Pull> {
    let output = shell
        .cmd("gh")
        .args(["api", "--hostname", "github.com", "--paginate"])
        .arg(format!("repos/{REPOSITORY}/commits/{commit}/pulls"))
        .args([
            "--jq",
            ".[] | select(.merged_at != null and .base.ref == \"trunk\" and .base.repo.full_name == \"gfx-rs/wgpu\") | {number, user: {login: .user.login}}",
        ])
        .read()
        .with_context(|| format!("GitHub PR lookup failed for trunk commit {commit}"))?;
    let mut pulls = serde_json::Deserializer::from_str(&output).into_iter::<Pull>();
    let pull = pulls.next().with_context(|| {
        format!("no merged trunk PR found for {commit}; set `prs` and `authors`")
    })??;
    ensure!(
        pulls.next().is_none(),
        "more than one merged trunk PR found for {commit}; set `prs` and `authors`"
    );
    Ok(pull)
}

fn resolve_credit(
    shell: &Shell,
    fragment: &Fragment,
    pulls: &mut BTreeMap<String, Pull>,
) -> Result<String> {
    let metadata = &fragment.metadata;
    let Some(prs) = metadata.prs.as_deref() else {
        let commit = fragment.commit.as_deref().context("missing trunk commit")?;
        let pull = match pulls.entry(commit.to_owned()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(lookup_pull(shell, commit)?),
        };
        let authors = metadata
            .authors
            .as_deref()
            .unwrap_or_else(|| std::slice::from_ref(&pull.user.login));
        return Ok(credit(Some(&[pull.number]), Some(authors)));
    };

    if let Some(authors) = &metadata.authors {
        return Ok(credit(Some(prs), Some(authors)));
    }

    let mut authors = Vec::new();
    for pr in prs {
        let name = shell
            .cmd("gh")
            .args(["api", "--hostname", "github.com"])
            .arg(format!("repos/{REPOSITORY}/pulls/{pr}"))
            .args(["--jq", ".user.login"])
            .read()
            .with_context(|| format!("GitHub author lookup failed for PR #{pr}"))?;
        ensure!(
            !name.is_empty() && name != "null",
            "no author found for PR #{pr}"
        );
        if !authors.contains(&name) {
            authors.push(name);
        }
    }
    Ok(credit(Some(prs), Some(&authors)))
}

fn release_offset(changelog: &str, version: &str) -> Result<(usize, &'static str)> {
    let (offset, newline) = ["\n", "\r\n"]
        .into_iter()
        .find_map(|newline| {
            let marker = UNRELEASED.replace('\n', newline);
            changelog
                .find(&marker)
                .map(|start| (start + marker.len(), newline))
        })
        .context("the fragment workflow is not active in CHANGELOG.md")?;
    ensure!(
        changelog[offset..].starts_with("## "),
        "move the old Unreleased entries into a release section before using fragments"
    );
    let mut fence = None;
    for line in changelog[offset..].lines() {
        let delimiter = line
            .chars()
            .next()
            .filter(|marker| matches!(marker, '`' | '~'))
            .map(|marker| (marker, line.len() - line.trim_start_matches(marker).len()))
            .filter(|&(_, width)| width >= 3);
        if let Some((marker, width)) = fence {
            if delimiter.is_some_and(|(end, end_width)| {
                end == marker && end_width >= width && line[end_width..].trim().is_empty()
            }) {
                fence = None;
            }
        } else if delimiter.is_some() {
            fence = delimiter;
        } else {
            let heading = line
                .strip_prefix("## ")
                .and_then(|heading| heading.split_ascii_whitespace().next());
            ensure!(
                heading != Some(version),
                "{version} already exists in CHANGELOG.md"
            );
        }
    }
    Ok((offset, newline))
}

fn clean_head(shell: &Shell) -> Result<String> {
    let status = shell
        .cmd("git")
        .args([
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            CHANGELOG,
            DIRECTORY,
        ])
        .read()?;
    ensure!(
        status.is_empty(),
        "commit changes to {CHANGELOG} and {DIRECTORY}/ before running release"
    );
    Ok(shell.cmd("git").args(["rev-parse", "HEAD"]).read()?)
}

pub(super) fn release(shell: &Shell, mut args: Arguments) -> Result<()> {
    let trunk: Option<String> = args.opt_value_from_str("--trunk")?;
    let version: String = args.free_from_str()?;
    ensure!(args.finish().is_empty(), "expected release <vX.Y.Z>");
    let version_pattern =
        regex_lite::Regex::new(r"\Av(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\z").unwrap();
    ensure!(
        version_pattern.is_match(&version),
        "expected a release version in the form vX.Y.Z"
    );

    let mut changelog = shell.read_file(CHANGELOG)?;
    let (offset, newline) = release_offset(&changelog, &version)?;
    let head = clean_head(shell)?;
    let fragments = ordered_fragments(shell, trunk.as_deref().unwrap_or("trunk"))?;
    for fragment in &fragments {
        ensure!(
            fragment.commit.is_some(),
            "{DIRECTORY}/{} was not found on trunk; update your trunk ref",
            fragment.name
        );
    }
    let mut pulls = BTreeMap::new();
    let body = render(&fragments, |fragment| {
        resolve_credit(shell, fragment, &mut pulls)
    })?;
    ensure!(
        clean_head(shell)? == head && shell.read_file(CHANGELOG)? == changelog,
        "release inputs changed during release"
    );
    let date = jiff::Timestamp::now()
        .to_zoned(jiff::tz::TimeZone::UTC)
        .date();
    let section = if body.is_empty() {
        format!("## {version} ({date})\n\n")
    } else {
        format!("## {version} ({date})\n\n{body}\n")
    };
    changelog.insert_str(offset, &section.replace('\n', newline));

    let root = shell.current_dir();
    let path = root.join(CHANGELOG);
    let mut output = tempfile::NamedTempFile::new_in(&root)?;
    output
        .as_file()
        .set_permissions(path.metadata()?.permissions())?;
    output.write_all(changelog.as_bytes())?;
    output
        .persist(path)
        .context("could not replace CHANGELOG.md")?;
    for fragment in &fragments {
        let path = root.join(DIRECTORY).join(&fragment.name);
        fs::remove_file(&path).with_context(|| {
            format!(
            "{CHANGELOG} was updated; remove the released fragments that remain, starting with {}",
            path.display()
        )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Arguments {
        Arguments::from_vec(values.iter().map(Into::into).collect())
    }

    #[test]
    fn preview_an_untracked_fragment() -> Result<()> {
        let shell = Shell::new()?;
        let directory = shell.create_temp_dir()?;
        shell.change_dir(directory.path());

        new(
            &shell,
            args(&["example", "bug-fixes", "general"]),
            io::empty(),
            io::sink(),
        )?;
        assert!(new(&shell, args(&["example", "major"]), io::empty(), io::sink()).is_err());
        let header = shell.read_file("changelog.d/example.md")?;
        assert_eq!(
            header,
            "+++\ncategory = \"bug-fixes\"\narea = \"general\"\n+++\n\n"
        );

        shell.write_file(
            "changelog.d/example.md",
            format!("{header}Fix the example.\n\n```rust\nexample();\n```\n"),
        )?;
        preview(&shell, args(&["changelog.d/example.md"]))?;

        let fragment = Fragment::read(&shell, "example.md".into())?;
        assert_eq!(
            render(&[fragment], |_| Ok(credit(None, None)))?,
            "### Bug Fixes\n\n#### General\n\n- Fix the example.\n\n  ```rust\n  example();\n  ```\n\n  By *PR author* in *PR link*.\n"
        );
        Ok(())
    }

    #[test]
    fn new_retries_invalid_prompt_answers() -> Result<()> {
        let shell = Shell::new()?;
        let directory = shell.create_temp_dir()?;
        shell.change_dir(directory.path());
        shell.create_dir(DIRECTORY)?;
        shell.write_file("changelog.d/taken.md", "Existing fragment.")?;

        let mut prompts = Vec::new();
        new(
            &shell,
            args(&[]),
            &b"../invalid\ntaken\nexample\nunknown\nadded\nunknown\n\n"[..],
            &mut prompts,
        )?;
        let prompts = String::from_utf8(prompts)?;
        assert_eq!(prompts.matches("Slug:").count(), 3);
        assert_eq!(prompts.matches("Category (").count(), 2);
        assert_eq!(prompts.matches("Area (").count(), 2);
        assert_eq!(
            shell.read_file("changelog.d/example.md")?,
            "+++\ncategory = \"added\"\narea = \"general\"\n+++\n\n"
        );
        assert_eq!(
            shell.read_file("changelog.d/taken.md")?,
            "Existing fragment."
        );
        Ok(())
    }

    #[test]
    fn new_prompts_only_for_missing_arguments() -> Result<()> {
        let shell = Shell::new()?;
        let directory = shell.create_temp_dir()?;
        shell.change_dir(directory.path());

        let mut prompts = Vec::new();
        new(&shell, args(&["breaking"]), &b"major\n"[..], &mut prompts)?;
        assert!(String::from_utf8(prompts)?.starts_with("Category ("));
        assert_eq!(
            shell.read_file("changelog.d/breaking.md")?,
            "+++\ncategory = \"major\"\n+++\n\n#### \n\n"
        );

        let error = new(
            &shell,
            args(&["unfinished", "added"]),
            io::empty(),
            io::sink(),
        )
        .unwrap_err();
        assert!(error.to_string().starts_with("missing input for Area"));
        assert!(!shell
            .current_dir()
            .join("changelog.d/unfinished.md")
            .exists());
        Ok(())
    }

    #[test]
    fn rejects_invalid_fragments() -> Result<()> {
        let shell = Shell::new()?;
        let directory = shell.create_temp_dir()?;
        shell.change_dir(directory.path());
        shell.create_dir(DIRECTORY)?;

        for (text, error) in [
            (
                "+++\ncategory = \"changed\"\n+++\n\nEntry.",
                "unknown variant",
            ),
            (
                "+++\ncategory = \"changes\"\n+++\n\nEntry.",
                "missing `area`",
            ),
            (
                "+++\ncategory = \"changes\"\narea = \"other\"\n+++\n\nEntry.",
                "unknown variant",
            ),
            (
                "+++\ncategory = \"major\"\nextra = 2\n+++\n\n#### Entry",
                "unknown field",
            ),
            (
                "+++\ncategory = \"major\"\n+++\n\nEntry.",
                "must start with a",
            ),
            (
                "+++\ncategory = \"major\"\n+++\n\n#### Bad\u{200b}character",
                "odd character",
            ),
            ("+++\ncategory = \"major\"\n", "expected a closing"),
        ] {
            shell.write_file("changelog.d/example.md", text)?;
            let result = check(&shell, args(&["--allow-released-changes"])).unwrap_err();
            assert!(format!("{result:#}").contains(error), "{result:#}");
        }
        Ok(())
    }

    #[test]
    fn releases_backported_fragments_in_trunk_order() -> Result<()> {
        let shell = Shell::new()?;
        let directory = shell.create_temp_dir()?;
        shell.change_dir(directory.path());
        shell.write_file("gitconfig", "")?;
        shell.set_var("GIT_CONFIG_GLOBAL", directory.path().join("gitconfig"));
        shell.set_var("GIT_CONFIG_NOSYSTEM", "1");
        shell.set_var("GIT_AUTHOR_NAME", "Test");
        shell.set_var("GIT_AUTHOR_EMAIL", "test@example.com");
        shell.set_var("GIT_COMMITTER_NAME", "Test");
        shell.set_var("GIT_COMMITTER_EMAIL", "test@example.com");
        let git = |args: &[&str]| shell.cmd("git").args(args).quiet().read();

        git(&["init", "--initial-branch=trunk"])?;
        let history = "## v1.0.0 (2026-01-01)\n\nPublished notes.\n\n```markdown\n## v1.0.1 (2026-01-02)\n```\n";
        let changelog = format!("# Change Log\n{UNRELEASED}{history}");
        shell.write_file(CHANGELOG, &changelog)?;
        shell.create_dir(DIRECTORY)?;
        shell.write_file("changelog.d/README.md", "Instructions.")?;
        git(&["add", CHANGELOG, DIRECTORY])?;
        git(&["commit", "-m", "Cut over"])?;
        git(&["branch", "v1"])?;

        let header = "+++\ncategory = \"added\"\narea = \"metal\"\nprs = [1, 2]\nauthors = [\"first-author\", \"second-author\"]\n+++\n\n";
        shell.write_file("changelog.d/z.md", format!("{header}First entry."))?;
        git(&["add", DIRECTORY])?;
        git(&["commit", "-m", "Add first entry"])?;
        let first = git(&["rev-parse", "HEAD"])?;

        git(&["switch", "-c", "topic"])?;
        shell.write_file("changelog.d/a.md", format!("{header}Second entry."))?;
        git(&["add", DIRECTORY])?;
        git(&["commit", "-m", "Add second entry"])?;
        git(&["switch", "trunk"])?;
        git(&["merge", "--no-ff", "--no-edit", "topic"])?;
        let second = git(&["rev-parse", "HEAD"])?;

        git(&["switch", "v1"])?;
        git(&["cherry-pick", "-m", "1", &second])?;
        git(&["cherry-pick", &first])?;
        let fragments = ordered_fragments(&shell, "trunk")?;
        assert_eq!(fragments[0].commit.as_deref(), Some(first.as_str()));
        assert_eq!(fragments[1].commit.as_deref(), Some(second.as_str()));

        assert!(release(&shell, args(&["v1.0.0"])).is_err());
        assert_eq!(shell.read_file(CHANGELOG)?, changelog);
        release(&shell, args(&["v1.0.1"]))?;
        let output = shell.read_file(CHANGELOG)?;
        assert!(output.starts_with(&format!("# Change Log\n{UNRELEASED}## v1.0.1 (")));
        assert!(output.ends_with(history));
        assert!(output.find("First entry.").unwrap() < output.find("Second entry.").unwrap());
        assert!(output.contains("By @first-author, @second-author in [#1](https://github.com/gfx-rs/wgpu/pull/1), [#2](https://github.com/gfx-rs/wgpu/pull/2)."));
        assert_eq!(shell.read_dir(DIRECTORY)?.len(), 1);
        assert!(check(&shell, args(&["trunk"])).is_err());
        check(&shell, args(&["trunk", "--allow-released-changes"]))?;

        git(&["add", CHANGELOG, DIRECTORY])?;
        git(&["commit", "-m", "Assemble release notes"])?;
        let release_commit = git(&["rev-parse", "HEAD"])?;
        git(&["switch", "trunk"])?;
        git(&["cherry-pick", &release_commit])?;
        assert_eq!(shell.read_file(CHANGELOG)?, output);
        assert_eq!(shell.read_dir(DIRECTORY)?.len(), 1);
        Ok(())
    }

    #[test]
    fn release_requires_cutover() {
        let changelog = "# Change Log\n\n## Unreleased\n\n### Major changes\n";
        assert!(release_offset(changelog, "v31.0.0").is_err());
        let changelog = format!("# Change Log\n{UNRELEASED}### Major changes\n");
        assert!(release_offset(&changelog, "v31.0.0").is_err());
    }
}
