use std::{cmp, sync::OnceLock};

use pico_args::Arguments;
use regex_lite::Regex;
use xshell::Shell;

pub(crate) fn check_changelog(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    const CHANGELOG_PATH_RELATIVE: &str = "./CHANGELOG.md";
    let allow_released_changes = args.contains("--allow-released-changes");

    let from_branch = args
        .free_from_str()
        .ok()
        .unwrap_or_else(|| "trunk".to_owned());
    let to_commit: Option<String> = args.free_from_str().ok();

    let from_commit = shell
        .cmd("git")
        .args(["merge-base", "--fork-point", &from_branch])
        .args(to_commit.as_ref())
        .read()
        .unwrap();

    let diff = shell
        .cmd("git")
        .args(["diff", &from_commit])
        // NOTE: If `to_commit` is not specified, we compare against the working tree, instead of
        // between commits.
        .args(to_commit.as_ref())
        .args(["--", CHANGELOG_PATH_RELATIVE])
        .read()
        .unwrap();

    // NOTE: If `to_commit` is not specified, we fetch from the file system, instead of `git show`.
    let changelog_contents = if let Some(to_commit) = to_commit.as_ref() {
        shell
            .cmd("git")
            .arg("show")
            .arg(format!("{to_commit}:{CHANGELOG_PATH_RELATIVE}"))
            .arg("--")
            .read()
            .unwrap()
    } else {
        shell.read_file(CHANGELOG_PATH_RELATIVE).unwrap()
    };

    let mut failed = false;

    let hunks_in_a_released_section = hunks_in_a_released_section(&changelog_contents, &diff);
    log::info!(
        "# of hunks in a released section of `{CHANGELOG_PATH_RELATIVE}`: {}",
        hunks_in_a_released_section.len()
    );
    if !hunks_in_a_released_section.is_empty() {
        failed = true;

        #[expect(clippy::uninlined_format_args)]
        {
            eprintln!(
                "Found hunk(s) in released sections of `{}`, which we don't want:\n",
                CHANGELOG_PATH_RELATIVE,
            );
        }

        for hunk in &hunks_in_a_released_section {
            eprintln!("{hunk}");
        }

        eprintln!();
        eprintln!(
            "hint: a release likely happened after this change was first added to `## Unreleased`. \
             Move the affected entries back up under `## Unreleased` in `{CHANGELOG_PATH_RELATIVE}` \
             and try again."
        );
    }

    if failed {
        #[expect(clippy::uninlined_format_args)]
        let msg = format!(
            "one or more checks against `{}` failed; see above for details",
            CHANGELOG_PATH_RELATIVE,
        );
        if allow_released_changes {
            log::warn!("{msg}");
            Ok(())
        } else {
            Err(anyhow::Error::msg(msg))
        }
    } else {
        Ok(())
    }
}

/// Given some `changelog_contents` (in Markdown) containing the full end state of the provided
/// `diff` (in [unified diff format]), return all hunks that are (1) below a `## Unreleased` section
/// _and_ (2) above all other second-level (i.e., `## …`) headings.
///
/// [unified diff format]: https://www.gnu.org/software/diffutils/manual/html_node/Detailed-Unified.html
///
/// This function makes a few assumptions that are necessary to uphold for correctness, in the
/// interest of a simple implementation:
///
/// - The provided `diff`'s end state _must_ correspond to `changelog_contents`.
/// - The provided `diff` must _only_ contain a single entry for the file containing
///   `changelog_contents`. using hunk information to compare against `changelog_contents`.
///
/// Failing to uphold these assumptons is not unsafe, but will yield incorrect results.
fn hunks_in_a_released_section<'a>(changelog_contents: &str, diff: &'a str) -> Vec<&'a str> {
    let mut changelog_lines = changelog_contents.lines();

    let changelog_unreleased_line_num =
        changelog_lines.position(|l| l == "## Unreleased").unwrap() as u64;

    let changelog_first_release_section_line_num = changelog_unreleased_line_num
        + 1
        + changelog_lines.position(|l| l.starts_with("## ")).unwrap() as u64;

    let hunks = {
        let first_hunk_match = diff.match_indices("\n@@").next();
        let Some((first_hunk_idx, _)) = first_hunk_match else {
            log::info!("no diff found");
            return vec![];
        };
        SplitPrefixInclusive::new("\n@@", &diff[first_hunk_idx..]).map(|s| &s['\n'.len_utf8()..])
    };
    let hunks_in_a_released_section = hunks
        .filter(|hunk| {
            let (hunk_header, hunk_contents) = hunk.split_once('\n').unwrap();

            // Reference: This is of the format `@@ -86,6 +88,10 @@ …`.
            static HUNK_HEADER_RE: OnceLock<Regex> = OnceLock::new();
            let hunk_header_re =
                HUNK_HEADER_RE.get_or_init(|| Regex::new(r"@@ -\d+,\d+ \+(\d+),\d+ @@.*").unwrap());
            let captures = hunk_header_re.captures_at(hunk_header, 0).unwrap();
            let post_change_hunk_start_offset =
                captures.get(1).unwrap().as_str().parse::<u64>().unwrap();

            let lines_until_first_change = hunk_contents
                .lines()
                .take_while(|l| l.starts_with(' '))
                .count()
                // NOTE: First line is the one-based index in the header, assume there's at least
                // one line and ignore it.
                .checked_sub(1)
                .unwrap() as u64;

            let first_hunk_change_start_offset =
                post_change_hunk_start_offset + lines_until_first_change;

            match first_hunk_change_start_offset.cmp(&changelog_first_release_section_line_num) {
                cmp::Ordering::Greater => true,
                cmp::Ordering::Equal => hunk_contents.lines().any(|l| l.starts_with('+')),
                _ => false,
            }
        })
        .collect::<Vec<_>>();

    hunks_in_a_released_section
}

struct SplitPrefixInclusive<'haystack, 'prefix> {
    haystack: Option<&'haystack str>,
    prefix: &'prefix str,
    current_pos: usize,
}

impl<'haystack, 'prefix> SplitPrefixInclusive<'haystack, 'prefix> {
    pub fn new(prefix: &'prefix str, haystack: &'haystack str) -> Self {
        assert!(haystack.starts_with(prefix));
        Self {
            haystack: Some(haystack),
            prefix,
            current_pos: 0,
        }
    }
}

impl<'haystack> Iterator for SplitPrefixInclusive<'haystack, '_> {
    type Item = &'haystack str;

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = &self.haystack?[self.current_pos..];

        let prefix_len = self.prefix.len();

        // NOTE: We've guaranteed that the prefix is always at the start of what remains. So, skip
        // the first match manually, and adjust match indices by `prefix_len` later.
        let to_search = &remaining[prefix_len..];

        match to_search.match_indices(self.prefix).next() {
            None => {
                self.haystack = None;
                Some(remaining)
            }
            Some((idx, _match)) => {
                let length = idx + prefix_len;
                self.current_pos += length;
                Some(&remaining[..length])
            }
        }
    }
}

#[cfg(test)]
mod test_split_prefix_inclusive {
    #[collapse_debuginfo(yes)]
    macro_rules! assert_chunks {
        ($prefix: expr, $haystack: expr, $expected: expr $(,)?) => {
            assert_eq!(
                super::SplitPrefixInclusive::new($prefix, $haystack).collect::<Vec<_>>(),
                $expected.into_iter().collect::<Vec<_>>(),
            );
        };
    }

    #[test]
    fn it_works() {
        assert_chunks! {
            "\n@@",
            "
@@ -1,4 +1,5 @@
 <!--
+    This change should be accepted.
     Pad out some changes here so we force multiple hunks in a diff.
     Pad out some changes here so we force multiple hunks in a diff.
     Pad out some changes here so we force multiple hunks in a diff.
@@ -17,6 +18,7 @@
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
+- This change should be accepted.
\u{0020}
 ## Recently released
\u{0020}
@@ -26,6 +28,7 @@
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
+- This change was added after release, reject me!
\u{0020}
 ## An older release
\u{0020}
",
        [
                "
@@ -1,4 +1,5 @@
 <!--
+    This change should be accepted.
     Pad out some changes here so we force multiple hunks in a diff.
     Pad out some changes here so we force multiple hunks in a diff.
     Pad out some changes here so we force multiple hunks in a diff.",
                "
@@ -17,6 +18,7 @@
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
+- This change should be accepted.
\u{0020}
 ## Recently released
\u{0020}",
            "
@@ -26,6 +28,7 @@
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
+- This change was added after release, reject me!
\u{0020}
 ## An older release
\u{0020}
",
        ]
                }
    }
}

#[cfg(test)]
mod test_hunks_in_a_released_section {
    #[collapse_debuginfo(yes)]
    macro_rules! assert_released_section_changes {
        ($changelog_contents: expr, $diff: expr, $expected: expr $(,)?) => {
            assert_eq!(
                super::hunks_in_a_released_section($changelog_contents, $diff),
                $expected
                    .map(|h: &str| h.to_owned())
                    .into_iter()
                    .collect::<Vec<_>>(),
            );
        };
    }

    #[test]
    fn change_in_a_release_section_rejects() {
        assert_released_section_changes! {
        "\
<!-- Some explanatory comment -->

## Unreleased

## Recently released

- This change actually went into the release.
- This change was added after release, reject me!

## An older release

- Yada yada.
",
        "\
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -5,6 +5,7 @@
 ## Recently released
\u{0020}
 - This change actually went into the release.
+- This change was added after release, reject me!
\u{0020}
 ## An older release
",
            [
        "\
@@ -5,6 +5,7 @@
 ## Recently released
\u{0020}
 - This change actually went into the release.
+- This change was added after release, reject me!
\u{0020}
 ## An older release
",
            ],
        }
    }

    #[test]
    fn change_in_unreleased_not_rejected() {
        assert_released_section_changes! {
        "\
<!-- Some explanatory comment -->

## Unreleased

- This change was due to the valiant efforts of a contributor.

## Recently released

- This change actually went into the release.

## An older release

- Yada yada.
",
        "\
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -2,6 +2,8 @@
\u{0020}
 ## Unreleased
\u{0020}
+- This change was due to the valiant efforts of a contributor.
+
 ## Recently released
\u{0020}
 - This change actually went into the release.
",
            [],
        }
    }

    #[test]
    fn change_above_unreleased_not_rejected() {
        assert_released_section_changes! {
        "\
<!-- Some explanatory comment -->
<!-- Turns out, we needed to explain things more. -->

## Unreleased

- This change was due to the valiant efforts of a contributor.

## Recently released

- This change actually went into the release.

## An older release

- Yada yada.
",
        "\
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -1,4 +1,5 @@
 <!-- Some explanatory comment -->
+<!-- Turns out, we needed to explain things more. -->
\u{0020}
 ## Unreleased
\u{0020}
",
            [],
        }
    }

    #[test]
    fn all_reject_and_not_reject_cases_at_once() {
        assert_released_section_changes! {
            "\
<!--
    This change should be accepted.
    Pad out some changes here so we force multiple hunks in a diff.
    Pad out some changes here so we force multiple hunks in a diff.
    Pad out some changes here so we force multiple hunks in a diff.
    Pad out some changes here so we force multiple hunks in a diff.
    Pad out some changes here so we force multiple hunks in a diff.
    Pad out some changes here so we force multiple hunks in a diff.
    Pad out some changes here so we force multiple hunks in a diff.
-->

## Unreleased

- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- This change should be accepted.

## Recently released

- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- Pad out some changes here so we force multiple hunks in a diff.
- This change was added after release, reject me!

## An older release

- Yada yada.
",
            "\
--- ../CHANGELOG.md
+++ ../CHANGELOG.md
@@ -1,4 +1,5 @@
 <!--
+    This change should be accepted.
     Pad out some changes here so we force multiple hunks in a diff.
     Pad out some changes here so we force multiple hunks in a diff.
     Pad out some changes here so we force multiple hunks in a diff.
@@ -17,6 +18,7 @@
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
+- This change should be accepted.
\u{0020}
 ## Recently released
\u{0020}
@@ -26,6 +28,7 @@
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
+- This change was added after release, reject me!
\u{0020}
 ## An older release
\u{0020}
",
            [
                "\
@@ -26,6 +28,7 @@
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
 - Pad out some changes here so we force multiple hunks in a diff.
+- This change was added after release, reject me!
\u{0020}
 ## An older release
\u{0020}
",
            ],
        }
    }

    #[test]
    fn rejection_ranges() {
        assert_released_section_changes! {
            "\
<!-- Some explanatory comment -->

## Unreleased

- This change was due to the valiant efforts of a contributor.

## Recently released

- This change actually went into the release.
- terribly
  long
  entry
  please
  save
  me

## An older release

- Yada yada.
",
            "\
diff --git a/../CHANGELOG-old.md b/../CHANGELOG-new.md
index a6bf3614a..c766d7225 100644
--- a/../CHANGELOG-old.md
+++ b/../CHANGELOG-new.md
@@ -7,6 +7,12 @@
 ## Recently released
\u{0020}
 - This change actually went into the release.
+- terribly
+  long
+  entry
+  please
+  save
+  me
\u{0020}
 ## An older release
\u{0020}
",
            [
                "\
@@ -7,6 +7,12 @@
 ## Recently released
\u{0020}
 - This change actually went into the release.
+- terribly
+  long
+  entry
+  please
+  save
+  me
\u{0020}
 ## An older release
\u{0020}
",
            ],
        }
        assert_released_section_changes! {
            "\
<!-- Some explanatory comment -->

## Unreleased

- This change was due to the valiant efforts of a contributor.

## Recently released

- This change actually went into the release.

## An older release

- Accidentally added before.
- Yada yada.
- Accidentally added after.
",
            "\
diff --git a/../CHANGELOG-old.md b/../CHANGELOG-new.md
index a6bf3614a..5c2dcdc4e 100644
--- a/../CHANGELOG-old.md
+++ b/../CHANGELOG-new.md
@@ -10,4 +10,6 @@
\u{0020}
 ## An older release
\u{0020}
+- Accidentally added before.
 - Yada yada.
+- Accidentally added after.
",
            [
                "\
@@ -10,4 +10,6 @@
\u{0020}
 ## An older release
\u{0020}
+- Accidentally added before.
 - Yada yada.
+- Accidentally added after.
",
            ]
        }
    }

    #[test]
    fn deletion_up_to_released() {
        assert_released_section_changes! {
        "\
<!-- Some explanatory comment -->

## Unreleased

WHADDUP FOLKS

## Released

- Blah blah blah.
",
        "\
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -4,8 +4,6 @@
\u{0020}
 WHADDUP FOLKS
\u{0020}
-HERE'S SOME STUFF THAT'S GONNA GET DELETED
-
 ## Released
\u{0020}
 - Blah blah blah.
",
            [],
        }
    }

    #[test]
    fn change_of_release_section() {
        assert_released_section_changes! {
        "\
<!-- Some explanatory comment -->

## Unreleased

WHADDUP FOLKS

## Released (hee hee hee)

- Blah blah blah.
",
        "\
--- a/CHANGELOG.md
+++ b/CHANGELOG.md
@@ -4,6 +4,6 @@
\u{0020}
 WHADDUP FOLKS
\u{0020}
-## Released
+## Released (hee hee hee)
\u{0020}
 - Blah blah blah.
",
            [
                "\
@@ -4,6 +4,6 @@
\u{0020}
 WHADDUP FOLKS
\u{0020}
-## Released
+## Released (hee hee hee)
\u{0020}
 - Blah blah blah.
",
            ],
        }
    }

    #[test]
    fn deletion_of_released_section() {
        // TODO: https://github.com/gfx-rs/wgpu/issues/9245
    }
}
