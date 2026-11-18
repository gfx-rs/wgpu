use std::io;

use anyhow::bail;
use pico_args::Arguments;
use xshell::Shell;

mod fragments;

pub(crate) fn print_help() {
    use fragments::{AREAS, CATEGORIES};

    println!(
        "\
Usage: cargo xtask changelog <COMMAND>

Commands:
  new [<slug> [<category> [<area>]]]
    Create changelog.d/<slug>.md. Prompts for missing arguments.
    Press Enter to use the general area. Major changes have no area.

    Categories: {CATEGORIES}
    Areas: {AREAS}

  preview [<slug> | changelog.d/<slug>.md] [--trunk <ref>]
    Validate and print one fragment or the full Unreleased section. Works offline.
    The full preview uses local trunk history for ordering. Drafts follow merged entries.

  check [<base-ref>] [--allow-released-changes]
    Validate fragments and reject changes to CHANGELOG.md since the merge base
    with <base-ref> (default: trunk). Use --allow-released-changes for PRs with
    the 'changelog: released entry changed' label.

  release <vX.Y.Z> [--trunk <ref>]
    Assemble committed fragments in CHANGELOG.md and remove the fragment files.
    Requires the fragment workflow to be active, current trunk history, and gh for PR credit.

Options:
  --trunk <ref>  Use this ref for trunk history (default: trunk).
  -h, --help     Print help"
    );
}

pub(crate) fn run(shell: Shell, mut args: Arguments) -> anyhow::Result<()> {
    match args.subcommand()?.as_deref() {
        Some("check") => fragments::check(&shell, args),
        Some("new") => fragments::new(&shell, args, io::stdin().lock(), io::stderr().lock()),
        Some("preview") => fragments::preview(&shell, args),
        Some("release") => fragments::release(&shell, args),
        None if args.finish().is_empty() => {
            print_help();
            Ok(())
        }
        _ => bail!("expected a changelog command; see `cargo xtask changelog --help`"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OddCharacter {
    line: u64,
    column: u64,
    character: char,
}

impl std::fmt::Display for OddCharacter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "- line {}, column {}: {}",
            self.line,
            self.column,
            self.character.escape_unicode(),
        )
    }
}

fn find_odd_characters(changelog_contents: &str) -> Vec<OddCharacter> {
    let mut odd_characters = vec![];

    let mut line = 1;
    let mut column = 1;
    for character in changelog_contents.chars() {
        if is_odd_character(character) {
            odd_characters.push(OddCharacter {
                line,
                column,
                character,
            });
        }

        if character == '\n' {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }

    odd_characters
}

fn is_odd_character(character: char) -> bool {
    match character {
        ' ' | '\t' | '\n' | '\r' => false,
        _ => {
            character.is_control()
                || character.is_whitespace()
                || matches!(
                    character,
                    '\u{00AD}' // soft hyphen
                        | '\u{200B}'..='\u{200F}' // zero-width characters and directional marks
                        | '\u{2028}'..='\u{202E}' // separators and bidi overrides
                        | '\u{2060}'..='\u{2064}' // invisible operators and joins
                        | '\u{FEFF}' // zero-width no-break space / byte order mark
                )
        }
    }
}

#[cfg(test)]
mod test_find_odd_characters {
    use super::{find_odd_characters, is_odd_character, OddCharacter};

    #[test]
    fn accepts_regular_text() {
        assert_eq!(
            find_odd_characters(
                "# Changelog\n\n- Em dashes—curly quotes'… accents é and emoji 🎉 are fine.\n"
            ),
            []
        );
    }

    #[test]
    fn accepts_ordinary_whitespace() {
        for character in [' ', '\t', '\n', '\r'] {
            assert!(!is_odd_character(character));
        }
    }

    #[test]
    fn rejects_invisible_characters() {
        for character in [
            '\u{00A0}', '\u{00AD}', '\u{200B}', '\u{200E}', '\u{202A}', '\u{2060}', '\u{FEFF}',
            '\u{0000}',
        ] {
            assert!(is_odd_character(character));
        }
    }

    #[test]
    fn reports_line_and_column() {
        assert_eq!(
            find_odd_characters("first line\nse\u{00A0}cond line"),
            [OddCharacter {
                line: 2,
                column: 3,
                character: '\u{00A0}',
            }],
        );
    }
}
