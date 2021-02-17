use crate::FastHashMap;
use thiserror::Error;

#[derive(Clone, Debug, Error)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Error {
    #[error("unmatched else")]
    UnmatchedElse,
    #[error("unmatched endif")]
    UnmatchedEndif,
    #[error("missing macro name")]
    MissingMacro,
}

#[derive(Clone, Debug)]
pub struct IfState {
    true_branch: bool,
    else_seen: bool,
}

#[derive(Clone, Debug)]
pub struct LinePreProcessor {
    defines: FastHashMap<String, String>,
    if_stack: Vec<IfState>,
    inside_comment: bool,
    in_preprocess: bool,
}

impl LinePreProcessor {
    pub fn new(defines: &FastHashMap<String, String>) -> Self {
        LinePreProcessor {
            defines: defines.clone(),
            if_stack: vec![],
            inside_comment: false,
            in_preprocess: false,
        }
    }

    fn subst_defines(&self, input: &str) -> String {
        //TODO: don't subst in commments, strings literals?
        self.defines
            .iter()
            .fold(input.to_string(), |acc, (k, v)| acc.replace(k, v))
    }

    pub fn process_line(&mut self, line: &str) -> Result<Option<String>, Error> {
        let mut skip = !self.if_stack.last().map(|i| i.true_branch).unwrap_or(true);
        let mut inside_comment = self.inside_comment;
        let mut in_preprocess = inside_comment && self.in_preprocess;
        // single-line comment
        let mut processed = line;
        if let Some(pos) = line.find("//") {
            processed = line.split_at(pos).0;
        }
        // multi-line comment
        let mut processed_string: String;
        loop {
            if inside_comment {
                if let Some(pos) = processed.find("*/") {
                    processed = processed.split_at(pos + 2).1;
                    inside_comment = false;
                    self.inside_comment = false;
                    continue;
                }
            } else if let Some(pos) = processed.find("/*") {
                if let Some(end_pos) = processed[pos + 2..].find("*/") {
                    // comment ends during this line
                    processed_string = processed.to_string();
                    processed_string.replace_range(pos..pos + end_pos + 4, "");
                    processed = &processed_string;
                } else {
                    processed = processed.split_at(pos).0;
                    inside_comment = true;
                }
                continue;
            }
            break;
        }
        // strip leading whitespace
        processed = processed.trim_start();
        if processed.starts_with('#') && !self.inside_comment {
            let mut iter = processed[1..]
                .trim_start()
                .splitn(2, |c: char| c.is_whitespace());
            if let Some(directive) = iter.next() {
                skip = true;
                in_preprocess = true;
                match directive {
                    "version" => {
                        skip = false;
                    }
                    "define" => {
                        let rest = iter.next().ok_or(Error::MissingMacro)?;
                        let pos = rest
                            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '(')
                            .unwrap_or_else(|| rest.len());
                        let (key, mut value) = rest.split_at(pos);
                        value = value.trim();
                        self.defines.insert(key.into(), self.subst_defines(value));
                    }
                    "undef" => {
                        let rest = iter.next().ok_or(Error::MissingMacro)?;
                        let key = rest.trim();
                        self.defines.remove(key);
                    }
                    "ifdef" => {
                        let rest = iter.next().ok_or(Error::MissingMacro)?;
                        let key = rest.trim();
                        self.if_stack.push(IfState {
                            true_branch: self.defines.contains_key(key),
                            else_seen: false,
                        });
                    }
                    "ifndef" => {
                        let rest = iter.next().ok_or(Error::MissingMacro)?;
                        let key = rest.trim();
                        self.if_stack.push(IfState {
                            true_branch: !self.defines.contains_key(key),
                            else_seen: false,
                        });
                    }
                    "else" => {
                        let if_state = self.if_stack.last_mut().ok_or(Error::UnmatchedElse)?;
                        if !if_state.else_seen {
                            // this is first else
                            if_state.true_branch = !if_state.true_branch;
                            if_state.else_seen = true;
                        } else {
                            return Err(Error::UnmatchedElse);
                        }
                    }
                    "endif" => {
                        self.if_stack.pop().ok_or(Error::UnmatchedEndif)?;
                    }
                    _ => {}
                }
            }
        }
        let res = if !skip && !self.inside_comment {
            Ok(Some(self.subst_defines(&line)))
        } else {
            Ok(if in_preprocess && !self.in_preprocess {
                Some("".to_string())
            } else {
                None
            })
        };
        self.in_preprocess = in_preprocess || skip;
        self.inside_comment = inside_comment;
        res
    }
}
