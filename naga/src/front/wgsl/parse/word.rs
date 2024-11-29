use std::fmt;
use std::sync::OnceLock;

use crate::FastHashMap;

use super::{ast, number};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Word<'a> {
    /// A WGSL identifier, with a possible global binding.
    Ident(&'a str, Option<GlobalBinding>),

    /// A boolean, `true` or `false.
    ///
    /// This is not simply an `Ident` with a `GlobalBinding`, because
    /// WGSL defines `true` and `false` as keywords with explicit
    /// productions in the expression grammar, not as identifiers with
    /// global bindings.
    Bool(bool),
    Underscore,
    Alias,
    Break,
    Case,
    Const,
    ConstAssert,
    Continue,
    Continuing,
    Default,
    Discard,
    Else,
    Fn,
    For,
    If,
    Let,
    Loop,
    Override,
    Return,
    Struct,
    Switch,
    Var,
    While,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum GlobalBinding {
    Literal(ast::Literal),
}

macro_rules! define_keywords {
    { $( $string:literal => $variant:tt ; )* } => {
        #[allow(unused_parens)]
        impl<'a> Word<'a> {
            pub const fn as_str(&self) -> &'a str {
                use Word::*;
                match *self {
                    $(
                        $variant => $string,
                    )*
                    Ident(s, _) => s,
                }
            }

            fn insert_keywords(map: &mut FastHashMap<&'static str, Word<'static>>) {
                use Word::*;
                $(
                    assert!(map.insert($string, $variant).is_none());
                )*
            }
        }
    };
}

define_keywords! {
    "_" => Underscore;
    "alias" => Alias;
    "break" => Break;
    "case" => Case;
    "const" => Const;
    "const_assert" => ConstAssert;
    "continue" => Continue;
    "continuing" => Continuing;
    "default" => Default;
    "discard" => Discard;
    "else" => Else;
    "false" => (Bool(false));
    "fn" => Fn;
    "for" => For;
    "if" => If;
    "let" => Let;
    "loop" => Loop;
    "override" => Override;
    "return" => Return;
    "struct" => Struct;
    "switch" => Switch;
    "true" => (Bool(true));
    "var" => Var;
    "while" => While;
}

impl fmt::Display for Word<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub type WordTable = FastHashMap<&'static str, Word<'static>>;
static KNOWN_WORDS: OnceLock<WordTable> = OnceLock::new();

pub fn get_table() -> &'static WordTable {
    KNOWN_WORDS.get_or_init(|| {
        use crate::common::wgsl;

        let mut map = FastHashMap::default();

        Word::insert_keywords(&mut map);

        for (value, name) in wgsl::RAYFLAG_NAMES.iter().cloned() {
            let lit = ast::Literal::Number(number::Number::U32(value.bits()));
            map.insert(name, Word::Ident(name, Some(GlobalBinding::Literal(lit))));
        }

        for (value, name) in wgsl::RAYQUERYINTERSECTION_NAMES.iter().cloned() {
            let lit = ast::Literal::Number(number::Number::U32(value as u32));
            map.insert(name, Word::Ident(name, Some(GlobalBinding::Literal(lit))));
        }

        map
    })
}
