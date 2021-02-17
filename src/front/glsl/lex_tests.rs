use super::{lex::Lexer, parser::Token::*, token::TokenMetadata};

#[test]
fn tokens() {
    let defines = crate::FastHashMap::default();

    // line comments
    let mut lex = Lexer::new("void main // myfunction\n//()\n{}", &defines);
    assert_eq!(
        lex.next().unwrap(),
        Void(TokenMetadata {
            line: 0,
            chars: 0..4
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 5..9
            },
            "main".into()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        LeftBrace(TokenMetadata {
            line: 2,
            chars: 0..1
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        RightBrace(TokenMetadata {
            line: 2,
            chars: 1..2
        })
    );
    assert_eq!(lex.next(), None);

    // multi line comment
    let mut lex = Lexer::new("void main /* comment [] {}\n/**\n{}*/{}", &defines);
    assert_eq!(
        lex.next().unwrap(),
        Void(TokenMetadata {
            line: 0,
            chars: 0..4
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 5..9
            },
            "main".into()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        LeftBrace(TokenMetadata {
            line: 2,
            chars: 4..5
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        RightBrace(TokenMetadata {
            line: 2,
            chars: 5..6
        })
    );
    assert_eq!(lex.next(), None);

    // identifiers
    let mut lex = Lexer::new("id123_OK 92No æNoø No¾ No好", &defines);
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 0..8
            },
            "id123_OK".into()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        IntConstant((
            TokenMetadata {
                line: 0,
                chars: 9..11
            },
            92
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 11..13
            },
            "No".into()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Unknown((
            TokenMetadata {
                line: 0,
                chars: 14..15
            },
            'æ'.to_string()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 15..17
            },
            "No".into()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Unknown((
            TokenMetadata {
                line: 0,
                chars: 17..18
            },
            'ø'.to_string()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 19..21
            },
            "No".into()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Unknown((
            TokenMetadata {
                line: 0,
                chars: 21..22
            },
            '¾'.to_string()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 23..25
            },
            "No".into()
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Unknown((
            TokenMetadata {
                line: 0,
                chars: 25..26
            },
            '好'.to_string()
        ))
    );
    assert_eq!(lex.next(), None);

    // version
    let mut lex = Lexer::new("#version 890 core", &defines);
    assert_eq!(
        lex.next().unwrap(),
        Version(TokenMetadata {
            line: 0,
            chars: 0..8
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        IntConstant((
            TokenMetadata {
                line: 0,
                chars: 9..12
            },
            890
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        Identifier((
            TokenMetadata {
                line: 0,
                chars: 13..17
            },
            "core".into()
        ))
    );
    assert_eq!(lex.next(), None);

    // operators
    let mut lex = Lexer::new(
        "+ - * | & % / += -= *= |= &= %= /= ++ -- || && ^^",
        &defines,
    );
    assert_eq!(
        lex.next().unwrap(),
        Plus(TokenMetadata {
            line: 0,
            chars: 0..1
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        Dash(TokenMetadata {
            line: 0,
            chars: 2..3
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        Star(TokenMetadata {
            line: 0,
            chars: 4..5
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        VerticalBar(TokenMetadata {
            line: 0,
            chars: 6..7
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        Ampersand(TokenMetadata {
            line: 0,
            chars: 8..9
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        Percent(TokenMetadata {
            line: 0,
            chars: 10..11
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        Slash(TokenMetadata {
            line: 0,
            chars: 12..13
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        AddAssign(TokenMetadata {
            line: 0,
            chars: 14..16
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        SubAssign(TokenMetadata {
            line: 0,
            chars: 17..19
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        MulAssign(TokenMetadata {
            line: 0,
            chars: 20..22
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        OrAssign(TokenMetadata {
            line: 0,
            chars: 23..25
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        AndAssign(TokenMetadata {
            line: 0,
            chars: 26..28
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        ModAssign(TokenMetadata {
            line: 0,
            chars: 29..31
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        DivAssign(TokenMetadata {
            line: 0,
            chars: 32..34
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        IncOp(TokenMetadata {
            line: 0,
            chars: 35..37
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        DecOp(TokenMetadata {
            line: 0,
            chars: 38..40
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        OrOp(TokenMetadata {
            line: 0,
            chars: 41..43
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        AndOp(TokenMetadata {
            line: 0,
            chars: 44..46
        })
    );
    assert_eq!(
        lex.next().unwrap(),
        XorOp(TokenMetadata {
            line: 0,
            chars: 47..49
        })
    );
    assert_eq!(lex.next(), None);

    // float constants
    let mut lex = Lexer::new("120.0 130.0lf 140.0Lf 150.0LF", &defines);
    assert_eq!(
        lex.next().unwrap(),
        FloatConstant((
            TokenMetadata {
                line: 0,
                chars: 0..5
            },
            120.0,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        DoubleConstant((
            TokenMetadata {
                line: 0,
                chars: 6..13
            },
            130.0,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        DoubleConstant((
            TokenMetadata {
                line: 0,
                chars: 14..21
            },
            140.0,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        DoubleConstant((
            TokenMetadata {
                line: 0,
                chars: 22..29
            },
            150.0,
        ))
    );
    assert_eq!(lex.next(), None);

    // Integer constants
    let mut lex = Lexer::new("120 130u 140U 150 0x1f 0xf2U 0xF1u", &defines);
    assert_eq!(
        lex.next().unwrap(),
        IntConstant((
            TokenMetadata {
                line: 0,
                chars: 0..3
            },
            120,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        UintConstant((
            TokenMetadata {
                line: 0,
                chars: 4..8
            },
            130,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        UintConstant((
            TokenMetadata {
                line: 0,
                chars: 9..13
            },
            140,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        IntConstant((
            TokenMetadata {
                line: 0,
                chars: 14..17
            },
            150,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        IntConstant((
            TokenMetadata {
                line: 0,
                chars: 18..22
            },
            31,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        UintConstant((
            TokenMetadata {
                line: 0,
                chars: 23..28
            },
            242,
        ))
    );
    assert_eq!(
        lex.next().unwrap(),
        UintConstant((
            TokenMetadata {
                line: 0,
                chars: 29..34
            },
            241,
        ))
    );
    assert_eq!(lex.next(), None);
}
