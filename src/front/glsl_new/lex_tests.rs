use super::lex::Lexer;
use super::parser::Token;

#[test]
fn glsl_lex_simple() {
    let source = "void main() {\n}";
    let lex = Lexer::new(source);
    let tokens: Vec<Token> = lex.collect();
    assert_eq!(tokens.len(), 6);

    let mut iter = tokens.iter();
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Void(TokenMetadata { line: 0, chars: 0..4 })"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Identifier((TokenMetadata { line: 0, chars: 5..9 }, \"main\"))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "LeftParen(TokenMetadata { line: 0, chars: 9..10 })"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "RightParen(TokenMetadata { line: 0, chars: 10..11 })"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "LeftBrace(TokenMetadata { line: 0, chars: 12..13 })"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "RightBrace(TokenMetadata { line: 1, chars: 0..1 })"
    );
}

#[test]
fn glsl_lex_line_comment() {
    let source = "void main // myfunction\n//()\n{}";
    let lex = Lexer::new(source);
    let tokens: Vec<Token> = lex.collect();
    assert_eq!(tokens.len(), 4);

    let mut iter = tokens.iter();
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Void(TokenMetadata { line: 0, chars: 0..4 })"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Identifier((TokenMetadata { line: 0, chars: 5..9 }, \"main\"))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "LeftBrace(TokenMetadata { line: 2, chars: 0..1 })"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "RightBrace(TokenMetadata { line: 2, chars: 1..2 })"
    );
}

#[test]
fn glsl_lex_identifier() {
    let source = "id123_OK 92No æNoø No¾ No好";
    let lex = Lexer::new(source);
    let tokens: Vec<Token> = lex.collect();
    assert_eq!(tokens.len(), 10);

    let mut iter = tokens.iter();
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Identifier((TokenMetadata { line: 0, chars: 0..8 }, \"id123_OK\"))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "IntConstant((TokenMetadata { line: 0, chars: 9..11 }, 92))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Identifier((TokenMetadata { line: 0, chars: 11..13 }, \"No\"))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Unknown((TokenMetadata { line: 0, chars: 14..15 }, \'æ\'))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Identifier((TokenMetadata { line: 0, chars: 15..17 }, \"No\"))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Unknown((TokenMetadata { line: 0, chars: 17..18 }, \'ø\'))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Identifier((TokenMetadata { line: 0, chars: 19..21 }, \"No\"))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Unknown((TokenMetadata { line: 0, chars: 21..22 }, \'¾\'))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Identifier((TokenMetadata { line: 0, chars: 23..25 }, \"No\"))"
    );
    assert_eq!(
        format!("{:?}", iter.next().unwrap()),
        "Unknown((TokenMetadata { line: 0, chars: 25..26 }, \'好\'))"
    );
}
