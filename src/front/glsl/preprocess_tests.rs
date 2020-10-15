use super::preprocess::{Error, LinePreProcessor};
use std::{iter::Enumerate, str::Lines};

#[derive(Clone, Debug)]
pub struct PreProcessor<'a> {
    lines: Enumerate<Lines<'a>>,
    input: String,
    line: usize,
    offset: usize,
    line_pp: LinePreProcessor,
}

impl<'a> PreProcessor<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut lexer = PreProcessor {
            lines: input.lines().enumerate(),
            input: "".to_string(),
            line: 0,
            offset: 0,
            line_pp: LinePreProcessor::new(),
        };
        lexer.next_line();
        lexer
    }

    fn next_line(&mut self) -> bool {
        if let Some((line, input)) = self.lines.next() {
            let mut input = String::from(input);

            while input.ends_with('\\') {
                if let Some((_, next)) = self.lines.next() {
                    input.pop();
                    input.push_str(next);
                } else {
                    break;
                }
            }

            self.input = input;
            self.line = line;
            self.offset = 0;
            true
        } else {
            false
        }
    }

    pub fn process(&mut self) -> Result<String, Error> {
        let mut res = String::new();
        loop {
            let line = &self.line_pp.process_line(&self.input)?;
            if let Some(line) = line {
                res.push_str(line);
            }
            if !self.next_line() {
                break;
            }
            if line.is_some() {
                res.push_str("\n");
            }
        }
        Ok(res)
    }
}

#[test]
fn preprocess() {
    // line continuation
    let mut pp = PreProcessor::new(
        "void main my_\
        func",
    );
    assert_eq!(pp.process().unwrap(), "void main my_func");

    // preserve #version
    let mut pp = PreProcessor::new(
        "#version 450 core\n\
        void main()",
    );
    assert_eq!(pp.process().unwrap(), "#version 450 core\nvoid main()");

    // simple define
    let mut pp = PreProcessor::new(
        "#define FOO 42 \n\
        fun=FOO",
    );
    assert_eq!(pp.process().unwrap(), "\nfun=42");

    // ifdef with else
    let mut pp = PreProcessor::new(
        "#define FOO\n\
        #ifdef FOO\n\
            foo=42\n\
        #endif\n\
        some=17\n\
        #ifdef BAR\n\
            bar=88\n\
        #else\n\
            mm=49\n\
        #endif\n\
        done=1",
    );
    assert_eq!(
        pp.process().unwrap(),
        "\n\
        foo=42\n\
        \n\
        some=17\n\
        \n\
        mm=49\n\
        \n\
        done=1"
    );

    // nested ifdef/ifndef
    let mut pp = PreProcessor::new(
        "#define FOO\n\
        #define BOO\n\
        #ifdef FOO\n\
            foo=42\n\
            #ifdef BOO\n\
                boo=44\n\
            #endif\n\
                ifd=0\n\
            #ifndef XYZ\n\
                nxyz=8\n\
            #endif\n\
        #endif\n\
        some=17\n\
        #ifdef BAR\n\
            bar=88\n\
        #else\n\
            mm=49\n\
        #endif\n\
        done=1",
    );
    assert_eq!(
        pp.process().unwrap(),
        "\n\
        foo=42\n\
        \n\
        boo=44\n\
        \n\
        ifd=0\n\
        \n\
        nxyz=8\n\
        \n\
        some=17\n\
        \n\
        mm=49\n\
        \n\
        done=1"
    );

    // undef
    let mut pp = PreProcessor::new(
        "#define FOO\n\
        #ifdef FOO\n\
            foo=42\n\
        #endif\n\
        some=17\n\
        #undef FOO\n\
        #ifdef FOO\n\
            foo=88\n\
        #else\n\
            nofoo=66\n\
        #endif\n\
        done=1",
    );
    assert_eq!(
        pp.process().unwrap(),
        "\n\
        foo=42\n\
        \n\
        some=17\n\
        \n\
        nofoo=66\n\
        \n\
        done=1"
    );

    // single-line comment
    let mut pp = PreProcessor::new(
        "#define FOO 42//1234\n\
        fun=FOO",
    );
    assert_eq!(pp.process().unwrap(), "\nfun=42");

    // multi-line comments
    let mut pp = PreProcessor::new(
        "#define FOO 52/*/1234\n\
        #define FOO 88\n\
        end of comment*/ /* one more comment */ #define FOO 56\n\
        fun=FOO",
    );
    assert_eq!(pp.process().unwrap(), "\nfun=56");

    // unmatched endif
    let mut pp = PreProcessor::new(
        "#ifdef FOO\n\
        foo=42\n\
        #endif\n\
        #endif",
    );
    assert_eq!(pp.process(), Err(Error::UnmatchedEndif));

    // unmatched else
    let mut pp = PreProcessor::new(
        "#ifdef FOO\n\
        foo=42\n\
        #else\n\
        bar=88\n\
        #else\n\
        bad=true\n\
        #endif",
    );
    assert_eq!(pp.process(), Err(Error::UnmatchedElse));
}
