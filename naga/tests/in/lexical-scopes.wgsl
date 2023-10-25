fn blockLexicalScope(a: bool) {
    {
        let a = 2;
        {
            let a = 2.0;
        }
        let test: i32 = a;
    }
    let test: bool = a;
}

fn ifLexicalScope(a: bool) {
    if (a) {
        let a = 2.0;
    }
    let test: bool = a;
}


fn loopLexicalScope(a: bool) {
    loop {
        let a = 2.0;
    }
    let test: bool = a;
}

fn forLexicalScope(a: f32) {
    for (var a = 0; a < 1; a++) {
        let a = true;
    }
    let test: f32 = a;
}

fn whileLexicalScope(a: i32) {
    while (a > 2) {
        let a = false;
    }
    let test: i32 = a;
}

fn switchLexicalScope(a: i32) {
    switch (a) {
        case 0 {
            let a = false;
        }
        case 1 {
            let a = 2.0;
        }
        default {
            let a = true;
        }
    }
    let test = a == 2;
}
