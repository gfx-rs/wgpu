fn blockLexicalScope(a: bool) {
    let a = 1.0;
    {
        let a = 2;
        {
            let a = true;
        }
        let test = a == 3;
    }
    let test = a == 2.0;
}

fn ifLexicalScope(a: bool) {
    let a = 1.0;
    if (a == 1.0) {
        let a = true;
    }
    let test = a == 2.0;
}


fn loopLexicalScope(a: bool) {
    let a = 1.0;
    loop {
        let a = true;
    }
    let test = a == 2.0;
}

fn forLexicalScope(a: f32) {
    let a = false;
    for (var a = 0; a < 1; a++) {
        let a = 3.0;
    }
    let test = a == true;
}

fn whileLexicalScope(a: i32) {
    while (a > 2) {
        let a = false;
    }
    let test = a == 1;
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
