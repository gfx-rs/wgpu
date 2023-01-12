fn blockLexicalScope(a: bool) {
    {
        {
        }
        let test = (2 == 3);
    }
    let test_1 = (1.0 == 2.0);
}

fn ifLexicalScope(a_1: bool) {
    if (1.0 == 1.0) {
    }
    let test_2 = (1.0 == 2.0);
}

fn loopLexicalScope(a_2: bool) {
    loop {
    }
    let test_3 = (1.0 == 2.0);
}

fn forLexicalScope(a_3: f32) {
    var a_4: i32;

    a_4 = 0;
    loop {
        let _e4 = a_4;
        if (_e4 < 1) {
        } else {
            break;
        }
        {
        }
        continuing {
            let _e8 = a_4;
            a_4 = (_e8 + 1);
        }
    }
    let test_4 = (false == true);
}

fn whileLexicalScope(a_5: i32) {
    loop {
        if (a_5 > 2) {
        } else {
            break;
        }
        {
        }
    }
    let test_5 = (a_5 == 1);
}

fn switchLexicalScope(a_6: i32) {
    switch a_6 {
        case 0: {
        }
        case 1: {
        }
        default: {
        }
    }
    let test_6 = (a_6 == 2);
}

