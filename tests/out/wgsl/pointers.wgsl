fn f() {
    var v: vec2<i32>;

    let px: ptr<function, i32> = (&v.x);
    (*px) = 10;
    return;
}

