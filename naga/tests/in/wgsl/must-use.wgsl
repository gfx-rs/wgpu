@must_use
fn use_me() -> i32 { return 10; }

fn use_return() -> i32 {
    return use_me();
}

fn use_assign_var() -> i32 {
    var q = use_me();
    return q;
}

fn use_assign_let() -> i32 {
    let q = use_me();
    return q;
}

fn use_phony_assign() {
    _ = use_me();
}

@compute @workgroup_size(1)
fn main() {
    use_return();
    use_assign_var();
    use_assign_let();
    use_phony_assign();
}
