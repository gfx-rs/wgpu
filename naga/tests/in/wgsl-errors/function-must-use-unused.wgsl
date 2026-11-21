@must_use
fn use_me(a: i32) -> i32 {
  return 10;
}

fn useless() -> i32 {
  use_me(1);
  return 0;
}
