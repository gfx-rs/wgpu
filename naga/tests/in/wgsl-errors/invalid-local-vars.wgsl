// Rejected in statement lowering
// There is a similar validator test in `validation.rs`.

override len: u32;
var<workgroup> arr: array<u32, len>;
fn f() {
    let x: array<u32, len> = arr;
}
