struct Atom {
    a: atomic<u32>
}
@group(0) @binding(0)
var<storage> atom: Atom;

fn return_atomic() -> Atom {
   return atom;
}
