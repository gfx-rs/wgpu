@compute @workgroup_size(1)
fn main() {}

fn breakIfEmpty() {
    loop {
        continuing {
            break if true;
        }
    }
}

fn breakIfEmptyBody(a: bool) {
    loop {
        continuing {
            var b = a;
            var c = a != b;

            break if a == c;
        }
    }
}

fn breakIf(a: bool) {
    loop {
        var d = a;
        var e = a != d;

        continuing {
            break if a == e;
        }
    }
}

fn breakIfSeparateVariable() {
    var counter = 0u;

    loop {
        counter += 1u;

        continuing {
            break if counter == 5u;
        }
    }
}
