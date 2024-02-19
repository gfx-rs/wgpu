var<private> v: i64 = 1li;
const k: u64 = 20lu;

fn fi(x: i64) -> i64 {
   let y: i64 = 31li - 1002003004005006li;
   var z = y + i64(5);
   return x + y + i64(k) + 50li;
}
fn fu(x: u64) -> u64 {
   let y: u64 = 31lu + 1002003004005006lu;
   let v = vec3<u64>(3lu,4lu,5lu);
   var z = y + u64(4);
   return x + y + k + 34lu + v.x + v.y + v.z;
}

@compute @workgroup_size(1)
fn main() {
   fu(67lu);
   fi(60li);
}
