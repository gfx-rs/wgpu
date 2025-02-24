// #6220: Don't generate unreachable SPIR-V blocks that branch into
// structured control flow constructs.
//
// Suppose we have Naga code like this:
//
//     Block {
//       ... prelude
//       Block { ... nested }
//       ... postlude
//     }
//
// The SPIR-V back end used to always generate three separate SPIR-V
// blocks for the sections labeled "prelude", "nested", and
// "postlude", each block ending with a branch to the next, even if
// they were empty.
//
// However, the function below generates code that includes the
// following structure:
//
//     Loop {
//       body: Block {
//         ... prelude
//         Block { Break }
//         ... postlude
//       }
//       continuing: ...
//     }
//
// In this case, even though the `Break` renders the "postlude"
// unreachable, we used to generate a SPIR-V block for it anyway,
// ending with a branch to the `Loop`'s "continuing" block. However,
// SPIR-V's structured control flow rules forbid branches to a loop
// construct's continue target from outside the loop, so the SPIR-V
// module containing the unreachable block didn't pass validation.
//
// One might assume that unreachable blocks shouldn't affect
// validation, but the spec doesn't clearly agree, and this doesn't
// seem to be the way validation has been implemented.
fn break_from_loop() {
    for (var i = 0; i < 4; i += 1) {
      break;
    }
}
