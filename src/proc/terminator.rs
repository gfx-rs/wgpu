/// Ensure that the given block has return statements
/// at the end of its control flow.
///
/// Note: we don't want to blindly append a return statement
/// to the end, because it may be either redundant or invalid,
/// e.g. when the user already has returns in if/else branches.
pub fn ensure_block_returns(block: &mut crate::Block) {
    use crate::Statement as S;
    match block.last_mut() {
        Some(&mut S::Block(ref mut b)) => {
            ensure_block_returns(b);
        }
        Some(&mut S::If {
            condition: _,
            ref mut accept,
            ref mut reject,
        }) => {
            ensure_block_returns(accept);
            ensure_block_returns(reject);
        }
        Some(&mut S::Switch {
            selector: _,
            ref mut cases,
            ref mut default,
        }) => {
            for case in cases.iter_mut() {
                if !case.fall_through {
                    ensure_block_returns(&mut case.body);
                }
            }
            ensure_block_returns(default);
        }
        Some(&mut S::Emit(_))
        | Some(&mut S::Break)
        | Some(&mut S::Continue)
        | Some(&mut S::Return { .. })
        | Some(&mut S::Kill) => (),
        Some(&mut S::Loop { .. })
        | Some(&mut S::Store { .. })
        | Some(&mut S::ImageStore { .. })
        | Some(&mut S::Call { .. })
        | None => block.push(S::Return { value: None }),
    }
}
