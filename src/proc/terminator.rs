/// Ensure that the given block has return statements
/// at the end of its control flow.
///
/// Note: we don't want to blindly append a return statement
/// to the end, because it may be either redundant or invalid,
/// e.g. when the user already has returns in if/else branches.
pub fn ensure_block_returns(block: &mut crate::Block) {
    match block.last_mut() {
        Some(&mut crate::Statement::Block(ref mut b)) => {
            ensure_block_returns(b);
        }
        Some(&mut crate::Statement::If {
            condition: _,
            ref mut accept,
            ref mut reject,
        }) => {
            ensure_block_returns(accept);
            ensure_block_returns(reject);
        }
        Some(&mut crate::Statement::Switch {
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
        Some(&mut crate::Statement::Break)
        | Some(&mut crate::Statement::Continue)
        | Some(&mut crate::Statement::Return { .. })
        | Some(&mut crate::Statement::Kill) => (),
        Some(&mut crate::Statement::Loop { .. })
        | Some(&mut crate::Statement::Store { .. })
        | Some(&mut crate::Statement::ImageStore { .. })
        | Some(&mut crate::Statement::Call { .. })
        | None => block.push(crate::Statement::Return { value: None }),
    }
}
