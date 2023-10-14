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
        }) => {
            for case in cases.iter_mut() {
                if !case.fall_through {
                    ensure_block_returns(&mut case.body);
                }
            }
        }
        Some(&mut (S::Emit(_) | S::Break | S::Continue | S::Return { .. } | S::Kill)) => (),
        Some(
            &mut (S::Loop { .. }
            | S::Store { .. }
            | S::ImageStore { .. }
            | S::Call { .. }
            | S::RayQuery { .. }
            | S::Atomic { .. }
            | S::WorkGroupUniformLoad { .. }
            | S::SubgroupBallot { .. }
            | S::SubgroupCollectiveOperation { .. }
            | S::SubgroupBroadcast { .. }
            | S::Barrier(_)),
        )
        | None => block.push(S::Return { value: None }, Default::default()),
    }
}
