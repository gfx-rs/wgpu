use crate::{
    conv,
    resource::BufferUsage,
    BufferId,
};
use super::{PendingTransition, ResourceState, Stitch, Unit};
use std::ops::Range;

//TODO: store `hal::buffer::State` here to avoid extra conversions
pub type BufferState = Unit<BufferUsage>;

impl PendingTransition<BufferState> {
    /// Produce the gfx-hal buffer states corresponding to the transition.
    pub fn to_states(&self) -> Range<hal::buffer::State> {
        conv::map_buffer_state(self.usage.start) ..
        conv::map_buffer_state(self.usage.end)
    }
}

impl Default for BufferState {
    fn default() -> Self {
        BufferState {
            init: BufferUsage::empty(),
            last: BufferUsage::empty(),
        }
    }
}

impl ResourceState for BufferState {
    type Id = BufferId;
    type Selector = ();
    type Usage = BufferUsage;

    fn query(
        &self,
        _selector: Self::Selector,
    ) -> Option<Self::Usage> {
        Some(self.last)
    }

    fn change(
        &mut self,
        id: Self::Id,
        _selector: Self::Selector,
        usage: Self::Usage,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let old = self.last;
        if usage != old {
            let pending = PendingTransition {
                id,
                selector: (),
                usage: old .. usage,
            };
            self.last = match output {
                Some(transitions) => {
                    transitions.push(pending);
                    usage
                }
                None =>  {
                    if !old.is_empty() && BufferUsage::WRITE_ALL.intersects(old | usage) {
                        return Err(pending);
                    }
                    old | usage
                }
            };
        }
        Ok(())
    }

    fn merge(
        &mut self,
        id: Self::Id,
        other: &Self,
        stitch: Stitch,
        output: Option<&mut Vec<PendingTransition<Self>>>,
    ) -> Result<(), PendingTransition<Self>> {
        let usage = other.select(stitch);
        self.change(id, (), usage, output)
    }

    fn optimize(&mut self) {
    }
}

#[cfg(test)]
mod test {
    use crate::TypedId;
    use super::*;

    #[test]
    fn change() {
        let mut bs = Unit {
            init: BufferUsage::INDEX,
            last: BufferUsage::STORAGE,
        };
        let id = TypedId::new(0, 0);
        assert!(bs.change(id, (), BufferUsage::VERTEX, None).is_err());
        bs.change(id, (), BufferUsage::VERTEX, Some(&mut Vec::new())).unwrap();
        bs.change(id, (), BufferUsage::INDEX, None).unwrap();
        assert_eq!(bs.last, BufferUsage::VERTEX | BufferUsage::INDEX);
    }
}
