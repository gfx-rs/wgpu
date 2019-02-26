use crate::{
    BindGroupHandle, Stored,
    BindGroupId, BindGroupLayoutId, PipelineLayoutId,
};


pub struct BindGroupPair {
    layout_id: BindGroupLayoutId,
    group_id: Stored<BindGroupId>,
}

#[derive(Default)]
pub struct BindGroupEntry {
    expected_layout_id: Option<BindGroupLayoutId>,
    provided: Option<BindGroupPair>,
}

impl BindGroupEntry {
    fn provide(&mut self, bind_group_id: BindGroupId, bind_group: &BindGroupHandle) -> bool {
        if let Some(BindGroupPair { ref layout_id, ref group_id }) = self.provided {
            if group_id.value == bind_group_id {
                assert_eq!(*layout_id, bind_group.layout_id);
                return false
            }
        }

        self.provided = Some(BindGroupPair {
            layout_id: bind_group.layout_id.clone(),
            group_id: Stored {
                value: bind_group_id,
                ref_count: bind_group.life_guard.ref_count.clone(),
            },
        });

        self.expected_layout_id == Some(bind_group.layout_id.clone())
    }

    pub fn expect_layout(
        &mut self, bind_group_layout_id: BindGroupLayoutId,
    ) -> Option<BindGroupId> {
        let some = Some(bind_group_layout_id);
        if self.expected_layout_id != some {
            self.expected_layout_id = some;
            match self.provided {
                Some(BindGroupPair { layout_id, ref group_id })
                    if layout_id == bind_group_layout_id => Some(group_id.value),
                Some(_) | None => None,
            }
        } else {
            None
        }
    }
}

#[derive(Default)]
pub struct Binder {
    pub(crate) pipeline_layout_id: Option<PipelineLayoutId>, //TODO: strongly `Stored`
    pub(crate) entries: Vec<BindGroupEntry>,
}

impl Binder {
    pub fn ensure_length(&mut self, length: usize) {
        while self.entries.len() < length {
            self.entries.push(BindGroupEntry::default());
        }
    }

    pub(crate) fn provide_entry(
        &mut self, index: usize, bind_group_id: BindGroupId, bind_group: &BindGroupHandle
    ) -> Option<PipelineLayoutId> {
        self.ensure_length(index + 1);
        if self.entries[index].provide(bind_group_id, bind_group) {
            self.pipeline_layout_id.as_ref().cloned()
        } else {
            None
        }
    }
}
