use crate::registry::{HUB, Items, ConcreteItems};
use crate::{
    B, Stored, WeaklyStored,
    BindGroup, PipelineLayout,
    BindGroupId, BindGroupLayoutId, PipelineLayoutId,
};

use hal;
use parking_lot::RwLockReadGuard;


#[derive(Clone, Default)]
struct BindGroupEntry {
    layout: Option<WeaklyStored<BindGroupLayoutId>>,
    data: Option<Stored<BindGroupId>>,
}

#[derive(Default)]
pub struct Binder {
    pipeline_layout_id: Option<WeaklyStored<PipelineLayoutId>>, //TODO: strongly `Stored`
    entries: Vec<BindGroupEntry>,
}

pub struct NewBind<'a, B: hal::Backend> {
    pipeline_layout_guard: RwLockReadGuard<'a, ConcreteItems<PipelineLayout<B>>>,
    pipeline_layout_id: PipelineLayoutId,
    bind_group_guard: RwLockReadGuard<'a, ConcreteItems<BindGroup<B>>>,
    bind_group_id: BindGroupId,
}

impl<'a, B: hal::Backend> NewBind<'a, B> {
    pub fn pipeline_layout(&self) -> &B::PipelineLayout {
        &self.pipeline_layout_guard.get(self.pipeline_layout_id).raw
    }

    pub fn descriptor_set(&self) -> &B::DescriptorSet {
        &self.bind_group_guard.get(self.bind_group_id).raw
    }
}

impl Binder {
    //Note: `'a` is need to avoid inheriting the lifetime from `self`
    pub fn bind_group<'a>(
        &mut self, index: u32, bind_group_id: BindGroupId
    ) -> Option<NewBind<'a, B>> {
        let bind_group_guard = HUB.bind_groups.read();
        let bind_group = bind_group_guard.get(bind_group_id);

        while self.entries.len() <= index as usize {
            self.entries.push(BindGroupEntry::default());
        }
        *self.entries.get_mut(index as usize).unwrap() = BindGroupEntry {
            layout: Some(bind_group.layout_id.clone()),
            data: Some(Stored {
                value: bind_group_id,
                ref_count: bind_group.life_guard.ref_count.clone(),
            }),
        };

        if let Some(WeaklyStored(pipeline_layout_id)) = self.pipeline_layout_id {
            //TODO: we can cache the group layout ids of the current pipeline in `Binder` itself
            let pipeline_layout_guard = HUB.pipeline_layouts.read();
            let pipeline_layout = pipeline_layout_guard.get(pipeline_layout_id);
            if pipeline_layout.bind_group_layout_ids[index as usize] == bind_group.layout_id {
                return Some(NewBind {
                    pipeline_layout_guard,
                    pipeline_layout_id,
                    bind_group_guard,
                    bind_group_id,
                })
            }
        }

        None
    }
}
