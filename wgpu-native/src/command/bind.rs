use crate::registry::{HUB, Items};
use crate::{
    B, Stored, WeaklyStored,
    BindGroupId, BindGroupLayoutId, PipelineLayoutId,
};

use hal;


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

//Note: we can probably make this much better than passing an `FnMut`

impl Binder {
    pub fn bind_group<F>(&mut self, index: usize, bind_group_id: BindGroupId, mut fun: F)
    where
        F: FnMut(&<B as hal::Backend>::PipelineLayout, &<B as hal::Backend>::DescriptorSet),
    {
        let bind_group_guard = HUB.bind_groups.read();
        let bind_group = bind_group_guard.get(bind_group_id);

        while self.entries.len() <= index {
            self.entries.push(BindGroupEntry::default());
        }
        *self.entries.get_mut(index).unwrap() = BindGroupEntry {
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
            if pipeline_layout.bind_group_layout_ids[index] == bind_group.layout_id {
                fun(&pipeline_layout.raw, &bind_group.raw);
            }
        }
    }

    pub fn change_layout<F>(&mut self, pipeline_layout_id: PipelineLayoutId, mut fun: F)
    where
        F: FnMut(&<B as hal::Backend>::PipelineLayout, usize, &<B as hal::Backend>::DescriptorSet),
    {
        if self.pipeline_layout_id == Some(WeaklyStored(pipeline_layout_id)) {
            return
        }

        self.pipeline_layout_id = Some(WeaklyStored(pipeline_layout_id));
        let pipeline_layout_guard = HUB.pipeline_layouts.read();
        let pipeline_layout = pipeline_layout_guard.get(pipeline_layout_id);
        let bing_group_guard = HUB.bind_groups.read();

        while self.entries.len() < pipeline_layout.bind_group_layout_ids.len() {
            self.entries.push(BindGroupEntry::default());
        }
        for (index, (entry, bgl_id)) in self.entries
            .iter_mut()
            .zip(&pipeline_layout.bind_group_layout_ids)
            .enumerate()
        {
            if entry.layout == Some(bgl_id.clone()) {
                continue
            }
            entry.layout = Some(bgl_id.clone());
            if let Some(ref bg_id) = entry.data {
                let bind_group = bing_group_guard.get(bg_id.value);
                fun(&pipeline_layout.raw, index, &bind_group.raw);
            }
        }
    }
}
