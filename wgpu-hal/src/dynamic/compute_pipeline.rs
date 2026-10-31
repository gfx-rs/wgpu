use core::fmt;

use crate::{ComputePipeline, DynResource};

pub trait DynComputePipeline: DynResource + fmt::Debug {
    fn get_subgroup_size(&self) -> Option<u32>;
}

impl<P: ComputePipeline + DynResource + fmt::Debug> DynComputePipeline for P {
    fn get_subgroup_size(&self) -> Option<u32> {
        P::get_subgroup_size(self)
    }
}
