pub fn map_naga_stage(stage: naga::ShaderStage) -> wgt::ShaderStage {
    match stage {
        naga::ShaderStage::Vertex => wgt::ShaderStage::VERTEX,
        naga::ShaderStage::Fragment => wgt::ShaderStage::FRAGMENT,
        naga::ShaderStage::Compute => wgt::ShaderStage::COMPUTE,
    }
}
