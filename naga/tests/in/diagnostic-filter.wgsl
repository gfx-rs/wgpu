diagnostic(off, derivative_uniformity);

fn thing() {}

@diagnostic(warning, derivative_uniformity)
fn with_diagnostic() {}
