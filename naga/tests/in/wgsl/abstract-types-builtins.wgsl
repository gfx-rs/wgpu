@compute @workgroup_size(1)
fn f() {
  // For calls that return abstract types, we can't write the type we
  // actually expect, but we can at least require an automatic
  // conversion.
  //
  // Error cases are covered in `wgsl_errors::more_inconsistent_type`.

  // start
  var clamp_aiaiai: i32 = clamp(1, 1, 1);
  var clamp_aiaiaf: f32 = clamp(1, 1, 1.0);
  var clamp_aiaii: i32 = clamp(1, 1, 1i);
  var clamp_aiaif: f32 = clamp(1, 1, 1f);
  var clamp_aiafai: f32 = clamp(1, 1.0, 1);
  var clamp_aiafaf: f32 = clamp(1, 1.0, 1.0);
//var clamp_aiafi: f32 = clamp(1, 1.0, 1i); error
  var clamp_aiaff: f32 = clamp(1, 1.0, 1f);
  var clamp_aiiai: i32 = clamp(1, 1i, 1);
//var clamp_aiiaf: f32 = clamp(1, 1i, 1.0); error
  var clamp_aiii: i32 = clamp(1, 1i, 1i);
//var clamp_aiif: f32 = clamp(1, 1i, 1f); error
  var clamp_aifai: f32 = clamp(1, 1f, 1);
  var clamp_aifaf: f32 = clamp(1, 1f, 1.0);
//var clamp_aifi: f32 = clamp(1, 1f, 1i); error
  var clamp_aiff: f32 = clamp(1, 1f, 1f);
  var clamp_afaiai: f32 = clamp(1.0, 1, 1);
  var clamp_afaiaf: f32 = clamp(1.0, 1, 1.0);
//var clamp_afaii: f32 = clamp(1.0, 1, 1i); error
  var clamp_afaif: f32 = clamp(1.0, 1, 1f);
  var clamp_afafai: f32 = clamp(1.0, 1.0, 1);
  var clamp_afafaf: f32 = clamp(1.0, 1.0, 1.0);
//var clamp_afafi: f32 = clamp(1.0, 1.0, 1i); error
  var clamp_afaff: f32 = clamp(1.0, 1.0, 1f);
//var clamp_afiai: f32 = clamp(1.0, 1i, 1); error
//var clamp_afiaf: f32 = clamp(1.0, 1i, 1.0); error
//var clamp_afii: f32 = clamp(1.0, 1i, 1i); error
//var clamp_afif: f32 = clamp(1.0, 1i, 1f); error
  var clamp_affai: f32 = clamp(1.0, 1f, 1);
  var clamp_affaf: f32 = clamp(1.0, 1f, 1.0);
//var clamp_affi: f32 = clamp(1.0, 1f, 1i); error
  var clamp_afff: f32 = clamp(1.0, 1f, 1f);
  var clamp_iaiai: i32 = clamp(1i, 1, 1);
//var clamp_iaiaf: f32 = clamp(1i, 1, 1.0); error
  var clamp_iaii: i32 = clamp(1i, 1, 1i);
//var clamp_iaif: f32 = clamp(1i, 1, 1f); error
//var clamp_iafai: f32 = clamp(1i, 1.0, 1); error
//var clamp_iafaf: f32 = clamp(1i, 1.0, 1.0); error
//var clamp_iafi: f32 = clamp(1i, 1.0, 1i); error
//var clamp_iaff: f32 = clamp(1i, 1.0, 1f); error
  var clamp_iiai: i32 = clamp(1i, 1i, 1);
//var clamp_iiaf: f32 = clamp(1i, 1i, 1.0); error
  var clamp_iii: i32 = clamp(1i, 1i, 1i);
//var clamp_iif: f32 = clamp(1i, 1i, 1f); error
//var clamp_ifai: f32 = clamp(1i, 1f, 1); error
//var clamp_ifaf: f32 = clamp(1i, 1f, 1.0); error
//var clamp_ifi: f32 = clamp(1i, 1f, 1i); error
//var clamp_iff: f32 = clamp(1i, 1f, 1f); error
  var clamp_faiai: f32 = clamp(1f, 1, 1);
  var clamp_faiaf: f32 = clamp(1f, 1, 1.0);
//var clamp_faii: f32 = clamp(1f, 1, 1i); error
  var clamp_faif: f32 = clamp(1f, 1, 1f);
  var clamp_fafai: f32 = clamp(1f, 1.0, 1);
  var clamp_fafaf: f32 = clamp(1f, 1.0, 1.0);
//var clamp_fafi: f32 = clamp(1f, 1.0, 1i); error
  var clamp_faff: f32 = clamp(1f, 1.0, 1f);
//var clamp_fiai: f32 = clamp(1f, 1i, 1); error
//var clamp_fiaf: f32 = clamp(1f, 1i, 1.0); error
//var clamp_fii: f32 = clamp(1f, 1i, 1i); error
//var clamp_fif: f32 = clamp(1f, 1i, 1f); error
  var clamp_ffai: f32 = clamp(1f, 1f, 1);
  var clamp_ffaf: f32 = clamp(1f, 1f, 1.0);
//var clamp_ffi: f32 = clamp(1f, 1f, 1i); error
  var clamp_fff: f32 = clamp(1f, 1f, 1f);
  // end


  var min_aiai: i32 = min(1,   1);
  var min_aiaf: f32 = min(1,   1.0);
  var min_aii:  i32 = min(1,   1i);
  var min_aif:  f32 = min(1,   1f);
  var min_afai: f32 = min(1.0, 1);
  var min_afaf: f32 = min(1.0, 1.0);
//var min_afi:  f32 = min(1.0, 1i); error
  var min_aff:  f32 = min(1.0, 1f);
  var min_iai:  i32 = min(1i,  1);
//var min_iaf:  f32 = min(1i,  1.0); error
  var min_ii:   i32 = min(1i,  1i);
//var min_if:   f32 = min(1i,  1f); error
  var min_fai:  f32 = min(1f,  1);
  var min_faf:  f32 = min(1f,  1.0);
//var min_fi:   f32 = min(1f,  1i); error
  var min_ff:   f32 = min(1f,  1f);

  var pow_aiai = pow(1, 1);
  var pow_aiaf = pow(1, 1.0);
  var pow_aif = pow(1, 1f);
  var pow_afai = pow(1.0, 1);
  var pow_afaf = pow(1.0, 1.0);
  var pow_aff = pow(1.0, 1f);
  var pow_fai = pow(1f, 1);
  var pow_faf = pow(1f, 1.0);
  var pow_ff = pow(1f, 1f);
}
