@id(0)    override has_point_light: bool = true;  // Algorithmic control
@id(1200) override specular_param: f32 = 2.3;     // Numeric control
@id(1300) override gain: f32;                     // Must be overridden
          override width: f32 = 0.0;              // Specified at the API level using
                                                  // the name "width".
          override depth: f32;                    // Specified at the API level using
                                                  // the name "depth".
                                                  // Must be overridden.
         // override height = 2 * depth;            // The default value
                                                  // (if not set at the API level),
                                                  // depends on another
                                                  // overridable constant.

override inferred_f32 = 2.718;
