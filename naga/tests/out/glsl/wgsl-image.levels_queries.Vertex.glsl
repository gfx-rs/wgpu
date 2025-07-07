#version 430 core
#extension GL_ARB_shader_texture_image_samples : require
uniform sampler2D _group_0_binding_1_vs;

uniform sampler2DArray _group_0_binding_4_vs;

uniform samplerCube _group_0_binding_5_vs;

uniform samplerCubeArray _group_0_binding_6_vs;

uniform sampler3D _group_0_binding_7_vs;

uniform sampler2DMS _group_0_binding_8_vs;


void main() {
    uint num_levels_2d = uint(textureQueryLevels(_group_0_binding_1_vs));
    uint num_layers_2d = uint(textureSize(_group_0_binding_4_vs, 0).z);
    uint num_levels_2d_array = uint(textureQueryLevels(_group_0_binding_4_vs));
    uint num_layers_2d_array = uint(textureSize(_group_0_binding_4_vs, 0).z);
    uint num_levels_cube = uint(textureQueryLevels(_group_0_binding_5_vs));
    uint num_levels_cube_array = uint(textureQueryLevels(_group_0_binding_6_vs));
    uint num_layers_cube = uint(textureSize(_group_0_binding_6_vs, 0).z);
    uint num_levels_3d = uint(textureQueryLevels(_group_0_binding_7_vs));
    uint num_samples_aa = uint(textureSamples(_group_0_binding_8_vs));
    uint sum = (((((((num_layers_2d + num_layers_cube) + num_samples_aa) + num_levels_2d) + num_levels_2d_array) + num_levels_3d) + num_levels_cube) + num_levels_cube_array);
    gl_Position = vec4(float(sum));
    return;
}

