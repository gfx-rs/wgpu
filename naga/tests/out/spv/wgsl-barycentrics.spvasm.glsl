///////////////////////////////////
// Entry point: "fs_main" (frag) //
///////////////////////////////////
#version 460
#extension GL_EXT_fragment_shader_barycentric : require

layout(location = 0) out vec4 _10;

void main()
{
    _10 = vec4(gl_BaryCoordEXT, 1.0);
}


//////////////////////////////////////////////////
// Entry point: "fs_main_no_perspective" (frag) //
//////////////////////////////////////////////////
#version 460
#extension GL_EXT_fragment_shader_barycentric : require

layout(location = 0) out vec4 _20;

void main()
{
    _20 = vec4(gl_BaryCoordNoPerspEXT, 1.0);
}

