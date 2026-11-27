struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct Input {
    uint vertex_index : SV_VertexID;
    uint instance_index : SV_InstanceID;
};

struct VertexInput_copy_struct_members {
    uint instance_index_1 : SV_InstanceID;
    uint vertex_index_1 : SV_VertexID;
};

struct VertexInput_pass_struct_to_function {
    uint instance_index_2 : SV_InstanceID;
    uint vertex_index_2 : SV_VertexID;
};

struct VertexInput_copy_parameters {
    uint instance_index_3 : SV_InstanceID;
    uint vertex_index_3 : SV_VertexID;
};

uint sum_indices(Input in_2)
{
    return (in_2.vertex_index + in_2.instance_index);
}

float4 copy_struct_members(VertexInput_copy_struct_members vertexinput_copy_struct_members) : SV_Position
{
    Input in_ = { (_NagaConstants.first_vertex + vertexinput_copy_struct_members.vertex_index_1), (_NagaConstants.first_instance + vertexinput_copy_struct_members.instance_index_1) };
    uint vertex = in_.vertex_index;
    uint instance = in_.instance_index;
    return float4(float(vertex), float(instance), 0.0, 1.0);
}

float4 pass_struct_to_function(VertexInput_pass_struct_to_function vertexinput_pass_struct_to_function) : SV_Position
{
    Input in_1 = { (_NagaConstants.first_vertex + vertexinput_pass_struct_to_function.vertex_index_2), (_NagaConstants.first_instance + vertexinput_pass_struct_to_function.instance_index_2) };
    const uint _e1 = sum_indices(in_1);
    return float4(float(_e1), 0.0, 0.0, 1.0);
}

float4 copy_parameters(VertexInput_copy_parameters vertexinput_copy_parameters) : SV_Position
{
    uint vertex_index = (_NagaConstants.first_vertex + vertexinput_copy_parameters.vertex_index_3);
    uint instance_index = (_NagaConstants.first_instance + vertexinput_copy_parameters.instance_index_3);
    return float4(float(vertex_index), float(instance_index), 0.0, 1.0);
}
