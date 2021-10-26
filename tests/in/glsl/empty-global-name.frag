layout(set = 1, binding = 1) uniform TextureData {
    vec4 material;
};

void main() {
	vec2 coords = vec2(material.xy);
}
