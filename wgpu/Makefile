# This Makefile generates SPIR-V shaders from GLSL shaders in the examples.

shader_compiler = glslangValidator

# All input shaders.
glsls = $(wildcard examples/*/*.vert examples/*/*.frag examples/*/*.comp)

# All SPIR-V targets.
spirvs = $(addsuffix .spv,$(glsls))

.PHONY: default
default: $(spirvs)

# Rule for making a SPIR-V target.
$(spirvs): %.spv: %
	$(shader_compiler) -V $< -o $@

.PHONY: clean
clean:
	rm -f $(spirvs)
