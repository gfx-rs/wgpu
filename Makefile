.PHONY: all clean
.SECONDARY: boids.metal quad.metal

all:

clean:
	rm *.metal *.air *.metallib *.vert *.frag *.comp

%.metal: test-data/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --example convert -- $< $@

%.air: %.metal
	xcrun -sdk macosx metal -c $< -mmacosx-version-min=10.11

%.metallib: %.air
	xcrun -sdk macosx metallib $< -o $@

%.vert: test-data/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --example convert --features glsl-out -- $< $@

%.frag: test-data/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --example convert --features glsl-out -- $< $@

%.comp: test-data/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --example convert --features glsl-out -- $< $@

%.glsl_validate: %.vert %.frag %.comp
	glslangValidator $<
