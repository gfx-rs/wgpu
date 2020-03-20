.PHONY: all clean
.SECONDARY: boids.metal quad.metal

all:

clean:
	rm *.metal *.air *.metallib

%.metal: test-data/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --example convert -- $< $@

%.air: %.metal
	xcrun -sdk macosx metal -c $< -mmacosx-version-min=10.11

%.metallib: %.air
	xcrun -sdk macosx metallib $< -o $@
