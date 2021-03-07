.PHONY: all clean validate-spv validate-msl validate-glsl validate-dot
.SECONDARY: boids.metal quad.metal
SNAPSHOTS_IN=tests/in
SNAPSHOTS_OUT=tests/out

all:
	cargo fmt
	cargo test --all-features
	cargo clippy --all-features

clean:
	rm *.metal *.air *.metallib *.vert *.frag *.comp *.spv

%.metal: $(SNAPSHOTS_IN)/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --features wgsl-in,msl-out -- $< $@

%.air: %.metal
	xcrun -sdk macosx metal -c $< -mmacosx-version-min=10.11

%.metallib: %.air
	xcrun -sdk macosx metallib $< -o $@

%.dot: $(SNAPSHOTS_IN)/%.wgsl $(wildcard src/*.rs src/front/wgsl/*.rs src/back/dot/*.rs bin/convert.rs)
	cargo run --features wgsl-in,dot-out -- $< $@

%.png: %.dot
	dot -Tpng $< -o $@

validate-spv: $(SNAPSHOTS_OUT)/*.spvasm.snap
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/snapshots__"};	\
		tail -n +5 $${file} | spirv-as --target-env vulkan1.0 -o - | spirv-val; \
	done

validate-msl: $(SNAPSHOTS_OUT)/*.msl.snap
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/snapshots__"};	\
		tail -n +5 $${file} | xcrun -sdk macosx metal -mmacosx-version-min=10.11 -x metal - -o /dev/null; \
	done

validate-glsl: $(SNAPSHOTS_OUT)/*.glsl.snap
	@set -e && for file in $(SNAPSHOTS_OUT)/*-Vertex.glsl.snap ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/snapshots__"};\
		tail -n +5 $${file} | glslangValidator --stdin -S vert; \
	done
	@set -e && for file in $(SNAPSHOTS_OUT)/*-Fragment.glsl.snap ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/snapshots__"};\
		tail -n +5 $${file} | glslangValidator --stdin -S frag; \
	done
	@set -e && for file in $(SNAPSHOTS_OUT)/*-Compute.glsl.snap ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/snapshots__"};\
		tail -n +5 $${file} | glslangValidator --stdin -S comp; \
	done

validate-dot: $(SNAPSHOTS_OUT)/*.dot.snap
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_OUT)/snapshots__"};	\
		tail -n +5 $${file} | dot -o /dev/null; \
	done
