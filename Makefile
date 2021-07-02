.PHONY: all clean validate-spv validate-msl validate-glsl validate-dot validate-wgsl validate-hlsl
.SECONDARY: boids.metal quad.metal
SNAPSHOTS_BASE_IN=tests/in
SNAPSHOTS_BASE_OUT=tests/out

all:
	cargo fmt
	cargo test --all-features --workspace
	cargo clippy --all-features --workspace -- -D warnings

clean:
	rm *.metal *.air *.metallib *.vert *.frag *.comp *.spv

%.metal: $(SNAPSHOTS_BASE_IN)/%.wgsl $(wildcard src/*.rs src/**/*.rs examples/*.rs)
	cargo run --features wgsl-in,msl-out -- $< $@

%.air: %.metal
	xcrun -sdk macosx metal -c $< -mmacosx-version-min=10.11

%.metallib: %.air
	xcrun -sdk macosx metallib $< -o $@

%.dot: $(SNAPSHOTS_BASE_IN)/%.wgsl $(wildcard src/*.rs src/front/wgsl/*.rs src/back/dot/*.rs bin/naga.rs)
	cargo run --features wgsl-in,dot-out -- $< $@

%.png: %.dot
	dot -Tpng $< -o $@

validate-spv: $(SNAPSHOTS_BASE_OUT)/spv/*.spvasm
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		cat $${file} | spirv-as --target-env vulkan1.0 -o - | spirv-val; \
	done

validate-msl: $(SNAPSHOTS_BASE_OUT)/msl/*.msl
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		cat $${file} | xcrun -sdk macosx metal -mmacosx-version-min=10.11 -x metal - -o /dev/null; \
	done

validate-glsl: $(SNAPSHOTS_BASE_OUT)/glsl/*.glsl
	@set -e && for file in $(SNAPSHOTS_BASE_OUT)/glsl/*.Vertex.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S vert; \
	done
	@set -e && for file in $(SNAPSHOTS_BASE_OUT)/glsl/*.Fragment.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S frag; \
	done
	@set -e && for file in $(SNAPSHOTS_BASE_OUT)/glsl/*.Compute.glsl ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};\
		cat $${file} | glslangValidator --stdin -S comp; \
	done

validate-dot: $(SNAPSHOTS_BASE_OUT)/dot/*.dot
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		cat $${file} | dot -o /dev/null; \
	done

validate-wgsl: $(SNAPSHOTS_BASE_OUT)/wgsl/*.wgsl
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"};	\
		cargo run $${file}; \
	done

validate-hlsl: $(SNAPSHOTS_BASE_OUT)/hlsl/*.hlsl
	@set -e && for file in $^ ; do \
		echo "Validating" $${file#"$(SNAPSHOTS_BASE_OUT)/"}; \
		config="$$(dirname $${file})/$$(basename $${file}).config"; \
		vertex=""\
		fragment="" \
		compute="" \
		. $${config}; \
		DXC_PARAMS="-Wno-parentheses-equality -Zi -Qembed_debug"; \
		[ ! -z "$${vertex}" ] && echo "Vertex Stage:" && dxc $${file} -T $${vertex} -E $${vertex_name} $${DXC_PARAMS} > /dev/null; \
		[ ! -z "$${fragment}" ] && echo "Fragment Stage:" && dxc $${file} -T $${fragment} -E $${fragment_name} $${DXC_PARAMS} > /dev/null; \
		[ ! -z "$${compute}" ] && echo "Compute Stage:" && dxc $${file} -T $${compute} -E $${compute_name} $${DXC_PARAMS} > /dev/null; \
		echo "======================"; \
	done
