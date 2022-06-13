# Change Log

## v0.9 (TBD)

- Fix minimal-versions of dependencies ([#1840](https://github.com/gfx-rs/naga/pull/1840)) **@teoxoy**
- Update MSRV to 1.56 ([#1838](https://github.com/gfx-rs/naga/pull/1838)) **@teoxoy**

API

- Rename `TypeFlags` `INTERFACE`/`HOST_SHARED` to `IO_SHARED`/`HOST_SHAREABLE` ([#1872](https://github.com/gfx-rs/naga/pull/1872)) **@jimblandy**
- Expose more error information ([#1827](https://github.com/gfx-rs/naga/pull/1827), [#1937](https://github.com/gfx-rs/naga/pull/1937)) **@jakobhellermann** **@nical** **@jimblandy**
- Do not unconditionally make error output colorful ([#1707](https://github.com/gfx-rs/naga/pull/1707)) **@rhysd**
- Rename `StorageClass` to `AddressSpace` ([#1699](https://github.com/gfx-rs/naga/pull/1699)) **@kvark**
- Add a way to emit errors to a path ([#1640](https://github.com/gfx-rs/naga/pull/1640)) **@laptou**

CLI

- Add `bincode` representation ([#1729](https://github.com/gfx-rs/naga/pull/1729)) **@kvark**
- Include file path in WGSL parse error ([#1708](https://github.com/gfx-rs/naga/pull/1708)) **@rhysd**
- Add `--version` flag ([#1706](https://github.com/gfx-rs/naga/pull/1706)) **@rhysd**
- Support reading input from stdin via `--stdin-file-path` ([#1701](https://github.com/gfx-rs/naga/pull/1701)) **@rhysd**
- Use `panic = "abort"` ([#1597](https://github.com/gfx-rs/naga/pull/1597)) **@jrmuizel**

DOCS

- Standardize some docs ([#1660](https://github.com/gfx-rs/naga/pull/1660)) **@NoelTautges**
- Document `TypeInner::BindingArray` ([#1859](https://github.com/gfx-rs/naga/pull/1859)) **@jimblandy**
- Clarify accepted types for `Expression::AccessIndex` ([#1862](https://github.com/gfx-rs/naga/pull/1862)) **@NoelTautges**
- Document `proc::layouter` ([#1693](https://github.com/gfx-rs/naga/pull/1693)) **@jimblandy**
- Document Naga's promises around validation and panics ([#1828](https://github.com/gfx-rs/naga/pull/1828)) **@jimblandy**
- `FunctionInfo` doc fixes ([#1726](https://github.com/gfx-rs/naga/pull/1726)) **@jimblandy**

VALIDATOR

- Forbid returning pointers and atomics from functions ([#911](https://github.com/gfx-rs/naga/pull/911)) **@jimblandy**
- Let validation check for more unsupported builtins ([#1962](https://github.com/gfx-rs/naga/pull/1962)) **@jimblandy**
- Fix `Capabilities::SAMPLER_NON_UNIFORM_INDEXING` bitflag ([#1915](https://github.com/gfx-rs/naga/pull/1915)) **@cwfitzgerald**
- Properly check that user-defined IO uses IO-shareable types ([#912](https://github.com/gfx-rs/naga/pull/912)) **@jimblandy**
- Validate `ValuePointer` exactly like a `Pointer` to a `Scalar` ([#1875](https://github.com/gfx-rs/naga/pull/1875)) **@jimblandy**
- Reject empty structs ([#1826](https://github.com/gfx-rs/naga/pull/1826)) **@jimblandy**
- Validate uniform address space layout constraints ([#1812](https://github.com/gfx-rs/naga/pull/1812)) **@teoxoy**
- Improve `AddressSpace` related error messages ([#1710](https://github.com/gfx-rs/naga/pull/1710)) **@kvark**

WGSL-IN

Main breaking changes

- Commas to separate struct members (comma after last member is optional)
  - `struct S { a: f32; b: i32; }` -> `struct S { a: f32, b: i32 }`
- Attribute syntax
  - `[[binding(0), group(0)]]` -> `@binding(0) @group(0)`
- Entry point stage attributes
  - `@stage(vertex)` -> `@vertex`
  - `@stage(fragment)` -> `@fragment`
  - `@stage(compute)` -> `@compute`
- Function renames
  - `smoothStep` -> `smoothstep`
  - `findLsb` -> `firstTrailingBit`
  - `findMsb` -> `firstLeadingBit`

Specification Changes (relavant changes have also been applied to the WGSL backend)

- Update number literal format ([#1863](https://github.com/gfx-rs/naga/pull/1863)) **@teoxoy**
- Allow non-ascii characters in identifiers ([#1849](https://github.com/gfx-rs/naga/pull/1849)) **@teoxoy**
- Update reserved keywords ([#1847](https://github.com/gfx-rs/naga/pull/1847), [#1870](https://github.com/gfx-rs/naga/pull/1870), [#1905](https://github.com/gfx-rs/naga/pull/1905)) **@teoxoy** **@Gordon-F**
- Update entry point stage attributes ([#1833](https://github.com/gfx-rs/naga/pull/1833)) **@Gordon-F**
- Make colon in case optional ([#1801](https://github.com/gfx-rs/naga/pull/1801)) **@Gordon-F**
- Rename `smoothStep` to `smoothstep` ([#1800](https://github.com/gfx-rs/naga/pull/1800)) **@Gordon-F**
- Make semicolon after struct declaration optional ([#1791](https://github.com/gfx-rs/naga/pull/1791)) **@stshine**
- Use commas to separate struct members instead of semicolons ([#1773](https://github.com/gfx-rs/naga/pull/1773)) **@Gordon-F**
- Rename `findLsb`/`findMsb` to `firstTrailingBit`/`firstLeadingBit` ([#1735](https://github.com/gfx-rs/naga/pull/1735)) **@kvark**
- Make parenthesis optional for `if` and `switch` statements ([#1725](https://github.com/gfx-rs/naga/pull/1725)) **@Gordon-F**
- Declare attribtues with `@attrib` instead of `[[attrib]]` ([#1676](https://github.com/gfx-rs/naga/pull/1676)) **@kvark**
- Allow non-structure buffer types ([#1682](https://github.com/gfx-rs/naga/pull/1682)) **@kvark**
- Remove `stride` attribute ([#1681](https://github.com/gfx-rs/naga/pull/1681)) **@kvark**

Improvements

- Implement `firstTrailingBit`/`firstLeadingBit` u32 overloads ([#1865](https://github.com/gfx-rs/naga/pull/1865)) **@teoxoy**
- Add error for non-floating-point matrix ([#1917](https://github.com/gfx-rs/naga/pull/1917)) **@grovesNL**
- Implement partial vector & matrix identity constructors ([#1916](https://github.com/gfx-rs/naga/pull/1916)) **@teoxoy**
- Implement phony assignment ([#1866](https://github.com/gfx-rs/naga/pull/1866), [#1869](https://github.com/gfx-rs/naga/pull/1869)) **@teoxoy**
- Fix being able to match `~=` as LogicalOperation ([#1849](https://github.com/gfx-rs/naga/pull/1849)) **@teoxoy**
- Implement Binding Arrays ([#1845](https://github.com/gfx-rs/naga/pull/1845)) **@cwfitzgerald**
- Implement unary vector operators ([#1820](https://github.com/gfx-rs/naga/pull/1820)) **@teoxoy**
- Implement zero value constructors and constructors that infer their type from their parameters ([#1790](https://github.com/gfx-rs/naga/pull/1790)) **@teoxoy**
- Implement invariant attribute ([#1789](https://github.com/gfx-rs/naga/pull/1789), [#1822](https://github.com/gfx-rs/naga/pull/1822)) **@teoxoy** **@jimblandy**
- Implement increment and decrement statements ([#1788](https://github.com/gfx-rs/naga/pull/1788), [#1912](https://github.com/gfx-rs/naga/pull/1912)) **@teoxoy**
- Implement `while` loop ([#1787](https://github.com/gfx-rs/naga/pull/1787)) **@teoxoy**
- Fix array size on globals ([#1717](https://github.com/gfx-rs/naga/pull/1717)) **@jimblandy**
- Implement integer vector overloads for `dot` function ([#1689](https://github.com/gfx-rs/naga/pull/1689)) **@francesco-cattoglio**
- Implement block comments ([#1675](https://github.com/gfx-rs/naga/pull/1675)) **@kocsis1david**
- Implement assignment binary operators ([#1662](https://github.com/gfx-rs/naga/pull/1662)) **@kvark**
- Implement `radians`/`degrees` builtin functions ([#1627](https://github.com/gfx-rs/naga/pull/1627)) **@encounter**
- Implement `findLsb`/`findMsb` builtin functions ([#1473](https://github.com/gfx-rs/naga/pull/1473)) **@fintelia**
- Implement `textureGather`/`textureGatherCompare` builtin functions ([#1596](https://github.com/gfx-rs/naga/pull/1596)) **@kvark**

SPV-IN

- Implement `OpBitReverse` and `OpBitCount` ([#1954](https://github.com/gfx-rs/naga/pull/1954)) **@JCapucho**
- Add `MultiView` to `SUPPORTED_CAPABILITIES` ([#1934](https://github.com/gfx-rs/naga/pull/1934)) **@expenses**
- Translate `OpSMod` and `OpFMod` correctly ([#1867](https://github.com/gfx-rs/naga/pull/1867)) **@teoxoy**
- Error on unsupported `MatrixStride` ([#1805](https://github.com/gfx-rs/naga/pull/1805)) **@teoxoy**
- Align array stride for undecorated arrays ([#1724](https://github.com/gfx-rs/naga/pull/1724)) **@JCapucho**

GLSL-IN

- Fix matrix multiplication check ([#1953](https://github.com/gfx-rs/naga/pull/1953)) **@JCapucho**
- Fix panic (stop emitter in conditional) ([#1952](https://github.com/gfx-rs/naga/pull/1952)) **@JCapucho**
- Translate `mod` fn correctly ([#1867](https://github.com/gfx-rs/naga/pull/1867)) **@teoxoy**
- Make the ternary operator behave as an if ([#1877](https://github.com/gfx-rs/naga/pull/1877)) **@JCapucho**
- Add support for `clamp` function ([#1502](https://github.com/gfx-rs/naga/pull/1502)) **@sjinno**
- Better errors for bad constant expression ([#1501](https://github.com/gfx-rs/naga/pull/1501)) **@sjinno**
- Error on a `matCx2` used with the `std140` layout ([#1806](https://github.com/gfx-rs/naga/pull/1806)) **@teoxoy**
- Allow nested accesses in lhs positions ([#1794](https://github.com/gfx-rs/naga/pull/1794)) **@JCapucho**
- Use forced conversions for vector/matrix constructors ([#1796](https://github.com/gfx-rs/naga/pull/1796)) **@JCapucho**
- Add support for `barrier` function ([#1793](https://github.com/gfx-rs/naga/pull/1793)) **@fintelia**
- Fix panic (resume expression emit after `imageStore`) ([#1795](https://github.com/gfx-rs/naga/pull/1795)) **@JCapucho**
- Allow multiple array specifiers ([#1780](https://github.com/gfx-rs/naga/pull/1780)) **@JCapucho**
- Fix memory qualifiers being inverted ([#1779](https://github.com/gfx-rs/naga/pull/1779)) **@JCapucho**
- Support arrays as input/output types ([#1759](https://github.com/gfx-rs/naga/pull/1759)) **@JCapucho**
- Fix freestanding constructor parsing ([#1758](https://github.com/gfx-rs/naga/pull/1758)) **@JCapucho**
- Fix matrix - scalar operations ([#1757](https://github.com/gfx-rs/naga/pull/1757)) **@JCapucho**
- Fix matrix - matrix division ([#1757](https://github.com/gfx-rs/naga/pull/1757)) **@JCapucho**
- Fix matrix comparisons ([#1757](https://github.com/gfx-rs/naga/pull/1757)) **@JCapucho**
- Add support for `texelFetchOffset` ([#1746](https://github.com/gfx-rs/naga/pull/1746)) **@JCapucho**
- Inject `sampler2DMSArray` builtins on use ([#1737](https://github.com/gfx-rs/naga/pull/1737)) **@JCapucho**
- Inject `samplerCubeArray` builtins on use ([#1736](https://github.com/gfx-rs/naga/pull/1736)) **@JCapucho**
- Add support for image builtin functions ([#1723](https://github.com/gfx-rs/naga/pull/1723)) **@JCapucho**
- Add support for image declarations ([#1723](https://github.com/gfx-rs/naga/pull/1723)) **@JCapucho**
- Texture builtins fixes ([#1719](https://github.com/gfx-rs/naga/pull/1719)) **@JCapucho**
- Type qualifiers rework ([#1713](https://github.com/gfx-rs/naga/pull/1713)) **@JCapucho**
- `texelFetch` accept multisampled textures ([#1715](https://github.com/gfx-rs/naga/pull/1715)) **@JCapucho**
- Fix panic when culling nested block ([#1714](https://github.com/gfx-rs/naga/pull/1714)) **@JCapucho**
- Fix composite constructors ([#1631](https://github.com/gfx-rs/naga/pull/1631)) **@JCapucho**
- Fix using swizzle as out arguments ([#1632](https://github.com/gfx-rs/naga/pull/1632)) **@JCapucho**

SPV-OUT

- Implement `reverseBits` and `countOneBits` ([#1897](https://github.com/gfx-rs/naga/pull/1897)) **@hasali19**
- Use `OpCopyObject` for matrix identity casts ([#1916](https://github.com/gfx-rs/naga/pull/1916)) **@teoxoy**
- Use `OpCopyObject` for bool - bool conversion due to `OpBitcast` not being feasible for booleans ([#1916](https://github.com/gfx-rs/naga/pull/1916)) **@teoxoy**
- Zero init variables in function and private address spaces ([#1871](https://github.com/gfx-rs/naga/pull/1871)) **@teoxoy**
- Use `SRem` instead of `SMod` ([#1867](https://github.com/gfx-rs/naga/pull/1867)) **@teoxoy**
- Add support for integer vector - scalar multiplication ([#1820](https://github.com/gfx-rs/naga/pull/1820)) **@teoxoy**
- Add support for matrix addition and subtraction ([#1820](https://github.com/gfx-rs/naga/pull/1820)) **@teoxoy**
- Emit required decorations on wrapper struct types ([#1815](https://github.com/gfx-rs/naga/pull/1815)) **@jimblandy**
- Decorate array and struct type layouts unconditionally ([#1815](https://github.com/gfx-rs/naga/pull/1815)) **@jimblandy**
- Fix wrong `MatrixStride` for `matCx2` and `mat2xR` ([#1781](https://github.com/gfx-rs/naga/pull/1781)) **@teoxoy**
- Use `OpImageQuerySize` for MS images ([#1742](https://github.com/gfx-rs/naga/pull/1742)) **@JCapucho**

MSL-OUT

- Fix pointers to private or workgroup address spaces possibly being read only ([#1901](https://github.com/gfx-rs/naga/pull/1901)) **@teoxoy**
- Zero init variables in function address space ([#1871](https://github.com/gfx-rs/naga/pull/1871)) **@teoxoy**
- Make binding arrays play nice with bounds checks ([#1855](https://github.com/gfx-rs/naga/pull/1855)) **@cwfitzgerald**
- Permit `invariant` qualifier on vertex shader outputs ([#1821](https://github.com/gfx-rs/naga/pull/1821)) **@jimblandy**
- Fix packed `vec3` stores ([#1816](https://github.com/gfx-rs/naga/pull/1816)) **@teoxoy**
- Actually test push constants to be used ([#1767](https://github.com/gfx-rs/naga/pull/1767)) **@kvark**
- Properly rename entry point arguments for struct members ([#1766](https://github.com/gfx-rs/naga/pull/1766)) **@jimblandy**
- Qualify read-only storage with const ([#1763](https://github.com/gfx-rs/naga/pull/1763)) **@kvark**
- Fix not unary operator for integer scalars ([#1760](https://github.com/gfx-rs/naga/pull/1760)) **@vincentisambart**
- Add bounds checks for `ImageLoad` and `ImageStore` ([#1730](https://github.com/gfx-rs/naga/pull/1730)) **@jimblandy**
- Fix resource bindings for non-structures ([#1718](https://github.com/gfx-rs/naga/pull/1718)) **@kvark**
- Always check whether _buffer_sizes arg is needed ([#1717](https://github.com/gfx-rs/naga/pull/1717)) **@jimblandy**
- WGSL storage address space should always correspond to MSL device address space ([#1711](https://github.com/gfx-rs/naga/pull/1711)) **@wtholliday**
- Mitigation for MSL atomic bounds check ([#1703](https://github.com/gfx-rs/naga/pull/1703)) **@glalonde**

HLSL-OUT

- Fix fallthrough in switch statements ([#1920](https://github.com/gfx-rs/naga/pull/1920)) **@teoxoy**
- Fix missing break statements ([#1919](https://github.com/gfx-rs/naga/pull/1919)) **@teoxoy**
- Fix `countOneBits` and `reverseBits` for signed integers ([#1928](https://github.com/gfx-rs/naga/pull/1928)) **@hasali19**
- Fix array constructor return type ([#1914](https://github.com/gfx-rs/naga/pull/1914)) **@teoxoy**
- Fix hlsl output for writes to scalar/vector storage buffer ([#1903](https://github.com/gfx-rs/naga/pull/1903)) **@hasali19**
- Use `fmod` instead of `%` ([#1867](https://github.com/gfx-rs/naga/pull/1867)) **@teoxoy**
- Use wrapped constructors when loading from storage address space ([#1893](https://github.com/gfx-rs/naga/pull/1893)) **@teoxoy**
- Zero init struct constructor ([#1890](https://github.com/gfx-rs/naga/pull/1890)) **@teoxoy**
- Flesh out matrix handling documentation ([#1850](https://github.com/gfx-rs/naga/pull/1850)) **@jimblandy**
- Emit `row_major` qualifier on matrix uniform globals ([#1846](https://github.com/gfx-rs/naga/pull/1846)) **@jimblandy**
- Fix bool splat ([#1820](https://github.com/gfx-rs/naga/pull/1820)) **@teoxoy**
- Add more padding when necessary ([#1814](https://github.com/gfx-rs/naga/pull/1814)) **@teoxoy**
- Support multidimensional arrays ([#1814](https://github.com/gfx-rs/naga/pull/1814)) **@teoxoy**
- Don't output interpolation modifier if it's the default ([#1809](https://github.com/gfx-rs/naga/pull/1809)) **@NoelTautges**
- Fix `matCx2` translation for uniform buffers ([#1802](https://github.com/gfx-rs/naga/pull/1802)) **@teoxoy**
- Fix modifiers not being written in the vertex output and fragment input structs ([#1789](https://github.com/gfx-rs/naga/pull/1789)) **@teoxoy**
- Fix matrix not being declared as transposed ([#1784](https://github.com/gfx-rs/naga/pull/1784)) **@teoxoy**
- Insert padding between struct members ([#1786](https://github.com/gfx-rs/naga/pull/1786)) **@teoxoy**
- Fix not unary operator for integer scalars ([#1760](https://github.com/gfx-rs/naga/pull/1760)) **@vincentisambart**

GLSL-OUT

- Fix type error for `countOneBits` implementation ([#1897](https://github.com/gfx-rs/naga/pull/1897)) **@hasali19**
- Fix storage format for `Rgba8Unorm` ([#1955](https://github.com/gfx-rs/naga/pull/1955)) **@JCapucho**
- Implement bounds checks for `ImageLoad` ([#1889](https://github.com/gfx-rs/naga/pull/1889)) **@JCapucho**
- Fix feature search in expressions ([#1887](https://github.com/gfx-rs/naga/pull/1887)) **@JCapucho**
- Emit globals of any type ([#1823](https://github.com/gfx-rs/naga/pull/1823)) **@jimblandy**
- Add support for boolean vector `~`, `|` and `&` ops ([#1820](https://github.com/gfx-rs/naga/pull/1820)) **@teoxoy**
- Fix array function arguments ([#1814](https://github.com/gfx-rs/naga/pull/1814)) **@teoxoy**
- Write constant sized array type for uniform ([#1768](https://github.com/gfx-rs/naga/pull/1768)) **@hatoo**
- Texture function fixes ([#1742](https://github.com/gfx-rs/naga/pull/1742)) **@JCapucho**
- Push constants use anonymous uniforms ([#1683](https://github.com/gfx-rs/naga/pull/1683)) **@JCapucho**
- Add support for push constant emulation ([#1672](https://github.com/gfx-rs/naga/pull/1672)) **@JCapucho**
- Skip unsized types if unused ([#1649](https://github.com/gfx-rs/naga/pull/1649)) **@kvark**
- Write struct and array initializers ([#1644](https://github.com/gfx-rs/naga/pull/1644)) **@JCapucho**


## v0.8.5 (2022-01-25)

MSL-OUT

- Make VS-output positions invariant on even more systems ([#1697](https://github.com/gfx-rs/naga/pull/1697)) **@cwfitzgerald**
- Improve support for point primitives ([#1696](https://github.com/gfx-rs/naga/pull/1696)) **@kvark**


## v0.8.4 (2022-01-24)

MSL-OUT

- Make VS-output positions invariant if possible ([#1687](https://github.com/gfx-rs/naga/pull/1687)) **@kvark**

GLSL-OUT

- Fix `floatBitsToUint` spelling ([#1688](https://github.com/gfx-rs/naga/pull/1688)) **@cwfitzgerald**
- Call proper memory barrier functions ([#1680](https://github.com/gfx-rs/naga/pull/1680)) **@francesco-cattoglio**


## v0.8.3 (2022-01-20)

- Don't pin `indexmap` version ([#1666](https://github.com/gfx-rs/naga/pull/1666)) **@a1phyr**

MSL-OUT

- Fix support for point primitives ([#1674](https://github.com/gfx-rs/naga/pull/1674)) **@kvark**

GLSL-OUT

- Fix sampler association ([#1671](https://github.com/gfx-rs/naga/pull/1671)) **@JCapucho**


## v0.8.2 (2022-01-11)

VALIDATOR

- Check structure resource types ([#1639](https://github.com/gfx-rs/naga/pull/1639)) **@kvark**

WGSL-IN

- Improve type mismatch errors ([#1658](https://github.com/gfx-rs/naga/pull/1658)) **@Gordon-F**

SPV-IN

- Implement more sign agnostic operations ([#1651](https://github.com/gfx-rs/naga/pull/1651), [#1650](https://github.com/gfx-rs/naga/pull/1650)) **@JCapucho**

SPV-OUT

- Fix modulo operator (use `OpFRem` instead of `OpFMod`) ([#1653](https://github.com/gfx-rs/naga/pull/1653)) **@JCapucho**

MSL-OUT

- Fix `texture1d` accesses ([#1647](https://github.com/gfx-rs/naga/pull/1647)) **@jimblandy**
- Fix data packing functions ([#1637](https://github.com/gfx-rs/naga/pull/1637)) **@phoekz**


## v0.8.1 (2021-12-29)

API

- Make `WithSpan` clonable ([#1620](https://github.com/gfx-rs/naga/pull/1620)) **@jakobhellermann**

MSL-OUT

- Fix packed vec access ([#1634](https://github.com/gfx-rs/naga/pull/1634)) **@kvark**
- Fix packed float support ([#1630](https://github.com/gfx-rs/naga/pull/1630)) **@kvark**

HLSL-OUT

- Support arrays of matrices ([#1629](https://github.com/gfx-rs/naga/pull/1629)) **@kvark**
- Use `mad` instead of `fma` function ([#1580](https://github.com/gfx-rs/naga/pull/1580)) **@parasyte**

GLSL-OUT

- Fix conflicting names for globals ([#1616](https://github.com/gfx-rs/naga/pull/1616)) **@Gordon-F**
- Fix `fma` function ([#1580](https://github.com/gfx-rs/naga/pull/1580)) **@parasyte**


## v0.8 (2021-12-18)
  - development release for wgpu-0.12
  - lots of fixes in all parts
  - validator:
    - now gated by `validate` feature
    - nicely detailed error messages with spans
  - API:
    - image gather operations
  - WGSL-in:
    - remove `[[block]]` attribute
    - `elseif` is removed in favor of `else if`
  - MSL-out:
    - full out-of-bounds checking

## v0.7.3 (2021-12-14)
  - API:
    - `view_index` builtin
  - GLSL-out:
    - reflect textures without samplers
  - SPV-out:
    - fix incorrect pack/unpack

## v0.7.2 (2021-12-01)
  - validator:
    - check stores for proper pointer class
  - HLSL-out:
    - fix stores into `mat3`
    - respect array strides
  - SPV-out:
    - fix multi-word constants
  - WGSL-in:
    - permit names starting with underscores
  - SPV-in:
    - cull unused builtins
    - support empty debug labels
  - GLSL-in:
    - don't panic on invalid integer operations

## v0.7.1 (2021-10-12)
  - implement casts from and to booleans in the backends

## v0.7 (2021-10-07)
  - development release for wgpu-0.11
  - API:
    - bit extraction and packing functions
    - hyperbolic trigonometry functions
    - validation is gated by a cargo feature
    - `view_index` builtin
    - separate bounds checking policies for locals/buffers/textures
  - IR:
    - types and constants are guaranteed to be unique
  - WGSL-in:
    - new hex literal parser
    - updated list of reserved words
    - rewritten logic for resolving references and pointers
    - `switch` can use unsigned selectors
  - GLSL-in:
    - better support for texture sampling
    - better logic for auto-splatting scalars
  - GLSL-out:
    - fixed storage buffer layout
    - fix module operator
  - HLSL-out:
    - fixed texture queries
  - SPV-in:
    - control flow handling is rewritten from scratch
  - SPV-out:
    - fully covered out-of-bounds checking
    - option to emit point size
    - option to clamp output depth

## v0.6.3 (2021-09-08)
  - Reduced heap allocations when generating WGSL, HLSL, and GLSL
  - WGSL-in:
    - support module-scope `let` type inference
  - SPV-in:
    - fix depth sampling with projection
  - HLSL-out:
    - fix local struct construction
  - GLSL-out:
    - fix `select()` order
  - SPV-out:
    - allow working around Adreno issue with `OpName`

## v0.6.2 (2021-09-01)
  - SPV-out fixes:
    - requested capabilities for 1D and cube images, storage formats
    - handling `break` and `continue` in a `switch` statement
    - avoid generating duplicate `OpTypeImage` types
  - HLSL-out fixes:
    - fix output struct member names
  - MSL-out fixes:
    - fix packing of fields in interface structs
  - GLSL-out fixes:
    - fix non-fallthrough `switch` cases
  - GLSL-in fixes:
    - avoid infinite loop on invalid statements

## v0.6.1 (2021-08-24)
  - HLSL-out fixes:
    - array arguments
    - pointers to array arguments
    - switch statement
    - rewritten interface matching
  - SPV-in fixes:
    - array storage texture stores
    - tracking sampling across function parameters
    - updated petgraph dependencies
  - MSL-out:
    - gradient sampling
  - GLSL-out:
    - modulo operator on floats

## v0.6 (2021-08-18)
  - development release for wgpu-0.10
  - API:
    - atomic types and functions
    - storage access is moved from global variables to the storage class and storage texture type
    - new built-ins: `primitive_index` and `num_workgroups`
    - support for multi-sampled depth images
  - WGSL:
    - `select()` order of true/false is swapped
  - HLSL backend is vastly improved and now usable
  - GLSL frontend is heavily reworked

## v0.5 (2021-06-18)
  - development release for wgpu-0.9
  - API:
    - barriers
    - dynamic indexing of matrices and arrays is only allowed on variables
    - validator now accepts a list of IR capabilities to allow
    - improved documentation
  - Infrastructure:
    - much richer test suite, focused around consuming or emitting WGSL
    - lazy testing on large shader corpuses
    - the binary is moved to a sub-crate "naga-cli"
  - Frontends:
    - GLSL frontend:
      - rewritten from scratch and effectively revived, no longer depends on `pomelo`
      - only supports 440/450/460 versions for now
      - has optional support for codespan messages
    - SPIRV frontend has improved CFG resolution (still with issues unresolved)
    - WGSL got better error messages, workgroup memory support
  - Backends:
    - general: better expression naming and emitting
    - new HLSL backend (in progress)
    - MSL:
      - support `ArraySize` expression
      - better texture sampling instructions
    - GLSL:
      - multisampling on GLES
    - WGSL is vastly improved and now usable

## v0.4.2 (2021-05-28)
  - SPIR-V frontend:
    - fix image stores
    - fix matrix stride check
  - SPIR-V backend:
    - fix auto-deriving the capabilities
  - GLSL backend:
    - support sample interpolation
    - write out swizzled vector accesses

## v0.4.1 (2021-05-14)
  - numerous additions and improvements to SPIR-V frontend:
    - int8, in16, int64
    - null constant initializers for structs and matrices
    - `OpArrayLength`, `OpCopyMemory`, `OpInBoundsAccessChain`, `OpLogicalXxxEqual`
    - outer product
    - fix struct size alignment
    - initialize built-ins with default values
    - fix read-only decorations on struct members
  - fix struct size alignment in WGSL
  - fix `fwidth` in WGSL
  - fix scalars arrays in GLSL backend

## v0.4 (2021-04-29)
  - development release for wgpu-0.8
  - API:
    - expressions are explicitly emitted with `Statement::Emit`
    - entry points have inputs in arguments and outputs in the result type
    - `input`/`output` storage classes are gone, but `push_constant` is added
    - `Interpolation` is moved into `Binding::Location` variant
    - real pointer semantics with required `Expression::Load`
    - `TypeInner::ValuePointer` is added
    - image query expressions are added
    - new `Statement::ImageStore`
    - all function calls are `Statement::Call`
    - `GlobalUse` is moved out into processing
    - `Header` is removed
    - entry points are an array instead of a map
    - new `Swizzle` and `Splat` expressions
    - interpolation qualifiers are extended and required
    - struct member layout is based on the byte offsets
  - Infrastructure:
    - control flow uniformity analysis
    - texture-sampler combination gathering
    - `CallGraph` processor is moved out into `glsl` backend
    - `Interface` is removed, instead the analysis produces `ModuleInfo` with all the derived info
    - validation of statement tree, expressions, and constants
    - code linting is more strict for matches
  - new GraphViz `dot` backend for pretty visualization of the IR
  - Metal support for inlined samplers
  - `convert` example is transformed into the default binary target named `naga`
  - lots of frontend and backend fixes

## v0.3.2 (2021-02-15)
  - fix logical expression types
  - fix _FragDepth_ semantics
  - spv-in:
    - derive block status of structures
  - spv-out:
    - add lots of missing math functions
    - implement discard

## v0.3.1 (2021-01-31)
  - wgsl:
    - support constant array sizes
  - spv-out:
    - fix block decorations on nested structures
    - fix fixed-size arrays
    - fix matrix decorations inside structures
    - implement read-only decorations

## v0.3 (2021-01-30)
  - development release for wgpu-0.7
  - API:
    - math functions
    - type casts
    - updated storage classes
    - updated image sub-types
    - image sampling/loading options
    - storage images
    - interpolation qualifiers
    - early and conservative depth
  - Processors:
    - name manager
    - automatic layout
    - termination analysis
    - validation of types, constants, variables, and entry points

## v0.2 (2020-08-17)
  - development release for wgpu-0.6

## v0.1 (2020-02-26)
  - initial release
