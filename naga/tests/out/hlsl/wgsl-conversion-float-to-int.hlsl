static const half MIN_F16_ = -65504.0h;
static const half MAX_F16_ = 65504.0h;
static const float MIN_F32_ = -3.4028235e38;
static const float MAX_F32_ = 3.4028235e38;
static const double MIN_F64_ = -1.7976931348623157e308L;
static const double MAX_F64_ = 1.7976931348623157e308L;

void test_const_eval()
{
    int min_f16_to_i32_ = int(-65504);
    int max_f16_to_i32_ = int(65504);
    uint min_f16_to_u32_ = 0u;
    uint max_f16_to_u32_ = 65504u;
    int64_t min_f16_to_i64_ = -65504L;
    int64_t max_f16_to_i64_ = 65504L;
    uint64_t min_f16_to_u64_ = 0uL;
    uint64_t max_f16_to_u64_ = 65504uL;
    int min_f32_to_i32_ = int(-2147483647 - 1);
    int max_f32_to_i32_ = int(2147483520);
    uint min_f32_to_u32_ = 0u;
    uint max_f32_to_u32_ = 4294967040u;
    int64_t min_f32_to_i64_ = (-9223372036854775807L - 1L);
    int64_t max_f32_to_i64_ = 9223371487098961920L;
    uint64_t min_f32_to_u64_ = 0uL;
    uint64_t max_f32_to_u64_ = 18446742974197923840uL;
    int64_t min_f64_to_i64_ = (-9223372036854775807L - 1L);
    int64_t max_f64_to_i64_ = 9223372036854774784L;
    uint64_t min_f64_to_u64_ = 0uL;
    uint64_t max_f64_to_u64_ = 18446744073709549568uL;
    int min_abstract_float_to_i32_ = int(-2147483647 - 1);
    int max_abstract_float_to_i32_ = int(2147483647);
    uint min_abstract_float_to_u32_ = 0u;
    uint max_abstract_float_to_u32_ = 4294967295u;
    int64_t min_abstract_float_to_i64_ = (-9223372036854775807L - 1L);
    int64_t max_abstract_float_to_i64_ = 9223372036854774784L;
    uint64_t min_abstract_float_to_u64_ = 0uL;
    uint64_t max_abstract_float_to_u64_ = 18446744073709549568uL;

    return;
}

int naga_f2i32(half value) {
    return int(clamp(value, -65504.0h, 65504.0h));
}

int test_f16_to_i32_(half f)
{
    return naga_f2i32(f);
}

uint naga_f2u32(half value) {
    return uint(clamp(value, 0.0h, 65504.0h));
}

uint test_f16_to_u32_(half f_1)
{
    return naga_f2u32(f_1);
}

int64_t naga_f2i64(half value) {
    return int64_t(clamp(value, -65504.0h, 65504.0h));
}

int64_t test_f16_to_i64_(half f_2)
{
    return naga_f2i64(f_2);
}

uint64_t naga_f2u64(half value) {
    return uint64_t(clamp(value, 0.0h, 65504.0h));
}

uint64_t test_f16_to_u64_(half f_3)
{
    return naga_f2u64(f_3);
}

int naga_f2i32(float value) {
    return int(clamp(value, -2147483600.0, 2147483500.0));
}

int test_f32_to_i32_(float f_4)
{
    return naga_f2i32(f_4);
}

uint naga_f2u32(float value) {
    return uint(clamp(value, 0.0, 4294967000.0));
}

uint test_f32_to_u32_(float f_5)
{
    return naga_f2u32(f_5);
}

int64_t naga_f2i64(float value) {
    return int64_t(clamp(value, -9.223372e18, 9.2233715e18));
}

int64_t test_f32_to_i64_(float f_6)
{
    return naga_f2i64(f_6);
}

uint64_t naga_f2u64(float value) {
    return uint64_t(clamp(value, 0.0, 1.8446743e19));
}

uint64_t test_f32_to_u64_(float f_7)
{
    return naga_f2u64(f_7);
}

int naga_f2i32(double value) {
    return int(clamp(value, -2147483648.0L, 2147483647.0L));
}

int test_f64_to_i32_(double f_8)
{
    return naga_f2i32(f_8);
}

uint naga_f2u32(double value) {
    return uint(clamp(value, 0.0L, 4294967295.0L));
}

uint test_f64_to_u32_(double f_9)
{
    return naga_f2u32(f_9);
}

int64_t naga_f2i64(double value) {
    return int64_t(clamp(value, -9.223372036854776e18L, 9.223372036854775e18L));
}

int64_t test_f64_to_i64_(double f_10)
{
    return naga_f2i64(f_10);
}

uint64_t naga_f2u64(double value) {
    return uint64_t(clamp(value, 0.0L, 1.844674407370955e19L));
}

uint64_t test_f64_to_u64_(double f_11)
{
    return naga_f2u64(f_11);
}

int2 naga_f2i32(half2 value) {
    return int2(clamp(value, -65504.0h, 65504.0h));
}

int2 test_f16_to_i32_vec(half2 f_12)
{
    return naga_f2i32(f_12);
}

uint2 naga_f2u32(half2 value) {
    return uint2(clamp(value, 0.0h, 65504.0h));
}

uint2 test_f16_to_u32_vec(half2 f_13)
{
    return naga_f2u32(f_13);
}

int64_t2 naga_f2i64(half2 value) {
    return int64_t2(clamp(value, -65504.0h, 65504.0h));
}

int64_t2 test_f16_to_i64_vec(half2 f_14)
{
    return naga_f2i64(f_14);
}

uint64_t2 naga_f2u64(half2 value) {
    return uint64_t2(clamp(value, 0.0h, 65504.0h));
}

uint64_t2 test_f16_to_u64_vec(half2 f_15)
{
    return naga_f2u64(f_15);
}

int2 naga_f2i32(float2 value) {
    return int2(clamp(value, -2147483600.0, 2147483500.0));
}

int2 test_f32_to_i32_vec(float2 f_16)
{
    return naga_f2i32(f_16);
}

uint2 naga_f2u32(float2 value) {
    return uint2(clamp(value, 0.0, 4294967000.0));
}

uint2 test_f32_to_u32_vec(float2 f_17)
{
    return naga_f2u32(f_17);
}

int64_t2 naga_f2i64(float2 value) {
    return int64_t2(clamp(value, -9.223372e18, 9.2233715e18));
}

int64_t2 test_f32_to_i64_vec(float2 f_18)
{
    return naga_f2i64(f_18);
}

uint64_t2 naga_f2u64(float2 value) {
    return uint64_t2(clamp(value, 0.0, 1.8446743e19));
}

uint64_t2 test_f32_to_u64_vec(float2 f_19)
{
    return naga_f2u64(f_19);
}

int2 naga_f2i32(double2 value) {
    return int2(clamp(value, -2147483648.0L, 2147483647.0L));
}

int2 test_f64_to_i32_vec(double2 f_20)
{
    return naga_f2i32(f_20);
}

uint2 naga_f2u32(double2 value) {
    return uint2(clamp(value, 0.0L, 4294967295.0L));
}

uint2 test_f64_to_u32_vec(double2 f_21)
{
    return naga_f2u32(f_21);
}

int64_t2 naga_f2i64(double2 value) {
    return int64_t2(clamp(value, -9.223372036854776e18L, 9.223372036854775e18L));
}

int64_t2 test_f64_to_i64_vec(double2 f_22)
{
    return naga_f2i64(f_22);
}

uint64_t2 naga_f2u64(double2 value) {
    return uint64_t2(clamp(value, 0.0L, 1.844674407370955e19L));
}

uint64_t2 test_f64_to_u64_vec(double2 f_23)
{
    return naga_f2u64(f_23);
}

[numthreads(1, 1, 1)]
void main()
{
    test_const_eval();
    const int _e1 = test_f16_to_i32_(1.0h);
    const uint _e3 = test_f16_to_u32_(1.0h);
    const int64_t _e5 = test_f16_to_i64_(1.0h);
    const uint64_t _e7 = test_f16_to_u64_(1.0h);
    const int _e9 = test_f32_to_i32_(1.0);
    const uint _e11 = test_f32_to_u32_(1.0);
    const int64_t _e13 = test_f32_to_i64_(1.0);
    const uint64_t _e15 = test_f32_to_u64_(1.0);
    const int _e17 = test_f64_to_i32_(1.0L);
    const uint _e19 = test_f64_to_u32_(1.0L);
    const int64_t _e21 = test_f64_to_i64_(1.0L);
    const uint64_t _e23 = test_f64_to_u64_(1.0L);
    const int2 _e27 = test_f16_to_i32_vec(half2(1.0h, 2.0h));
    const uint2 _e31 = test_f16_to_u32_vec(half2(1.0h, 2.0h));
    const int64_t2 _e35 = test_f16_to_i64_vec(half2(1.0h, 2.0h));
    const uint64_t2 _e39 = test_f16_to_u64_vec(half2(1.0h, 2.0h));
    const int2 _e43 = test_f32_to_i32_vec(float2(1.0, 2.0));
    const uint2 _e47 = test_f32_to_u32_vec(float2(1.0, 2.0));
    const int64_t2 _e51 = test_f32_to_i64_vec(float2(1.0, 2.0));
    const uint64_t2 _e55 = test_f32_to_u64_vec(float2(1.0, 2.0));
    const int2 _e59 = test_f64_to_i32_vec(double2(1.0L, 2.0L));
    const uint2 _e63 = test_f64_to_u32_vec(double2(1.0L, 2.0L));
    const int64_t2 _e67 = test_f64_to_i64_vec(double2(1.0L, 2.0L));
    const uint64_t2 _e71 = test_f64_to_u64_vec(double2(1.0L, 2.0L));
    return;
}
