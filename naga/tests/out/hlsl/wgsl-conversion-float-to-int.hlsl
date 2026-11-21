static const half MIN_F16_ = -65504.0h;
static const half MAX_F16_ = 65504.0h;
static const float MIN_F32_ = -3.4028235e38;
static const float MAX_F32_ = 3.4028235e38;
static const double MIN_F64_ = -1.7976931348623157e308L;
static const double MAX_F64_ = 1.7976931348623157e308L;

int naga_f2i32(half value) {
    return int(clamp(value, -65504.0h, 65504.0h));
}

uint naga_f2u32(half value) {
    return uint(clamp(value, 0.0h, 65504.0h));
}

int64_t naga_f2i64(half value) {
    return int64_t(clamp(value, -65504.0h, 65504.0h));
}

uint64_t naga_f2u64(half value) {
    return uint64_t(clamp(value, 0.0h, 65504.0h));
}

int naga_f2i32(float value) {
    return int(clamp(value, -2147483600.0, 2147483500.0));
}

uint naga_f2u32(float value) {
    return uint(clamp(value, 0.0, 4294967000.0));
}

int64_t naga_f2i64(float value) {
    return int64_t(clamp(value, -9.223372e18, 9.2233715e18));
}

uint64_t naga_f2u64(float value) {
    return uint64_t(clamp(value, 0.0, 1.8446743e19));
}

int naga_f2i32(double value) {
    return int(clamp(value, -2147483648.0L, 2147483647.0L));
}

uint naga_f2u32(double value) {
    return uint(clamp(value, 0.0L, 4294967295.0L));
}

int64_t naga_f2i64(double value) {
    return int64_t(clamp(value, -9.223372036854776e18L, 9.223372036854775e18L));
}

uint64_t naga_f2u64(double value) {
    return uint64_t(clamp(value, 0.0L, 1.844674407370955e19L));
}

int2 naga_f2i32(half2 value) {
    return int2(clamp(value, -65504.0h, 65504.0h));
}

uint2 naga_f2u32(half2 value) {
    return uint2(clamp(value, 0.0h, 65504.0h));
}

int64_t2 naga_f2i64(half2 value) {
    return int64_t2(clamp(value, -65504.0h, 65504.0h));
}

uint64_t2 naga_f2u64(half2 value) {
    return uint64_t2(clamp(value, 0.0h, 65504.0h));
}

int2 naga_f2i32(float2 value) {
    return int2(clamp(value, -2147483600.0, 2147483500.0));
}

uint2 naga_f2u32(float2 value) {
    return uint2(clamp(value, 0.0, 4294967000.0));
}

int64_t2 naga_f2i64(float2 value) {
    return int64_t2(clamp(value, -9.223372e18, 9.2233715e18));
}

uint64_t2 naga_f2u64(float2 value) {
    return uint64_t2(clamp(value, 0.0, 1.8446743e19));
}

int2 naga_f2i32(double2 value) {
    return int2(clamp(value, -2147483648.0L, 2147483647.0L));
}

uint2 naga_f2u32(double2 value) {
    return uint2(clamp(value, 0.0L, 4294967295.0L));
}

int64_t2 naga_f2i64(double2 value) {
    return int64_t2(clamp(value, -9.223372036854776e18L, 9.223372036854775e18L));
}

uint64_t2 naga_f2u64(double2 value) {
    return uint64_t2(clamp(value, 0.0L, 1.844674407370955e19L));
}

[numthreads(1, 1, 1)]
void main()
{
    int min_f16_to_i32_ = (int)0;
    int max_f16_to_i32_ = (int)0;
    uint min_f16_to_u32_ = (uint)0;
    uint max_f16_to_u32_ = (uint)0;
    int64_t min_f16_to_i64_ = (int64_t)0;
    int64_t max_f16_to_i64_ = (int64_t)0;
    uint64_t min_f16_to_u64_ = (uint64_t)0;
    uint64_t max_f16_to_u64_ = (uint64_t)0;
    int min_f32_to_i32_ = (int)0;
    int max_f32_to_i32_ = (int)0;
    uint min_f32_to_u32_ = (uint)0;
    uint max_f32_to_u32_ = (uint)0;
    int64_t min_f32_to_i64_ = (int64_t)0;
    int64_t max_f32_to_i64_ = (int64_t)0;
    uint64_t min_f32_to_u64_ = (uint64_t)0;
    uint64_t max_f32_to_u64_ = (uint64_t)0;
    int64_t min_f64_to_i64_ = (int64_t)0;
    int64_t max_f64_to_i64_ = (int64_t)0;
    uint64_t min_f64_to_u64_ = (uint64_t)0;
    uint64_t max_f64_to_u64_ = (uint64_t)0;
    int min_abstract_float_to_i32_ = (int)0;
    int max_abstract_float_to_i32_ = (int)0;
    uint min_abstract_float_to_u32_ = (uint)0;
    uint max_abstract_float_to_u32_ = (uint)0;
    int64_t min_abstract_float_to_i64_ = (int64_t)0;
    int64_t max_abstract_float_to_i64_ = (int64_t)0;
    uint64_t min_abstract_float_to_u64_ = (uint64_t)0;
    uint64_t max_abstract_float_to_u64_ = (uint64_t)0;
    half f = (half)0;
    int local = (int)0;
    half f_1 = (half)0;
    uint local_1 = (uint)0;
    half f_2 = (half)0;
    int64_t local_2 = (int64_t)0;
    half f_3 = (half)0;
    uint64_t local_3 = (uint64_t)0;
    float f_4 = (float)0;
    int local_4 = (int)0;
    float f_5 = (float)0;
    uint local_5 = (uint)0;
    float f_6 = (float)0;
    int64_t local_6 = (int64_t)0;
    float f_7 = (float)0;
    uint64_t local_7 = (uint64_t)0;
    double f_8 = (double)0;
    int local_8 = (int)0;
    double f_9 = (double)0;
    uint local_9 = (uint)0;
    double f_10 = (double)0;
    int64_t local_10 = (int64_t)0;
    double f_11 = (double)0;
    uint64_t local_11 = (uint64_t)0;
    half2 f_12 = (half2)0;
    int2 local_12 = (int2)0;
    half2 f_13 = (half2)0;
    uint2 local_13 = (uint2)0;
    half2 f_14 = (half2)0;
    int64_t2 local_14 = (int64_t2)0;
    half2 f_15 = (half2)0;
    uint64_t2 local_15 = (uint64_t2)0;
    float2 f_16 = (float2)0;
    int2 local_16 = (int2)0;
    float2 f_17 = (float2)0;
    uint2 local_17 = (uint2)0;
    float2 f_18 = (float2)0;
    int64_t2 local_18 = (int64_t2)0;
    float2 f_19 = (float2)0;
    uint64_t2 local_19 = (uint64_t2)0;
    double2 f_20 = (double2)0;
    int2 local_20 = (int2)0;
    double2 f_21 = (double2)0;
    uint2 local_21 = (uint2)0;
    double2 f_22 = (double2)0;
    int64_t2 local_22 = (int64_t2)0;
    double2 f_23 = (double2)0;
    uint64_t2 local_23 = (uint64_t2)0;

    min_f16_to_i32_ = int(-65504);
    max_f16_to_i32_ = int(65504);
    min_f16_to_u32_ = 0u;
    max_f16_to_u32_ = 65504u;
    min_f16_to_i64_ = -65504LL;
    max_f16_to_i64_ = 65504LL;
    min_f16_to_u64_ = 0ULL;
    max_f16_to_u64_ = 65504ULL;
    min_f32_to_i32_ = int(-2147483647 - 1);
    max_f32_to_i32_ = int(2147483520);
    min_f32_to_u32_ = 0u;
    max_f32_to_u32_ = 4294967040u;
    min_f32_to_i64_ = (-9223372036854775807LL - 1LL);
    max_f32_to_i64_ = 9223371487098961920LL;
    min_f32_to_u64_ = 0ULL;
    max_f32_to_u64_ = 18446742974197923840ULL;
    min_f64_to_i64_ = (-9223372036854775807LL - 1LL);
    max_f64_to_i64_ = 9223372036854774784LL;
    min_f64_to_u64_ = 0ULL;
    max_f64_to_u64_ = 18446744073709549568ULL;
    min_abstract_float_to_i32_ = int(-2147483647 - 1);
    max_abstract_float_to_i32_ = int(2147483647);
    min_abstract_float_to_u32_ = 0u;
    max_abstract_float_to_u32_ = 4294967295u;
    min_abstract_float_to_i64_ = (-9223372036854775807LL - 1LL);
    max_abstract_float_to_i64_ = 9223372036854774784LL;
    min_abstract_float_to_u64_ = 0ULL;
    max_abstract_float_to_u64_ = 18446744073709549568ULL;
    f = 1.0h;
    half _e105 = f;
    local = naga_f2i32(_e105);
    f_1 = 1.0h;
    half _e109 = f_1;
    local_1 = naga_f2u32(_e109);
    f_2 = 1.0h;
    half _e113 = f_2;
    local_2 = naga_f2i64(_e113);
    f_3 = 1.0h;
    half _e117 = f_3;
    local_3 = naga_f2u64(_e117);
    f_4 = 1.0;
    float _e121 = f_4;
    local_4 = naga_f2i32(_e121);
    f_5 = 1.0;
    float _e125 = f_5;
    local_5 = naga_f2u32(_e125);
    f_6 = 1.0;
    float _e129 = f_6;
    local_6 = naga_f2i64(_e129);
    f_7 = 1.0;
    float _e133 = f_7;
    local_7 = naga_f2u64(_e133);
    f_8 = 1.0L;
    double _e137 = f_8;
    local_8 = naga_f2i32(_e137);
    f_9 = 1.0L;
    double _e141 = f_9;
    local_9 = naga_f2u32(_e141);
    f_10 = 1.0L;
    double _e145 = f_10;
    local_10 = naga_f2i64(_e145);
    f_11 = 1.0L;
    double _e149 = f_11;
    local_11 = naga_f2u64(_e149);
    f_12 = half2(1.0h, 2.0h);
    half2 _e153 = f_12;
    local_12 = naga_f2i32(_e153);
    f_13 = half2(1.0h, 2.0h);
    half2 _e157 = f_13;
    local_13 = naga_f2u32(_e157);
    f_14 = half2(1.0h, 2.0h);
    half2 _e161 = f_14;
    local_14 = naga_f2i64(_e161);
    f_15 = half2(1.0h, 2.0h);
    half2 _e165 = f_15;
    local_15 = naga_f2u64(_e165);
    f_16 = float2(1.0, 2.0);
    float2 _e169 = f_16;
    local_16 = naga_f2i32(_e169);
    f_17 = float2(1.0, 2.0);
    float2 _e173 = f_17;
    local_17 = naga_f2u32(_e173);
    f_18 = float2(1.0, 2.0);
    float2 _e177 = f_18;
    local_18 = naga_f2i64(_e177);
    f_19 = float2(1.0, 2.0);
    float2 _e181 = f_19;
    local_19 = naga_f2u64(_e181);
    f_20 = double2(1.0L, 2.0L);
    double2 _e185 = f_20;
    local_20 = naga_f2i32(_e185);
    f_21 = double2(1.0L, 2.0L);
    double2 _e189 = f_21;
    local_21 = naga_f2u32(_e189);
    f_22 = double2(1.0L, 2.0L);
    double2 _e193 = f_22;
    local_22 = naga_f2i64(_e193);
    f_23 = double2(1.0L, 2.0L);
    double2 _e197 = f_23;
    local_23 = naga_f2u64(_e197);
    return;
}
