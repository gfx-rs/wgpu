static const bool has_point_light = false;
static const float specular_param = 2.3;
static const float gain = 1.1;
static const float width = 0.0;
static const float depth = 2.3;
static const float height = 4.6;
static const float inferred_f32_ = 2.718;
static const uint auto_conversion = 0u;

static float gain_x_10_ = 11.0;
static float store_override = (float)0;

[numthreads(1, 1, 1)]
void main()
{
    float t = 23.0;
    bool x = (bool)0;
    float gain_x_100_ = (float)0;

    x = true;
    float _e9 = gain_x_10_;
    gain_x_100_ = (_e9 * 10.0);
    store_override = gain;
    return;
}
