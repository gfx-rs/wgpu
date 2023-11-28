static const double k = 2.0L;

static double v = 1.0L;

double f(double x)
{
    double z = (double)0;

    double y = (30.0L + 400.0L);
    z = (y + 5.0L);
    return (((x + y) + k) + 5.0L);
}

[numthreads(1, 1, 1)]
void main()
{
    const double _e1 = f(6.0L);
    return;
}
