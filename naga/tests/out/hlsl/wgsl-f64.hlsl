static const double k = 2.0L;

static double v = 1.0L;

double ZeroValuedouble() {
    return (double)0;
}

[numthreads(1, 1, 1)]
void main()
{
    double x = (double)0;
    double z = (double)0;
    double w = (double)0;
    double local = (double)0;

    x = 6.0L;
    double _e2 = x;
    z = ZeroValuedouble();
    w = -1.0L;
    double _e5 = (30.0L + 400.0L);
    z = (_e5 + 5.0L);
    local = (((_e2 + _e5) + k) + 5.0L);
    return;
}
