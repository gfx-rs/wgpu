
static float a = (float)0;

void main_1()
{
    float b = (float)0;
    float c = (float)0;
    float d = (float)0;

    float _expr8 = a;
    b = log(_expr8 + sqrt(_expr8 * _expr8 + 1.0));
    float _expr10 = a;
    c = log(_expr10 + sqrt(_expr10 * _expr10 - 1.0));
    float _expr12 = a;
    d = 0.5 * log((1.0 + _expr12) / (1.0 - _expr12));
    return;
}

void main()
{
    main_1();
}
