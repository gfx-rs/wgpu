static float a = (float)0;

void main_1()
{
    float b = (float)0;
    float c = (float)0;
    float d = (float)0;

    float _e4 = a;
    b = log(_e4 + sqrt(_e4 * _e4 + 1.0));
    float _e6 = a;
    c = log(_e6 + sqrt(_e6 * _e6 - 1.0));
    float _e8 = a;
    d = 0.5 * log((1.0 + _e8) / (1.0 - _e8));
    return;
}

void main()
{
    main_1();
}
