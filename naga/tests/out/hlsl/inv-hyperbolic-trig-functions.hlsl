static float a = (float)0;

void main_1()
{
    float b = (float)0;
    float c = (float)0;
    float d = (float)0;

    float _expr4 = a;
    b = log(_expr4 + sqrt(_expr4 * _expr4 + 1.0));
    float _expr6 = a;
    c = log(_expr6 + sqrt(_expr6 * _expr6 - 1.0));
    float _expr8 = a;
    d = 0.5 * log((1.0 + _expr8) / (1.0 - _expr8));
    return;
}

void main()
{
    main_1();
}
