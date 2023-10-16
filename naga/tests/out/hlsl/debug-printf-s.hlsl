void main_1()
{
    printf("%d",42);
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    main_1();
}
