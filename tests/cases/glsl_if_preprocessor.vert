#version 460 core

#define TEST 3
#define TEST_EXPR 2 && 2

#if TEST_EXPR - 2 == 0
#error 0
#elif TEST_EXPR - 2 == 1
#error 1
#elif TEST_EXPR - 2 == 2
#error 2
#else
#error You shouldn't do that
#endif
