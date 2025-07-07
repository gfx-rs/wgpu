#version 460 core

void switchEmpty(int a) {
    switch (a) {}

    return;
}

void switchNoDefault(int a) {
    switch (a) {
        case 0:
            break;
    }

    return;
}

void switchCaseImplConv(uint a) {
    switch (a) {
        case 0:
            break;
    }

    return;
}

void switchNoLastBreak(int a) {
    switch (a) {
        default:
            int b = a;
    }

    return;
}

void main() {
    switchEmpty(1);
    switchNoDefault(2);
    switchCaseImplConv(3);
    switchNoLastBreak(4);
}
