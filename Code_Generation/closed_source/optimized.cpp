#include <iostream>
#include <cmath>
#include <limits>

long long add(long long a, long long b) {
    return a + b;
}

int main() {
    long long a = 1000000000000;
    long long b = 2000000000000;
    std::cout << add(a, b) << std::endl;
    return 0;
}