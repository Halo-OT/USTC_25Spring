#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;

int main() {
    double x_values[] = {0.0, 0.5, 1.0, sqrt(2), 10.0, 100.0, 300.0};
    int num_x = sizeof(x_values) / sizeof(x_values[0]);
    const int max_k = 10000000; // 1e7次循环以确保误差小于1e-6

    for (int i = 0; i < num_x; ++i) {
        double x = x_values[i];
        double sum = 0.0;

        for (int k = 1; k <= max_k; ++k) {
            sum += 1.0 / (k * (k + x));
        }

        // 输出时保留足够的小数位
        cout << "x = " << fixed << setprecision(3) << x << ", y = ";
        cout << setprecision(15) << sum << endl;
    }

    return 0;
}