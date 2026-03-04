#include <iostream>
#include <cmath>
#include <iomanip> // 引入头文件用于设置输出精度
using namespace std;

double f(double x) {
    return sin(x);
}

double exact_integral() {
    return -cos(5.0) + cos(1.0);
}

double trapezoidal(int n) {
    double h = 4.0 / n;
    double sum = f(1.0) + f(5.0);
    for (int k = 1; k < n; k++) {
        sum += 2.0 * f(1.0 + k * h);
    }
    return h / 2.0 * sum;
}

double simpson(int n) {
    double h = 4.0 / n;
    double sum_odd = 0.0, sum_even = 0.0;
    for (int k = 1; k < n; k++) {
        double x = 1.0 + k * h;
        if (k % 2 == 1) {
            sum_odd += f(x);
        } else {
            sum_even += f(x);
        }
    }
    return h / 3.0 * (f(1.0) + 4.0 * sum_odd + 2.0 * sum_even + f(5.0));
}

void print_results(double (*method)(int), string method_name) {
    double exact = exact_integral();
    double prev_error = 0.0;
    for (int l = 1; l <= 12; l++) {
        int n = 1 << l; // 计算2^l
        double approx = method(n);
        double error = fabs(exact - approx);
        double order = 0.0;
        if (l > 1) {
            order = log(prev_error / error) / log(2.0);
        }
        cout << method_name << " l=" << l << ": ";
        cout << fixed << setprecision(15); // 设置输出保留15位小数
        cout << "数值积分值: " << approx << ", 误差: " << error;
        if (l > 1) {
            cout << ", 误差阶: " << order;
        }
        cout << endl;
        prev_error = error;
    }
}

int main() {
    print_results(trapezoidal, "复化梯形公式");
    print_results(simpson, "复化Simpson公式");
    return 0;
}
