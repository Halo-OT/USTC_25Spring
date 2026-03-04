#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

int main() {
    // 输入数据（x和y的值）
    vector<double> xs;
    xs.push_back(0.25);
    xs.push_back(0.50);
    xs.push_back(0.75);
    xs.push_back(1.00);
    xs.push_back(1.25);
    xs.push_back(1.50);
    xs.push_back(1.75);
    xs.push_back(2.00);
    xs.push_back(2.25);
    xs.push_back(2.50);

    vector<double> ys;
    ys.push_back(1.284);
    ys.push_back(1.648);
    ys.push_back(2.117);
    ys.push_back(2.718);
    ys.push_back(3.427);
    ys.push_back(2.798);
    ys.push_back(3.534);
    ys.push_back(4.456);
    ys.push_back(5.465);
    ys.push_back(5.894);
    
    int n = xs.size();
    if (n != ys.size()) {
        cout << "数据维度不一致！" << endl;
        return -1;
    }

    double sum_sin_sq = 0.0, sum_cos_sq = 0.0;
    double sum_sin_cos = 0.0, sum_sin_y = 0.0, sum_cos_y = 0.0;

    for (int i = 0; i < n; ++i) {
        double x = xs[i];
        double y = ys[i];
        double sinx = sin(x);
        double cosx = cos(x);

        sum_sin_sq += sinx * sinx;
        sum_cos_sq += cosx * cosx;
        sum_sin_cos += sinx * cosx;
        sum_sin_y += y * sinx;
        sum_cos_y += y * cosx;
    }

    double denom = sum_sin_sq * sum_cos_sq - sum_sin_cos * sum_sin_cos;
    if (abs(denom) < 1e-8) {
        cout << "行列式为零，无法求解！" << endl;
        return -1;
    }

    double a = (sum_cos_sq * sum_sin_y - sum_sin_cos * sum_cos_y) / denom;
    double b = (sum_sin_sq * sum_cos_y - sum_sin_cos * sum_sin_y) / denom;

    // 计算均方误差
    double mse = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = xs[i];
        double pred = a * sin(x) + b * cos(x);
        double err = ys[i] - pred;
        mse += err * err;
    }
    mse /= n;
    
    cout << fixed << setprecision(15);
    cout << "a = " << a << ", b = " << b << ", 均方误差 = " << mse << endl;

    return 0;
}
