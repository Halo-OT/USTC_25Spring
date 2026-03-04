#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
using namespace std;

double f(double x, double y) {
    return -x*x * y * y;
}

// 精确解 y(x)=3/(1 + x^3)
double exact(double x) {
    return 3.0 / (1.0 + x*x*x);
}

// 4阶 Runge-Kutta 单步
double rungeKuttaStep(double x, double y, double h) {
    double k1 = f(x, y);
    double k2 = f(x + 0.5*h, y + 0.5*h*k1);
    double k3 = f(x + 0.5*h, y + 0.5*h*k2);
    double k4 = f(x + h, y + h*k3);
    return y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4);
}

// 4步隐式 Adams 校正（预测用四阶显式Adams公式）
double adamsMoulton4(const vector<double>& xs, const vector<double>& ys, double h) {
    int n = xs.size();
    
    // 预测 y_{n+1} 使用四阶显式Adams公式
    // y_{n+1} = y_n + h/24 * [55f(x_n,y_n) - 59f(x_{n-1},y_{n-1}) + 37f(x_{n-2},y_{n-2}) - 9f(x_{n-3},y_{n-3})]
    double f_nm3 = f(xs[n-4], ys[n-4]); // f(x_{n-3}, y_{n-3})
    double f_nm2 = f(xs[n-3], ys[n-3]); // f(x_{n-2}, y_{n-2})
    double f_nm1 = f(xs[n-2], ys[n-2]); // f(x_{n-1}, y_{n-1})
    double f_n   = f(xs[n-1], ys[n-1]); // f(x_n, y_n)
    
    double y_pred = ys[n-1] + h/24.0 * (55*f_n - 59*f_nm1 + 37*f_nm2 - 9*f_nm3);
    double x_new = xs[n-1] + h;
    
    // 校正: y_{n+1} = y_n + h/24*(9f_{n+1}+19f_n-5f_{n-1}+f_{n-2})
    double f_np1 = f(x_new, y_pred);
    return ys[n-1] + h/24.0*(9*f_np1 + 19*f_n - 5*f_nm1 + f_nm2);
}

int main() {
    double x0 = 0.0, y0 = 3.0;
    double x_end = 1.5;
    
    // 使用C++98/03兼容的初始化方式
    vector<double> hs;
    hs.push_back(0.1);
    hs.push_back(0.1/2);
    hs.push_back(0.1/4);
    hs.push_back(0.1/8);

    cout << fixed << setprecision(15);
    
    // 输出RK4结果
    cout << "四阶Runge-Kutta公式的误差和误差阶:" << endl;
    double prev_err = 0;
    
    // 使用传统for循环替代范围for
    for (size_t i = 0; i < hs.size(); ++i) {
        double h = hs[i];
        int steps = int((x_end - x0)/h + 0.5);
        double x = x0, y = y0;
        for (int j = 0; j < steps; ++j) {
            y = rungeKuttaStep(x, y, h);
            x += h;
        }
        double err = fabs(y - exact(x_end));
        double order = (prev_err > 0 ? log(prev_err/err)/log(2.0) : 0.0);
        cout << "h = " << h << ", err = " << err << ", ok = " << order << endl;
        prev_err = err;
    }
    
    cout << endl;
    
    // 输出隐式Adams结果
    cout << "四阶隐式Adams公式的误差和误差阶:" << endl;
    prev_err = 0;
    
    for (size_t i = 0; i < hs.size(); ++i) {
        double h = hs[i];
        int steps = int((x_end-x0)/h + 0.5);
        vector<double> xs(4), ys(4);
        // 用 RK4 预启动四个值
        xs[0]=x0; ys[0]=y0;
        for(int j=1; j<4; j++){
            xs[j] = xs[j-1] + h;
            ys[j] = rungeKuttaStep(xs[j-1], ys[j-1], h);
        }
        // 继续用隐式 Adams
        for(int j=4; j<=steps; j++){
            double x_new = x0 + j*h;
            // 取最后4个点
            vector<double> seg_x(xs.end()-4, xs.end());
            vector<double> seg_y(ys.end()-4, ys.end());
            double y_new = adamsMoulton4(seg_x, seg_y, h);
            xs.push_back(x_new);
            ys.push_back(y_new);
        }
        double y_end = ys.back();
        double err = fabs(y_end - exact(x_end));
        double order = (prev_err>0 ? log(prev_err/err)/log(2.0) : 0.0);
        cout << "h = " << h << ", err = " << err << ", ok = " << order << endl;
        prev_err = err;
    }

    return 0;
}