#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <utility>
#include <iomanip>

using namespace std;

// 定义函数 f(x) = x³/3 - x
double f(double x) {
    return (x * x * x) / 3.0 - x;
}

// 定义导数 f'(x) = x² - 1
double df(double x) {
    return x * x - 1.0;
}

// Newton 迭代法
template<typename Func, typename Deriv>
double newton(Func f, Deriv df, double x0, double eps, int& steps, vector<double>& iterations) {
    const int MAX_ITER = 1000; // 最大迭代次数
    double x_prev = x0;
    iterations.clear();
    iterations.push_back(x_prev);
    steps = 0;

    for (steps = 0; steps <= MAX_ITER; ++steps) {
        double fx = f(x_prev);
        double dfx = df(x_prev);
        if (dfx == 0) {
            return numeric_limits<double>::quiet_NaN(); // 导数为零
        }
        double x_new = x_prev - fx / dfx;
        iterations.push_back(x_new);

        // 检查终止条件：x的变化量
        if (abs(x_new - x_prev) < eps) {
            return x_new;
        }

        x_prev = x_new;
    }

    // 超过最大迭代次数
    return numeric_limits<double>::quiet_NaN();
}


// 弦截法
template<typename Func>
double secant(Func f, double x0, double x1, double eps, int& steps, vector<double>& iterations) {
    const int MAX_ITER = 1000; // 最大迭代次数
    double xn_prev = x0;
    double xn = x1;
    steps = 0;
    iterations.clear();
    iterations.push_back(xn_prev);
    iterations.push_back(xn);
    
    double f_prev = f(xn_prev);
    double f_curr = f(xn);
    
    while (steps < MAX_ITER) {
        steps++; // 进入循环即增加步数
        double denominator = f_curr - f_prev;
        if (denominator == 0) {
            return numeric_limits<double>::quiet_NaN(); // 除零错误
        }
        double delta = f_curr * (xn - xn_prev) / denominator;
        double xn_next = xn - delta;
        double f_next = f(xn_next);
        iterations.push_back(xn_next);
        
        // 检查终止条件：x变化量或函数值
        if (abs(xn_next - xn) < eps || abs(f_next) < eps) {
            return xn_next;
        }
        
        // 更新迭代变量
        xn_prev = xn;
        xn = xn_next;
        f_prev = f_curr;
        f_curr = f_next;
    }
    
    // 超过最大迭代次数
    return numeric_limits<double>::quiet_NaN();
}

// 收敛阶计算函数
/*
double calculateConvergenceOrder(const vector<double>& iterations) {
    if (iterations.size() < 4) {
        return numeric_limits<double>::quiet_NaN();
    }
    
    int n = iterations.size();
    // 使用已知真实解，对于本例 f(x) = x³/3 - x，解为 x = 0 或 x = ±√3
    double x_star;
    double last_val = iterations[n-1];
    
    // 确定最接近哪个解
    if (abs(last_val) < 1.0) {
        x_star = 0.0;
    } else if (last_val > 0) {
        x_star = sqrt(3.0);  // 精确值为√3
    } else {
        x_star = -sqrt(3.0);
    }
    
    vector<double> errors;
    for (size_t i = 0; i < n; ++i) {
        errors.push_back(abs(iterations[i] - x_star));
    }
    
    // 只使用收敛后期的几次迭代（最后的1/3部分）
    int start_idx = max(0, n - n/3 - 3);
    
    double sum_p = 0.0;
    int valid_count = 0;
    
    for (size_t i = start_idx; i < n-3; ++i) {
        double e_n = errors[i];
        double e_n1 = errors[i+1];
        double e_n2 = errors[i+2];
        
        // 添加更严格的有效性检查
        if (e_n > e_n1 && e_n1 > e_n2 && e_n1/e_n2 > 1.01 && e_n/e_n1 > 1.01) {
            double p = log(e_n1/e_n2) / log(e_n/e_n1);
            if (!isnan(p) && !isinf(p) && p > 0 && p < 10) {  // 合理范围检查
                sum_p += p;
                valid_count++;
            }
        }
    }
    
    return valid_count > 0 ? sum_p / valid_count : numeric_limits<double>::quiet_NaN();
}
*/


int main() {
    const double eps = 1e-8;
    
    // 使用push_back初始化向量
    vector<double> newton_x0;
    newton_x0.push_back(0.1);
    newton_x0.push_back(0.2);
    newton_x0.push_back(0.9);
    newton_x0.push_back(9.0);
    
    // 使用push_back初始化向量
    vector<pair<double, double> > secant_pairs; // 注意此处尖括号间有空格
    secant_pairs.push_back(make_pair(-0.1, 0.1));
    secant_pairs.push_back(make_pair(-0.2, 0.2));
    secant_pairs.push_back(make_pair(-2.0, 0.9));
    secant_pairs.push_back(make_pair(0.9, 9.0));

    // 设置输出精度
    cout << fixed << setprecision(15);

    // Newton 迭代法测试
    cout << "Newton 方法结果：" << endl;
    for (size_t i = 0; i < newton_x0.size(); ++i) {
        double x0 = newton_x0[i];
        int steps = 0;
        vector<double> iterations;
        double root = newton(f, df, x0, eps, steps, iterations);
        // double order = calculateConvergenceOrder(iterations);
        cout << "初值 = " << x0 << ", 根 = " << root << ", 迭代步数 = " << steps << endl;
        // cout << ", 收敛阶 ≈ " << (isnan(order) ? "无法计算" : to_string(order)) << endl;
    }

    // 弦截法测试
    cout << "\n弦截法结果：" << endl;
    for (size_t i = 0; i < secant_pairs.size(); ++i) {
        double x0 = secant_pairs[i].first;
        double x1 = secant_pairs[i].second;
        int steps = 0;
        vector<double> iterations;
        double root = secant(f, x0, x1, eps, steps, iterations);
        // double order = calculateConvergenceOrder(iterations);
        cout << "初值 = " << x0 << ", " << x1 << ", 根 = " << root << ", 迭代步数 = " << steps << endl;
        // cout << ", 收敛阶 ≈ " << (isnan(order) ? "无法计算" : to_string(order)) ;
    }

    return 0;
}