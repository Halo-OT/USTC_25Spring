#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>  // 添加文件操作头文件

using namespace std;

// 向量类型定义
typedef vector<double> Vector;

// 目标函数 f(x1, x2) = 100(x2 - x1^2)^2 + (1 - x1)^2
double objective_function(const Vector& x) {
    double x1 = x[0], x2 = x[1];
    return 100 * pow(x2 - x1*x1, 2) + pow(1 - x1, 2);
}

// 计算梯度
Vector gradient(const Vector& x) {
    double x1 = x[0], x2 = x[1];
    Vector grad(2);
    grad[0] = -400 * x1 * (x2 - x1*x1) - 2 * (1 - x1);  // ∂f/∂x1
    grad[1] = 200 * (x2 - x1*x1);                        // ∂f/∂x2
    return grad;
}

// 计算Hessian矩阵
vector<Vector> hessian(const Vector& x) {
    double x1 = x[0], x2 = x[1];
    vector<Vector> H(2, Vector(2));
    H[0][0] = -400 * (x2 - 3*x1*x1) + 2;   // ∂²f/∂x1²
    H[0][1] = H[1][0] = -400 * x1;         // ∂²f/∂x1∂x2
    H[1][1] = 200;                         // ∂²f/∂x2²
    return H;
}

// 计算向量的范数
double norm(const Vector& v) {
    double sum = 0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

// 向量加法
Vector vector_add(const Vector& a, const Vector& b) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

// 向量数乘
Vector vector_scale(const Vector& v, double scalar) {
    Vector result(v.size());
    for (int i = 0; i < v.size(); i++) {
        result[i] = v[i] * scalar;
    }
    return result;
}

// 求解线性方程组 Hx = b（使用高斯消元法）
Vector solve_linear_system(vector<Vector> A, Vector b) {
    int n = A.size();
    
    // 前向消元
    for (int i = 0; i < n; i++) {
        // 找主元
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(A[k][i]) > abs(A[max_row][i])) {
                max_row = k;
            }
        }
        swap(A[i], A[max_row]);
        swap(b[i], b[max_row]);
        
        // 消元
        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // 回代
    Vector x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
    
    return x;
}

// 一维搜索（简单的黄金分割法）
double golden_section_search(const Vector& x, const Vector& direction, double tol = 1e-6) {
    double a = 0, b = 1;
    double phi = (1 + sqrt(5)) / 2;
    double resphi = 2 - phi;
    
    double x1 = a + resphi * (b - a);
    double x2 = a + (1 - resphi) * (b - a);
    
    Vector temp1 = vector_add(x, vector_scale(direction, x1));
    Vector temp2 = vector_add(x, vector_scale(direction, x2));
    double f1 = objective_function(temp1);
    double f2 = objective_function(temp2);
    
    while (abs(b - a) > tol) {
        if (f1 > f2) {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + (1 - resphi) * (b - a);
            temp2 = vector_add(x, vector_scale(direction, x2));
            f2 = objective_function(temp2);
        } else {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + resphi * (b - a);
            temp1 = vector_add(x, vector_scale(direction, x1));
            f1 = objective_function(temp1);
        }
    }
    
    return (a + b) / 2;
}

// 最速下降法
void steepest_descent(Vector x, ofstream& outfile, double tolerance = 1e-4) {
    outfile << "最速下降法:" << endl;
    outfile << fixed << setprecision(15);
    
    int iteration = 0;
    Vector grad = gradient(x);
    
    while (norm(grad) > tolerance) {
        iteration++;
        double f_val = objective_function(x);
        
        outfile << "第" << iteration << "次迭代 f(x_i) = " << f_val 
                << ", x_1 = " << x[0] << ", x_2 = " << x[1] << endl;
        
        // 计算搜索方向（负梯度方向）
        Vector direction = vector_scale(grad, -1.0);
        
        // 一维搜索确定步长
        double alpha = golden_section_search(x, direction);
        
        // 更新x
        x = vector_add(x, vector_scale(direction, alpha));
        grad = gradient(x);
        
        if (iteration > 10000) {
            outfile << "达到最大迭代次数" << endl;
            break;
        }
    }
    
    outfile << "收敛结果: f(x) = " << objective_function(x) 
            << ", x_1 = " << x[0] << ", x_2 = " << x[1] << endl;
    outfile << "总迭代次数: " << iteration << endl << endl;
}

// 牛顿法
void newton_method(Vector x, ofstream& outfile, double tolerance = 1e-4) {
    outfile << "牛顿法:" << endl;
    outfile << fixed << setprecision(15);
    
    int iteration = 0;
    Vector grad = gradient(x);
    
    while (norm(grad) > tolerance) {
        iteration++;
        double f_val = objective_function(x);
        
        outfile << "第" << iteration << "次迭代 f(x_i) = " << f_val 
                << ", x_1 = " << x[0] << ", x_2 = " << x[1] << endl;
        
        // 计算Hessian矩阵
        vector<Vector> H = hessian(x);
        
        // 求解 H * d = -grad
        Vector neg_grad = vector_scale(grad, -1.0);
        Vector direction = solve_linear_system(H, neg_grad);
        
        // 一维搜索确定步长
        double alpha = golden_section_search(x, direction);
        
        // 更新x
        x = vector_add(x, vector_scale(direction, alpha));
        grad = gradient(x);
        
        if (iteration > 1000) {
            outfile << "达到最大迭代次数" << endl;
            break;
        }
    }
    
    outfile << "收敛结果: f(x) = " << objective_function(x) 
            << ", x_1 = " << x[0] << ", x_2 = " << x[1] << endl;
    outfile << "总迭代次数: " << iteration << endl << endl;
}

int main() {
    // 创建输出文件
    ofstream outfile("optimization_results.txt");
    if (!outfile) {
        cerr << "无法创建输出文件!" << endl;
        return 1;
    }
    
    // 初始点
    Vector x0(2);
    x0[0] = 0.0;  // x1
    x0[1] = 0.0;  // x2
    
    outfile << "求解函数 f(x1, x2) = 100(x2 - x1^2)^2 + (1 - x1)^2 的极小值" << endl;
    outfile << "初始点: (" << x0[0] << ", " << x0[1] << ")" << endl;
    outfile << "收敛条件: 梯度范数 < 1.0E-4" << endl << endl;
    
    // 使用最速下降法
    steepest_descent(x0, outfile);
    
    // 使用牛顿法
    newton_method(x0, outfile);
    
    outfile.close();
    cout << "计算完成，结果已保存到 optimization_results.txt 文件中" << endl;
    
    return 0;
}