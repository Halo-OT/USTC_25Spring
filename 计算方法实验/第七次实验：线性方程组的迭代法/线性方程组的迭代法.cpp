#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <climits>
#include <iomanip> 

void readData(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    std::ifstream in("data Ab.txt");
    if (!in) {
        std::cerr << "文件打开失败" << std::endl;
        return;
    }
    A.resize(9, std::vector<double>(9));
    b.resize(9);
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            in >> A[i][j];
        }
        in >> b[i]; // 读取每行最后一个元素作为 b 的分量
    }
    in.close();
}

void gaussSeidel(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    int n = 9;
    std::vector<double> x(n, 1.0); // 初值设为 (1,1,...,1)
    std::vector<double> x_new(n);
    int count = 0;
    double tol = 1e-7;
    int max_iter = 1000;
    while (true) {
        count++;
        x_new = x;
        for (int i = 0; i < n; ++i) {
            double s1 = 0, s2 = 0;
            for (int j = 0; j < i; ++j) {
                s1 += A[i][j] * x_new[j];
            }
            for (int j = i + 1; j < n; ++j) {
                s2 += A[i][j] * x[j];
            }
            x_new[i] = (b[i] - s1 - s2) / A[i][i];
        }
        double norm = 0;
        for (int i = 0; i < n; ++i) {
            norm = std::max(norm, std::fabs(x_new[i] - x[i]));
        }
        if (norm < tol) {
            std::cout << "Gauss - Seidel迭代步数：" << count << std::endl;
            std::cout << "解：" << std::endl;
            std::cout << std::fixed << std::setprecision(15); 

            for (int i = 0; i < n; ++i) {
                std::cout << x_new[i] << " ";
            }
            std::cout << std::endl;
            return;
        }
        x = x_new;
        if (count > max_iter) {
            std::cout << "Gauss - Seidel不收敛" << std::endl;
            return;
        }
    }
}

void sor(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    int n = 9;
    double best_omega;
    int min_steps = INT_MAX;
    std::vector<double> best_solution(n);
    for (int i = 1; i <= 100; ++i) { // 遍历 ω = i / 50（i 从 1 到 100，即 ω 从 0.02 到 2.0）
        double omega = (double)i / 50;
        std::vector<double> x(n, 1.0); // 初值设为 (1,1,...,1)
        std::vector<double> x_new(n);
        int count = 0;
        double tol = 1e-7;
        int max_iter = 1000;
        bool converged = true;
        while (true) {
            count++;
            x_new = x;
            for (int j = 0; j < n; ++j) {
                double s1 = 0, s2 = 0;
                for (int k = 0; k < j; ++k) {
                    s1 += A[j][k] * x_new[k];
                }
                for (int k = j + 1; k < n; ++k) {
                    s2 += A[j][k] * x[k];
                }
                double temp = (b[j] - s1 - s2) / A[j][j];
                x_new[j] = x[j] + omega * (temp - x[j]);
            }
            double norm = 0;
            for (int j = 0; j < n; ++j) {
                norm = std::max(norm, std::fabs(x_new[j] - x[j]));
            }
            if (norm < tol) {
                std::cout << "ω=" << omega << " 迭代步数：" << count << std::endl;
                if (count < min_steps) {
                    min_steps = count;
                    best_omega = omega;
                    best_solution = x_new;
                }
                converged = true;
                break;
            }
            x = x_new;
            if (count > max_iter) {
                std::cout << "ω=" << omega << " 不收敛" << std::endl;
                converged = false;
                break;
            }
        }
    }
    if (min_steps != INT_MAX) {
        // 正常精度输出ω值
        std::cout << "最佳松弛因子ω=" << best_omega << " 对应的解：" << std::endl;
        // 设置高精度后再输出解
        std::cout << std::fixed << std::setprecision(15);
        for (int i = 0; i < n; ++i) {
            std::cout << best_solution[i] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<std::vector<double>> A;
    std::vector<double> b;
    readData(A, b);
    gaussSeidel(A, b);
    sor(A, b);
    return 0;
}
