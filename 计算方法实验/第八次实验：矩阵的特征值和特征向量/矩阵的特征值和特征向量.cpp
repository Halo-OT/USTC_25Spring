#include <iostream>
#include <cmath>
#include <iomanip>

const int N = 5;
const double TOL = 1e-4;

// 计算 Frobenius 范数：off-diagonal 部分
double offFrobenius(double A[N][N]) {
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) sum += A[i][j] * A[i][j];
        }
    }
    return std::sqrt(sum);
}

// 在 A 中找到最大的非对角元素位置 (p,q)
void maxOffDiagonal(double A[N][N], int &p, int &q) {
    double maxVal = 0.0;
    p = 0; q = 1;
    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            if (std::fabs(A[i][j]) > maxVal) {
                maxVal = std::fabs(A[i][j]);
                p = i; q = j;
            }
        }
    }
}

int main() {
    // 初始矩阵 A
    double A[N][N] = {
        {3,   2,  5,  4,  6},
        {2,   1,  3, -7,  8},
        {5,   3,  2,  5, -4},
        {4,  -7,  5,  1,  3},
        {6,   8, -4,  3,  8}
    };
    // 特征向量矩阵，初始化为单位矩阵
    double V[N][N] = {0};
    for (int i = 0; i < N; ++i) V[i][i] = 1.0;

    double norm = offFrobenius(A);
    int iter = 0;

    while (norm > TOL) {
        int p, q;
        maxOffDiagonal(A, p, q);
        double app = A[p][p];
        double aqq = A[q][q];
        double apq = A[p][q];

        // 计算旋转参数
        double tau = (aqq - app) / (2.0 * apq);
        double t = (tau >= 0 ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1 + tau*tau));
        double c = 1.0 / std::sqrt(1 + t*t);
        double s = c * t;

        // 更新 A
        A[p][p] = app - t * apq;
        A[q][q] = aqq + t * apq;
        A[p][q] = A[q][p] = 0.0;
        for (int k = 0; k < N; ++k) {
            if (k != p && k != q) {
                double aik = A[k][p];
                double aiq = A[k][q];
                A[k][p] = A[p][k] = c * aik - s * aiq;
                A[k][q] = A[q][k] = c * aiq + s * aik;
            }
        }
        // 更新特征向量矩阵 V
        for (int k = 0; k < N; ++k) {
            double vip = V[k][p];
            double viq = V[k][q];
            V[k][p] = c * vip - s * viq;
            V[k][q] = s * vip + c * viq;
        }

        norm = offFrobenius(A);
        ++iter;
        if (iter > 10000) break; // 安全退出
    }

    // 输出结果
    std::cout << std::fixed << std::setprecision(15);
    for (int i = 0; i < N; ++i) {
        std::cout << "r" << (i+1) << " = " << A[i][i] << "(特征值)" << ", ";
        std::cout << "v" << (i+1) << "(";
        for (int j = 0; j < N; ++j) {
            std::cout << V[j][i] << (j < N-1 ? ", " : "");
            std::cout << "(特征向量)";
        }
        std::cout << ")\n";
    }
    return 0;
}
