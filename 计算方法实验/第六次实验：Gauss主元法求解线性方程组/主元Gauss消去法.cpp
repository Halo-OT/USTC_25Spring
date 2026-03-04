#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>

using namespace std;

// 设置数值稳定性阈值
const double epsilon = 1e-10;

int main() {
    ifstream inFile("data Ab.txt");
    if (!inFile) {
        cerr << "无法打开文件 data Ab.txt" << endl;
        return 1;
    }

    // 读取第一行以确定阶数
    string line;
    getline(inFile, line);
    istringstream iss(line);
    vector<double> temp;
    double num;
    while (iss >> num) {
        temp.push_back(num);
    }
    int n = temp.size() - 1; // 阶数 = 每行元素数 - 1（最后一个是 b 的元素）
    inFile.clear();           // 清除流状态
    inFile.seekg(0, ios::beg); // 回到文件开头

    // 动态分配内存存储系数矩阵 A 和常数项 b
    vector<vector<double> > A(n, vector<double>(n));
    vector<double> b(n);

    // 读取文件内容到矩阵 A 和向量 b
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inFile >> A[i][j];
        }
        inFile >> b[i];
    }
    inFile.close();

    // 列主元 Gauss 消去法
    for (int k = 0; k < n; ++k) {
        // 选主元
        int maxRow = k;
        double maxVal = fabs(A[k][k]);

        for (int i = k + 1; i < n; ++i) {
            if (fabs(A[i][k]) > fabs(A[maxRow][k])) {
                maxVal = fabs(A[i][k]);
                maxRow = i;
            }
        }

         // 检查主元是否接近零，如果是，可能是矩阵奇异
         if (maxVal < epsilon) {
            // 检查剩余右侧常数项是否全为零
            bool allZero = true;
            for (int i = k; i < n; i++) {
                if (fabs(b[i]) > epsilon) {
                    allZero = false;
                    break;
                }
            }
            
            if (allZero) {
                cout << "警告：系数矩阵奇异，方程组有无穷多解。" << endl;
            } else {
                cout << "错误：系数矩阵奇异，方程组无解。" << endl;
            }
            return 1;
        }

        // 交换行
        if (maxRow != k) {
            swap(A[k], A[maxRow]);
            swap(b[k], b[maxRow]);
        }
        // 消元
        for (int i = k + 1; i < n; ++i) {
            double m = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] -= m * A[k][j];
            }
            b[i] -= m * b[k];
        }
    }

    // 回代求解
    vector<double> x(n);
    x[n - 1] = b[n - 1] / A[n - 1][n - 1];
    for (int i = n - 2; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }

    // 输出结果，保留15位有效数字
    cout << "输出结果：" << endl;
    for (int i = 0; i < n; ++i) {
        cout << fixed << setprecision(15);
        cout << "x" << i + 1 << " = " << x[i];
        if (i != n - 1) {
            cout << "，";
        }
    }
    cout << endl;


    return 0;
}
