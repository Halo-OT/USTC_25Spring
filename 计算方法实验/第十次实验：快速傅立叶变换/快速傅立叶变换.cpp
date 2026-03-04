#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>

using namespace std;
typedef complex<double> Complex;
const double PI = acos(-1.0);

// 递归实现FFT算法
vector<Complex> FFT(const vector<Complex>& f) {
    int n = f.size();
    
    // 基本情况：如果长度为1，直接返回
    if (n == 1) {
        return f;
    }
    
    // 计算旋转因子
    Complex wn(cos(2*PI/n), -sin(2*PI/n));
    Complex w(1, 0);
    
    // 将偶数和奇数位置的元素分别存储
    vector<Complex> f0(n/2), f1(n/2);
    for (int i = 0; i < n/2; i++) {
        f0[i] = f[2*i];      // 偶数位置
        f1[i] = f[2*i + 1];  // 奇数位置
    }
    
    // 递归调用FFT
    vector<Complex> g0 = FFT(f0);
    vector<Complex> g1 = FFT(f1);
    
    // 合并结果
    vector<Complex> g(n);
    for (int k = 0; k < n/2; k++) {
        g[k] = (g0[k] + w * g1[k]) / 2.0;
        g[k + n/2] = (g0[k] - w * g1[k]) / 2.0;
        w *= wn;
    }
    
    return g;
}

// 确保输入长度是2的幂
vector<Complex> padToPowerOfTwo(const vector<Complex>& input) {
    int n = 1;
    while (n < input.size()) {
        n *= 2;
    }
    
    vector<Complex> padded = input;
    padded.resize(n, Complex(0, 0));
    return padded;
}

// 采样函数 f(t) = 0.7sin(2π×2t) + sin(2π×5t)
double f(double t) {
    return 0.7 * sin(2 * PI * 2 * t) + sin(2 * PI * 5 * t);
}

int main() {
    // 设置输出精度
    cout << fixed << setprecision(15);
    
    // 分别对128点和256点采样进行FFT
    int sampleNums[] = {128, 256};
    
    for (int i = 0; i < 2; i++) {
        int n = sampleNums[i];
        cout << "采样点数 n = " << n << endl;
        
        // 采样
        vector<Complex> samples(n);
        for (int j = 0; j < n; j++) {
            double t = (double)j / n;  // t ∈ [0,1)
            samples[j] = f(t);
        }
        
        // 确保长度是2的幂
        samples = padToPowerOfTwo(samples);
        
        // 执行FFT
        vector<Complex> result = FFT(samples);
        
        // 输出结果
        cout << "向量g的分量:" << endl;
        for (int j = 0; j < n; j++) {
            cout << "向量g的第" << j << "个分量, x_" << j << " = " << 
                    real(result[j]) << ", y_" << j << " = " << 
                    imag(result[j]) << endl;
            
            // 可以只输出主要分量
            if (abs(result[j]) < 1.0 && j > 10) {
                cout << "..." << endl;
                break;
            }
        }
        cout << endl;
    }
    
    return 0;
}