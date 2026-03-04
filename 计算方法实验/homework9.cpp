
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// 定义微分方程函数
double differentialEquation(double x, double y) {
    return -pow(x, 2) * pow(y, 2);
}

// 四阶龙格-库塔方法
double rungeKuttaMethod(double stepSize) {
    int steps = static_cast<int>(1.5 / stepSize);
    double x = 0.0;
    double y = 3.0;
    
    for (int i = 0; i < steps; ++i) {
        double k1 = stepSize * differentialEquation(x, y);
        double k2 = stepSize * differentialEquation(x + stepSize / 2, y + k1 / 2);
        double k3 = stepSize * differentialEquation(x + stepSize / 2, y + k2 / 2);
        double k4 = stepSize * differentialEquation(x + stepSize, y + k3);
        
        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        x += stepSize;
    }
    
    return y;
}

// 四阶Adams预估-校正方法
double adamsMethod(double stepSize) {
    int steps = static_cast<int>(1.5 / stepSize);
    vector<double> x(steps + 1);
    vector<double> y(steps + 1);
    
    x[0] = 0.0;
    y[0] = 3.0;
    
    // 使用RK4计算前4个点
    for (int i = 0; i < 4; ++i) {
        double k1 = stepSize * differentialEquation(x[i], y[i]);
        double k2 = stepSize * differentialEquation(x[i] + stepSize / 2, y[i] + k1 / 2);
        double k3 = stepSize * differentialEquation(x[i] + stepSize / 2, y[i] + k2 / 2);
        double k4 = stepSize * differentialEquation(x[i] + stepSize, y[i] + k3);
        
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
        x[i + 1] = x[i] + stepSize;
    }
    
    // Adams预估-校正方法
    for (int i = 4; i < steps; ++i) {
        double nextX = x[i] + stepSize;
        
        // 预估
        double predictedY = y[i] + stepSize / 24 * (
            55 * differentialEquation(x[i], y[i]) - 
            59 * differentialEquation(x[i - 1], y[i - 1]) + 
            37 * differentialEquation(x[i - 2], y[i - 2]) - 
            9 * differentialEquation(x[i - 3], y[i - 3])
        );
        
        // 校正
        double correctedY = y[i] + stepSize / 24 * (
            9 * differentialEquation(nextX, predictedY) + 
            19 * differentialEquation(x[i], y[i]) - 
            5 * differentialEquation(x[i - 1], y[i - 1]) + 
            differentialEquation(x[i - 2], y[i - 2])
        );
        
        y[i + 1] = correctedY;
        x[i + 1] = nextX;
    }
    
    return y[steps];
}

// 精确解函数
double exactSolution(double x) {
    return 3.0 / (1 + pow(x, 3));
}

int main() {
    const vector<double> stepSizes = {0.1, 0.05, 0.025, 0.0125};
    const int iterations = 3;
    
    // 输出龙格-库塔方法的结果
    cout << "四阶龙格-库塔方法的误差和误差阶:" << endl;
    double previousRKError = 0;
    
    for (int i = 0; i <= iterations; ++i) {
        double h = stepSizes[i];
        double rkResult = rungeKuttaMethod(h);
        double exactValue = exactSolution(1.5);
        double error = fabs(rkResult - exactValue);
        
        cout << "h = " << h << ", 误差 = " << scientific << error;
        
        if (i > 0) {
            double order = fabs(log(error / previousRKError) / log(2));
            cout << ", 误差阶 = " << order << endl;
        } else {
            cout << endl;
        }
        
        previousRKError = error;
    }
    
    // 输出Adams方法的结果
    cout << "\n四阶隐式Adams方法的误差和误差阶:" << endl;
    double previousAdamsError = 0;
    
    for (int i = 0; i <= iterations; ++i) {
        double h = stepSizes[i];
        double adamsResult = adamsMethod(h);
        double exactValue = exactSolution(1.5);
        double error = fabs(adamsResult - exactValue);
        
        cout << "h = " << h << ", 误差 = " << scientific << error;
        
        if (i > 0) {
            double order = fabs(log(error / previousAdamsError) / log(2));
            cout << ", 误差阶 = " << order << endl;
        } else {
            cout << endl;
        }
        
        previousAdamsError = error;
    }
    
    return 0;
}    