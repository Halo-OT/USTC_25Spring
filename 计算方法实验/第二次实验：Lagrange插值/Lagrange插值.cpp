#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>  

using namespace std;

double f(double x) {
    return 1.0 / (1.0 + x * x);
}

vector<double> generate_uniform_nodes(int n, vector<double>& f_values) {
    vector<double> nodes;
    f_values.clear();
    for (int i = 0; i <= n; ++i) {
        double xi = -5.0 + (10.0 * i) / n;
        nodes.push_back(xi);
        f_values.push_back(f(xi));
    }
    return nodes;
}

vector<double> generate_chebyshev_nodes(int n, vector<double>& f_values) {
    vector<double> nodes;
    f_values.clear();
    double pi = acos(-1.0);
    for (int i = 0; i <= n; ++i) {
        double angle = (2 * i + 1) * pi / (2 * (n + 1));
        double xi = -5.0 * cos(angle);
        nodes.push_back(xi);
        f_values.push_back(f(xi));
    }
    return nodes;
}

double lagrange_interpolation(double y, const vector<double>& nodes, const vector<double>& f_values) {
    int n_nodes = nodes.size();
    double result = 0.0;
    for (int k = 0; k < n_nodes; ++k) {
        double term = f_values[k];
        double product = 1.0;
        for (int m = 0; m < n_nodes; ++m) {
            if (m != k) {
                product *= (y - nodes[m]) / (nodes[k] - nodes[m]);
            }
        }
        result += term * product;
    }
    return result;
}

double compute_max_error(const vector<double>& nodes, const vector<double>& f_values, const vector<double>& y_values) {
    double max_err = 0.0;
    for (size_t j = 0; j < y_values.size(); ++j) {
        double y = y_values[j];
        double f_true = f(y);
        double f_interp = lagrange_interpolation(y, nodes, f_values);
        double err = abs(f_true - f_interp);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

vector<double> generate_y_values() {
    vector<double> ys;
    for (int j = 0; j <= 500; ++j) {
        double yj = -5.0 + (10.0 * j) / 500;
        ys.push_back(yj);
    }
    return ys;
}

int main() {
    vector<int> ns;
    ns.push_back(5);
    ns.push_back(10);
    ns.push_back(20);
    ns.push_back(40);
    vector<double> y_values = generate_y_values();

    for (size_t i = 0; i < ns.size(); ++i) {  
        int n = ns[i];
        vector<double> f_uni, f_cheb;
        vector<double> uniform_nodes = generate_uniform_nodes(n, f_uni);
        vector<double> cheb_nodes = generate_chebyshev_nodes(n, f_cheb);

        double max_uni = compute_max_error(uniform_nodes, f_uni, y_values);
        double max_cheb = compute_max_error(cheb_nodes, f_cheb, y_values);

        
        cout << "线性节点 " << uniform_nodes.size() << " " << setprecision(15) << max_uni << endl;
        cout << "cos节点 " << cheb_nodes.size() << " " << setprecision(15) << max_cheb << endl;
    }

    return 0;
}
