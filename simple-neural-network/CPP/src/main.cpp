#include <iostream>
#include "neuralNetwork.h"

double a(functioncall f) {
    double value = 4;
    return f(value);
}

double f(double x) {
    const double m = 0.3;
    const double b = 0.2;
    return m * x + b;
}

int main() {
    // Matrix m = Matrix(3, 2, 1);
    // m(0, 1) = 2;
    // m(1, 0) = 3;
    // m(1, 1) = 4;
    // m(2, 0) = 5;
    // m(2, 1) = 6;

    // Matrix n = m;
    // m.print();
    // n.print();
    // std::cout << m(2,0) << std::endl;

    // m = m + n;

    // m.print();

    NeuralNetwork n = NeuralNetwork(2, 1, 2);

    vector<double> inputs;
    inputs.push_back(1.0);
    inputs.push_back(2.0);

    std::cout << "Inputs" << std::endl;
    for (int i = 0; i < inputs.size(); i++)
        std::cout << inputs[i] << std::endl;

    std::cout << "Results" << std::endl;
    for (int i = 0; i < inputs.size(); i++)
        std::cout << f(inputs[i]) << std::endl;

    vector<double> out = n.predict(inputs);

    std::cout << "Outputs" << std::endl;
    for (int i = 0; i < out.size(); i++)
        std::cout << out[i] << std::endl;

    std::cout << "pass" << std::endl;
    return 0;
}