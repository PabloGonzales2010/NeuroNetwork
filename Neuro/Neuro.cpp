#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

class Perceptron {
    std::vector<std::vector<double>> allLayers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> empty;
    std::vector<std::vector<double>> allErrors;
    double success;
    double all;
    double averageerror;
    double successrate;
    double learningRate;
    double indexoflastlayer;
    double sigma(const double x) { //вычисление сигмоидной функции
        return 1 / (1 + exp(-x));
    }

    void forward(const std::vector<double>& x) { //прямой ход одной итерации
        allLayers = empty;
        allLayers[0] = x;
        for (int i = 0; i < weights.size(); i++) {
            for (int j = 0; j < weights[i].size(); j++) {
                for (int k = 0; k < weights[i][j].size(); k++) {
                    allLayers[i + 1][k] += weights[i][j][k] * allLayers[i][j];
                }
            }
            for (int j = 0; j < allLayers[i + 1].size(); j++) {
                allLayers[i + 1][j] = sigma(allLayers[i + 1][j]);
            }
        }
    }

    void back(const std::vector<double>& y) { //обратное распространение ошибки одной итерации
        allErrors = empty;
        for (int i = 0; i < allErrors[indexoflastlayer].size(); i++) {
            allErrors[indexoflastlayer][i] = y[i] - allLayers[indexoflastlayer][i];
        }
        for (int i = indexoflastlayer - 1; i > 0; i--) {
            for (int j = 0; j < allErrors[i].size(); j++) {
                for (int k = 0; k < allErrors[i + 1].size(); k++) {
                    allErrors[i][j] += allErrors[i + 1][k] * weights[i][j][k];
                }
            }
        }
        for (int i = 0; i < allLayers.size() - 1; i++) {
            for (int k = 0; k < allLayers[i + 1].size(); k++) {
                double t = learningRate * allErrors[i + 1][k] * allLayers[i + 1][k] * (1 - allLayers[i + 1][k]);
                for (int j = 0; j < allLayers[i].size(); j++) {
                    weights[i][j][k] += t * allLayers[i][j];
                }
            }
        }
        //-------------------------------------------------------
        std::vector<double> output = allLayers[indexoflastlayer];
        int imax = 0;
        double max = output[0];
        for (int j = 1; j < output.size(); j++) {
            averageerror += abs(output[j] - y[j]);
            if (output[j] > max) {
                max = output[j];
                imax = j;
            }
        }
        for (int j = 0; j < output.size(); j++) {
            output[j] = 0;
        }
        output[imax] = 1;
        if (output == y) success++;
    }
public:
    void train(std::vector<std::vector<double>>& x, std::vector<std::vector<double>>& y, int epochs) {
        for (int i = 0; i < epochs; i++) {
            success = 0;
            averageerror = 0;
            for (int j = 0; j < 100; j++) {
                std::cout << ".";
            }
            std::cout << "\n";
            for (int j = 0; j < x.size(); j++) {
                forward(x[j]);
                back(y[j]);
                if (j % (x.size() / 100) == 0) std::cout << "|";
            }
            std::cout << "\n";
            all = x.size();
            successrate = success / all;
            averageerror /= all * y[0].size();
            std::cout << "Epoch " << i + 1 << " Average error is " << averageerror << " Success rate is " << successrate * 100 << "%\n";
        }
    }
    std::vector<double> predict(std::vector<double> x) {
        forward(x);
        return allLayers[indexoflastlayer];
    }
    void test(std::vector<std::vector<double>>& xtest, std::vector<std::vector<double>>& ytest) {
        success = 0;
        averageerror = 0;
        for (int i = 0; i < xtest.size(); i++) {
            std::vector<double> output = predict(xtest[i]);
            int imax = 0;
            double max = output[0];
            for (int j = 1; j < output.size(); j++) {
                averageerror += abs(output[j] - ytest[i][j]);
                if (output[j] > max) {
                    max = output[j];
                    imax = j;
                }
            }
            for (int j = 0; j < output.size(); j++) {
                output[j] = 0;
            }
            output[imax] = 1;
            if (output == ytest[i]) success++;
        }
        all = xtest.size();
        successrate = success / all;
        averageerror /= all * ytest[0].size();
        std::cout << "Average error is " << averageerror << " Success rate is " << successrate * 100 << "%\n";
    }
    Perceptron(const std::vector<int>& layers) { //структура перцептрона
        learningRate = 0.05;
        success = 0;
        all = 0;
        successrate = 0;
        averageerror = 0;
        for (int i = 0; i < layers.size(); i++) {
            std::vector<double> temp(layers[i]);
            allLayers.push_back(temp);
        }
        empty = allLayers;
        indexoflastlayer = allLayers.size() - 1;
        for (int i = 0; i < layers.size()-1; i++) {
            std::vector<std::vector<double>> temp1;
            for (int j = 0; j < layers[i]; j++) {
                std::vector<double> temp;
                for (int k = 0; k < layers[i + 1]; k++) {
                    double a = (double)rand() / RAND_MAX / sqrt(layers[i]);
                    temp.push_back(a);
                }
                temp1.push_back(temp);
            }
            weights.push_back(temp1);
        }
    }
};

std::vector<std::string> split(std::string const& original, char separator)
{
    std::vector<std::string> results;
    std::string::const_iterator start = original.begin();
    std::string::const_iterator end = original.end();
    std::string::const_iterator next = std::find(start, end, separator);
    while (next != end) {
        results.push_back(std::string(start, next));
        start = next + 1;
        next = std::find(start, end, separator);
    }
    results.push_back(std::string(start, next));
    return results;
}

int main()
{
    setlocale(LC_CTYPE, "Russian");

    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
    std::vector<std::vector<std::string>> xs;
    std::ifstream in("../mnist_train.csv");
    std::string line;
    for (int i = 0; i < 3000; i++){
        getline(in, line);
        xs.push_back(split(line,','));
        if (i % 100 == 0) std::cout << i << " подгружено\n";
    }
    for (int i = 0; i < xs.size(); i++) {
        std::vector<double> temp;
        for (int j = 1; j < xs[i].size(); j++) {
            temp.push_back(stod(xs[i][j]) / 255);
        }
        x.push_back(temp);
        std::vector<double> temp1(10);
        temp1[stoi(xs[i][0])] = 1;
        y.push_back(temp1);
        if (i % 100 == 0) std::cout << i << " предподготовлено\n";
    }
    std::cout << "Чистка памяти... ";
    xs.clear();
    xs.shrink_to_fit();
    std::cout << "завершена!\n";
    std::vector<std::vector<double>> xtest;
    std::vector<std::vector<double>> ytest;
    std::vector<std::vector<std::string>> tests;
    std::ifstream intest("../mnist_test.csv");
    for (int i = 0; i < 2000; i++) {
        getline(intest, line);
        tests.push_back(split(line, ','));
        if (i % 100 == 0) std::cout << i << " тестов подгружено\n";
    }
    for (int i = 0; i < tests.size(); i++) {
        std::vector<double> temp;
        for (int j = 1; j < tests[i].size(); j++) {
            temp.push_back(stod(tests[i][j]) / 255);
        }
        xtest.push_back(temp);
        std::vector<double> temp1(10);
        temp1[stoi(tests[i][0])] = 1;
        ytest.push_back(temp1);
        if (i % 100 == 0) std::cout << i << " тестов предподготовлено\n";
    }
    std::cout << "Чистка памяти... ";
    tests.clear();
    tests.shrink_to_fit();
    std::cout << "завершена!\n";
    Perceptron perceptron({784,800,10});
    std::cout << "Идёт обучение...!\n";
    perceptron.train(x, y, 3);
    std::cout << "Тесты:\n";
    perceptron.test(xtest, ytest);
    
}
