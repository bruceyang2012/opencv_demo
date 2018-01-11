#ifndef FUNCTION_H
#define FUNCTION_H

#include<opencv2/core.hpp>
#include<iostream>

#endif // FUNCTION_H
using namespace cv;
using namespace std;

// Sigmoid function
Mat sigmoid(Mat &x);

// Tanh functin
Mat tanh(Mat &x);

// ReLU function
Mat ReLU(Mat &x);

// Derivative function
Mat derivativeFunction(Mat &fx, string fuc_type);

// Objective function
void calcLoss(Mat &output, Mat &target, Mat &output_error, float &loss);
