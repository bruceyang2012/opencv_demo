#include "Function.h"

// Sigmoid function
Mat sigmoid(Mat &x)
{
    Mat exp_x, fx;
    exp(-x, exp_x);
    fx = 1.0/(1.0 + exp_x);
    return fx;
}

// Tanh functin
Mat tanh(Mat &x)
{
    Mat exp_x, exp_x_, fx;
    exp(-x, exp_x_);
    exp(x, exp_x);
    fx = (exp_x - exp_x_) / (exp_x + exp_x_);
    return fx;
}

// ReLU function
Mat ReLU(Mat &x)
{
    Mat fx = x;
    for(int i = 0; i < fx.rows; i++){
        for(int j = 0; j < fx.cols; j++){
            if(fx.at<float>(i,j) < 0){
                fx.at<float>(i,j) = 0;
            }
        }
    }
    return fx;
}

// Derivative function (BP)
Mat derivativeFunction(Mat &fx, string func_type)
{
    Mat dx;
    if(func_type == "sigmoid"){
        dx = sigmoid(fx).mul(1-sigmoid(fx)); //?
    }
    if(func_type == "tanh"){
        Mat tanh_2;
        pow(tanh(fx),2.,tanh_2);
        dx = 1 - tanh_2;
    }
    if(func_type == "ReLU"){
        dx = fx;
        for(int i = 0; i < fx.rows; i++){
            for(int j = 0;j < fx.cols;j++){
                if(fx.at<float>(i, j) > 0){
                    dx.at<float>(i,j) = 1;
                }
            }
        }
    }
    return dx;
}

// Objective function(MSE)
void calcLoss(Mat &output, Mat &target, Mat &output_error, float &loss)
{
    if(target.empty())
    {
        cout << "Can't find the target Matrix" << std::endl;
        return;
    }
    output_error = target - output;
    Mat err_sqrare;
    pow(output_error,2.,err_sqrare);
    Scalar err_sqr_sum = sum(err_sqrare);
    loss = err_sqr_sum[0]/(float)(output.rows);
}
