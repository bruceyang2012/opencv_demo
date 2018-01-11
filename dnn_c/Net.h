#ifndef NET_H
#define NET_H

#endif // NET_H

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "Function.h"

using namespace std;
using namespace cv;

class Net{
public:
    // Integer vector specifying the number of neurons of input and output layers.
    vector<int> layer_neuron_num;
    string activation_function = "sigmoid";
    int output_interval = 10;
    float learning_rate;
    float accuracy = 0.;
    vector<double> loss_vec;
    float fine_tune_factor = 1.01; //?

protected:
    vector<Mat> layer;
    vector<Mat> weights;
    vector<Mat> bias;
    vector<Mat> delta_err; //bp

    Mat output_error;
    Mat target;
    Mat board; //?
    float loss;

public:
    Net(){};
    ~Net(){};

    // Initialize net
    void initNet(vector<int> layer_neuron_num_);

    void initWeights(int type=0, double a=0., double b=0.1);

    void initBias(Scalar& bias);

    // Forward
    void forward();

    // Backward
    void backward();

    // Train with accuray_threshold
    void train(Mat input, Mat target_, float accuracy_threshold);

    // Train with loss_threshold (?)
    void train(Mat input, Mat target_, float loss_threshold, bool draw_loss_curve=false);

    // Test
    void test(Mat &input, Mat &target_); // why use &

    // Predict one sample
    int predict_one(Mat &input);

    // Predict more samples
    vector<int> predict(Mat &input);

    // Save model
    void save(std::string filename);

    // Load model
    void load(std::string filename);

protected:
    // Methods of initWeights. if type=0,Gaussian, else uniform.
    void initWeight(Mat &dst, int type, double a, double b);

    // Activation fuction
    Mat activationFunction(Mat &x, string func_type);

    // Compute delta error
    void deltaError();

    // Update weights
    void updateWeights();
};

// Get the message of samples in XML file.
void get_input_label(string filename, Mat &input, Mat &label, int sample_num, int start = 0);

// Draw loss curve
void draw_curve(Mat &board, vector<double> points);
