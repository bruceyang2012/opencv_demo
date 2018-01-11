#include "Net.h"
#include <opencv2/imgproc.hpp>

// Initialize net
void Net::initNet(vector<int> layer_neuron_num_)
{
    layer_neuron_num = layer_neuron_num_;
    // build every layer
    layer.resize(layer_neuron_num.size());
    for(int i = 0; i < layer.size(); i++){
        layer[i].create(layer_neuron_num[i], 1, CV_32FC1);
    }
    cout << "Generate layers, successfully!" << endl;

    // build weights matrix and bias
    weights.resize(layer.size() - 1);  // no number
    bias.resize(layer.size() - 1);
    for(int i = 0; i < (layer.size() - 1); i++){
        weights[i].create(layer[i + 1].rows, layer[i].rows, CV_32FC1);
        bias[i] = Mat::zeros(layer[i + 1].rows, 1, CV_32FC1);
    }
}

// Methods of initWeights. if type=0,Gaussian, else uniform.
void Net::initWeight(Mat &dst, int type, double a, double b)
{
    if(type == 0){
        randn(dst, a, b);
    }else{
        randu(dst, a, b);
    }
}

void Net::initWeights(int type, double a, double b)
{
    for(int i = 0; i < weights.size(); i++){
        initWeight(weights[i], 0 ,0 , 0.1);
    }
}

void Net::initBias(Scalar& bias_)
{
    for(int i = 0; i < bias.size(); i++){
        bias[i] = bias_;
    }
}

// Activation fuction
Mat Net::activationFunction(Mat &x, string func_type)
{
    activation_function = func_type;
    Mat fx;
    if(func_type == "sigmoid"){
        fx = sigmoid(x);
    }
    if(func_type == "tanh"){
        fx = tanh(x);
    }
    if(func_type == "ReLU"){
        fx = ReLU(x);
    }
    return fx;
}

// Forward
void Net::forward()
{
    for(int i = 0; i < layer_neuron_num.size() - 1; i++){
        Mat product = weights[i] * layer[i] + bias[i];
        layer[i + 1] = activationFunction(product, activation_function);
    }
    calcLoss(layer[layer.size()-1], target, output_error, loss);
}

// DeltaError = derivative_value * output_error
void Net::deltaError()
{
    delta_err.resize(layer.size() - 1);
    for (int i = delta_err.size() - 1; i >= 0; i--)
    {
        delta_err[i].create(layer[i + 1].size(), layer[i+1].type());
        Mat dx = derivativeFunction(layer[i + 1], activation_function);
        if(i == delta_err.size() - 1){  // output layer
            delta_err[i] = dx.mul(output_error);
        }else{  // hidden layer
            Mat weight = weights[i];
            delta_err[i] = dx.mul((weights[i + 1]).t() * delta_err[i + 1]);
        }
    }
}

// Update weights according to the deltaError
void Net::updateWeights()
{
    for(int i = 0; i < weights.size(); i++){
        Mat delta_weights = learning_rate * (delta_err[i] * layer[i].t());
        Mat delta_bias = learning_rate * delta_err[i];
        weights[i] = weights[i] + delta_weights;
        bias[i] = bias[i] + delta_bias;
    }
}

// Backward
void Net::backward()
{
    deltaError();
    updateWeights();
}

// Train with accuray_threshold
void Net::train(Mat input, Mat target_, float accuracy_threshold)
{
    if(input.empty())
    {
        cout << "Input is empty!" << endl;
        return;
    }
    cout << "Start training..." << endl;

    Mat sample;
    if(input.rows == layer[0].rows && input.cols == 1)
    {
        target = target_;
        sample = input;
        layer[0] = sample;
        forward();
        int num_of_train = 0;
        while(accuracy < accuracy_threshold)
        {
            backward();
            forward();
            num_of_train++;
            if(num_of_train % 500 == 0){
                cout << "Train " << num_of_train << " times" << endl;
                cout << "LossL " << loss <<endl;
            }
        }
        cout << endl <<"Train " << num_of_train << " times" << endl;
        cout << "Loss : " << loss << endl;
        cout << "Train successfully!" << endl;
    }
    else if(input.rows == layer[0].rows && input.cols > 1)
    {
        double batch_loss = 0.;
        int epoch = 0;
        while(accuracy < accuracy_threshold)
        {
            batch_loss = 0.;
            for(int i = 0; i < input.cols; ++i)
            {
                target = target_.col(i);
                sample = input.col(i);
                layer[0] = sample;

                forward();
                batch_loss += loss;
                backward();
            }
            test(input, target_);
            epoch++;
            if(epoch % 10 == 0)
            {
                cout << "Number of epoch : " << epoch << endl;
                cout << "Loss sum : " << batch_loss <<endl;
            }
            //if (epoch % 100 == 0)
            //{
            //	learning_rate*= 1.01;
            //}
        }
        cout << endl << "Number of epoch : " << epoch << endl;
        cout << "Loss sum : " << batch_loss <<endl;
        cout << "Train successfully!" << endl;
    }
    else
    {
        cout << "Rows of input don't match the number of input!" << endl;
    }
}

// Train with loss_threshold (?)
void Net::train(Mat input, Mat target_, float loss_threshold, bool draw_loss_curve)
{
    if (input.empty())
    {
        std::cout << "Input is empty!" << std::endl;
        return;
    }

    std::cout << "Train begain!" << std::endl;

    cv::Mat sample;
    if (input.rows == (layer[0].rows) && input.cols == 1)
    {
        target = target_;
        sample = input;
        layer[0] = sample;
        forward();
        //backward();
        int num_of_train = 0;
        while (loss > loss_threshold)
        {
            backward();
            forward();
            num_of_train++;
            if (num_of_train % 500 == 0)
            {
                std::cout << "Train " << num_of_train << " times" << std::endl;
                std::cout << "Loss: " << loss << std::endl;
            }
        }
        std::cout << std::endl << "Train " << num_of_train << " times" << std::endl;
        std::cout << "Loss: " << loss << std::endl;
        std::cout << "Train sucessfully!" << std::endl;
    }
    else if (input.rows == (layer[0].rows) && input.cols > 1)
    {
        double batch_loss = loss_threshold + 0.01;
        int epoch = 0;
        while (batch_loss > loss_threshold)
        {
            batch_loss = 0.;
            for (int i = 0; i < input.cols; ++i)
            {
                target = target_.col(i);
                sample = input.col(i);
                layer[0] = sample;

                forward();
                backward();

                batch_loss += loss;
            }

            loss_vec.push_back(batch_loss);

            if (loss_vec.size() >= 2 && draw_loss_curve)
            {
                draw_curve(board, loss_vec);
            }
            epoch++;
            if (epoch % output_interval == 0)
            {
                std::cout << "Number of epoch: " << epoch << std::endl;
                std::cout << "Loss sum: " << batch_loss << std::endl;
            }
            if (epoch % 100 == 0)
            {
                learning_rate *= fine_tune_factor;
            }
        }
        std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
        std::cout << "Loss sum: " << batch_loss << std::endl;
        std::cout << "Train sucessfully!" << std::endl;
    }
    else
    {
        std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
    }
}

// Test
void Net::test(Mat &input, Mat &target_) // why use &
{
    if (input.empty())
    {
        std::cout << "Input is empty!" << std::endl;
        return;
    }
    std::cout << std::endl << "Predict,begain!" << std::endl;

    if (input.rows == (layer[0].rows) && input.cols == 1)
    {
        int predict_number = predict_one(input);

        cv::Point target_maxLoc;
        minMaxLoc(target_, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
        int target_number = target_maxLoc.y;

        std::cout << "Predict: " << predict_number << std::endl;
        std::cout << "Target:  " << target_number << std::endl;
        std::cout << "Loss: " << loss << std::endl;
    }
    else if (input.rows == (layer[0].rows) && input.cols > 1)
    {
        double loss_sum = 0;
        int right_num = 0;
        cv::Mat sample;
        for (int i = 0; i < input.cols; ++i)
        {
            sample = input.col(i);
            int predict_number = predict_one(sample);
            loss_sum += loss;

            target = target_.col(i);
            cv::Point target_maxLoc;
            minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
            int target_number = target_maxLoc.y;

            std::cout << "Test sample: " << i << "   " << "Predict: " << predict_number << std::endl;
            std::cout << "Test sample: " << i << "   " << "Target:  " << target_number << std::endl << std::endl;
            if (predict_number == target_number)
            {
                right_num++;
            }
        }
        accuracy = (double)right_num / input.cols;
        std::cout << "Loss sum: " << loss_sum << std::endl;
        std::cout << "accuracy: " << accuracy << std::endl;
    }
    else
    {
        std::cout << "Rows of input don't cv::Match the number of input!" << std::endl;
        return;
    }
}

// Predict one sample
int Net::predict_one(Mat &input)
{
    if(input.empty()){
        cout << "Input is empty !" << endl;
        return -1;
    }

    if(input.rows == layer[0].rows && input.cols == 1)
    {
        layer[0] = input;
        forward();

        Mat layer_out = layer[layer.size() - 1];
        Point predict_maxLoc;

        minMaxLoc(layer_out, NULL, NULL, NULL, &predict_maxLoc, noArray());
        return predict_maxLoc.y;
    }
    else
    {
        std::cout << "Please give one sample alone and ensure input.rows = layer[0].rows" << std::endl;
        return -1;
    }
}

//// Predict more samples
//vector<int> Net::predict(Mat &input);

//// Save model
//void Net::save(std::string filename);

//// Load model
//void Net::load(std::string filename);

// Get the message of samples in XML file.
void get_input_label(string filename, Mat &input, Mat &label, int sample_num, int start)
{
    cv::FileStorage fs;
    fs.open(filename, cv::FileStorage::READ);
    cv::Mat input_, target_;
    fs["input"] >> input_;
    fs["target"] >> target_;
    fs.release();
    input = input_(cv::Rect(start, 0, sample_num, input_.rows));
    label = target_(cv::Rect(start, 0, sample_num, target_.rows));
}

// Draw loss curve
void draw_curve(Mat &board, vector<double> points)
{
    Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
    board = board_;

    line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
    line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);

    for (size_t i = 0; i < points.size() - 1; i++)
    {
        cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
        cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
        line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);
        if (i >= 1000)
        {
            return;
        }
    }
    cv::imshow("Loss", board);
    cv::waitKey(1);
}
