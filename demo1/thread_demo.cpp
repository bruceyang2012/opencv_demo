#include <opencv2/highgui.hpp>// 提供imread读取图片函数
using namespace cv;

#include <iostream>
#include <thread> // 提供线程类
#include <mutex> // 提供互斥锁类(对部分和进行累加的时候需要加锁)，很好用

using namespace std;

mutex mtx;// 定义一个互斥锁
long totalSum;// 总和
enum RangeSpecify { LEFT_UP, LEFT_DOWN, RIGHT_UP, RIGHT_DOWN };

void ImageAverage(Mat &img, int r)// 线程代码
{
    int startRow, startCol, endRow, endCol;
    switch (r){
    case 1:
        startRow = 0;
        endRow = img.rows / 2;
        startCol = 0;
        endCol = img.cols / 2;
        break;
    case 2:
        startRow = img.rows / 2;
        endRow = img.rows;
        startCol = 0;
        endCol = img.cols / 2;
        break;
    case 3:
        startRow = 0;
        endRow = img.rows / 2;
        startCol = img.cols / 2;
        endCol = img.cols;
        break;
    case 4:
        startRow = img.rows / 2;
        endRow = img.rows;
        startCol = img.cols / 2;
        endCol = img.cols;
        break;
    }
    double t = (double)getTickCount();
    long sum = 0;
    for (int i = startRow; i < endRow; i++) {
        for (int j = startCol; j < endCol; j++) {
            sum += img.at<unsigned char>(i, j);
        }
    }
    mtx.lock();// 在访问公共变量totalSum 之前对其进行加锁
    totalSum += sum;
    mtx.unlock();// 访问完毕立刻解锁

    cout << r << " : " << sum << endl;
    cout << "task completed! Time elapsed " << (double)getTickCount() -t << endl;// 打印本次线程时间花费
}

//int main()
//{
//    Mat src = imread("yuan.jpg", CV_LOAD_IMAGE_GRAYSCALE);

//    double t = (double)getTickCount();
//    thread* t0 = new thread(ImageAverage, ref(src), 1);
//    thread* t1 = new thread(ImageAverage, ref(src), 2);
//    thread* t2 = new thread(ImageAverage, ref(src), 3);
//    thread* t3 = new thread(ImageAverage, ref(src), 4);

//    t0->join();// 等待子线程t0执行完毕
//    t1->join();
//    t2->join();
//    t3->join();

//    cout << endl <<"多线程总时间花费：" << (double)getTickCount() - t << endl;
//    cout << "图像均值(多线程）: " << totalSum*1.0 / (src.cols*src.rows) << endl << endl;

//    // 验证准确性
//    long sum =0;

//    t = (double)getTickCount();
//    for (int i = 0; i < src.rows; i++) {
//        for (int j = 0; j < src.cols; j++) {
//            sum += src.at<unsigned char>(i, j);
//        }
//    }

//    cout << "参照时间花费：" << (double)getTickCount() - t << endl;
//    cout << "参照均值： " << sum*1.0 / (src.rows*src.cols) << endl<<endl;

//    system("pause");
//    return 0;
//}
