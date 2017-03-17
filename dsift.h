#ifndef DSIFT_H_INCLUDED
#define DSIFT_H_INCLUDED

#include <opencv2/opencv.hpp>
#include "opencv2/nonfree/nonfree.hpp"

extern "C" {
#include "vl/dsift.h"
}

//利用opencv自带的dsift提取函数提取特征
void dsift_opencv(cv::Mat &img, int step, int patchSize, cv::Mat &dsiftFeature);

void dsift_vlfeat(cv::Mat &img, int step, int patchSize, cv::Mat &dsiftFeature);

//将图像等比例缩小到maxImgSize
void img_resize(cv::Mat &image, int maxImgSize);

//生成采样点网格
void meshgrid(const cv::Range &xgv, const cv::Range &ygv, int step, cv::Mat &X, cv::Mat &Y);

//计算feaSetX, feaSetY
void calculateSiftXY_opencv(int width, int height, int patchSize, int binSize, int step, cv::Mat &feaSet_x, cv::Mat &feaSet_y);
void calculateSiftXY_vlfeat(int width, int height, int patchSize, int binSize, int step, cv::Mat &feaSet_x, cv::Mat &feaSet_y);

//对全部图像提取dsift特征
void extract_dsift_feature(std::string imgDir, std::string dsiftDir, int step, int binSize, int patchSize);

#endif // DSIFT_H_INCLUDED
