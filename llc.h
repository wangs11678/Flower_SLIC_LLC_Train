#ifndef LLC_H_INCLUDED
#define LLC_H_INCLUDED

#include <opencv2/opencv.hpp>

void llc_coding(cv::Mat &B, cv::Mat &X, cv::Mat &llcCodes, int knn);

void llc_pooling(cv::Mat &tdictionary,
                         cv::Mat &tinput,
                         cv::Mat &tllccodes,
                         cv::Mat &llcFeature,
                         cv::Mat &feaSet_x,
                         cv::Mat &feaSet_y,
                         int width,
                         int height
                        );

void llc_coding_pooling(std::string imgDir,
                                std::string dsiftDir,
                                std::string llcDir,
                                cv::Mat &dictionary,
                                int knn
                                );

#endif // LLC_H_INCLUDED
