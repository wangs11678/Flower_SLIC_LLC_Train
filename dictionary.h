#ifndef DICTIONARY_H_INCLUDED
#define DICTIONARY_H_INCLUDED

#include <opencv2/opencv.hpp>

extern "C" {
#include "vl/kmeans.h"
}

void gen_dictionary_bowKMeans(const std::string& dsiftDir, const std::string& dictionaryFile, int wordSize, cv::Mat &dictionary);

void gen_dictionary_cvKMeans(const std::string& dsiftDir, const std::string& dictionaryFile, int wordSize, cv::Mat &dictionary);

void gen_dictionary_vlfeatKMeans(const std::string& dsiftDir, const std::string& dictionaryFile, int wordSize, cv::Mat &dictionary);
#endif // DICTIONARY_H_INCLUDED
