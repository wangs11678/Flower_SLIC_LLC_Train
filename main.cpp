#include "utils.h"
#include "dsift.h"
#include "dictionary.h"
#include "llc.h"
#include "train.h"
#include "predict.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
    //生成目录=====================================================
    MakeDir("result");
    MakeDir("result/dsift"); //生成存放dsift特征目录
    MakeDir("result/llc"); //生成存放llc特征目录

    //特征提取=====================================================
    int step = 6;
    int binSize = 4;
    int patchSize = 16;
    string imgDir = "images";
    string dsiftDir = "result/dsift";
    extract_dsift_feature(imgDir, dsiftDir, step, binSize, patchSize);

    //训练字典=====================================================
    Mat dictionary;
    int wordSize = 1024;
    string dictionaryFile = "result/dictionary.xml.gz";
    //gen_dictionary_bowKMeans(dsiftDir, dictionaryFile, wordSize, dictionary);
    //gen_dictionary_cvKMeans(dsiftDir, dictionaryFile, wordSize, dictionary);
    gen_dictionary_vlfeatKMeans(dsiftDir, dictionaryFile, wordSize, dictionary);

    //降维算法=====================================================
    int knn = 5; //number of neighbors for local coding
    string llcDir = "result/llc";
    llc_coding_pooling(imgDir, dsiftDir, llcDir, dictionary, knn);

    //生成文档=====================================================
    int trNum = 700;
    char trainFile[] = "result/train.txt";
    char testFile[] = "result/test.txt";
    gen_txt_file(llcDir, trainFile, testFile, trNum);

    //训练数据=====================================================
    char modelFile[] = "result/model.txt";
    SVM_train(argc, argv, trainFile, modelFile);

    //测试数据=====================================================
    char resultFile[] = "result/result.txt";
    vector<int> labels;
    labels = SVM_predict(argc, argv, testFile, modelFile, resultFile);

    return 0;
}





