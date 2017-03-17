#include "dictionary.h"
#include "utils.h"

using namespace std;
using namespace cv;

/*
 * 从提取的dsift特征中读取descriptors进行聚类
 * train a dictionary
 *下面有三种聚类方法，选其中一种即可
 */

//利用opencv自带的BOWKMeansTrainer进行KMeans聚类
void gen_dictionary_bowKMeans(const string& dsiftDir, const string& dictionaryFile, int wordSize, Mat &dictionary)
{
	FileStorage fs(dictionaryFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["dictionary"] >> dictionary;
	}
	else
	{
		vector<string> categories;
		GetDirList(dsiftDir, &categories);

		Mat allDescriptors;

		for (int i = 0; i != categories.size(); i++)
		{
			cout << "Processing category " << categories[i] << endl;
			string currentCategory = dsiftDir + "/" + categories[i];

			vector<string> fileList;
			GetFileList(currentCategory, &fileList);

			for (int j = 0; j != fileList.size(); j++)
			{
				string filePath = currentCategory + "/" + fileList[j];

				Mat dsiftFeature;

				FileStorage fs2(filePath, FileStorage::READ);
				if(fs2.isOpened())
				{
					fs2["dsiftFeature"] >> dsiftFeature; //从提取的dsift特征里读取特征到descriptors
				}

				if (allDescriptors.empty())
				{
					allDescriptors.create(0, dsiftFeature.cols, dsiftFeature.type());
				}
				allDescriptors.push_back(dsiftFeature); //将所有读取的dsift特征挨个存放到allDescriptors
			}
		}
		assert(!allDescriptors.empty());
		cout << "Build dictionary..." << endl;

		BOWKMeansTrainer bowTrainer(wordSize);
		dictionary = bowTrainer.cluster(allDescriptors); //进行聚类

		fs.open(dictionaryFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "dictionary" << dictionary;
		}
		cout << "Done build dictionary..." << endl;
	}
}

//利用opencv自带的cvKMeans2进行KMeans聚类
void gen_dictionary_cvKMeans(const string& dsiftDir, const string& dictionaryFile, int wordSize, Mat &dictionary)
{
	FileStorage fs(dictionaryFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["dictionary"] >> dictionary;
	}
	else
	{
		vector<string> categories;
		GetDirList(dsiftDir, &categories);

		Mat allDescriptors;

		for (int i = 0; i != categories.size(); i++)
		{
			cout << "Processing category " << categories[i] << endl;
			string currentCategory = dsiftDir + "/" + categories[i];

			vector<string> fileList;
			GetFileList(currentCategory, &fileList);

			for (int j = 0; j != fileList.size(); j++)
			{
				string filePath = currentCategory + "/" + fileList[j];

				Mat dsiftFeature;

				FileStorage fs2(filePath, FileStorage::READ);
				if(fs2.isOpened())
				{
					fs2["dsiftFeature"] >> dsiftFeature; //从提取的dsift特征里读取特征到descriptors
				}

				if (allDescriptors.empty())
				{
					allDescriptors.create(0, dsiftFeature.cols, dsiftFeature.type());
				}

				allDescriptors.push_back(dsiftFeature); //将所有读取的dsift特征挨个存放到allDescriptors
			}
		}
		assert(!allDescriptors.empty());
		cout << "Build dictionary..." << endl;

        CvMat *samples=cvCreateMat(allDescriptors.rows, 128, CV_32FC1); //包含所有图片的所有feature信息的矩阵，featureNum个feature，每个feature为dims（128）维向量，每一维的元素类型为32位浮点数
		CvMat temp = allDescriptors;
		cvCopy(&temp, samples);
		CvMat *clusters=cvCreateMat(allDescriptors.rows, 1, CV_32SC1); //每个feature所在“质心”的指针（实际上本例程中没有用到该信息）
        //dictionary = Mat(wordSize, 128, CV_32FC1); //“质心”信息的数组，wordSize个“质心”每个质心都是dims（128）维向量，每一维的元素类型为32位浮点数
        CvMat *centers=cvCreateMat(wordSize, 128, CV_32FC1); //“质心”信息的数组，k个“质心”每个质心都是dims（128）维向量，每一维的元素类型为32位浮点数
        cvSetZero(clusters); //将矩阵初始化为0
        cvSetZero(centers); //将矩阵初始化为0

        cvKMeans2(samples, wordSize, clusters, cvTermCriteria(CV_TERMCRIT_EPS, 10, 1.0),
                            3, (CvRNG *)0, KMEANS_USE_INITIAL_LABELS, centers); //Kmeans聚类

        dictionary = Mat(centers, true);
		fs.open(dictionaryFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "dictionary" << dictionary;
		}
		cout << "Done build dictionary..." << endl;
	}
}


//利用vlfeat库进行KMeans聚类
void gen_dictionary_vlfeatKMeans(const string& dsiftDir, const string& dictionaryFile, int wordSize, Mat &dictionary)
{
	FileStorage fs(dictionaryFile, FileStorage::READ);
	if(fs.isOpened())
	{
		fs["dictionary"] >> dictionary;
	}
	else
	{
		vector<string> categories;
		GetDirList(dsiftDir, &categories);

		Mat allDescriptors;

		for (int i = 0; i != categories.size(); i++)
		{
			cout << "Processing category " << categories[i] << endl;
			string currentCategory = dsiftDir + "/" + categories[i];

			vector<string> fileList;
			GetFileList(currentCategory, &fileList);

			for (int j = 0; j != fileList.size(); j++)
			{
				string filePath = currentCategory + "/" + fileList[j];

				Mat dsiftFeature;

				FileStorage fs2(filePath, FileStorage::READ);
				if(fs2.isOpened())
				{
					fs2["dsiftFeature"] >> dsiftFeature; //从提取的dsift特征里读取特征到descriptors
				}

				if (allDescriptors.empty())
				{
					allDescriptors.create(0, dsiftFeature.cols, dsiftFeature.type());
				}

				allDescriptors.push_back(dsiftFeature); //将所有读取的dsift特征挨个存放到allDescriptors
			}
		}

        cout << "Build dictionary..." << endl;

        int dsiftNum = allDescriptors.rows;
        int dsiftDim = allDescriptors.cols;
        float *data = new float[dsiftNum*dsiftDim];

        for(int i = 0; i < dsiftNum; i++)
        {
            for(int j = 0; j < dsiftDim; j++)
            {
                data[j+i*dsiftDim] = allDescriptors.at<float>(i, j);
            }
        }

        float * centers ;
        // Use float data and the L2 distance for clustering
        VlKMeans * kmeans = vl_kmeans_new (VL_TYPE_FLOAT, VlDistanceL2) ;
        // Use Lloyd algorithm
        vl_kmeans_set_algorithm (kmeans, VlKMeansLloyd) ;
        // Initialize the cluster centers by randomly sampling the data
        vl_kmeans_init_centers_with_rand_data (kmeans, data, dsiftDim, dsiftNum, wordSize) ;
        // Run at most 100 iterations of cluster refinement using Lloyd algorithm
        vl_kmeans_set_max_num_iterations (kmeans, 100) ;
        vl_kmeans_refine_centers (kmeans, data, dsiftNum) ;

        // Obtain the cluster centers
        centers = (float*)vl_kmeans_get_centers(kmeans) ;

        dictionary = Mat(wordSize, dsiftDim, CV_32FC1);
        for (int i = 0; i < wordSize; i++)
        {
            for (int j = 0; j < dsiftDim; j++)
            {
                dictionary.at<float>(i, j) = centers[j+i*dsiftDim];
            }
        }
		fs.open(dictionaryFile, FileStorage::WRITE);
		if (fs.isOpened())
		{
			fs << "dictionary" << dictionary;
		}
		cout << "Done build dictionary..." << endl;
	}
}
