#include "llc.h"
#include "utils.h"

using namespace std;
using namespace cv;

/*
 * Locality-constrained linear coding(LLC)算法
 *包括coding和pooing过程
 */

//llc approximation coding
void llc_coding(Mat &B, Mat &X, Mat &llcCodes, int knn)
{
    //find k nearest neighbors
    int nframe = X.rows; //X(前面提取的dsift特征)矩阵的行
    int nbase = B.rows; //B(字典)矩阵的行

	Mat XX, BB;
	//reduce矩阵变向量，相当于matlab中的sum
	cv::reduce(X.mul(X), XX, 1, CV_REDUCE_SUM, CV_32FC1);
	cv::reduce(B.mul(B), BB, 1, CV_REDUCE_SUM, CV_32FC1);

	//repeat相当于matlab中的repmat
	Mat D1 = cv::repeat(XX, 1, nbase);
    Mat Bt;
    transpose(B, Bt); //注意转置阵不能返回给原Mat本身
    Mat D2 = 2*X*Bt;
    Mat BBt;
    transpose(BB, BBt);
	Mat D3 = cv::repeat(BBt, nframe, 1);
    Mat D = D1 - D2 + D3;

	Mat SD(nframe, nbase, CV_16UC1);
	//对D所有行升序排列后，将索引赋给SD
    cv::sortIdx(D, SD, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

    Mat IDX(nframe, knn, CV_16UC1);
	//将SD的第i列赋值给IDX的第i列
	for (int i = 0; i < knn; i++)
	{
		SD.col(i).copyTo(IDX.col(i));
	}

	float beta = 1e-4;
	int nxcol = X.cols; //特征行

    Mat II = Mat::eye(knn, knn, CV_32FC1);

    llcCodes = Mat::zeros(nframe, nbase, CV_32FC1);

    Mat z, zt;
    Mat z1(knn, nxcol, CV_32FC1);
    Mat z2(knn, nxcol, CV_32FC1);

    Mat C, C_inv;
    Mat w, wt;

    for (int i = 0; i < nframe; i++)
	{
        for (int j = 0; j < knn; j++)
		{
			B.row(IDX.ptr<ushort>(i)[j]).copyTo(z1.row(j));
			X.row(i).copyTo(z2.row(j));
        }
        z = z1 - z2;

        transpose(z, zt);
        C = z*zt;
        C = C + II*beta*trace(C)[0]; //trace(C)[0]求矩阵的迹
        invert(C, C_inv);

        w = C_inv*Mat::ones(knn, 1, CV_32FC1); //相当于matlab中的w = C\ones(knn,1);

        float sum_w = 0;
		sum_w = cv::sum(w)[0];
        w = w/sum_w;
        transpose(w, wt);

        for (int j = 0; j < knn; j++)
		{
            llcCodes.at<float>(i, IDX.ptr<ushort>(i)[j]) = wt.at<float>(0, j);
        }
    }

    XX.release();
    BB.release();
	Bt.release();
	BBt.release();
    D.release();
    D1.release();
    D2.release();
	D3.release();
    SD.release();
    II.release();
    z.release();
	zt.release();
    z1.release();
    z2.release();
    C.release();
	C_inv.release();
    w.release();
    wt.release();
}

void llc_pooling(Mat &tdictionary,
                            Mat &tinput,
                            Mat &tllccodes,
                            Mat &llcFeature,
                            Mat &feaSet_x,
                            Mat &feaSet_y,
                            int width,
                            int height
                            )
{
	Mat dictionary, input;
	transpose(tdictionary, dictionary);
	transpose(tinput, input);

	int dSize = dictionary.cols;
	int nSmp = input.cols;

	Mat idxBin = Mat::zeros(nSmp, 1, CV_32FC1);
	Mat llccodes;
	transpose(tllccodes, llccodes);

	Mat pyramid(1, 3, CV_32FC1);
	pyramid.at<float>(0, 0) = 1;
	pyramid.at<float>(0, 1) = 2;
	pyramid.at<float>(0, 2) = 4;

	int pLevels = pyramid.cols;
	Mat pBins(1,3,CV_32FC1);
	int tBins = 0;
	for(int i = 0; i < 3; i++)
	{
		pBins.at<float>(0, i) = pyramid.at<float>(0, i)*pyramid.at<float>(0, i);
		tBins += pBins.at<float>(0, i);
	}
	Mat beta = Mat::zeros(dSize, tBins, CV_32FC1);
	int bId = 0;
	int betacol = -1; //beta的列


	for (int iter1 = 0; iter1 != pLevels; iter1++)
	{
		int nBins = pBins.at<float>(0, iter1);
		float wUnit = width / pyramid.at<float>(0, iter1);
		float hUnit = height / pyramid.at<float>(0, iter1);

		//find to which spatial bin each local descriptor belongs
		Mat xBin(nSmp, 1, CV_32FC1);
		Mat yBin(nSmp, 1, CV_32FC1);

		for(int i = 0; i < nSmp; i++)
		{
			xBin.at<float>(i, 0) = ceil(feaSet_x.at<float>(i, 0) / wUnit);
		    yBin.at<float>(i, 0) = ceil(feaSet_y.at<float>(i, 0) / hUnit);
			idxBin.at<float>(i, 0) = (yBin.at<float>(i, 0) - 1) * pyramid.at<float>(0, iter1) + xBin.at<float>(i, 0);
		}

		for(int iter2 = 1; iter2 <= nBins; iter2++)
		{
			bId = bId + 1;
			betacol = betacol + 1;

			int nsbrows = 0; //统计每次循环sidxBin的行总数
			for(int i = 0; i < nSmp; i++)
			{
				if(idxBin.at<float>(i, 0) == iter2)
				{
					nsbrows++;
				}
			}

			Mat sidxBin(nsbrows, 1, CV_16UC1);
			int sbrow = 0; //sidxBin的行
			for(int i = 0; i < nSmp; i++)
			{
				if(idxBin.at<float>(i, 0) == iter2)
				{
					sidxBin.ptr<ushort>(sbrow++)[0] = i;
				}
			}
			if(sidxBin.empty())
			{
				continue;
			}

			//beta(:, bId) = max(llc_codes(:, sidxBin), [], 2);
			float iRowMax = 0; //每一行的最大值
			for(int i = 0; i < llccodes.rows; i++)
			{
				iRowMax = llccodes.at<float>(i, sidxBin.ptr<ushort>(0)[0]);
				for(int j = 0; j < nsbrows; j++)
				{
					if(llccodes.at<float>(i, sidxBin.ptr<ushort>(j)[0]) > iRowMax)
					{
						iRowMax = llccodes.at<float>(i, sidxBin.ptr<ushort>(j)[0]);
					}
				}
				beta.at<float>(i, betacol) = iRowMax;
			}
		}
	}

	if(bId != tBins)
	{
		cout<<"Index number error!"<<endl;
		exit;
	}

    llcFeature = Mat(dSize*tBins, 1,  CV_32FC1);

	for(int i = 0; i < tBins; i++)
	{
		for(int j = 0; j < dSize; j++)
		{
			llcFeature.at<float>(j+i*dSize, 0) = beta.at<float>(j, i);
		}
	}

	float sum = 0; //注意类型是float不是int
	for(int i = 0; i < dSize*tBins; i++)
	{
		sum += llcFeature.at<float>(i, 0) * llcFeature.at<float>(i, 0);
	}
	llcFeature = llcFeature/sqrt(sum);

	dictionary.release();
	input.release();
	idxBin.release();
	idxBin.release();
	llccodes.release();
	pyramid.release();
	pBins.release();
	beta.release();
	feaSet_x.release();
	feaSet_y.release();
}

void llc_coding_pooling(string imgDir, string dsiftDir, string llcDir, Mat &dictionary, int knn)
{
	int width, height;
	Mat dsiftFeature, feaSet_x, feaSet_y, llcCodes, llcFeature;

	vector<string> categories;
	GetDirList(imgDir, &categories);

	for (int i = 0; i != categories.size(); i++)
	{
		cout<<"Locality-constrained linear coding: "<<categories[i]<<"..."<<endl;
		string currentCategoryDatabase = imgDir + "/" + categories[i];
		string currentCategoryDsift = dsiftDir + "/" + categories[i];

		vector<string> fileList;
		GetFileList(currentCategoryDatabase, &fileList);

		for (int j = 0; j != fileList.size(); j++)
		{
			string llcFileName = llcDir + "/" + categories[i] + "/" + fileList[j] + ".xml.gz";

			MakeDir(llcDir + "/" + categories[i]);

			FileStorage fs(llcFileName, FileStorage::READ);
			if (fs.isOpened())
			{
				// already cached
				fs["llcFeature"] >> llcFeature;
			}
			else
			{
				string dsiftFileName = currentCategoryDsift + "/" + fileList[j] + ".xml.gz";

				FileStorage fs2(dsiftFileName, FileStorage::READ);
				if (fs2.isOpened())
				{
					// already cached
					fs2["dsiftFeature"] >> dsiftFeature;
					fs2["feaSet_x"] >> feaSet_x;
					fs2["feaSet_y"] >> feaSet_y;
					fs2["width"] >> width;
					fs2["height"] >> height;
				}

				llc_coding(dictionary, dsiftFeature, llcCodes, knn);

                llc_pooling(dictionary, dsiftFeature, llcCodes, llcFeature, feaSet_x, feaSet_y, width, height);

				fs.open(llcFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "llcFeature" << llcFeature;
				}
			}
		}
	}
	dsiftFeature.release();
	feaSet_x.release();
	feaSet_y.release();
	llcCodes.release();
	llcFeature.release();
}
