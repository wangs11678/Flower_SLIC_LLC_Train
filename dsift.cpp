#include "dsift.h"
#include "utils.h"

using namespace std;
using namespace cv;

//利用opencv自带的sift特征提取
void dsift_opencv(Mat &img, int step, int patchSize, Mat &dsiftFeature)
{
	SiftDescriptorExtractor sift;
	vector<KeyPoint> keypoints; // keypoint storage

    int width = img.cols;
    int height = img.rows;

    int remX = (width - patchSize) % step;
	int offsetX = floor(remX/2) + 1;
	int remY = (height - patchSize) % step;
	int offsetY = floor(remY/2) + 1;

    /*
     * 利用opencv自带的提取sift特征代码，
     * 自定义KeyPoint(关键点)，改编成
     * dense sift特征提取
     */
	// manual keypoint grid
	for (int y=offsetY+patchSize/2; y<=height-patchSize/2+1; y+=step)
    {
		for (int x=offsetX+patchSize/2; x<=width-patchSize/2+1; x+=step)
		{
			// x,y,radius
			keypoints.push_back(KeyPoint(float(x), float(y), float(step)));
		}
	}
	sift.compute(img, keypoints, dsiftFeature);
}


//利用vlfeat库提取dsift特征
void dsift_vlfeat(Mat &img, int step, int binSize, Mat &dsiftFeature)
{
	VlDsiftFilter * vlf = vl_dsift_new_basic(img.rows, img.cols, step, binSize);
	// transform image in cv::Mat to float vector
	std::vector<float> imgvec;

	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			imgvec.push_back(img.at<unsigned char>(i,j) / 255.0f);
		}
	}
	// call processing function of vl
	vl_dsift_process(vlf, &imgvec[0]);

	//定义Mat存放提取到的dsift特征
    dsiftFeature = Mat(vl_dsift_get_keypoint_num(vlf), vlf->descrSize, CV_32FC1);

	for(int i = 0; i < vl_dsift_get_keypoint_num(vlf); i++)
	{
		for(int j = 0; j < vlf->descrSize; j++)
		{
			//将提取到的特征赋值给dsiftFeature保存
			dsiftFeature.at<float>(i,j) = vlf->descrs[j+i*vlf->descrSize];
		}
	}

	// Extract keypoints
    //const VlDsiftKeypoint * vlkeypoints;
    //vlkeypoints = vl_dsift_get_keypoints(vlf);
	//for (int i = 0; i < vl_dsift_get_keypoint_num(vlf); i++)
	//{
	//	cout << vlkeypoints[i].x << ", ";
	//	cout << vlkeypoints[i].y << endl;
	//}

	vl_dsift_delete(vlf); //释放资源，不然会挤爆内存
}


void img_resize(Mat &image, int maxImgSize)
{
	Mat grayImg;
	cvtColor(image, grayImg, CV_BGR2GRAY);
	image = grayImg;

	if(image.rows > maxImgSize || image.rows >maxImgSize)
	{
		int max_size = image.rows>image.cols?image.rows:image.cols;
		float scale = float(maxImgSize)/max_size;
		Size dsize = Size(image.cols*scale, image.rows*scale);
		Mat destImg = Mat(dsize, CV_32S);
		resize(image, destImg, dsize);
		image = destImg;
	}

	grayImg.release();
}


void meshgrid(const cv::Range &xgv, const cv::Range &ygv, int step, cv::Mat &X, cv::Mat &Y)
{
    std::vector<int> t_x, t_y;
    for(int i = xgv.start; i <= xgv.end; i += step) t_x.push_back(i);
    for(int j = ygv.start; j <= ygv.end; j += step) t_y.push_back(j);

    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, X);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), Y);
}


void calculateSiftXY_opencv(int width, int height, int patchSize, int binSize, int step, Mat &feaSet_x, Mat &feaSet_y)
{
	int remX = (width - patchSize) % step;
	int offsetX = floor(remX/2) + 1;
	int remY = (height - patchSize) % step;
	int offsetY = floor(remY/2) + 1;

	cv::Mat gridX, gridY, gridXX, gridYY;
	meshgrid(cv::Range(offsetX, width-patchSize+1), cv::Range(offsetY, height-patchSize+1), step, gridXX, gridYY);

	transpose(gridXX, gridX);
	transpose(gridYY, gridY);

	for(int i = 0; i < gridX.rows; i++)
	{
		for(int j = 0; j < gridX.cols; j++)
		{
			feaSet_x.at<float>(j+i*gridX.cols, 0) = gridX.ptr<int>(i)[j] + patchSize/2;
		}
	}

	for(int i = 0; i < gridY.rows; i++)
	{
		for(int j = 0; j < gridY.cols; j++)
		{
			feaSet_y.at<float>(j+i*gridY.cols, 0) = gridY.ptr<int>(i)[j] + patchSize/2;
		}
	}

	gridX.release();
	gridY.release();
	gridXX.release();
	gridYY.release();
}


void calculateSiftXY_vlfeat(int width, int height, int patchSize, int binSize, int step, Mat &feaSet_x, Mat &feaSet_y)
{
    float offsetX = 0.5*binSize*(4-1);
    float offsetY = 0.5*binSize*(4-1);

	cv::Mat gridX, gridY, gridXX, gridYY;
	meshgrid(cv::Range(offsetX, width-patchSize/2+1), cv::Range(offsetY, height-patchSize/2+1), step, gridXX, gridYY);

	transpose(gridXX, gridX);
	transpose(gridYY, gridY);

	for(int i = 0; i < gridX.rows; i++)
	{
		for(int j = 0; j < gridX.cols; j++)
		{
			feaSet_x.at<float>(j+i*gridX.cols, 0) = gridX.ptr<int>(i)[j];
		}
	}

	for(int i = 0; i < gridY.rows; i++)
	{
		for(int j = 0; j < gridY.cols; j++)
		{
			feaSet_y.at<float>(j+i*gridY.cols, 0) = gridY.ptr<int>(i)[j];
		}
	}

	gridX.release();
	gridY.release();
	gridXX.release();
	gridYY.release();
}


void extract_dsift_feature(string imgDir, string dsiftDir, int step, int binSize, int patchSize)
{
	int maxImgSize = 300;

	vector<string> categories;
	GetDirList(imgDir, &categories);

	for (int i = 0; i != categories.size(); i++)
	{
		cout<<"Dense SIFT: "<<categories[i]<<"..."<<endl;
		string currentCategory = imgDir + "/" + categories[i];

		vector<string> fileList;
		GetFileList(currentCategory, &fileList);

		for (int j = 0; j != fileList.size(); j++)
		{
			MakeDir(dsiftDir + "/" + categories[i]);
			string dsiftFileName = dsiftDir + "/" + categories[i] + "/" + fileList[j] + ".xml.gz";

			Mat dsiftFeature;

			FileStorage fs(dsiftFileName, FileStorage::READ);
			if (fs.isOpened())
			{
				// already cached
				continue;
			}
			else
			{
				string filepath = currentCategory + "/" + fileList[j];
				Mat img = imread(filepath);

				if (img.empty())
				{
					continue; // maybe not an image file
				}

                img_resize(img, maxImgSize);

				int width = img.cols;
				int height = img.rows;

				//dsift_opencv(img, step, patchSize, dsiftFeature);
				dsift_vlfeat(img, step, binSize, dsiftFeature);

                Mat feaSet_x(dsiftFeature.rows, 1, CV_32FC1);
                Mat feaSet_y(dsiftFeature.rows, 1, CV_32FC1);

				//calculateSiftXY_opencv(width, height, patchSize, binSize, step, feaSet_x, feaSet_y);
				calculateSiftXY_vlfeat(width, height, patchSize, binSize, step, feaSet_x, feaSet_y);

				fs.open(dsiftFileName, FileStorage::WRITE);
				if (fs.isOpened())
				{
					fs << "dsiftFeature" << dsiftFeature;
					fs << "feaSet_x" << feaSet_x;
					fs << "feaSet_y" << feaSet_y;
					fs << "width" << width;
					fs << "height" << height;
				}
			}
		}
	}
}

