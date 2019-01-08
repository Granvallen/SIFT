#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "sift.h"

using namespace std;
using namespace cv;


// 优选匹配点
vector<DMatch> chooseGood(Mat descriptor, vector<DMatch> matches)
{
	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptor.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	vector<DMatch> goodMatches;
	for (int i = 0; i < descriptor.rows; i++)
	{
		if (matches[i].distance < 0.2 * max_dist)
			goodMatches.push_back(matches[i]);
	}
	return goodMatches;
}

// 匹配对称性检测
void symmetryTest(vector<cv::DMatch>& matches1, vector<cv::DMatch>& matches2, vector<cv::DMatch>& symMatches)
{
	for (auto m1 : matches1)
		for (auto m2 : matches2)
		{
			// 进行匹配测试
			if (m1.queryIdx == m2.trainIdx  &&	m2.queryIdx == m1.trainIdx)
			{
				symMatches.push_back(m1);
				break;
			}
		}
}

// SIFT Demo
int main()
{
	Mat img1 = imread("lena1.jpg");
	Mat img2 = imread("lena2.jpg");

	sift siftlab(3, 1.6);

	vector<keypoint> kpts1, kpts2; // 关键点
	Mat fvec1, fvec2; // 特征矩阵

	siftlab.detect(img1, kpts1, fvec1);
	siftlab.detect(img2, kpts2, fvec2);

	cout << "img1 keypoint: " << kpts1.size() << endl;
	cout << "img2 keypoint: " << kpts2.size() << endl;

	vector<KeyPoint> KPts1, KPts2; // 转化到openCV 的 KeyPoint类型
	keypoint::coinvertToKeyPoint(kpts1, KPts1);
	keypoint::coinvertToKeyPoint(kpts2, KPts2);

	// 绘制特征点(关键点)
	Mat fpic1, fpic2;
	drawKeypoints(img1, KPts1, fpic1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img2, KPts2, fpic2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//drawKeypoints(img1, KPts1, fpic1, Scalar(0, 0, 255));
	//drawKeypoints(img2, KPts2, fpic2, Scalar(0, 0, 255));

	imshow("fpic1", fpic1);
	imshow("fpic2", fpic2);
	cvMoveWindow("fpic1", 100, 100);
	cvMoveWindow("fpic2", 100, 100);

	// 图像匹配
	BFMatcher matcher;
	vector<Mat> train_dest_collection(1, fvec1);
	matcher.add(train_dest_collection);
	matcher.train();

	vector<DMatch> matches1, matches2;   //定义连接对象
	matcher.match(fvec1, fvec2, matches1);  //生成匹配对
	matcher.match(fvec2, fvec1, matches2);

	// 匹配优化
	vector<DMatch> goodMatches1, goodMatches2, symMatches;
	goodMatches1 = chooseGood(fvec1, matches1);
	goodMatches2 = chooseGood(fvec2, matches2);

	Mat img_matches;
	symmetryTest(goodMatches1, goodMatches2, symMatches);
	drawMatches(img1, KPts1, img2, KPts2, symMatches, img_matches);

	imshow("matches", img_matches);
	//imwrite("lenademo.jpg", img_matches);

	waitKey(0);


    return 0;
}
