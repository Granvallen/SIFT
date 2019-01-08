#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm> // for min

using namespace cv;
using namespace std;

// keypoint class
class keypoint
{
public:
	keypoint(int oct = 0, int lyr = 0, Point p = Point(0, 0), double scl = 0, float ang = 0):
		octave(oct), layer(lyr), scale(scl), pt(p), angle(ang) {}

	static void coinvertToKeyPoint(vector<keypoint>& kpts, vector<KeyPoint>& KPts);

public:
	int octave;
	int layer; // ע���0��ʼ
	Point pt;
	double scale; // �߶ȿռ�����
	float angle;
	float response;
	int oct_info;
};

// pyramid class
class pyramid
{
public:
	void appendTo(int oct, Mat& img); // ����������ͼ��
	void build(int oct) { pyr.resize(oct); }
	void clear() { pyr.clear(); }
	int octaves() { return pyr.size(); } // ���� 
	vector<Mat>& operator[] (int oct); // ����[]

private:
	vector<vector<Mat> > pyr;
};



// sift class
class sift
{
public:
	sift(int s = 3, double sigma = 1.6) : S(s), Sigma(sigma), debug(0) { Layers = s + 2; K = pow(2., 1. / s); }

	// sift��⿪ʼ���
	// 0. �������� + ͼ��Ԥ����
	bool detect(const Mat& img, vector<keypoint>& kpts, Mat& fvec);

	void info();

private:

	// 1. ���������� pyr_G  pyr_DoG
	void buildPyramid();

	// 2. ���������
	void findFeaturePoints(vector<keypoint>& kpts);
	// 2.1 Ѱ��DoGͼ��ֵ��
	// 2.2 ��ֵ��ɸѡ
	bool filterExtrema(keypoint& kpt);
	// 2.3 ����������������
	void calcMainOrientation(keypoint& kpt, vector<float>& angs);

	// 3. ��ȡ�����㴦��128ά��������
	void calcFeatureVector(vector<keypoint>& kpts, Mat& fvec);


public:
	int debug;

private:
	int Octaves;
	int Layers; // LayersΪ��˹ģ��������һ��octave��ͼ����-1 �� S = 3  0 1 2 3 4 5 ����ͼ�� Layers = 6 - 1 = 5
	double Sigma;
	int S;
	double K;

	// ͼ��
	Mat img_org; // ԭͼ��
	Mat img; // �Ŵ������Ҷ�ͼ

	// ������
	pyramid pyr_G; // ��˹ģ��������
	pyramid pyr_DoG; // DoG������
};

