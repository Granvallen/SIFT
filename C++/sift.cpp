#include "sift.h"

// keypoint 
void keypoint::coinvertToKeyPoint(vector<keypoint>& kpts, vector<KeyPoint>& KPts)
{
	for (auto kpt : kpts)
	{
		KeyPoint KPt;
		KPt.pt.x = kpt.pt.y * pow(2.f, float(kpt.octave - 1));
		KPt.pt.y = kpt.pt.x * pow(2.f, float(kpt.octave - 1));
		KPt.size = kpt.scale * pow(2.f, float(kpt.octave - 1)) * 2;
		KPt.angle = 360 - kpt.angle;
		KPt.response = kpt.response;
		KPt.octave = kpt.oct_info;
		KPts.push_back(KPt);
	}
}


// pyramid
void pyramid::appendTo(int oct, Mat& img)
{
	Mat temp;
	img.copyTo(temp);
	pyr[oct].push_back(temp);
}

vector<Mat>& pyramid::operator[](int oct)
{
	return pyr[oct];
}


// sift 
// ��⿪ʼ
bool sift::detect(const Mat& image, vector<keypoint>& kpts, Mat& fvec)
{
	image.copyTo(img_org);

	// ����Octave Ĭ�Ͻ��������ϲ� ͼ�񳤿��СֵΪ8
	Octaves = round(log( float(min(img_org.cols, img_org.rows)) ) / log(2.f) - 2.f);

	// ͼ��ҶȻ� ���� ����ת��Ϊfloat����
	Mat img_gray, img_gray_f;
	if (img_org.channels() == 3 || img_org.channels() == 4)
		cvtColor(img_org, img_gray, COLOR_BGR2GRAY);
	else
		img_org.copyTo(img_gray);
	img_gray.convertTo(img_gray_f, CV_32FC1);

	// INTER_LINEAR ˫���Բ�ֵ
	resize(img_gray_f, img, Size(img_gray_f.cols * 2, img_gray_f.rows * 2), 0, 0, INTER_LINEAR);
	// ��ֵ���˲�
	double sigma_init = sqrt(max(Sigma * Sigma - 0.5 * 0.5 * 4, 0.01));
	GaussianBlur(img, img, Size(2 * cvCeil(2 * sigma_init) + 1, 2 * cvCeil(2 * sigma_init) + 1), sigma_init, sigma_init);

	//cout << 1 << endl;

	// Ԥ������� ���������� ����pyr_G �� pyr_DoG
	buildPyramid();

	//cout << 2 << endl;

	// ���������
	findFeaturePoints(kpts);

	//cout << 3 << endl;

	// ��ȡ�����㴦��128ά��������
	calcFeatureVector(kpts, fvec);

	//cout << 4 << endl;

	return true;
}

// ����������
void sift::buildPyramid()
{
	// �ȼ����sigma
	vector<double> sigma_i(Layers + 1);
	sigma_i[0] = Sigma;
	for (int lyr_i = 1; lyr_i < Layers + 1; lyr_i++) // 0  1 2 3 4 5
	{
		double sigma_prev = pow(K, lyr_i - 1) * Sigma;
		double sigma_curr = K * sigma_prev;
		sigma_i[lyr_i] = sqrt(sigma_curr*sigma_curr - sigma_prev*sigma_prev);
	}

	// ��������ʼ��
	pyr_G.clear();
	pyr_DoG.clear();

	// ���ɽ�����
	Mat img_i, img_DoG;
	img.copyTo(img_i);
	pyr_G.build(Octaves); // ȷ������������
	pyr_DoG.build(Octaves);
	for (int oct_i = 0; oct_i < Octaves; oct_i++)
	{
		pyr_G.appendTo(oct_i, img_i); // ÿ���һ��ͼ����Ҫ��˹ģ��
		// ����һ��octave������˹ģ��ͼ�� ͬʱ������һ��octaveDoGͼ��
		for (int lyr_i = 1; lyr_i < Layers + 1; lyr_i++) // 0    1 2 3 4 5
		{
			GaussianBlur(img_i, img_i, Size(2 * cvCeil(2 * sigma_i[lyr_i]) + 1, 2 * cvCeil(2 * sigma_i[lyr_i]) + 1), sigma_i[lyr_i], sigma_i[lyr_i]);
			pyr_G.appendTo(oct_i, img_i);
			subtract(img_i, pyr_G[oct_i][lyr_i - 1], img_DoG, noArray(), CV_32FC1);
			pyr_DoG.appendTo(oct_i, img_DoG);
		}

		// ������������һ��octave�ĵ�һ��ͼ��
		resize(pyr_G[oct_i][Layers - 2], img_i, Size(img_i.cols / 2, img_i.rows / 2), 0, 0, INTER_NEAREST);
	}
}


// ��������� ����Ѱ�Ҽ�ֵ��-> ɸѡ��ֵ��-> ����������������
void sift::findFeaturePoints(vector<keypoint>& kpts)
{
	// ���kpts
	kpts.clear();

	// ��������
	int oct_i, lyr_i, r, c, pr, pc, kpt_i, ang_i, k;

	// 27�����ص�����
	float pxs[27];

	// ��ֵ���ݴ�����
	vector<keypoint> kpts_temp;

	// Ѱ�Ҽ�ֵ��
	int threshold = cvFloor(0.5 * 0.04 / S * 255);
	for (oct_i = 0; oct_i < Octaves; oct_i++)
	{
		for (lyr_i = 1; lyr_i < Layers - 1; lyr_i++) // 0  123  4
		{
			const Mat& img_curr = pyr_DoG[oct_i][lyr_i];
			// �������� �ų����������
			for (r = 1; r < img_curr.rows - 1; r++)
			{
				for (c = 1; c < img_curr.cols - 1; c++)
				{
					// ȡ���Ƚ����ؿ�
					const Mat& prev = pyr_DoG[oct_i][lyr_i - 1];
					const Mat& curr = pyr_DoG[oct_i][lyr_i];
					const Mat& next = pyr_DoG[oct_i][lyr_i + 1];
					float px = curr.at<float>(r, c);

					if (abs(px) >= threshold)
					{
						// ȡ���Ƚ����� -1 0 1
						for (pr = -1, k = 0; pr < 2; pr++)
							for (pc = -1; pc < 2; pc++)
							{
								pxs[k] = prev.at<float>(r + pr, c + pc);
								pxs[k + 1] = curr.at<float>(r + pr, c + pc);
								pxs[k + 2] = next.at<float>(r + pr, c + pc);
								k += 3;
							}

						if ((px >= pxs[0] && px >= pxs[1] && px >= pxs[2] && px >= pxs[3] && px >= pxs[4] &&
								px >= pxs[5] && px >= pxs[6] && px >= pxs[7] && px >= pxs[8] && px >= pxs[9] &&
								px >= pxs[10] && px >= pxs[11] && px >= pxs[12] && px >= pxs[14] && px >= pxs[15] &&
								px >= pxs[16] && px >= pxs[17] && px >= pxs[18] && px >= pxs[19] && px >= pxs[20] &&
								px >= pxs[21] && px >= pxs[22] && px >= pxs[23] && px >= pxs[24] && px >= pxs[25] &&
								px >= pxs[26]) ||
								(px <= pxs[0] && px <= pxs[1] && px <= pxs[2] && px <= pxs[3] && px <= pxs[4] &&
								px <= pxs[5] && px <= pxs[6] && px <= pxs[7] && px <= pxs[8] && px <= pxs[9] &&
								px <= pxs[10] && px <= pxs[11] && px <= pxs[12] && px <= pxs[14] && px <= pxs[15] &&
								px <= pxs[16] && px <= pxs[17] && px <= pxs[18] && px <= pxs[19] && px <= pxs[20] &&
								px <= pxs[21] && px <= pxs[22] && px <= pxs[23] && px <= pxs[24] && px <= pxs[25] &&
								px <= pxs[26]))
						{
							keypoint kpt(oct_i, lyr_i, Point(r, c));
							kpts_temp.push_back(kpt);// ����Ǽ�ֵ���ȱ���
						}
					}

				} 
			}

		}
	}

	for (kpt_i = 0; kpt_i < kpts_temp.size(); kpt_i++)
	{
		if (!filterExtrema(kpts_temp[kpt_i]))
			continue;

		vector<float> angs;
		calcMainOrientation(kpts_temp[kpt_i], angs);

		for (ang_i = 0; ang_i < angs.size(); ang_i++)
		{
			kpts_temp[kpt_i].angle = angs[ang_i];
			kpts.push_back(kpts_temp[kpt_i]);
		}
	}

}


// ɸѡ��ֵ��  �����ϸ񷵻�false
bool sift::filterExtrema(keypoint& kpt)
{
	// �õ�����ֵ
	float biasThreshold = 0.5f;
	float contrastThreshold = 0.04f;
	float edgeThreshold = 10.f;

	float normalscl = 1.f / 255; // ��ͼ��� 0~255 ��һ���� 0~1 ������

	// ɸѡ���� 1 ȥ���뾫ȷ��ֵƫ�ƽϴ�ĵ�
	bool isdrop = true; // �Ƿ�ɾ���ü�ֵ���ʶ

	// ����ɸѡ�����õ��ϵı��� Ԥ������
	// ȡ��ֵ������
	int kpt_r = kpt.pt.x;
	int kpt_c = kpt.pt.y;
	int kpt_lyr = kpt.layer;
	// ����
	Vec3f dD;
	float dxx, dyy, dss, dxy, dxs, dys;
	// ƫ��ֵ
	Vec3f x_hat;

	// 5�β�ֵ�ƽ���ʵ��ֵ
	for (int try_i = 0; try_i < 5; try_i++)
	{
		// ȡDoGͼ�� ÿ�ζ�Ҫ����ȡ ��Ϊlayer���
		const Mat& DoG_prev = pyr_DoG[kpt.octave][kpt_lyr - 1];
		const Mat& DoG_curr = pyr_DoG[kpt.octave][kpt_lyr];
		const Mat& DoG_next = pyr_DoG[kpt.octave][kpt_lyr + 1];

		// ����һ�׵�
		dD = Vec3f((DoG_curr.at<float>(kpt_r, kpt_c + 1) - DoG_curr.at<float>(kpt_r, kpt_c - 1)) * normalscl * 0.5f,
				(DoG_curr.at<float>(kpt_r + 1, kpt_c) - DoG_curr.at<float>(kpt_r - 1, kpt_c)) * normalscl * 0.5f,
				(DoG_prev.at<float>(kpt_r, kpt_c) - DoG_next.at<float>(kpt_r, kpt_c)) * normalscl * 0.5f);

		// ������׵�(Hessian����)   ע���ĸΪ1
		dxx = (DoG_curr.at<float>(kpt_r, kpt_c + 1) + DoG_curr.at<float>(kpt_r, kpt_c - 1) - 2 * DoG_curr.at<float>(kpt_r, kpt_c)) * normalscl;
		dyy = (DoG_curr.at<float>(kpt_r + 1, kpt_c) + DoG_curr.at<float>(kpt_r - 1, kpt_c) - 2 * DoG_curr.at<float>(kpt_r, kpt_c)) * normalscl;
		dss = (DoG_next.at<float>(kpt_r, kpt_c) + DoG_prev.at<float>(kpt_r, kpt_c) - 2 * DoG_curr.at<float>(kpt_r, kpt_c)) * normalscl;


		// ��ϵ� ע���ĸΪ4
		dxy = (DoG_curr.at<float>(kpt_r + 1, kpt_c + 1) - DoG_curr.at<float>(kpt_r + 1, kpt_c - 1)
					- DoG_curr.at<float>(kpt_r - 1, kpt_c + 1) + DoG_curr.at<float>(kpt_r - 1, kpt_c - 1)) * normalscl * 0.25f;
		dxs = (DoG_next.at<float>(kpt_r, kpt_c + 1) - DoG_next.at<float>(kpt_r, kpt_c - 1)
					- DoG_prev.at<float>(kpt_r, kpt_c + 1) + DoG_prev.at<float>(kpt_r, kpt_c - 1)) * normalscl * 0.25f;
		dys = (DoG_next.at<float>(kpt_r + 1, kpt_c) - DoG_next.at<float>(kpt_r - 1, kpt_c)
					- DoG_prev.at<float>(kpt_r + 1, kpt_c) + DoG_prev.at<float>(kpt_r - 1, kpt_c)) * normalscl * 0.25f;

		// �ɶ��׵��ϳ�Hessian����
		Matx33f H(dxx, dxy, dxs,
				  dxy, dyy, dys,
				  dxs, dys, dss);

		// ���ƫ��ֵ
		x_hat = Vec3f(H.solve(dD, DECOMP_LU));

		for (int x = 0; x < 3; x++)// ע�������и� - ��
			x_hat[x] *= -1;

		//if (std::abs(x_hat[0]) >(float)(INT_MAX / 3) ||
		//	std::abs(x_hat[1]) > (float)(INT_MAX / 3) ||
		//	std::abs(x_hat[2]) > (float)(INT_MAX / 3))
		//	return false;

		// ������ֵ������ �Ա������һ�β�ֵ���� 
		kpt_c += round(x_hat[0]);
		kpt_r += round(x_hat[1]);
		kpt_lyr += round(x_hat[2]);

		// �ж�ƫ�Ƴ̶�  ע��sigma��ƫ��ҲҪ����  ����ֱ��ͨ��ɸѡ  ͨ��ɸѡΨһ����
		if (abs(x_hat[0]) < biasThreshold && abs(x_hat[1]) < biasThreshold && abs(x_hat[2]) < biasThreshold) // ��ֵΪ 0.5
		{
			isdrop = false;
			break;
		}

		// �ж��µ�����������Ƿ񳬹��߽�(������һȦ����Ҳ����)   ��������  �����ٴ���������ƫ��
		if (kpt_r < 1 || kpt_r > DoG_curr.rows - 2 ||
			kpt_c < 1 || kpt_c > DoG_curr.cols - 2 ||
			kpt_lyr < 1 || kpt_lyr > Layers - 2) // ����ȡ��һ�������һ�� 0   1 2 3   4
		{
			break;
		}
	}

	// ����õ���ɾ�� ����false
	if (isdrop)
		return false;

	// ɸѡ���� 2 ȥ����Ӧ��С�ļ�ֵ�� ��ֵԽ��Ź��ĵ�Խ��
	// ����ȡDoGͼ��

	const Mat& DoG_curr = pyr_DoG[kpt.octave][kpt_lyr];

	float D_hat = DoG_curr.at<float>(kpt_r, kpt_c) * normalscl + dD.dot(x_hat) * 0.5f; // ����???  vec3fתMatx31f
	if (abs(D_hat) * S < contrastThreshold) // ��opencv��SIFTʵ������ "* s"  ��ԭ����???
		return false;


	// ɸѡ���� 3 ȥ����Ե�ؼ���
	// ����Hessian����ļ�������ʽ
	float trH = dxx + dyy;
	float detH = dxx * dyy - dxy * dxy;

	if (detH <= 0 || trH*trH * edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1) * detH)
		return false;


	// ������ ��ֵ����Ա����� ���¼�ֵ����Ϣ
	kpt.pt = Point(kpt_r, kpt_c);
	kpt.layer = kpt_lyr;
	kpt.scale = pow(K, kpt_lyr) * Sigma; // ���������ڳ߶ȿռ�ĳ߶�
	kpt.response = D_hat;
	kpt.oct_info = kpt.octave + (kpt.layer << 8) + (cvRound((x_hat[2] + 0.5) * 255) << 16);
	return true;
}


// ����������������
void sift::calcMainOrientation(keypoint& kpt, vector<float>& angs)
{
	// ȡ����������Ϣ
	int kpt_oct = kpt.octave;
	int kpt_lyr = kpt.layer;
	int kpt_r = kpt.pt.x;
	int kpt_c = kpt.pt.y;
	double kpt_scl = kpt.scale;

	int radius = round(3 * 1.5 * kpt_scl); // ���������������뾶
	const Mat& pyr_G_i = pyr_G[kpt_oct][kpt_lyr]; // ȡ����˹ģ��ͼ��


	int px_r, px_c; // ���������صľ�������(img_r, img_c)


	float histtemp[36 + 4] = { 0 }; // ���߸������2����λ���ڲ�ֵ����    0  1  2~37  38  39


	// �������������������   (radius * 2 + 1)*(radius * 2 + 1)
	for (int i = -radius; i <= radius; i++) // ���ϵ��� i������
	{
		px_r = kpt_r + i;
		if (px_r <= 0 || px_r >= pyr_G_i.rows - 1) // ���곬��ͼ�� �Ҳ�����ͼ�����Ե����
			continue;

		for (int j = -radius; j <= radius; j++) // ������ j������
		{

			px_c = kpt_c + j;
			if (px_c <= 0 || px_c >= pyr_G_i.cols - 1)
				continue;

			// �����ݶ� ע��dy�ļ���
			float dx = pyr_G_i.at<float>(px_r, px_c + 1) - pyr_G_i.at<float>(px_r, px_c - 1);
			float dy = pyr_G_i.at<float>(px_r - 1, px_c) - pyr_G_i.at<float>(px_r + 1, px_c);

			// �����ݶ� ��ֵ �� ����
			float mag = sqrt(dx * dx + dy * dy);
			float ang = fastAtan2(dy, dx);

			// ���������ֱ��ͼ����
			int bin = round(ang * 36.f / 360.f); 

			if (bin >= 36) // binȡ 0~35
				bin -= 36;
			else if (bin < 0)
				bin += 36;

			// �����˹��Ȩ��
			float w_G = exp( -(i * i + j * j) / (2 * (1.5 * kpt_scl) * (1.5 * kpt_scl)) );

			// ����histtemp   0  1  2~37  38  39
			histtemp[bin + 2] += mag * w_G;

		}
	}

	// ��ֱ��ͼƽ������ ���histtemp�Ŀ�λ     0  1  2~37  38  39
	float hist[36] = { 0 };// ���߸������2����λ���ڲ�ֵ����    0  1  2~37  38  39  

	histtemp[0] = histtemp[36];
	histtemp[1] = histtemp[37];
	histtemp[38] = histtemp[2];
	histtemp[39] = histtemp[3];

	// ��Ȩ�ƶ�ƽ�� �� ����ֵ
	float hist_max = 0;
	for (int k = 2; k < 40 - 2; k++)
	{
		hist[k - 2] = (histtemp[k - 2] + histtemp[k + 2]) * (1.f / 16) + 
				      (histtemp[k - 1] + histtemp[k + 1]) * (4.f / 16) +
					   histtemp[k] * (6.f / 16);
		// ˳��������ֵ
		if (hist[k - 2] > hist_max)
			hist_max = hist[k - 2];
	}

	// ����ֱ��ͼ �� ������ �� ������
	float histThreshold = 0.8f * hist_max; // ���������ֵ

	for (int k = 0; k < 36; k++) // 0 ~ 35
	{
		int kl = k > 0 ? k - 1 : 36 - 1;
		int kr = k < 36 - 1 ? k + 1 : 0;

		if (hist[k] > hist[kl] && hist[k] > hist[kr] && hist[k] >= histThreshold)
		{
			// ͨ�� �����߲�ֵ ���㾫ȷ���ǹ�ʽ  ��ȷֵ��Χ  0 ~ 35...  |36 
			float bin = k + 0.5f * (hist[kl] - hist[kr]) / (hist[kl] - 2 * hist[k] + hist[kr]);

			if (bin < 0) // binԽ�紦��
				bin += 36;
			else if (bin >= 36)
				bin -= 36;

			// ���㾫ȷ����  angΪ�Ƕ���
			float ang = bin * (360.f / 36);
			if (abs(ang - 360.f) < FLT_EPSILON)
				ang = 0.f;

			// ����ü�ֵ�� ������ �� ������
			angs.push_back(ang);
		}
	}
}


// ������������
void sift::calcFeatureVector(vector<keypoint>& kpts, Mat& fvec)
{
	// fvec������������ɵľ��� ÿһ����һ���������128ά���� �������ڴ�ռ�
	fvec.create(kpts.size(), 128, CV_32FC1);

	for (int kpt_i = 0; kpt_i < kpts.size(); kpt_i++)
	{
		// ��������
		int i, j, k, ri, ci, oi;

		// ȡ����������Ϣ
		int kpt_oct = kpts[kpt_i].octave;
		int kpt_lyr = kpts[kpt_i].layer;
		int kpt_r = kpts[kpt_i].pt.x;
		int kpt_c = kpts[kpt_i].pt.y;
		double kpt_scl = kpts[kpt_i].scale;
		float kpt_ang = kpts[kpt_i].angle;

		int d = 4; // ��������Ŀ ���� 4 * 4 * 8 �е�4
		int n = 8; // �ݶ�ֱ��ͼ����Ŀ 8����������

		// ȡ����Ӧ�߶ȵĸ�˹ģ��ͼ��
		const Mat& pyr_G_i = pyr_G[kpt_oct][kpt_lyr];

		float hist_width = 3 * kpt_scl; // ������߳�  ����

		// ������ת������� cos���ܵ��ǻ�����
		// ������ת����������������߶Ƚ��й�һ�� /hist_width  �Ա���֮����������ת�������(r_rot, c_rot)������������߶ȵ�
		float cos_t = cosf(kpt_ang * CV_PI / 180) / hist_width;
		float sin_t = sinf(kpt_ang * CV_PI / 180) / hist_width;

		int radius = round(hist_width * 1.4142135623730951f * (d + 1) * 0.5); // �������в��������������뾶   Բ��

		// �ж��¼�����İ뾶��ͼ��Խ��߳� ���뾶ȡС��
		// ���뾶��ͼ��Խ���һ����ʱ��һ������ �ٴ󲻹�Ҳ�Ǳ���ͼ���������� ����û������
		radius = min(radius, (int)sqrt(pyr_G_i.rows * pyr_G_i.rows + pyr_G_i.cols * pyr_G_i.cols));


		// Ԥ����ռ�
		int histlen = (d + 2) * (d + 2) * (n + 2);
		AutoBuffer<float> hist(histlen);
		// ��ʼ��hist
		memset(hist, 0, sizeof(float) * histlen);


		// ���ɱ����������������ص� ����ֱ��ͼ
		for (i = -radius; i <= radius; i++) // ���ϵ��� i������
		{
			for (j = -radius; j <= radius; j++) // ������ j������
			{

				// ������ת����������(ע��������������߶ȵ�)
				float c_rot = j * cos_t - i * sin_t;
				float r_rot = j * sin_t + i * cos_t;

				float rbin = r_rot + d / 2.f - 0.5f; // ��������������߶��µľ������� ������0.5��ƽ��
				float cbin = c_rot + d / 2.f - 0.5f; // ȡֵ��Χ  -1 ~ 3... |4

				// 0.5��ƽ��ʹ�������������ύ�㶼�������������Ͻ� ����֮���ֵ ����һ�����ص����Χ�ĸ��������ֱ��ͼ����
				// ��������ͼ���������(px_r, px_c)
				int px_r = kpt_r + i;
				int px_c = kpt_c + j;

				// �����ת����������ͼ���� ����ֱ��ͼ����
				if (-1 < rbin && rbin < d && -1 < cbin && cbin < d &&
					0 < px_r && px_r < pyr_G_i.rows - 1 && 0 < px_c && px_c < pyr_G_i.cols - 1)
				{
					// �����ݶ� ע�� dy�ļ���
					float dx = pyr_G_i.at<float>(px_r, px_c + 1) - pyr_G_i.at<float>(px_r, px_c - 1);
					float dy = pyr_G_i.at<float>(px_r - 1, px_c) - pyr_G_i.at<float>(px_r + 1, px_c);
					
					// �����ݶ� ��ֵ �� ����
					float mag = sqrt(dx * dx + dy * dy);
					float ang = fastAtan2(dy, dx);

					// �жϷ��������ĸ�ֱ��ͼ������
					float obin = (ang - kpt_ang) * (n / 360.f); // ȡֵ 0 ~ 7... | 8
					
					// �����˹��Ȩ��ķ�ֵ
					float w_G = expf(-(r_rot * r_rot + c_rot * c_rot) / (0.5f * d * d)); // �����˹��Ȩ�� - 1 / ((d / 2) ^ 2 * 2)   -> - 1 / (d ^ 2 * 0.5)
					mag *= w_G;

					int r0 = cvFloor(rbin); // -1 0 1 2 3
					int c0 = cvFloor(cbin);
					int o0 = cvFloor(obin); // 0 ~ 7

					// �൱���������ڵ�����
					rbin -= r0;
					cbin -= c0;
					obin -= o0;

					// ����o0Խ��ѭ��
					if (o0 < 0)
						o0 += n;
					else if (o0 >= n)
						o0 -= n;

					// �����Բ�ֵ ���hist������ �����ص��ֵ���׵���Χ�ĸ��������ֱ��ͼ��ȥ
					// ����8������ֵ    0 < rbin cbin obin < 1
					// �ȼ��㹱��Ȩֵ
					float v_rco000 = rbin * cbin * obin;
					float v_rco001 = rbin * cbin * (1 - obin);

					float v_rco010 = rbin * (1 - cbin) * obin;
					float v_rco011 = rbin * (1 - cbin) * (1 - obin);

					float v_rco100 = (1 - rbin) * cbin * obin;
					float v_rco101 = (1 - rbin) * cbin * (1 - obin);

					float v_rco110 = (1 - rbin) * (1 - cbin) * obin;
					float v_rco111 = (1 - rbin) * (1 - cbin) * (1 - obin);

					// rbin     0 ~ 5          r0  -1 ~ 3
					// cbin     0 ~ 5          c0  -1 ~ 3
					// obin     0 ~ 7 | 8 9  ��ֵ�ᵽ8     o0   0 ~ 7
					hist[60 * (r0+1) + 10 * (c0+1) + o0] += mag * v_rco000;
					hist[60 * (r0+1) + 10 * (c0+1) + (o0+1)] += mag * v_rco001;

					hist[60 * (r0+1) + 10 * (c0+2) + o0] += mag * v_rco010;
					hist[60 * (r0+1) + 10 * (c0+2) + (o0+1)] += mag * v_rco011;
					
					hist[60 * (r0+2) + 10 * (c0+1) + o0] += mag * v_rco100;
					hist[60 * (r0+2) + 10 * (c0+1) + (o0+1)] += mag * v_rco101;

					hist[60 * (r0+2) + 10 * (c0+2) + o0] += mag * v_rco110;
					hist[60 * (r0+2) + 10 * (c0+2) + (o0+1)] += mag * v_rco111;
				}
			}
		}

	
		// ���� 4 * 4 ������  ��histֱ��ͼ�е�����������
		float fvec_i[128] = { 0 };
		for (ri = 1, k = 0; ri <= 4; ri++)
			for (ci = 1; ci <= 4; ci++)
			{
				hist[60 * ri + 10 * ci + 0] += hist[60 * ri + 10 * ci + 8];
				hist[60 * ri + 10 * ci + 1] += hist[60 * ri + 10 * ci + 9];
				
				for (oi = 0; oi < 8; oi++)
					fvec_i[k++] = hist[60 * ri + 10 * ci + oi];
			}


		// ���������Ż�
		float scl;
		float fvec_norm = 0, fvecThreshold;
		for (k = 0; k < 128; k++)
			fvec_norm += fvec_i[k] * fvec_i[k];

		fvecThreshold = 0.2f * sqrtf(fvec_norm);

		// ������������ֵ��ķ������д���  ���Ʒ����Թ��ոı�Ӱ��
		for (k = 0, fvec_norm = 0; k < 128; k++)
		{
			if (fvec_i[k] > fvecThreshold)
				fvec_i[k] = fvecThreshold;
			fvec_norm += fvec_i[k] * fvec_i[k];
		}

		// ��һ�� ���洦�������������
		scl = 1 / max(std::sqrt(fvec_norm), FLT_EPSILON);
		float* fvec_temp = fvec.ptr<float>(kpt_i);
		for (k = 0; k < 128; k++)
			fvec_temp[k] = fvec_i[k] * scl;

	}
}


// ��ӡsift����Ϣ
void sift::info()
{
	cout << "Sigma" << Sigma << endl;
	cout << "S" << S << endl;
	cout << "Octaves" << Octaves << endl;
	cout << "Layers" << Layers << endl;
}



