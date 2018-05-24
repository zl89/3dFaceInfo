

#include "SWFaceMatchApi.h"
#include "facerecon.h"
#include "dataprocess.h"

#include <fstream>
#include <time.h>

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <list>


using namespace cv;
using namespace std;

double g_meanface[15] = {	-31.1297 , 34.0010 , 122.4674,
							30.9267   , 33.7814  , 122.1970 ,
							 0.0132  ,   4.4303  , 161.5600 ,
							-29.3909  , -33.8801  , 128.1061 ,
							 29.1029  , -33.5093  , 127.8227};

double g_meanfacelm2[10] = {	30.2946, 51.6963 - 25,
								65.5318, 51.5014 - 25,
								48.0252, 71.7366,
								33.5493, 92.3615,
								62.7299, 92.2041 };
int g_img_size = 450;
Size m_cropimg_size = Size(96,112);

extern int g_bnormalize ;
extern int g_bnormalizeX ;
extern int g_bnormalizeY ;


//vector<Point2f> g_meanface2D = {Point2f(30.2946, 51.6963),
//								Point2f(65.5318, 51.5014),
//								Point2f(48.0252, 71.7366),
//								Point2f(33.5493, 92.3655),
//								Point2f(62.7299, 92.2041)};


Mat tformfwd(const Mat& trans, const Mat& uv)
{
	Mat uv_h = Mat::ones(uv.rows, 3, CV_64FC1);
	uv.copyTo(uv_h(Rect(0, 0, 2, uv.rows)));
	Mat xv_h = uv_h*trans;
	return xv_h(Rect(0, 0, 2, uv.rows));
}

Mat find_none_flectives_similarity(const Mat& uv, const Mat& xy)
{
	Mat A = Mat::zeros(2 * xy.rows, 4, CV_64FC1);
	Mat b = Mat::zeros(2 * xy.rows, 1, CV_64FC1);
	Mat x = Mat::zeros(4, 1, CV_64FC1);

	xy(Rect(0, 0, 1, xy.rows)).copyTo(A(Rect(0, 0, 1, xy.rows)));//x
	xy(Rect(1, 0, 1, xy.rows)).copyTo(A(Rect(1, 0, 1, xy.rows)));//y
	A(Rect(2, 0, 1, xy.rows)).setTo(1.);

	xy(Rect(1, 0, 1, xy.rows)).copyTo(A(Rect(0, xy.rows, 1, xy.rows)));//y
	(xy(Rect(0, 0, 1, xy.rows))).copyTo(A(Rect(1, xy.rows, 1, xy.rows)));//-x
	A(Rect(1, xy.rows, 1, xy.rows)) *= -1;
	A(Rect(3, xy.rows, 1, xy.rows)).setTo(1.);

	uv(Rect(0, 0, 1, uv.rows)).copyTo(b(Rect(0, 0, 1, uv.rows)));
	uv(Rect(1, 0, 1, uv.rows)).copyTo(b(Rect(0, uv.rows, 1, uv.rows)));

	cv::solve(A, b, x, cv::DECOMP_SVD);
	Mat trans_inv = (Mat_<double>(3, 3) << x.at<double>(0), -x.at<double>(1), 0,
		x.at<double>(1), x.at<double>(0), 0,
		x.at<double>(2), x.at<double>(3), 1);
	Mat trans = trans_inv.inv(cv::DECOMP_SVD);
	trans.at<double>(0, 2) = 0;
	trans.at<double>(1, 2) = 0;
	trans.at<double>(2, 2) = 1;

	return trans;
}

Mat find_similarity(const Mat& uv, const Mat& xy)
{
	Mat trans1 = find_none_flectives_similarity(uv, xy);
	Mat xy_reflect = xy;
	xy_reflect(Rect(0, 0, 1, xy.rows)) *= -1;
	Mat trans2r = find_none_flectives_similarity(uv, xy_reflect);
	Mat reflect = (Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, 1);

	Mat trans2 = trans2r*reflect;
	Mat xy1 = tformfwd(trans1, uv);

	double norm1 = cv::norm(xy1 - xy);

	Mat xy2 = tformfwd(trans2, uv);
	double norm2 = cv::norm(xy2 - xy);

	Mat trans;
	if (norm1 < norm2){
		trans = trans1;
	}
	else {
		trans = trans2;
	}
	return trans;
}

Mat get_similarity_transform(const vector<Point2f>& src_points, const vector<Point2f>& dst_points, bool reflective = true)
{
	Mat trans;
	Mat src((int)src_points.size(), 2, CV_32FC1, (void*)(&src_points[0].x));
	src.convertTo(src, CV_64FC1);

	Mat dst((int)dst_points.size(), 2, CV_32FC1, (void*)(&dst_points[0].x));
	dst.convertTo(dst, CV_64FC1);

	if (reflective){
		trans = find_similarity(src, dst);
	}
	else {
		trans = find_none_flectives_similarity(src, dst);
	}
	Mat trans_cv2 = trans(Rect(0, 0, 2, trans.rows)).t();

	return trans_cv2;
}

Mat align_face(const cv::Mat& src, const double* landmark)
{

	vector<Point2f> detect_points;
	for (int i = 0; i < 5; i++)
	{
		detect_points.push_back(Point2f(landmark[i], landmark[i + 5]));
	}
	vector<Point2f> reference_points;

	reference_points.push_back(cv::Point2f(30.2946, 51.6963 - 25));
	reference_points.push_back(cv::Point2f(65.5318, 51.5014 - 25));
	reference_points.push_back(cv::Point2f(48.0252, 71.7366));
	reference_points.push_back(cv::Point2f(33.5493, 92.3615));
	reference_points.push_back(cv::Point2f(62.7299, 92.2041));



	Mat tfm = get_similarity_transform(detect_points, reference_points);
	Mat aligned_face;
	warpAffine(src, aligned_face, tfm, m_cropimg_size);
	return aligned_face;
}
Mat align_face(const cv::Mat& src, const double* landmark, int ptCount, int lossindex)
{

	std::vector<Point2f> detect_points;
	for (int i = 0; i < ptCount; i++)
	{
		detect_points.push_back(Point2f(landmark[i], landmark[i + ptCount]));
	}
	std::vector<Point2f> reference_points;

	for (int i = 0, n = 0; i < ptCount; n++, i++)
	{
		if (n == lossindex)
			n++;
		reference_points.push_back(cv::Point2f(g_meanfacelm2[2 * n], g_meanfacelm2[2 * n + 1]));
	}



	Mat tfm = get_similarity_transform(detect_points, reference_points);
	Mat aligned_face;
	warpAffine(src, aligned_face, tfm, m_cropimg_size);
	return aligned_face;
}
Mat align_face2(const cv::Mat& src, const double* landmark, int ptCount, int lossindex)
{

	std::vector<Point2f> detect_points;
	for (int i = 0; i < ptCount; i++)
	{
		detect_points.push_back(Point2f(landmark[i * 2], landmark[i * 2 + 1]));
	}
	std::vector<Point2f> reference_points;

	for (int i = 0, n = 0; i < ptCount; n++, i++)
	{
		if (n == lossindex)
			n++;
		reference_points.push_back(cv::Point2f(g_meanfacelm2[2 * n], g_meanfacelm2[2 * n + 1]));
	}



	Mat tfm = get_similarity_transform(detect_points, reference_points);
	Mat aligned_face;
	warpAffine(src, aligned_face, tfm, m_cropimg_size);
	return aligned_face;
}
Mat align_face224(const cv::Mat& src, const double* landmark, int ptCount, int lossindex)
{

	std::vector<Point2f> detect_points;
	for (int i = 0; i < ptCount; i++)
	{
		detect_points.push_back(Point2f(landmark[i * 2], landmark[i * 2 + 1]));
	}
	std::vector<Point2f> reference_points;

	for (int i = 0, n = 0; i < ptCount; n++, i++)
	{
		if (n == lossindex)
			n++;
		reference_points.push_back(cv::Point2f(g_meanfacelm2[2 * n] * 2, g_meanfacelm2[2 * n + 1] * 2));
	}
	Mat tfm = get_similarity_transform(detect_points, reference_points);
	Mat aligned_face;
	warpAffine(src, aligned_face, tfm, m_cropimg_size * 2);
	return aligned_face;
}
template<typename Ty>
void savedata(const char* filename, Ty *data,int num,int dim)
{
	ofstream ofile;               //定义输出文件
	ofile.open(filename, ios::out);     //作为输出文件打开

	for(int i =0;i<num;i++)
	{	for(int j =0;j<dim;j++)
			ofile << data[i+j*num] << " ";
		ofile<<endl;
	}
	ofile.close();
}

Mat procrustes(double *meanfacelm3, double *facelm3, int keycount, double *data, int nPtCount, int dim = 3)
{
	Mat meanface(keycount, 1, CV_64FC3, meanfacelm3);
	meanface = meanface.t();
	Mat face(keycount, 1, CV_64FC3, facelm3);
	face = face.t();

	Scalar mu_x = cv::mean(meanface);
	Mat tempmeanface = meanface - Mat(meanface.size(), meanface.type(), mu_x);
	double normX = cv::norm(tempmeanface);
	tempmeanface = tempmeanface / normX;
	tempmeanface = tempmeanface.reshape(1, keycount);

	Scalar mu_y = cv::mean(face);
	Mat tempface = face - Mat(face.size(), face.type(), mu_y);
	double normY = cv::norm(tempface);
	tempface = tempface / normY;
	tempface = tempface.reshape(1, keycount);

	Mat A = tempmeanface.t()*tempface;

	Mat w, u, vt;
	SVDecomp(A, w, u, vt);

	Mat T = vt.t()*u.t();

	double traceTA = cv::sum(w)[0];
	double b = traceTA * normX / normY;
	double d = 1 - traceTA *traceTA;

	Mat muY = Mat(Size(3, 1), CV_64FC3, mu_y);
	muY = muY.reshape(1, 3);
	Mat muX = Mat(Size(3, 1), CV_64FC3, mu_x);
	muX = muX.reshape(1, 3);
	Mat c = muX - b*muY*T;

	Mat trans = -1 / b *c*T.t();

	Mat transM = Mat::zeros(4, 4, CV_64FC1);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			transM.at<double>(i, j) = 1.0 / b * T.at<double>(i, j);
		}
	}

	transM.at<double>(0, 3) = trans.at<double>(0, 0);
	transM.at<double>(1, 3) = trans.at<double>(0, 1);
	transM.at<double>(2, 3) = trans.at<double>(0, 2);
	transM.at<double>(3, 3) = 1;

	Mat transMinv;
	cv::invert(transM, transMinv);


	Mat tempss = Mat(nPtCount, 3, CV_64FC1, data);

	Mat ss = Mat(4, nPtCount, CV_64FC1);

	//tempss.copyTo(ss(Rect(0, 0, 3, nPtCount)));
	for (int i = 0; i < nPtCount; i++)
	{
		ss.at<double>(0, i) = tempss.at<double>(i, 0);
		ss.at<double>(1, i) = tempss.at<double>(i, 1);
		ss.at<double>(2, i) = tempss.at<double>(i, 2);
		ss.at<double>(3, i) = 1;
	}

	//Mat res = ss*transMinv.t();
	//res=res.t();
	Mat res = transMinv*ss;
	Mat re = res(Rect(0, 0, nPtCount, 3));

	return re;

}
bool PointInTri(Point2d&p, Point2d& a, Point2d& b, Point2d& c)
{

	float signOfTrig = (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
	float signOfAB = (b.x - a.x)*(p.y - a.y) - (b.y - a.y)*(p.x - a.x);
	float signOfCA = (a.x - c.x)*(p.y - c.y) - (a.y - c.y)*(p.x - c.x);
	float signOfBC = (c.x - b.x)*(p.y - c.y) - (c.y - b.y)*(p.x - c.x);

	bool d1 = (signOfAB * signOfTrig > 0);
	bool d2 = (signOfCA * signOfTrig > 0);
	bool d3 = (signOfBC * signOfTrig > 0);

	return d1 && d2 && d3;

}
void ZBuffer(double* vertex, int nver, int* tri, int ntri, double *marklandlm3, int nKey, double* depth, double *marklandlm2)
{
	float f = 0.9;
	Mat proj(3, 4, CV_64FC1);
	double r[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
	Mat R(3, 3, CV_64FC1, r);
	R = f*R;
	R.copyTo(proj(Rect(0, 0, 3, 3)));
	proj.at<double>(0, 3) = 230;
	proj.at<double>(1, 3) = 200;
	proj.at<double>(2, 3) = 0;
	int img_size = g_img_size;
	int width = img_size;
	int height = img_size;

	Mat mark(nKey, 3, CV_64FC1, marklandlm3);
	mark = mark.t();
	Mat mark4n = Mat::ones(4, nKey, CV_64FC1);

	mark.copyTo(mark4n(Rect(0, 0, nKey, 3)));

	Mat lm2 = proj*mark4n;
	for (int i = 0; i < nKey; i++)
	{
		//lm2.at<double>(1, i) = img_size + 1 - lm2.at<double>(1, i);
		marklandlm2[i * 2] = lm2.at<double>(0, i);
		marklandlm2[i * 2 + 1] = img_size + 1 - lm2.at<double>(1, i);
	}


	Mat ver(nver, 3, CV_64FC1, vertex);
	Mat Ver = Mat::ones(nver, 4, CV_64FC1);
	ver.copyTo(Ver(Rect(0, 0, 3, nver)));


	Mat proVerT = Ver*proj.t();

	for (int i = 0; i < nver; i++)
	{
		proVerT.at<double>(i, 1) = img_size + 1 - proVerT.at<double>(i, 1);
	}


	double *pData = (double*)(proVerT.data);




	double* point1 = new double[2 * ntri];// 
	double* point2 = new double[2 * ntri];// 
	double* point3 = new double[2 * ntri];// 
	double* h = new double[ntri];

	for (int i = 0; i < width * height; i++)
	{
		depth[i] = -99999999999999;
	}

	for (int i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3 * i]);
		int p2 = int(tri[3 * i + 1]);
		int p3 = int(tri[3 * i + 2]);

		point1[2 * i] = pData[3 * p1];	point1[2 * i + 1] = pData[3 * p1 + 1];
		point2[2 * i] = pData[3 * p2];	point2[2 * i + 1] = pData[3 * p2 + 1];
		point3[2 * i] = pData[3 * p3];	point3[2 * i + 1] = pData[3 * p3 + 1];

		double cent3d_z = (pData[3 * p1 + 2] + pData[3 * p2 + 2] + pData[3 * p3 + 2]) / 3;

		h[i] = cent3d_z;

	}


	Point2d point;
	Point2d pt1, pt2, pt3;

	//init image
	for (int i = 0; i < ntri; i++)
	{
		pt1.x = point1[2 * i]; pt1.y = point1[2 * i + 1];
		pt2.x = point2[2 * i]; pt2.y = point2[2 * i + 1];
		pt3.x = point3[2 * i]; pt3.y = point3[2 * i + 1];

		int x_min = (int)ceil((double)min(min((double)pt1.x, (double)pt2.x), (double)pt3.x));
		int x_max = (int)floor((double)max(max((double)pt1.x, (double)pt2.x), (double)pt3.x));

		int y_min = (int)ceil((double)min(min((double)pt1.y, (double)pt2.y), (double)pt3.y));
		int y_max = (int)floor((double)max(max((double)pt1.y, (double)pt2.y), (double)pt3.y));

		if (x_max < x_min || y_max < y_min || x_max > width - 1 || x_min < 0 || y_max > height - 1 || y_min < 0)
			continue;

		for (int x = x_min; x <= x_max; x++)
		{
			for (int y = y_min; y <= y_max; y++)
			{
				point.x = x;
				point.y = y;
				if ((depth[x * height + y] < h[i]) && (PointInTri(point, pt1, pt2, pt3) == true))
				{
					depth[x*height + y] = h[i];
					//cout << h[i] << endl;
				}
			}
		}
	}


	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] h;
}
/*
vertext	:点云坐标
nver	:点云个数
tri		:点云三角面片
ntri	:点云三角面片个数
tex_img	:纹理图像
teximgwidht	:纹理图像宽度
teximgheight	:纹理图像高度
nChannels	:纹理图像通道
dstimg		:结果图像
*/
void ZBufferTex(double* vertex, int nver, int* tri, int ntri, float* tex, int ntex, int *textri,
	unsigned char* tex_img, int teximgwidth, int teximgheight, int nChannels,
	double angle,
	double *marklandlm3, int nKey, double *marklandlm2,
	unsigned char* dstimg)
{
	float f = 2.5;
	Mat proj(3, 4, CV_64FC1);
	double degree = angle*CV_PI / 180;
	double r[9] = { cos(degree), 0, sin(degree), 0, 1, 0, -sin(degree), 0, cos(degree) };
	Mat R(3, 3, CV_64FC1, r);
	R = f*R;
	R.copyTo(proj(Rect(0, 0, 3, 3)));
	proj.at<double>(0, 3) = 230;
	proj.at<double>(1, 3) = 200;
	proj.at<double>(2, 3) = 0;
	int img_size = g_img_size;
	int width = img_size;
	int height = img_size;

	Mat mark(nKey, 3, CV_64FC1, marklandlm3);
	mark = mark.t();
	Mat mark4n = Mat::ones(4, nKey, CV_64FC1);

	mark.copyTo(mark4n(Rect(0, 0, nKey, 3)));

	Mat lm2 = proj*mark4n;
	for (int i = 0; i < nKey; i++)
	{
		marklandlm2[i * 2] = lm2.at<double>(0, i);
		marklandlm2[i * 2 + 1] = img_size + 1 - lm2.at<double>(1, i);
	}

	Mat ver(nver, 3, CV_64FC1, vertex);
	Mat Ver = Mat::ones(nver, 4, CV_64FC1);
	ver.copyTo(Ver(Rect(0, 0, 3, nver)));


	Mat proVerT = Ver*proj.t();

	for (int i = 0; i < nver; i++)
	{
		proVerT.at<double>(i, 1) = img_size + 1 - proVerT.at<double>(i, 1);
	}


	double *pData = (double*)(proVerT.data);

	int faceNum = ntex / nver;

	double* point1 = new double[2 * ntri];
	double* point2 = new double[2 * ntri];
	double* point3 = new double[2 * ntri];
	double* h = new double[ntri];
	double* imgh = new double[width * height];
	unsigned char* tritex = new unsigned char[ntri * nChannels];

	for (int i = 0; i < width * height; i++)
	{
		imgh[i] = -99999999999999;
	}

	for (int i = 0; i < ntri; i++)
	{
		// 3 point index of triangle
		int p1 = int(tri[3 * i]);
		int p2 = int(tri[3 * i + 1]);
		int p3 = int(tri[3 * i + 2]);

		int textrip1 = textri[3 * i];
		int textrip2 = textri[3 * i + 1];
		int textrip3 = textri[3 * i + 2];

		point1[2 * i] = pData[3 * p1];	point1[2 * i + 1] = pData[3 * p1 + 1];
		point2[2 * i] = pData[3 * p2];	point2[2 * i + 1] = pData[3 * p2 + 1];
		point3[2 * i] = pData[3 * p3];	point3[2 * i + 1] = pData[3 * p3 + 1];

		double cent3d_z = (pData[3 * p1 + 2] + pData[3 * p2 + 2] + pData[3 * p3 + 2]) / 3;

		h[i] = cent3d_z;

		double x1 = tex[textrip1 * 2];
		double y1 = tex[textrip1 * 2 + 1];
		int texx1 = int(x1*teximgwidth);
		int texy1 = teximgheight - int(y1*teximgheight) ;
		if (texx1 > teximgwidth || texx1 < 0 || texy1<0 || texy1>teximgheight)
			continue;

		double x2 = tex[textrip2 * 2];
		double y2 = tex[textrip2 * 2 + 1];
		int texx2 = int(x2*teximgwidth);
		int texy2 = teximgheight - int(y2*teximgheight) ;
		if (texx2 > teximgwidth || texx2 < 0 || texy2<0 || texy2>teximgheight)
			continue;

		double x3 = tex[textrip3 * 2];
		double y3 = tex[textrip3 * 2 + 1];
		int texx3 = int(x3*teximgwidth);
		int texy3 = teximgheight - int(y3*teximgheight) ;
		if (texx3> teximgwidth || texx3 < 0 || texy3<0 || texy3>teximgheight)
			continue;

		for (int j = 0; j < nChannels; j++)
		{
			//tritex[nChannels*i + j] = tex_img[(texx1 + texy1*teximgwidth)*nChannels + j];
			tritex[nChannels*i + j] = (int(tex_img[(texx1 + texy1*teximgwidth)*nChannels + j]) + \
				int(tex_img[(texx2 + texy2*teximgwidth)*nChannels + j]) + \
				int(tex_img[(texx3 + texy3*teximgwidth)*nChannels + j])) / 3;
		}
	}


	Point2d point;
	Point2d pt1, pt2, pt3;

	//init image
	memset(dstimg, 127, sizeof(unsigned char)* width*height*nChannels);

	for (int i = 0; i < ntri; i++)
	{
		pt1.x = point1[2 * i]; pt1.y = point1[2 * i + 1];
		pt2.x = point2[2 * i]; pt2.y = point2[2 * i + 1];
		pt3.x = point3[2 * i]; pt3.y = point3[2 * i + 1];

		int x_min = (int)ceil((double)min(min((double)pt1.x, (double)pt2.x), (double)pt3.x));
		int x_max = (int)floor((double)max(max((double)pt1.x, (double)pt2.x), (double)pt3.x));

		int y_min = (int)ceil((double)min(min((double)pt1.y, (double)pt2.y), (double)pt3.y));
		int y_max = (int)floor((double)max(max((double)pt1.y, (double)pt2.y), (double)pt3.y));

		if (x_max < x_min || y_max < y_min || x_max > width - 1 || x_min < 0 || y_max > height - 1 || y_min < 0)
			continue;

		for (int x = x_min; x <= x_max; x++)
		{
			for (int y = y_min; y <= y_max; y++)
			{
				point.x = x;
				point.y = y;
				if (imgh[x + y* width] < h[i] && PointInTri(point, pt1, pt2, pt3))
				{
					imgh[x + y* width] = h[i];
					for (int j = 0; j < nChannels; j++)
					{
						//dstimg[j + x + y*width*nChannels] = tritex[nChannels * i + j];
						dstimg[j + x*nChannels + y*width*nChannels] = tritex[nChannels*i + j];
					}
				}
			}
		}
	}

	delete[] point1;
	delete[] point2;
	delete[] point3;
	delete[] h;
	delete[] imgh;
	delete[] tritex;
}

int GenDepth(swPointsCloud &pc, swTriangle &tri, swLandmarkIndex3D &PID, swImageData&dstimg, swLandmark2D&dstpoint)
{
	int lossindex = -1;
	for (int i = 0; i < SW_MAXLANDMARKNUM; i++)
	{
		if (PID.Index3D[i] <= 0)
		{
			if (lossindex == -1)
				lossindex = i;
			else
				return -1;
		}
	}

	//lossindex = 2;

	if (lossindex == 2)
	{
		return -1;
	}

	int ptCount = 5;

	if (lossindex != -1)
		ptCount = 4;


	int dim = pc.dim;

	int PointsCount = pc.nCount;
	//savedata("face.txt",landmark,5,3);
	//savedata("ptc.txt",pc->data,PointsCount,dim);

	double *ptdata = new double[PointsCount*dim];
	for (int j = 0; j < dim; j++)
	{
		for (int i = 0; i < PointsCount; i++)
		{
			ptdata[i*dim + j] = pc.data[i*dim + j];
		}
	}

	double *landmark = new double[ptCount * 3];
	for (int n = 0, i = 0; i<ptCount; i++, n++)
	{
		if (n == lossindex)
			n++;
		landmark[i * 3] = pc.data[PID.Index3D[n] * dim];
		landmark[i * 3 + 1] = pc.data[PID.Index3D[n] * dim + 1];
		landmark[i * 3 + 2] = pc.data[PID.Index3D[n] * dim + 2];
	}
	double *meanfacelm3 = new double[ptCount * 3];
	for (int n = 0, i = 0; i<ptCount; i++, n++)
	{
		if (n == lossindex)
			n++;
		meanfacelm3[i * 3] = g_meanface[n*3];
		meanfacelm3[i * 3 + 1] = g_meanface[n * 3+1];
		meanfacelm3[i * 3 + 2] = g_meanface[n * 3+2];
	}

	Mat newshape = procrustes(meanfacelm3, landmark, ptCount, ptdata, PointsCount, 3);
	//savedata("shape.txt",shape->data,shape->size[0],shape->size[1]);
	//savedata("newshape.txt",new_shape->data,new_shape->size[0],new_shape->size[1]);

	// 人脸关键点对齐
	Mat newface = procrustes(meanfacelm3, landmark, ptCount, landmark, ptCount, 3);

	//savedata("newface.txt",newfacelm->data,newfacelm->size[0],newfacelm->size[1]);
	//生成深度图
	int triCount = tri.nCount;

	double *depthmap = new double[450 * 450];
	Mat new_shape = newshape.t();
	Mat newfacelm = newface.t();
	double marklandlm2[10] = { 0 };
	ZBuffer((double*)new_shape.data, PointsCount, tri.data, triCount, (double*)newfacelm.data, ptCount, depthmap, marklandlm2);

	Mat depthface(450, 450, CV_64FC1, depthmap);
	Mat depthimg;
	depthface.convertTo(depthimg, CV_8UC1);
	depthimg = depthimg.t();

	Mat alignface = align_face2(depthimg, marklandlm2, ptCount, lossindex);

	if (dstimg.data == NULL)
		dstimg.data = new unsigned char[alignface.cols*alignface.rows];

	memcpy(dstimg.data, alignface.data, alignface.cols*alignface.rows*sizeof(char));
	dstimg.height = alignface.rows;
	dstimg.width = alignface.cols;
	dstimg.channel = 1;

	for (int i = 0; i<ptCount; i++)
	{
		dstpoint.point[i].x = marklandlm2[i * 2];
		dstpoint.point[i].y = marklandlm2[i * 2 + 1];
	}

	delete depthmap;
	depthmap = NULL;

	delete landmark;
	landmark = NULL;
	delete meanfacelm3;
	meanfacelm3 = NULL;

	delete ptdata;
	ptdata = NULL;



	return 1;

}

int GetImageType(int channel)
{
	int imagetype = 0;

	switch (channel)
	{
	case 1:
		imagetype = CV_8UC1;
		break;
	case 3:
		imagetype = CV_8UC3;
		break;
	case 4:
		imagetype = CV_8UC4;
		break;

	default:
		return -1;
		break;
	}

	return imagetype;
}

double cosine(const float* arrayA, const float* arrayB, int length)
{
	if (!arrayA || !arrayB)
		return 0;
	double sumarrayA = 0, sumarrayB = 0;
	double cosine = 0;
	for (int i = 0; i < length; i++){
		sumarrayA += arrayA[i] * arrayA[i];
		sumarrayB += arrayB[i] * arrayB[i];
		cosine += arrayA[i] * arrayB[i];
	}
	sumarrayA = sqrt(sumarrayA);
	sumarrayB = sqrt(sumarrayB);
	if ((sumarrayA - 0 < 0.0001) || (sumarrayB - 0 < 0.0001)){
		return 0;
	}
	cosine /= (sumarrayA*sumarrayB);
	//  cout<<sumarrayA<<' '<<sumarrayB<<' '<<cosine<<endl;
	return cosine;
}

void getNormalFaceRGBData(swImageData &src, swRect &faceRect, swImageData &faceData)
{
	int type = GetImageType(src.channel);
	if (type < 0 || src.data == NULL)
	{
		return ;
	}
	Mat src_img(src.height, src.width, type, src.data);

	int addx = 50;
	int addy = 50;

	int x = max(faceRect.x - addx, 0);
	int y = max(faceRect.y - addy, 0);
	int width = faceRect.width + faceRect.x + addx > src.width ? src.width - x : faceRect.width + faceRect.x + addx - x;
	width = (width * 8 + 31) / 32 * 4;
	int height = faceRect.height + faceRect.y + addy > src.height ? src.height - y : faceRect.height + faceRect.y + addy - y;
	height = (height * 8 + 31) / 32 * 4;
	Rect facerect = Rect(x,y,width,height);
	if (facerect.x <0 || facerect.y<0 || facerect.br().x>src_img.cols || facerect.tl().y>src_img.rows)
	{
		return;
	}
	Mat facemat = src_img(facerect).clone();

	// normallize
	if (g_bnormalize == 1)
	{
		resize(facemat, facemat, Size(g_bnormalizeX, g_bnormalizeY));
	}
	cv::cvtColor(facemat, facemat, CV_BGR2RGB);

	faceData = swImageData(facemat.data, facemat.cols, facemat.rows, facemat.channels());
}
void getNormalFaceBGRData(swImageData &src, swRect &faceRect, swImageData &faceData)
{
	int type = GetImageType(src.channel);
	if (type < 0 || src.data == NULL)
	{
		return;
	}
	Mat src_img(src.height, src.width, type, src.data);

	int addx = 50;
	int addy = 50;

	int x = max(faceRect.x - addx, 0);
	int y = max(faceRect.y - addy, 0);
	int width = faceRect.width + faceRect.x + addx > src.width ? src.width - x : faceRect.width + faceRect.x + addx - x;
	width = (width * 8 + 31) / 32 * 4;
	int height = faceRect.height + faceRect.y + addy > src.height ? src.height - y : faceRect.height + faceRect.y + addy - y;
	height = (height * 8 + 31) / 32 * 4;
	Rect facerect = Rect(x, y, width, height);
	if (facerect.x <0 || facerect.y<0 || facerect.br().x>src_img.cols || facerect.tl().y>src_img.rows)
	{
		return;
	}
	Mat facemat = src_img(facerect).clone();

	// normallize
	if (g_bnormalize == 1)
	{
		resize(facemat, facemat, Size(g_bnormalizeX, g_bnormalizeY));
	}
	faceData = swImageData(facemat.data, facemat.cols, facemat.rows, facemat.channels());
}
int GenTexFace(swPointsCloud&pc, swTriangle &tri, swTexCord &tex, swTriangle &textri, swTexRGB &texRgb, swLandmarkIndex3D &PID, int angle, swImageData &dstimg)
{
	int lossindex = -1;
	for (int i = 0; i < SW_MAXLANDMARKNUM; i++)
	{
		if (PID.Index3D[i] < 0)
		{
			if (lossindex == -1)
				lossindex = i;
			else
				return -1;
		}
	}

	//lossindex = 2;

	if (lossindex == 2)
	{
		return -1;
	}

	int ptCount = 5;

	if (lossindex != -1)
		ptCount = 4;

	int dim = pc.dim;

	int PointsCount = pc.nCount;
	//savedata("face.txt",landmark,5,3);
	//savedata("ptc.txt",pc->data,PointsCount,dim);

	double *ptdata = new double[PointsCount*dim];
	for (int j = 0; j < dim; j++)
	{
		for (int i = 0; i < PointsCount; i++)
		{
			ptdata[i*dim + j] = pc.data[i*dim + j];
		}
	}
	double *landmark = new double[ptCount * 3];
	for (int n = 0, i = 0; i<ptCount; i++, n++)
	{
		if (n == lossindex)
			n++;
		landmark[i * 3] = pc.data[PID.Index3D[n] * dim];
		landmark[i * 3 + 1] = pc.data[PID.Index3D[n] * dim + 1];
		landmark[i * 3 + 2] = pc.data[PID.Index3D[n] * dim + 2];
	}
	double *meanfacelm3 = new double[ptCount * 3];
	for (int n = 0, i = 0; i<ptCount; i++, n++)
	{
		if (n == lossindex)
			n++;
		meanfacelm3[i * 3] = g_meanface[n*3];
		meanfacelm3[i * 3 + 1] = g_meanface[n * 3+1];
		meanfacelm3[i * 3 + 2] = g_meanface[n * 3+2] -120;
	}

	Mat newshape = procrustes(meanfacelm3, landmark, ptCount, ptdata, PointsCount, 3);
	//savedata("shape.txt",shape->data,shape->size[0],shape->size[1]);
	//savedata("newshape.txt",new_shape->data,new_shape->size[0],new_shape->size[1]);

	// 人脸关键点对齐
	Mat newface = procrustes(meanfacelm3, landmark, ptCount, landmark, ptCount, 3);

	//savedata("newface.txt",newfacelm->data,newfacelm->size[0],newfacelm->size[1]);
	//生成深度图
	int triCount = tri.nCount;

	Mat texface(450, 450, CV_8UC3);
	Mat new_shape = newshape.t();
	Mat newfacelm = newface.t();
	double marklandlm2[10] = { 0 };
	ZBufferTex((double*)new_shape.data, PointsCount, tri.data, tri.nCount, tex.data, tex.nCount, textri.data,
		texRgb.rgb, texRgb.width, texRgb.height, 3, angle, (double*)newfacelm.data, ptCount, marklandlm2, texface.data);

	if (dstimg.data == NULL)
		dstimg.data = new unsigned char[texface.cols*texface.rows * texface.channels()];

	memcpy(dstimg.data, texface.data, texface.cols*texface.rows*sizeof(unsigned char) * texface.channels());
	dstimg.height = texface.rows;
	dstimg.width = texface.cols;
	dstimg.channel = texface.channels();

	delete landmark;
	landmark = NULL;

	delete ptdata;
	ptdata = NULL;

	delete meanfacelm3;
	meanfacelm3 = NULL;
	return 1;

}
