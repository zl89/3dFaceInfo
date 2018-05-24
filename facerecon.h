#pragma once
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>

using namespace caffe;
using namespace cv;
class CFaceRecon
{
public:
	CFaceRecon();
	~CFaceRecon();
	int init(const string& proto_odel_dir, int cuponly, int use2Dmodel = false);

private:
	int m_feature_size;
	vector<cv::Point2f> m_meancoord5point;
	cv::Size	m_cropimg_size;


private:
	boost::shared_ptr<Net<float> > Net_;
	void WrapInputLayer(std::vector<cv::Mat>* input_channels, Blob<float>* input_layer,
		const int height, const int width);

	float simd_dot(const float* x, const float* y, const long& len);

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

		return trans_cv2;/**/
		//return Mat();
	}

	Mat align_face(const cv::Mat& src, const vector<Point2f>& landmark);
	double cosine(const float* arrayA, const float* arrayB, int length);
public:
	int GetFeatureSize()
	{
		return m_feature_size;
	}
	// Extract feature with a cropping face.
	// 'feats' must be initialized with size of feature_size().
	int ExtractFeatureWithCrop(const cv::Mat &src_image, vector<Point2f> points, float *feature);

	int ExtractFeature(const cv::Mat &crop_img, float* const feat);
	// Calculate similarity of face features fc1 and fc2.
	// dim = -1 default feature size
	float CalcSimilarity(const float* fc1, const float* fc2, long dim = -1);
};