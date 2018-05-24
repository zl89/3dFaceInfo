#pragma once
#include <caffe/caffe.hpp>
#include "opencv2/opencv.hpp"

#define INTER_FAST
using namespace caffe;

typedef struct FaceRect {
	float x1;
	float y1;
	float x2;
	float y2;
	float score; /**< Larger score should mean higher confidence. */
	float Width()
	{
		return x2 - x1+1;
	}
	float Height()
	{
		return y2 - y1+1;
	}
} FaceRect;

typedef struct FacePts {
	float x[5], y[5];
} FacePts;

typedef struct FaceInfo {
	FaceRect bbox;
	cv::Vec4f regression;
	FacePts facePts;
	double roll;
	double pitch;
	double yaw;
} FaceInfo;

class CFaceDetetor {
public:
	CFaceDetetor();
	int Init(const string& proto_model_dir,int cuponly = false);
	void Detect(const cv::Mat& img, std::vector<FaceInfo> &faceInfo);

private:
	bool CvMatToDatumSignalChannel(const cv::Mat& cv_mat, Datum* datum);
	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels, Blob<float>* input_layer,
		const int height, const int width);
	void SetMean();
	void GenerateBoundingBox(Blob<float>* confidence, Blob<float>* reg,
		float scale, float thresh, int image_width, int image_height);
	void ClassifyFace(const std::vector<FaceInfo>& regressed_rects, cv::Mat &sample_single,
		boost::shared_ptr<Net<float> >& net, double thresh, char netName);
	void ClassifyFace_MulImage(const std::vector<FaceInfo> &regressed_rects, cv::Mat &sample_single,
		boost::shared_ptr<Net<float> >& net, double thresh, char netName);
	std::vector<FaceInfo> NonMaximumSuppression(std::vector<FaceInfo>& bboxes, float thresh, char methodType);
	void Bbox2Square(std::vector<FaceInfo>& bboxes);
	void Padding(int img_w, int img_h);
	std::vector<FaceInfo> BoxRegress(std::vector<FaceInfo> &faceInfo_, int stage);
	void RegressPoint(const std::vector<FaceInfo>& faceInfo);

	void clear();
private:
	boost::shared_ptr<Net<float> > PNet_;
	boost::shared_ptr<Net<float> > RNet_;
	boost::shared_ptr<Net<float> > ONet_;

	// x1,y1,x2,t2 and score
	std::vector<FaceInfo> condidate_rects_;
	std::vector<FaceInfo> total_boxes_;
	std::vector<FaceInfo> regressed_rects_;
	std::vector<FaceInfo> regressed_pading_;

	std::vector<cv::Mat> crop_img_;
	int curr_feature_map_w_;
	int curr_feature_map_h_;
	int num_channels_;

	double threshold[3];
	double factor = 0.709;
	int minSize = 40;
	bool m_cpuonly;
};
