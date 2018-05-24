#include "facerecon.h"
#ifdef _DEBUG
#pragma comment(lib,"../depends//caffe/caffelib/libcaffed.lib")
#else
#pragma comment(lib,"../depends//caffe/caffelib/libcaffe.lib")
//#pragma comment(lib,"../depends//caffe/caffelib/caffe.lib")
#pragma comment(lib,"opencv_core2410")
#pragma comment(lib,"opencv_contrib2410")
#pragma comment(lib,"opencv_highgui2410")
#pragma comment(lib,"opencv_imgproc2410.lib")
#pragma comment(lib,"cuda")
#pragma comment(lib,"curand")
#pragma comment(lib,"cudart")
#pragma comment(lib,"cublas")
#pragma comment(lib,"libopenblas.dll.a")
#pragma comment(lib,"hdf5.lib")
#pragma comment(lib,"hdf5_hl.lib")
#pragma comment(lib,"libprotobuf.lib")
#pragma comment(lib,"libglog.lib")
#pragma comment(lib,"gflagsd.lib")
//
//#pragma comment(lib,"opencv_contrib2410.lib")
//#pragma comment(lib,"opencv_core2410.lib")
//#pragma comment(lib,"opencv_features2d2410.lib")
//#pragma comment(lib,"opencv_flann2410.lib")
//#pragma comment(lib,"opencv_gpu2410.lib")
//#pragma comment(lib,"opencv_highgui2410.lib")
//#pragma comment(lib,"opencv_imgproc2410.lib")
//#pragma comment(lib,"opencv_legacy2410.lib")
//#pragma comment(lib,"opencv_ml2410.lib")
//#pragma comment(lib,"opencv_nonfree2410.lib")
//#pragma comment(lib,"opencv_objdetect2410.lib")
//#pragma comment(lib,"opencv_ocl2410.lib")
//#pragma comment(lib,"opencv_photo2410.lib")
//#pragma comment(lib,"opencv_stitching2410.lib")
//#pragma comment(lib,"opencv_superres2410.lib")
//#pragma comment(lib,"opencv_ts2410.lib")
//#pragma comment(lib,"libprotobuf.lib")
//#pragma comment(lib,"libglog.lib")
//#pragma comment(lib,"gflagsd.lib")
//#pragma comment(lib,"hdf5.lib")
//#pragma comment(lib,"hdf5_cpp.lib")
//#pragma comment(lib,"hdf5_f90cstub.lib")
//#pragma comment(lib,"hdf5_fortran.lib")
//#pragma comment(lib,"hdf5_hl.lib")
//#pragma comment(lib,"hdf5_hl_cpp.lib")
//#pragma comment(lib,"hdf5_hl_f90cstub.lib")
//#pragma comment(lib,"hdf5_hl_fortran.lib")
//#pragma comment(lib,"hdf5_tools.lib")
//#pragma comment(lib,"szip.lib")
//#pragma comment(lib,"zlib.lib")
//#pragma comment(lib,"libopenblas.dll.a")
//#pragma comment(lib,"cudnn.lib")
//#pragma comment(lib,"cudart.lib")
//#pragma comment(lib,"cublas.lib")
//#pragma comment(lib,"curand.lib")
//#pragma comment(lib,"cuda.lib")

#endif

CFaceRecon::CFaceRecon()
{
	m_feature_size = 1024;
	m_meancoord5point.push_back(cv::Point2f(30.2946, 51.6963));
	m_meancoord5point.push_back(cv::Point2f(65.5318, 51.5014));
	m_meancoord5point.push_back(cv::Point2f(48.0252, 71.7366));
	m_meancoord5point.push_back(cv::Point2f(33.5493, 92.3655));
	m_meancoord5point.push_back(cv::Point2f(62.7299, 92.2041));
	m_cropimg_size = cv::Size(96, 112);
}


CFaceRecon::~CFaceRecon()
{
	
}
float CFaceRecon::simd_dot(const float* x, const float* y, const long& len)
{
	float inner_prod = 0.0f;
	__m128 X, Y; // 128-bit values
	__m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
	float temp[4];

	long i;
	for (i = 0; i + 4 < len; i += 4) {
		X = _mm_loadu_ps(x + i); // load chunk of 4 floats
		Y = _mm_loadu_ps(y + i);
		acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
	}
	_mm_storeu_ps(&temp[0], acc); // store acc into an array
	inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

	// add the remaining values
	for (; i < len; ++i) {
		inner_prod += x[i] * y[i];
	}
	return inner_prod;
}

int CFaceRecon::init(const std::string &proto_model_dir, int cuponly, int use2Dmodel)
{
	if (cuponly == 1)
	{
		Caffe::set_mode(Caffe::CPU);
	}
	else
	{
		Caffe::set_mode(Caffe::GPU);
	}

	if (use2Dmodel)
	{
		Net_.reset(new Net<float>((proto_model_dir + "/sphereface_deploy.prototxt"), TEST));
		Net_->CopyTrainedLayersFrom(proto_model_dir + "/sphereface_model_iter_28000.caffemodel");
	}
	else
	{
		Net_.reset(new Net<float>((proto_model_dir + "/centerface_deploy.prototxt"), TEST));
		Net_->CopyTrainedLayersFrom(proto_model_dir + "/centerface_model.caffemodel");
	}

	if (Net_->num_inputs() != 1 || Net_->num_outputs() != 1)
	{
		return 0;
	}

	return 1;
}

void CFaceRecon::WrapInputLayer(std::vector<cv::Mat>* input_channels, Blob<float>* input_layer, const int height, const int width)
{
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

Mat CFaceRecon::align_face(const cv::Mat& src, const vector<Point2f>& landmark)
{

	vector<Point2f> detect_points = landmark;

	vector<Point2f> reference_points = m_meancoord5point;

	Mat tfm = get_similarity_transform(detect_points, reference_points);
	Mat aligned_face;

	warpAffine(src, aligned_face, tfm, m_cropimg_size);
	return aligned_face;
}

int CFaceRecon::ExtractFeature(const cv::Mat &crop_image, float* feature)
{
	if (feature == NULL) {
		std::cout << "Face Recognizer: 'feats' must be initialized with size \
					 					            of GetFeatureSize(). " << std::endl;
		return 0;
	}
	cv::Mat sample_single;
	crop_image.convertTo(sample_single, CV_32FC3);
	sample_single.convertTo(sample_single, CV_32FC3, 0.0078125, -127.5*0.0078125);



	Blob<float>* input_layer = Net_->input_blobs()[0];
	int hs = sample_single.rows;// height
	int ws = sample_single.cols;// width
	input_layer->Reshape(1, 3, hs, ws);
	Net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels, Net_->input_blobs()[0], hs, ws);
	cv::split(sample_single, input_channels);
	// check data transform right
	CHECK(reinterpret_cast<float*>(input_channels.at(0).data) == Net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
	Net_->Forward();

	Blob<float>* reg = Net_->output_blobs()[0];
	const float* reg_data = reg->cpu_data();
	int featuresize = m_feature_size/2;

	memcpy(feature, reg_data, featuresize * sizeof(float));

	cv::Mat flip_sample_single;
	cv::flip(sample_single, flip_sample_single, 2);

	std::vector<cv::Mat> flip_input_channels;
	WrapInputLayer(&flip_input_channels, Net_->input_blobs()[0], hs, ws);
	cv::split(flip_sample_single, flip_input_channels);
	// check data transform right
	CHECK(reinterpret_cast<float*>(flip_input_channels.at(0).data) == Net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
	Net_->Forward();


	Blob<float> *flip_reg = Net_->output_blobs()[0];
	const float *flip_reg_data = flip_reg->cpu_data();

	memcpy(feature + featuresize, flip_reg_data, featuresize * sizeof(float));
	
	return m_feature_size;

}

int CFaceRecon::ExtractFeatureWithCrop(const cv::Mat &src_image, vector<Point2f> landmark, float *feature)
{
	if (feature == NULL) {
		std::cout << "Face Recognizer: 'feats' must be initialized with size of GetFeatureSize(). " << std::endl;
		return 0;
	}

	Mat cropimg = align_face(src_image, landmark);

	int re = ExtractFeature(cropimg, feature);


	return re;
}
double CFaceRecon::cosine(const float* arrayA, const float* arrayB, int length)
{
	if (!arrayA || !arrayB)
		return 0;
	double sumarrayA = 0, sumarrayB = 0;
	double cosine = 0;
	for (int i = 0; i<length; i++){
		sumarrayA += arrayA[i] * arrayA[i];
		sumarrayB += arrayB[i] * arrayB[i];
		cosine += arrayA[i] * arrayB[i];
	}
	sumarrayA = sqrt(sumarrayA);
	sumarrayB = sqrt(sumarrayB);
	if ((sumarrayA - 0<0.0001) || (sumarrayB - 0<0.0001)){
		return 0;
	}
	cosine /= (sumarrayA*sumarrayB);
	//  cout<<sumarrayA<<' '<<sumarrayB<<' '<<cosine<<endl;
	return cosine;
}

float CFaceRecon::CalcSimilarity(const float* fc1, const float* fc2, long dim)

{
	if (dim == -1) {
		dim = GetFeatureSize();
	}
	//return simd_dot(fc1, fc2, dim)/(sqrt(simd_dot(fc1, fc1, dim))*sqrt(simd_dot(fc2, fc2, dim)));

	return cosine(fc1, fc2, dim);


}