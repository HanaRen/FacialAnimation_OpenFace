///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED �AS IS?FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called �open source?software licenses (�Open Source
// Components?, which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee�s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltru�aitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////


// FeatureExtraction.cpp : Defines the entry point for the feature extraction console application.

// System includes
#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// openGL include
#include <GL/glut.h>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

// OpenNI include
//#include <XnCppWrapper.h>  //OpenNI��ͷ�ļ�


// Local includes
#include "LandmarkCoreIncludes.h"
#include "OpenGLDemo.h"
#include "glm.h"

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>


#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort(const std::string & error)
{
	std::cout << error << std::endl;
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;
//using namespace xn; // OpenNI�������ռ�
using namespace boost::filesystem;
using namespace cv;

LandmarkDetector::FaceModelParameters* det_parameters;
LandmarkDetector::CLNF* face_model;
FaceAnalysis::FaceAnalyser* face_analyser;
int device = 0;
vector<string> arguments;
boost::filesystem::path config_path;
boost::filesystem::path parent_path;
vector<string> input_files, depth_directories, output_files, tracked_videos_output;
bool use_world_coordinates;
string output_codec; //not used but should
bool video_input = true;
bool verbose = true;
bool images_as_video = false;

vector<vector<string> > input_image_files;
// Grab camera parameters, if they are not defined (approximate values will be used)
float fx = 0, fy = 0, cx = 0, cy = 0;
int d = 0;
// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
bool cx_undefined = false;
bool fx_undefined = false;
vector<string> output_similarity_align;
vector<string> output_hog_align_files;

double sim_scale = 0.7;
int sim_size = 112;
bool grayscale = false;
bool video_output = false;
bool dynamic = true; // Indicates if a dynamic AU model should be used (dynamic is useful if the video is long enough to include neutral expressions)
int num_hog_rows;
int num_hog_cols;

// By default output all parameters, but these can be turned off to get smaller files or slightly faster processing times
// use -no2Dfp, -no3Dfp, -noMparams, -noPose, -noAUs, -noGaze to turn them off
bool output_2D_landmarks = true;
bool output_3D_landmarks = true;
bool output_model_params = true;
bool output_pose = true;
bool output_AUs = true;
bool output_gaze = true;
// If multiple video files are tracked, use this to indicate if we are done
bool done = false;
int f_n = -1;
int curr_img = -1;

string au_loc;
string au_loc_local;

string current_file;
cv::VideoCapture video_capture;
cv::Mat captured_image;
int total_frames = -1;
int reported_completion = 0;
double fps_vid_in = -1.0;

//XnStatus result = XN_STATUS_OK;  //OpenNI�����ķ��ؽ��
//ImageMetaData imageMD; //OpenNI��ɫ����

std::ofstream output_file;
std::ofstream hog_output_file;
cv::VideoWriter writerFace;

//Context context;
//ImageGenerator imageGenerator;

int initialOpenCV();

extern int openCVAnalysis();

void get_output_feature_params(vector<string> &output_similarity_aligned, vector<string> &output_hog_aligned_files, double &similarity_scale,
	int &similarity_size, bool &grayscale, bool& verbose, bool& dynamic, bool &output_2D_landmarks, bool &output_3D_landmarks,
	bool &output_model_params, bool &output_pose, bool &output_AUs, bool &output_gaze, vector<string> &arguments);

void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments);

void output_HOG_frame(std::ofstream* hog_file, bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_rows, int num_cols);

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetCorrectedPoseWorld(face_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar(0, (1 - vis_certainty)*255.0, vis_certainty * 255), thickness, fx, fy, cx, cy);

		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			FaceAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	//cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);

	if (!det_parameters.quiet_mode)
	{
		cv::namedWindow("tracking_result", 1);
		cv::imshow("tracking_result", captured_image);
		cv::moveWindow("tracking_result",120,240);
	}
}

void prepareOutputFile(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_landmarks, int num_model_modes, vector<string> au_names_class, vector<string> au_names_reg);

// Output all of the information into one file in one go (quite a few parameters, but simplifies the flow)
void outputAllFeatures(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	const LandmarkDetector::CLNF& face_model, int frame_count, double time_stamp, bool detection_success,
	cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, const cv::Vec6d& pose_estimate, double fx, double fy, double cx, double cy,
	const FaceAnalysis::FaceAnalyser& face_analyser);

void post_process_output_file(FaceAnalysis::FaceAnalyser& face_analyser, string output_file, bool dynamic);

extern GLuint texGround;


vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	// First argument is reserved for the name of the executable
	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Useful utility for creating directories for storing the output files
void create_directory_from_file(string output_path)
{

	// Creating the right directory structure

	// First get rid of the file
	auto p = path(path(output_path).parent_path());

	if (!p.empty() && !boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);
		if (!success)
		{
			cout << "Failed to create a directory... " << p.string() << endl;
		}
	}
}

void create_directory(string output_path)
{

	// Creating the right directory structure
	auto p = path(output_path);

	if (!boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);

		if (!success)
		{
			cout << "Failed to create a directory..." << p.string() << endl;
		}
	}
}

int initialOpenCV()
{
	//// ��������ʼ���豸������ for asus depth camera
	//result = context.Init();
	//if (XN_STATUS_OK != result)
	//	cerr << "�豸�����ĳ�ʼ������" << endl;
	//result = imageGenerator.Create(context);
	//if (XN_STATUS_OK != result)
	//	cerr << "������ɫ����������" << endl;
	//ͨ��ӳ��ģʽ��������������������ֱ��ʡ�֡��
	//XnMapOutputMode mapMode;
	//mapMode.nXRes = 640;
	//mapMode.nYRes = 480;
	//mapMode.nFPS = 30;
	//result = imageGenerator.SetMapOutputMode(mapMode);
	//result = context.StartGeneratingAll();

	arguments.push_back("../../x64/Release/FeatureExtractionVid.exe");
	// Search paths
	config_path = boost::filesystem::path(CONFIG_DIR);
	parent_path = boost::filesystem::path(arguments[0]).parent_path();

	// Some initial parameters that can be overriden from command line	
	arguments.push_back("-f");
	arguments.push_back("../../videos/default.wmv");

	arguments.push_back("-of");
	arguments.push_back("./output_features_vid/001.txt");
	//arguments.push_back("-no3Dfp");
	arguments.push_back("-no2Dfp");
	arguments.push_back("-noMparams");
	//arguments.push_back("-noPose");
	arguments.push_back("-noAUs");
	arguments.push_back("-noGaze");

	det_parameters = new LandmarkDetector::FaceModelParameters(arguments);
	// Always track gaze in feature extraction
	det_parameters->track_gaze = true;
	// Get the input output file parameters

	// Indicates that rotation should be with respect to camera or world coordinates

	LandmarkDetector::get_video_input_output_params(input_files, depth_directories, output_files, tracked_videos_output, use_world_coordinates, output_codec, arguments);

	// Adding image support for reading in the files
	if (input_files.empty())
	{
		vector<string> d_files;
		vector<string> o_img;
		vector<cv::Rect_<double>> bboxes;
		get_image_input_output_params_feats(input_image_files, images_as_video, arguments);

		if (!input_image_files.empty())
		{
			video_input = false;
		}
	}

	// Get camera parameters
	LandmarkDetector::get_camera_params(d, fx, fy, cx, cy, arguments);

	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// The modules that are being used for tracking
	face_model = new LandmarkDetector::CLNF(det_parameters->model_location);


	get_output_feature_params(output_similarity_align, output_hog_align_files, sim_scale, sim_size, grayscale, verbose, dynamic,
		output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze, arguments);
	// Used for image masking
	string tri_loc;
	boost::filesystem::path tri_loc_path = boost::filesystem::path("model/tris_68_full.txt");
	if (boost::filesystem::exists(tri_loc_path))
	{
		tri_loc = tri_loc_path.string();
	}
	else if (boost::filesystem::exists(parent_path / tri_loc_path))
	{
		tri_loc = (parent_path / tri_loc_path).string();
	}
	else if (boost::filesystem::exists(config_path / tri_loc_path))
	{
		tri_loc = (config_path / tri_loc_path).string();
	}
	else
	{
		cout << "Can't find triangulation files, exiting" << endl;
		return 1;
	}

	// Will warp to scaled mean shape
	cv::Mat_<double> similarity_normalised_shape = face_model->pdm.mean_shape * sim_scale;
	// Discard the z component
	similarity_normalised_shape = similarity_normalised_shape(cv::Rect(0, 0, 1, 2 * similarity_normalised_shape.rows / 3)).clone();


	if (dynamic)
	{
		au_loc_local = "AU_predictors/AU_all_best.txt";
	}
	else
	{
		au_loc_local = "AU_predictors/AU_all_static.txt";
	}

	boost::filesystem::path au_loc_path = boost::filesystem::path(au_loc_local);
	if (boost::filesystem::exists(au_loc_path))
	{
		au_loc = au_loc_path.string();
	}
	else if (boost::filesystem::exists(parent_path / au_loc_path))
	{
		au_loc = (parent_path / au_loc_path).string();
	}
	else if (boost::filesystem::exists(config_path / au_loc_path))
	{
		au_loc = (config_path / au_loc_path).string();
	}
	else
	{
		cout << "Can't find AU prediction files, exiting" << endl;
		return 1;
	}

	// Creating a  face analyser that will be used for AU extraction
	face_analyser = new FaceAnalysis::FaceAnalyser(vector<cv::Vec3d>(), 0.7, 112, 112, au_loc, tri_loc);

	if (video_input)
	{
		// We might specify multiple video files as arguments
		if (input_files.size() > 0)
		{
			f_n++;
			current_file = input_files[f_n];
		}
		else
		{
			// If we want to write out from webcam
			f_n = 0;
		}
		// Do some grabbing
		if (current_file.size() > 0)
		{
			INFO_STREAM("Attempting to read from file: " << current_file);
			video_capture = cv::VideoCapture(current_file);
			total_frames = (int)video_capture.get(CV_CAP_PROP_FRAME_COUNT);
			fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

			// Check if fps is nan or less than 0
			if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
			{
				INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
				fps_vid_in = 30;
			}
		}
		else
		{
			INFO_STREAM("Attempting to capture from device: " << device);
			video_capture = cv::VideoCapture(device);
			fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

			//fps_vid_in = 30;
			//if (XN_STATUS_OK == result)
			//{
			//	//��ȡһ֡��ɫͼ��ת��ΪOpenCV�е�ͼ���ʽ
			//	imageGenerator.GetMetaData(imageMD);
			//	Mat cvRGBImg(imageMD.FullYRes(), imageMD.FullXRes(), CV_8UC3, (char *)imageMD.Data());
			//	cvtColor(cvRGBImg, captured_image, CV_RGB2BGR);
			//}

			//// Check if fps is nan or less than 0
			//if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
			//{
			//	INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
			//	fps_vid_in = 30;
			//} && XN_STATUS_OK != result

		}
		if (!video_capture.isOpened())
		{
			FATAL_STREAM("Failed to open video source, exiting");
			return 1;
		}
		else
		{
			INFO_STREAM("Device or file opened");
		}

		if (video_capture.isOpened())
		{
			video_capture >> captured_image;//grab video frame--.read(frame);.grab() retrieve(frame)
		}

		std::cout << "...caputured_image" << std::endl;

	}
	else
	{
		f_n++;
		curr_img++;
		if (!input_image_files[f_n].empty())
		{
			string curr_img_file = input_image_files[f_n][curr_img];
			captured_image = cv::imread(curr_img_file, -1);
		}
		else
		{
			FATAL_STREAM("No .jpg or .png images in a specified drectory, exiting");
			return 1;
		}

	}

	// If optical centers are not defined just use center of image
	if (cx_undefined)
	{
		cx = captured_image.cols / 2.0f;
		cy = captured_image.rows / 2.0f;
	}
	// Use a rough guess-timate of focal length
	if (fx_undefined)
	{
		fx = 500 * (captured_image.cols / 640.0);
		fy = 500 * (captured_image.rows / 480.0);

		fx = (fx + fy) / 2.0;
		fy = fx;
	}

	// Creating output files

	if (!output_files.empty())
	{
		output_file.open(output_files[f_n], ios_base::out);
		prepareOutputFile(&output_file, output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze, face_model->pdm.NumberOfPoints(), face_model->pdm.NumberOfModes(), face_analyser->GetAUClassNames(), face_analyser->GetAURegNames());
	}

	// Saving the HOG features
	if (!output_hog_align_files.empty())
	{
		hog_output_file.open(output_hog_align_files[f_n], ios_base::out | ios_base::binary);
	}

	// saving the videos
	if (!tracked_videos_output.empty())
	{
		try
		{
			writerFace = cv::VideoWriter(tracked_videos_output[f_n], CV_FOURCC(output_codec[0], output_codec[1], output_codec[2], output_codec[3]), fps_vid_in, captured_image.size(), true);
		}
		catch (cv::Exception e)
		{
			WARN_STREAM("Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
		}
	}
}

int frame_count = 0;
// This is useful for a second pass run (if want AU predictions)
vector<cv::Vec6d> params_global_video;
vector<bool> successes_video;
vector<cv::Mat_<double>> params_local_video;
vector<cv::Mat_<double>> detected_landmarks_video;

// Use for timestamping if using a webcam
int64 t_initial = cv::getTickCount();

bool visualise_hog = verbose;

// Timestamp in seconds of current processing
double time_stamp = 0;

int openCVAnalysis()
{
	if (!captured_image.empty())
	{
		//std::cout << "...ifEmpty" << std::endl;

		// Grab the timestamp first
		if (video_input)
		{
			time_stamp = (double)frame_count * (1.0 / fps_vid_in);
		}
		else
		{
			// if loading images assume 30fps
			time_stamp = (double)frame_count * (1.0 / 30.0);
		}

		// Reading the images
		cv::Mat_<uchar> grayscale_image;

		if (captured_image.channels() == 3)
		{
			cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
		}
		else
		{
			grayscale_image = captured_image.clone();
		}

		// The actual facial landmark detection / tracking
		bool detection_success;

		if (video_input || images_as_video)
		{
			detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, *face_model, *det_parameters);
		}
		else
		{
			detection_success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, *face_model, *det_parameters);
		}

		// Gaze tracking, absolute gaze direction
		cv::Point3f gazeDirection0(0, 0, -1);
		cv::Point3f gazeDirection1(0, 0, -1);

		if (det_parameters->track_gaze && detection_success && face_model->eye_model)
		{
			FaceAnalysis::EstimateGaze(*face_model, gazeDirection0, fx, fy, cx, cy, true);
			FaceAnalysis::EstimateGaze(*face_model, gazeDirection1, fx, fy, cx, cy, false);
		}

		// Do face alignment
		cv::Mat sim_warped_img;
		cv::Mat_<double> hog_descriptor;

		// But only if needed in output
		if (!output_similarity_align.empty() || hog_output_file.is_open() || output_AUs)
		{
			face_analyser->AddNextFrame(captured_image, *face_model, time_stamp, false, !det_parameters->quiet_mode);
			face_analyser->GetLatestAlignedFace(sim_warped_img);

			if (!det_parameters->quiet_mode)
			{
				cv::imshow("sim_warp", sim_warped_img);
			}
			if (hog_output_file.is_open())
			{
				FaceAnalysis::Extract_FHOG_descriptor(hog_descriptor, sim_warped_img, num_hog_rows, num_hog_cols);

				if (visualise_hog && !det_parameters->quiet_mode)
				{
					cv::Mat_<double> hog_descriptor_vis;
					FaceAnalysis::Visualise_FHOG(hog_descriptor, num_hog_rows, num_hog_cols, hog_descriptor_vis);
					cv::imshow("hog", hog_descriptor_vis);
				}
			}
		}

		// Work out the pose of the head from the tracked model
		cv::Vec6d pose_estimate;
		if (use_world_coordinates)
		{
			pose_estimate = LandmarkDetector::GetCorrectedPoseWorld(*face_model, fx, fy, cx, cy);

		}
		else
		{
			pose_estimate = LandmarkDetector::GetCorrectedPoseCamera(*face_model, fx, fy, cx, cy);
		}

		if (hog_output_file.is_open())
		{
			output_HOG_frame(&hog_output_file, detection_success, hog_descriptor, num_hog_rows, num_hog_cols);
		}

		// Write the similarity normalised output
		if (!output_similarity_align.empty())
		{

			if (sim_warped_img.channels() == 3 && grayscale)
			{
				cvtColor(sim_warped_img, sim_warped_img, CV_BGR2GRAY);
			}

			char name[100];

			// output the frame number
			std::sprintf(name, "frame_det_%06d.bmp", frame_count);

			// Construct the output filename
			boost::filesystem::path slash("/");

			std::string preferredSlash = slash.make_preferred().string();

			string out_file = output_similarity_align[f_n] + preferredSlash + string(name);
			bool write_success = imwrite(out_file, sim_warped_img);

			if (!write_success)
			{
				cout << "Could not output similarity aligned image image" << endl;
				return 1;
			}
		}

		// Visualising the tracker
		visualise_tracking(captured_image, *face_model, *det_parameters, gazeDirection0, gazeDirection1, frame_count, fx, fy, cx, cy);

		// Output the landmarks, pose, gaze, parameters and AUs
		outputAllFeatures(&output_file, output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose, output_AUs, output_gaze,
			*face_model, frame_count, time_stamp, detection_success, gazeDirection0, gazeDirection1,
			pose_estimate, fx, fy, cx, cy, *face_analyser);

		// output the tracked video
		if (!tracked_videos_output.empty())
		{
			writerFace << captured_image;
		}

		if (video_input)//web camera input exist
		{
			video_capture >> captured_image;
		}
		else//videos or image sequences
		{
			curr_img++;
			if (curr_img < (int)input_image_files[f_n].size())
			{
				string curr_img_file = input_image_files[f_n][curr_img];
				captured_image = cv::imread(curr_img_file, -1);
			}
			else
			{
				captured_image = cv::Mat();
			}
		}

		//result = context.WaitNoneUpdateAll();
		//if (XN_STATUS_OK == result)
		//{
		//	imageGenerator.GetMetaData(imageMD);
		//	Mat cvRGBImg(imageMD.FullYRes(), imageMD.FullXRes(), CV_8UC3, (char *)imageMD.Data());
		//	cvtColor(cvRGBImg, captured_image, CV_RGB2BGR);
		//}

		// detect key presses
		char character_press = cv::waitKey(1);

		// restart the tracker
		if (character_press == 'r')
		{
			face_model->Reset();
		}
		// quit the application
		else if (character_press == 'q')
		{
			return(0);
		}

		// Update the frame count
		frame_count++;

		if (total_frames != -1)
		{
			if ((double)frame_count / (double)total_frames >= reported_completion / 10.0)
			{
				cout << reported_completion * 10 << "% ";
				reported_completion = reported_completion + 1;
			}
		}

	}

	//output_file.close();

	if (output_files.size() > 0 && output_AUs)
	{
		cout << "Postprocessing the Action Unit predictions" << endl;
		post_process_output_file(*face_analyser, output_files[f_n], dynamic);
	}
	// Reset the models for the next video
	//face_analyser->Reset();
	//face_model->Reset();

	//frame_count = 0;
	//curr_img = -1;

	if (total_frames != -1)
	{
		cout << endl;
	}

	// break out of the loop if done with all the files (or using a webcam)
	if ((video_input && f_n == input_files.size() - 1) || (!video_input && f_n == input_image_files.size() - 1))
	{
		done = true;
	}

}

// Allows for post processing of the AU signal
void post_process_output_file(FaceAnalysis::FaceAnalyser& face_analyser, string output_file, bool dynamic)
{

	vector<double> certainties;
	vector<bool> successes;
	vector<double> timestamps;
	vector<std::pair<std::string, vector<double>>> predictions_reg;
	vector<std::pair<std::string, vector<double>>> predictions_class;

	// Construct the new values to overwrite the output file with
	face_analyser.ExtractAllPredictionsOfflineReg(predictions_reg, certainties, successes, timestamps, dynamic);
	face_analyser.ExtractAllPredictionsOfflineClass(predictions_class, certainties, successes, timestamps, dynamic);

	int num_class = predictions_class.size();
	int num_reg = predictions_reg.size();

	// Extract the indices of writing out first
	vector<string> au_reg_names = face_analyser.GetAURegNames();
	std::sort(au_reg_names.begin(), au_reg_names.end());
	vector<int> inds_reg;

	// write out ar the correct index
	for (string au_name : au_reg_names)
	{
		for (int i = 0; i < num_reg; ++i)
		{
			if (au_name.compare(predictions_reg[i].first) == 0)
			{
				inds_reg.push_back(i);
				break;
			}
		}
	}

	vector<string> au_class_names = face_analyser.GetAUClassNames();
	std::sort(au_class_names.begin(), au_class_names.end());
	vector<int> inds_class;

	// write out ar the correct index
	for (string au_name : au_class_names)
	{
		for (int i = 0; i < num_class; ++i)
		{
			if (au_name.compare(predictions_class[i].first) == 0)
			{
				inds_class.push_back(i);
				break;
			}
		}
	}
	// Read all of the output file in
	vector<string> output_file_contents;

	std::ifstream infile(output_file);
	string line;

	while (std::getline(infile, line))
		output_file_contents.push_back(line);

	infile.close();

	// Read the header and find all _r and _c parts in a file and use their indices
	std::vector<std::string> tokens;
	boost::split(tokens, output_file_contents[0], boost::is_any_of(","));

	int begin_ind = -1;

	for (size_t i = 0; i < tokens.size(); ++i)
	{
		if (tokens[i].find("AU") != string::npos && begin_ind == -1)
		{
			begin_ind = i;
			break;
		}
	}
	int end_ind = begin_ind + num_class + num_reg;

	// Now overwrite the whole file
	std::ofstream outfile(output_file, ios_base::out);
	// Write the header
	outfile << output_file_contents[0].c_str() << endl;

	// Write the contents
	for (int i = 1; i < (int)output_file_contents.size(); ++i)
	{
		std::vector<std::string> tokens;
		boost::split(tokens, output_file_contents[i], boost::is_any_of(","));

		outfile << tokens[0];

		for (int t = 1; t < (int)tokens.size(); ++t)
		{
			if (t >= begin_ind && t < end_ind)
			{
				if (t - begin_ind < num_reg)
				{
					outfile << ", " << predictions_reg[inds_reg[t - begin_ind]].second[i - 1];
				}
				else
				{
					outfile << ", " << predictions_class[inds_class[t - begin_ind - num_reg]].second[i - 1];
				}
			}
			else
			{
				outfile << ", " << tokens[t];
			}
		}
		outfile << endl;
	}
}

void prepareOutputFile(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_landmarks, int num_model_modes, vector<string> au_names_class, vector<string> au_names_reg)
{

	*output_file << "frame, timestamp, confidence, success";

	if (output_gaze)
	{
		*output_file << ", gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z";
	}

	if (output_pose)
	{
		*output_file << ", pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz";
	}

	if (output_2D_landmarks)
	{
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", x_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", y_" << i;
		}
	}

	if (output_3D_landmarks)
	{
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", X_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", Y_" << i;
		}
		for (int i = 0; i < num_landmarks; ++i)
		{
			*output_file << ", Z_" << i;
		}
	}

	// Outputting model parameters (rigid and non-rigid), the first parameters are the 6 rigid shape parameters, they are followed by the non rigid shape parameters
	if (output_model_params)
	{
		*output_file << ", p_scale, p_rx, p_ry, p_rz, p_tx, p_ty";
		for (int i = 0; i < num_model_modes; ++i)
		{
			*output_file << ", p_" << i;
		}
	}

	if (output_AUs)
	{
		std::sort(au_names_reg.begin(), au_names_reg.end());
		for (string reg_name : au_names_reg)
		{
			*output_file << ", " << reg_name << "_r";
		}

		std::sort(au_names_class.begin(), au_names_class.end());
		for (string class_name : au_names_class)
		{
			*output_file << ", " << class_name << "_c";
		}
	}

	*output_file << endl;

}

// Output all of the information into one file in one go (quite a few parameters, but simplifies the flow)
void outputAllFeatures(std::ofstream* output_file, bool output_2D_landmarks, bool output_3D_landmarks,
	bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	const LandmarkDetector::CLNF& face_model, int frame_count, double time_stamp, bool detection_success,
	cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, const cv::Vec6d& pose_estimate, double fx, double fy, double cx, double cy,
	const FaceAnalysis::FaceAnalyser& face_analyser)
{

	double confidence = 0.5 * (1 - face_model.detection_certainty);

	*output_file << frame_count + 1 << ", " << time_stamp << ", " << confidence << ", " << detection_success;

	// Output the estimated gaze
	if (output_gaze)
	{
		*output_file << ", " << gazeDirection0.x << ", " << gazeDirection0.y << ", " << gazeDirection0.z
			<< ", " << gazeDirection1.x << ", " << gazeDirection1.y << ", " << gazeDirection1.z;
	}

	// Output the estimated head pose
	if (output_pose)
	{
		if (face_model.tracking_initialised)
		{
			*output_file << ", " << pose_estimate[0] << ", " << pose_estimate[1] << ", " << pose_estimate[2]
				<< ", " << pose_estimate[3] << ", " << pose_estimate[4] << ", " << pose_estimate[5];
			for (int i = 0; i < 3; i++)
			{
				transPos.push_back(pose_estimate[i]);
			}
			for (int i = 3; i < 6; i++)
			{
				rotPos.push_back(pose_estimate[i]);
			}
		}
		else
		{
			*output_file << ", 0, 0, 0, 0, 0, 0";
		}
	}

	// Output the detected 2D facial landmarks
	if (output_2D_landmarks)
	{
		for (int i = 0; i < face_model.pdm.NumberOfPoints() * 2; ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << face_model.detected_landmarks.at<double>(i);
			}
			else
			{
				*output_file << ", 0";
			}
		}
	}

	// Output the detected 3D facial landmarks
	if (output_3D_landmarks)
	{
		cv::Mat_<double> shape_3D = face_model.GetShape(fx, fy, cx, cy);

		//std::cout << "original matrix��\n" << shape_3D.t() << "\n";

		//Transfer the coordinate system to the 31st feature point
		//related to the model i use, the 31st feature point is my center point 

		shape_3D = shape_3D.t();//transpose matrix��68 rows 3 cols
		cv::Mat_<double> outShape(68, 3, 0.0);
		double X = shape_3D.at<double>(30, 0);
		double Y = shape_3D.at<double>(30, 1);
		double Z = shape_3D.at<double>(30, 2);
		//std::cout << "X:"<< X << "\tY:" << Y <<  "\tZ:" << Z <<"\n";

		double X1 = shape_3D.at<double>(27, 0);
		double Y1 = shape_3D.at<double>(27, 1);
		double Z1 = shape_3D.at<double>(27, 2);
		//std::cout << "X1:" << X1 << "\tY1:" << Y1 << "\tZ1:" << Z1 << "\n";

		double distance = sqrt((X - X1)*(X - X1) + (Y - Y1)*(Y - Y1) + (Z - Z1)*(Z - Z1));
		//std::cout << "distance��" << distance << "\n";

		for (int i = 0; i < 68; i++)
		{
			outShape.at<double>(i, 0) = shape_3D.at<double>(i, 0) - (double)X;
			outShape.at<double>(i, 1) = shape_3D.at<double>(i, 1) - (double)Y;
			outShape.at<double>(i, 2) = shape_3D.at<double>(i, 2) - (double)Z;
		}
		//std::cout << "matrix��\n" << outShape<< "\n";
		outShape = outShape.t(); //transpose matrix��3 rows 68 cols X��Y��Z

		for (int i = 0; i < face_model.pdm.NumberOfPoints() * 3; ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << outShape.at<double>(i) / (distance / modelScale); //human:0.858875;female:0.585156;faceonly:0.203175;myface:0.042409
				allPos.push_back(outShape.at<double>(i) / (distance / modelScale));
			}
			else
			{
				*output_file << ", 0";
			}
		}
		if (allPos.size() == 204)
		{
			for (int i = 0; i < 68; i++)
			{
				xPos.push_back(allPos[i]);
			}
			for (int i = 68; i < 136; i++)
			{
				yPos.push_back(allPos[i]);
			}
			for (int i = 136; i < 204; i++)
			{
				zPos.push_back(allPos[i]);
			}
		}
	}

	if (output_model_params)
	{
		for (int i = 0; i < 6; ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << face_model.params_global[i];
			}
			else
			{
				*output_file << ", 0";
			}
		}
		for (int i = 0; i < face_model.pdm.NumberOfModes(); ++i)
		{
			if (face_model.tracking_initialised)
			{
				*output_file << ", " << face_model.params_local.at<double>(i, 0);
			}
			else
			{
				*output_file << ", 0";
			}
		}
	}

	if (output_AUs)
	{
		auto aus_reg = face_analyser.GetCurrentAUsReg();

		vector<string> au_reg_names = face_analyser.GetAURegNames();
		std::sort(au_reg_names.begin(), au_reg_names.end());

		// write out ar the correct index
		for (string au_name : au_reg_names)
		{
			for (auto au_reg : aus_reg)
			{
				if (au_name.compare(au_reg.first) == 0)
				{
					*output_file << ", " << au_reg.second;
					break;
				}
			}
		}

		if (aus_reg.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAURegNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}

		auto aus_class = face_analyser.GetCurrentAUsClass();

		vector<string> au_class_names = face_analyser.GetAUClassNames();
		std::sort(au_class_names.begin(), au_class_names.end());

		// write out ar the correct index
		for (string au_name : au_class_names)
		{
			for (auto au_class : aus_class)
			{
				if (au_name.compare(au_class.first) == 0)
				{
					*output_file << ", " << au_class.second;
					break;
				}
			}
		}

		if (aus_class.size() == 0)
		{
			for (size_t p = 0; p < face_analyser.GetAUClassNames().size(); ++p)
			{
				*output_file << ", 0";
			}
		}
	}
	*output_file << endl;
}


void get_output_feature_params(vector<string> &output_similarity_aligned, vector<string> &output_hog_aligned_files, double &similarity_scale,
	int &similarity_size, bool &grayscale, bool& verbose, bool& dynamic,
	bool &output_2D_landmarks, bool &output_3D_landmarks, bool &output_model_params, bool &output_pose, bool &output_AUs, bool &output_gaze,
	vector<string> &arguments)
{
	output_similarity_aligned.clear();
	output_hog_aligned_files.clear();

	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string output_root = "";

	// By default the model is dynamic
	dynamic = true;

	string separator = string(1, boost::filesystem::path::preferred_separator);

	// First check if there is a root argument (so that videos and outputs could be defined more easilly)
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0)
		{
			output_root = arguments[i + 1] + separator;
			i++;
		}
		if (arguments[i].compare("-outroot") == 0)
		{
			output_root = arguments[i + 1] + separator;
			i++;
		}
	}

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-simalign") == 0)
		{
			output_similarity_aligned.push_back(output_root + arguments[i + 1]);
			create_directory(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-hogalign") == 0)
		{
			output_hog_aligned_files.push_back(output_root + arguments[i + 1]);
			create_directory_from_file(output_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-verbose") == 0)
		{
			verbose = true;
		}
		else if (arguments[i].compare("-au_static") == 0)
		{
			dynamic = false;
		}
		else if (arguments[i].compare("-g") == 0)
		{
			grayscale = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-simscale") == 0)
		{
			similarity_scale = stod(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-simsize") == 0)
		{
			similarity_size = stoi(arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-no2Dfp") == 0)
		{
			output_2D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-no3Dfp") == 0)
		{
			output_3D_landmarks = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noMparams") == 0)
		{
			output_model_params = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noPose") == 0)
		{
			output_pose = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noAUs") == 0)
		{
			output_AUs = false;
			valid[i] = false;
		}
		else if (arguments[i].compare("-noGaze") == 0)
		{
			output_gaze = false;
			valid[i] = false;
		}
	}

	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

}

// Can process images via directories creating a separate output file per directory
void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fdir") == 0)
		{

			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			path image_directory(arguments[i + 1]);

			try
			{
				// does the file exist and is it a directory
				if (exists(image_directory) && is_directory(image_directory))
				{

					vector<path> file_in_directory;
					copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

					// Sort the images in the directory first
					sort(file_in_directory.begin(), file_in_directory.end());

					vector<string> curr_dir_files;

					for (vector<path>::const_iterator file_iterator(file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
					{
						// Possible image extension .jpg and .png
						if (file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0)
						{
							curr_dir_files.push_back(file_iterator->string());
						}
					}

					input_image_files.push_back(curr_dir_files);
				}
			}
			catch (const filesystem_error& ex)
			{
				cout << ex.what() << '\n';
			}

			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-asvid") == 0)
		{
			as_video = true;
		}
	}

	// Clear up the argument list
	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

}

void output_HOG_frame(std::ofstream* hog_file, bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_rows, int num_cols)
{

	// Using FHOGs, hence 31 channels
	int num_channels = 31;

	hog_file->write((char*)(&num_cols), 4);
	hog_file->write((char*)(&num_rows), 4);
	hog_file->write((char*)(&num_channels), 4);

	// Not the best way to store a bool, but will be much easier to read it
	float good_frame_float;
	if (good_frame)
		good_frame_float = 1;
	else
		good_frame_float = -1;

	hog_file->write((char*)(&good_frame_float), 4);

	cv::MatConstIterator_<double> descriptor_it = hog_descriptor.begin();

	for (int y = 0; y < num_cols; ++y)
	{
		for (int x = 0; x < num_rows; ++x)
		{
			for (unsigned int o = 0; o < 31; ++o)
			{

				float hog_data = (float)(*descriptor_it++);
				hog_file->write((char*)&hog_data, 4);
			}
		}
	}
}

int main(int argc, char **argv)
{
	// Initialize the GLUT	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(640, 480);
	glutInitWindowPosition(760, 240);
	glutCreateWindow("Facial Animation");
	init();

	//read the mapping between model and feature points
	readMap();
	//load OBJ file
	openObjFile(objFilePath);
	cout << "vertices num of model:" << pModel->numvertices << endl;
	clock_t b = clock();
	//obtain the neighbours of each vertex
	allVertexneighbors = getAllVertexNeighbors();
	//obtain the Laplacian matrix of model
	laplacianMatrixEigen = getLaplacianMatrixUmbrella(pModel, anchorsIdx);
	clock_t e = clock();
	std::cout << "neighbor & lapM time cost:" << e - b << std::endl;
	//calculate��Lt*L��reverse
	calculateLTL(laplacianMatrixEigen);

	//Change the position of the origin of the model coordinates
	changeOriginPoint();

	// Set the callback function
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);

	initialOpenCV();

	glutIdleFunc(idle);
	glutMainLoop();
	return 0;
}
