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

#include "stdafx.h"

#include "LandmarkDetectorParameters.h"

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

// System includes
#include <sstream>
#include <iostream>
#include <cstdlib>

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

using namespace std;

using namespace LandmarkDetector;

FaceModelParameters::FaceModelParameters()
{
	// initialise the default values
	init();
}

FaceModelParameters::FaceModelParameters(vector<string> &arguments)
{
	// initialise the default values
	init();

	// First element is reserved for the executable location (useful for finding relative model locs)
	boost::filesystem::path root = boost::filesystem::path(arguments[0]).parent_path();

	bool* valid = new bool[arguments.size()];
	valid[0] = true;

	for (size_t i = 1; i < arguments.size(); ++i)
	{
		valid[i] = true;

		if (arguments[i].compare("-mloc") == 0)
		{
			string model_loc = arguments[i + 1];
			model_location = model_loc;
			valid[i] = false;
			valid[i + 1] = false;
			i++;

		}
		if (arguments[i].compare("-sigma") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> sigma;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-w_reg") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> weight_factor;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-reg") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> reg_factor;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-multi_view") == 0)
		{

			stringstream data(arguments[i + 1]);
			int m_view;
			data >> m_view;

			multi_view = (bool)(m_view != 0);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-validate_detections") == 0)
		{
			stringstream data(arguments[i + 1]);
			int v_det;
			data >> v_det;

			validate_detections = (bool)(v_det != 0);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-n_iter") == 0)
		{
			stringstream data(arguments[i + 1]);
			data >> num_optimisation_iteration;

			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-gaze") == 0)
		{
			track_gaze = true;

			valid[i] = false;
			i++;
		}
		else if (arguments[i].compare("-q") == 0)
		{

			quiet_mode = true;

			valid[i] = false;
		}
		else if (arguments[i].compare("-wild") == 0)
		{
			// For in the wild fitting these parameters are suitable
			window_sizes_init = vector<int>(4);
			window_sizes_init[0] = 15; window_sizes_init[1] = 13; window_sizes_init[2] = 11; window_sizes_init[3] = 9;

			sigma = 1.25;
			reg_factor = 35;
			weight_factor = 2.5;
			num_optimisation_iteration = 10;

			valid[i] = false;

			// For in-the-wild images use an in-the wild detector				
			curr_face_detector = HOG_SVM_DETECTOR;

		}
	}

	for (int i = (int)arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

	// Make sure model_location is valid
	// First check working directory, then the executable's directory, then the config path set by the build process.
	boost::filesystem::path config_path = boost::filesystem::path(CONFIG_DIR);
	boost::filesystem::path model_path = boost::filesystem::path(model_location);
	if (boost::filesystem::exists(model_path))
	{
		model_location = model_path.string();
	}
	else if (boost::filesystem::exists(root/model_path))
	{
		model_location = (root/model_path).string();
	}
	else if (boost::filesystem::exists(config_path/model_path))
	{
		model_location = (config_path/model_path).string();
	}
	else
	{
		std::cout << "Could not find the landmark detection model to load" << std::endl;
	}
}

void FaceModelParameters::init()
{

	// number of iterations that will be performed at each scale
	num_optimisation_iteration = 5;

	// using an external face checker based on SVM
	validate_detections = true;

	// Using hierarchical refinement by default (can be turned off)
	refine_hierarchical = true;

	// Refining parameters by default
	refine_parameters = true;

	window_sizes_small = vector<int>(4);
	window_sizes_init = vector<int>(4);

	// For fast tracking
	window_sizes_small[0] = 0;
	window_sizes_small[1] = 9;
	window_sizes_small[2] = 7;
	window_sizes_small[3] = 5;

	// Just for initialisation
	window_sizes_init.at(0) = 11;
	window_sizes_init.at(1) = 9;
	window_sizes_init.at(2) = 7;
	window_sizes_init.at(3) = 5;

	face_template_scale = 0.3;
	// Off by default (as it might lead to some slight inaccuracies in slowly moving faces)
	use_face_template = false;

	// For first frame use the initialisation
	window_sizes_current = window_sizes_init;

	model_location = "model/main_clnf_general.txt";

	sigma = 1.5;
	reg_factor = 25;
	weight_factor = 0; // By default do not use NU-RLMS for videos as it does not work as well for them

	validation_boundary = -0.45;

	limit_pose = true;
	multi_view = false;

	reinit_video_every = 4;

	// Face detection
#if OS_UNIX
	face_detector_location = "classifiers/haarcascade_frontalface_alt.xml";
#else
	face_detector_location = "classifiers/haarcascade_frontalface_alt.xml";
#endif

	quiet_mode = false;

	// By default use HOG SVM
	curr_face_detector = HOG_SVM_DETECTOR;

	// The gaze tracking has to be explicitly initialised
	track_gaze = false;
}

