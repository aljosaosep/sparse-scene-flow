/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
RWTH Aachen University, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep (osep -at- vision.rwth-aachen.de)

rwth_mot framework is free software; you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or any later version.

rwth_mot framework is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
rwth_mot framework; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#ifndef GOT_DATASETS_DIRTY_UTILS_H
#define GOT_DATASETS_DIRTY_UTILS_H

// OpenCV
#include <opencv2/highgui/highgui.hpp>

// Boost
#include <boost/program_options.hpp>

// KITTI
#include <libkitti/kitti.h>

#include "camera.h"

namespace po = boost::program_options;

namespace SUN {
    namespace utils {
        namespace dirty {

            class DatasetAssitantDirty {
            public:
                DatasetAssitantDirty(const po::variables_map &config_variables_map);
                bool LoadData__KITTI(int current_frame);
                bool LoadData__OXFORD(int current_frame);
                bool LoadData(int current_frame, const std::string dataset_string);

                cv::Mat left_image_;
                cv::Mat right_image_;
                SUN::utils::Camera left_camera_;
                SUN::utils::Camera right_camera_;
                double stereo_baseline_;
                po::variables_map variables_map_;
            };
        }
    }
}


#endif //GOT_DATASETS_DIRTY_UTILS_H
