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

#include "datasets_dirty_utils.h"

// boost
#include <boost/filesystem.hpp>

// opencv
#include <opencv2/imgproc.hpp>

// utils
#include "sun_utils/utils_io.h"

#define MAX_PATH_LEN 500

namespace SUN {
    namespace utils {
        namespace dirty {

            // -------------------------------------------------------------------------------
            // +++ DATASET ASSISTANT IMPLEMENTATION +++
            // -------------------------------------------------------------------------------
            DatasetAssitantDirty::DatasetAssitantDirty(const po::variables_map &config_variables_map) {
                this->variables_map_ = config_variables_map;
                stereo_baseline_ = -1;
            }

            bool DatasetAssitantDirty::LoadData(int current_frame, const std::string dataset_string) {
                std::string dataset_str_lower = dataset_string;
                std::transform(dataset_str_lower.begin(), dataset_str_lower.end(), dataset_str_lower.begin(), ::tolower);

                bool status = false;
                if (dataset_str_lower=="kitti")
                    status = this->LoadData__KITTI(current_frame);
                else if (dataset_str_lower=="oxford")
                    status = this->LoadData__OXFORD(current_frame);
                else
                    std::cout << "DatasetAssitantDirty error: no such dataset: " << dataset_string << std::endl;

                return status;
            }

            bool DatasetAssitantDirty::LoadData__KITTI(int current_frame) {
                assert (this->variables_map_.count("right_image_path"));
                assert (this->variables_map_.count("left_image_path"));
                assert (this->variables_map_.count("calib_path"));

                // KITTI camera calibration
                SUN::utils::KITTI::Calibration calibration;
                const std::string calib_path = this->variables_map_["calib_path"].as<std::string>();
                if (!calibration.Open(calib_path)) {
                    printf("DatasetAssitantDirty error: Can't Open calibration file: %s\r\n", calib_path.c_str());
                    return false;
                }

                // Image data
                char left_image_path_buff[MAX_PATH_LEN];
                snprintf(left_image_path_buff, MAX_PATH_LEN, this->variables_map_["left_image_path"].as<std::string>().c_str(), current_frame);
                left_image_ = cv::imread(left_image_path_buff, CV_LOAD_IMAGE_COLOR);
                if (left_image_.data == nullptr) {
                    printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                    return false;
                }

                // Right image
                char right_image_path_buff[MAX_PATH_LEN];
                snprintf(right_image_path_buff, MAX_PATH_LEN, this->variables_map_["right_image_path"].as<std::string>().c_str(), current_frame);
                right_image_ = cv::imread(right_image_path_buff, CV_LOAD_IMAGE_COLOR);

                if (right_image_.data == nullptr) {
                    printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                    return false;
                }

                // Init cameras
                left_camera_.init(calibration.GetProjCam2(), Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                right_camera_.init(calibration.GetProjCam3(), Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                stereo_baseline_ = calibration.b();

                return true;
            }

            bool DatasetAssitantDirty::LoadData__OXFORD(int current_frame) {

                assert (this->variables_map_.count("right_image_path"));
                assert (this->variables_map_.count("left_image_path"));
                assert (this->variables_map_.count("calib_path"));

                // Left image
                char left_image_path_buff[MAX_PATH_LEN];
                snprintf(left_image_path_buff, MAX_PATH_LEN, this->variables_map_["left_image_path"].as<std::string>().c_str(), current_frame);
                left_image_ = cv::imread(left_image_path_buff, CV_LOAD_IMAGE_COLOR);
                if (left_image_.data == nullptr) {
                    printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                    return false;
                }

                // Right image
                char right_image_path_buff[MAX_PATH_LEN];
                snprintf(right_image_path_buff, MAX_PATH_LEN, this->variables_map_["right_image_path"].as<std::string>().c_str(), current_frame);
                right_image_ = cv::imread(right_image_path_buff, CV_LOAD_IMAGE_COLOR);

                if (right_image_.data == nullptr) {
                    printf("DatasetAssitantDirty error: could not load image: %s\r\n", left_image_path_buff);
                    return false;
                }

                /// Init camera
                const std::string calib_path = this->variables_map_["calib_path"].as<std::string>();
                auto getIntrinsicMatOxford = [calib_path](char *which_cam)->Eigen::Matrix3d {
                    char buff[500];
                    snprintf(buff, 500, "%s/stereo_wide_%s.txt", calib_path.c_str(), which_cam);
                    Eigen::MatrixXd stupid_intrinsics;
                    SUN::utils::IO::ReadEigenMatrixFromTXT(buff, stupid_intrinsics);
                    Eigen::Matrix3d K;
                    K.setIdentity();
                    K(0,0) = stupid_intrinsics(0,0); // fx
                    K(1,1) = stupid_intrinsics(0,1); // fy
                    K(0,2) = stupid_intrinsics(0,2); // cx
                    K(1,2) = stupid_intrinsics(0,3); // cx
                    return K;
                };

                Eigen::Matrix3d K_left = getIntrinsicMatOxford("left");
                Eigen::Matrix3d K_right = getIntrinsicMatOxford("right");

                const float oxford_f    = (float)K_left(0,0);
                const float oxford_c_u  = (float)K_left(0,2);
                const float oxford_c_v  = (float)K_left(1,2);
                const float oxford_b    = 0.24;

                Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
                Eigen::Matrix<double, 3, 4> T_left, T_right;
                T_left.setZero();
                T_right.setZero();
                T_left.block(0,0,3,3) = Eigen::Matrix3d::Identity();
                T_right.block(0,0,3,3) = Eigen::Matrix3d::Identity();
                T_right(0, 3) = -oxford_b;

                Eigen::Matrix<double, 3, 4> P_left = K_left * T_left * R;
                Eigen::Matrix<double, 3, 4> P_right = K_right * T_right * R;

                left_camera_.init(P_left, Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                right_camera_.init(P_right, Eigen::Matrix4d::Identity(), left_image_.cols, left_image_.rows);
                stereo_baseline_ = oxford_b;

                return true;
            }

        }
    }
}