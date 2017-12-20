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
#ifndef GOT_SPARSE_FLOW_H
#define GOT_SPARSE_FLOW_H

// libviso
#include <libviso2/matcher.h>

// eigen
#include <Eigen/Core>

// cv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// utils
#include "camera.h"

namespace  libviso2 { class VisualOdometryStereo; }

namespace SUN {
    namespace utils {
        namespace scene_flow {

            struct VelocityInfo {
                Eigen::Vector2i p;
                Eigen::Vector3d p_3d;
                Eigen::Vector3d p_vel;

                Eigen::Vector3d p_prev;
                Eigen::Vector3d p_prev_to_curr;
            };

            libviso2::Matcher *InitMatcher();

            Eigen::Vector4d ProjTo3D(float u1, float u2, float v, const SUN::utils::Camera &camera, float baseline);

            std::vector<libviso2::Matcher::p_match>
            GetMatches(libviso2::Matcher *M, const cv::Mat &color_left, const cv::Mat &color_right,
                       const SUN::utils::Camera &camera,
                       float baseline, bool only_push);

            std::tuple<cv::Mat, std::vector<VelocityInfo> >
            GetSceneFlow(std::vector<libviso2::Matcher::p_match> quad_matches, const Eigen::Matrix4d Tr, const SUN::utils::Camera &camera,
                         float baseline, float dt, float max_velocity_ms=40.0);

            Eigen::Matrix4d EstimateEgomotion(libviso2::VisualOdometryStereo &viso, const cv::Mat &color_left, const cv::Mat &color_right);

            }
    }
}


#endif //GOT_SPARSE_FLOW_H
