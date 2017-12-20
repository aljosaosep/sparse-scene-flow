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

#ifndef SUN_UTILS_VISUALIZATION
#define SUN_UTILS_VISUALIZATION

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// project
#include "camera.h"

// Forward declarations
namespace SUN { namespace utils { class Detection; } }

namespace SUN {
    namespace utils {
        namespace visualization {

            struct BirdEyeVizProperties {
                // Bird-eye visualization
                float birdeye_scale_factor_;
                int birdeye_far_plane_;
                int birdeye_left_plane_;
                int birdeye_right_plane_;

                BirdEyeVizProperties() {
                    // Bird-eye visualization module params
                    birdeye_scale_factor_ = 13.0;
                    birdeye_far_plane_ = 50;
                    birdeye_left_plane_ = -20;
                    birdeye_right_plane_ = 20;
                }
            };

            // -------------------------------------------------------------------------------
            // +++ PRIMITIVES +++
            // -------------------------------------------------------------------------------
            void DrawLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const SUN::utils::Camera &camera, cv::Mat &ref_image,
                          const cv::Scalar &color, int thickness=1, int line_type=1, const cv::Point2i &offset=cv::Point2i(0,0));

            void ArrowedLine(cv::Point2d pt1, cv::Point2d pt2, const cv::Scalar& color, cv::Mat &ref_image, int thickness=1, int line_type=8, int shift=0,
                             double tipLength=0.1);
            void DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image);

            // -------------------------------------------------------------------------------
            // +++ COVARIANCE MATRICES +++
            // -------------------------------------------------------------------------------
            /**
              * @brief Draws an iso-contour of the covariance matrix (iso-contour is picked via chisquare_val)
              * @author Aljosa (osep@vision.rwth-aachen.de)
              */
            void DrawCovarianceMatrix2dEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image, cv::Vec3f color);

            /**
              * @brief Draws smooth representation of covariance matrix (via particles).
              * @author Dirk (klostermann@rwth-aachen.de)
              */
            void DrawCovarianceMatrix2dSmooth(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image, cv::Vec3f color);

            void DrawSparseFlowBirdeye(const std::vector<Eigen::Vector3d> &pts, const std::vector<Eigen::Vector3d> &vel,
                                       const SUN::utils::Camera &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image);

            /// Bird-eye visualization tools
            void TransformPointToScaledFrustum(double &pose_x, double &pose_z, const BirdEyeVizProperties &viz_props);
            void DrawGridBirdeye(double res_x, double res_z, const SUN::utils::Camera &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image);

        }
    }
}

#endif
