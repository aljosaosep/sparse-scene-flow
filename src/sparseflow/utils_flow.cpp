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


#include <libviso2/viso_stereo.h>

#include "utils_flow.h"

namespace SUN {
    namespace utils {
        namespace scene_flow {
            libviso2::Matcher *InitMatcher() {
                // parameters
                libviso2::Matcher::parameters matcher_params;

                matcher_params.nms_n = 5;   // non-max-suppression: min. distance between maxima (in pixels)
                matcher_params.nms_tau = 50;  // non-max-suppression: interest point peakiness threshold
                matcher_params.match_binsize = 50;  // matching bin width/height (affects efficiency only)
                matcher_params.match_radius = 200; // matching radius (du/dv in pixels)
                matcher_params.match_disp_tolerance = 1;   // du tolerance for stereo matches (in pixels)
                matcher_params.outlier_disp_tolerance = 5;   // outlier removal: disparity tolerance (in pixels)
                matcher_params.outlier_flow_tolerance = 5;   // outlier removal: flow tolerance (in pixels)
                matcher_params.multi_stage = 1;   // 0=disabled,1=multistage matching (denser and faster)
                matcher_params.half_resolution = 0;   // 0=disabled,1=match at half resolution, refine at full resolution
                matcher_params.refinement = 2;   // refinement (0=none,1=pixel,2=subpixel)

                // bucketing parameters
                //matcher_params.max_features         = 4;
                //matcher_params.bucket_width         = 10;
                //matcher_params.bucket_height        = 10;

                // create matcher instance
                auto *M = new libviso2::Matcher(matcher_params);
                return M;
            }

            Eigen::Vector4d ProjTo3D(float u1, float u2, float v, const SUN::utils::Camera &camera, float baseline) {
                const double f = camera.f_u(); // Focal
                const double cu = camera.c_u();
                const double cv = camera.c_v();
                const double d = std::fabs(u2 - u1); // Disparity

                Eigen::Vector4d ret(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());

                if (d > 0.0001) {
                    double X = ((u1 - cu) * baseline) / d;
                    double Y = ((v - cv) * baseline) / d;
                    double Z = f * baseline / d;
                    ret = Eigen::Vector4d(X, Y, Z, 1.0);
                }

                return ret;
            }

//            Eigen::Vector4d ProjTo3D_(float u1, float u2, float v, float f, float cu, float cv, float baseline) {
//
//                const double d = std::fabs(u2 - u1); // Disparity
//
//                Eigen::Vector4d ret(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
//                                    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
//
//                if (d > 0.0001) {
//                    double X = ((u1 - cu) * baseline) / d;
//                    double Y = ((v - cv) * baseline) / d;
//                    double Z = f * baseline / d;
//                    ret = Eigen::Vector4d(X, Y, Z, 1.0);
//                }
//
//                return ret;
//            }

//            std::vector<libviso2::Matcher::p_match>
//            GetMatches(libviso2::Matcher *M, const cv::Mat &color_left, const cv::Mat &color_right, const SUN::utils::Camera &camera, float baseline,
//                       bool only_push) {
//
//                cv::Mat grayscale_left, grayscale_right;
//                cv::cvtColor(color_left, grayscale_left, CV_BGR2GRAY);
//                cv::cvtColor(color_right, grayscale_right, CV_BGR2GRAY);
//                const int32_t width = color_left.cols;
//                const int32_t height = color_left.rows;
//
//                // Convert input images to uint8_t buffer
//                uint8_t *left_img_data = (uint8_t *) malloc(width * height * sizeof(uint8_t));
//                uint8_t *right_img_data = (uint8_t *) malloc(width * height * sizeof(uint8_t));
//                int32_t k = 0;
//                for (int32_t v = 0; v < height; v++) {
//                    for (int32_t u = 0; u < width; u++) {
//                        left_img_data[k] = (uint8_t) grayscale_left.at<uchar>(v, u);
//                        right_img_data[k] = (uint8_t) grayscale_right.at<uchar>(v, u);
//                        k++;
//                    }
//                }
//
//                int32_t dims[] = {width, height, width};
//
//                // Push images
//                M->pushBack(left_img_data, right_img_data, dims, false);
//
//                std::vector<libviso2::Matcher::p_match> matches;
//                if (!only_push) {
//
//                    // do matching
//                    M->matchFeatures(2); // 2 ... quad matching
//
//                    // Get matches
//                    // quad matching
//                    matches = M->getMatches();
//                }
//
//                free(left_img_data);
//                free(right_img_data);
//
//                return matches;
//            }
//
//            std::tuple<cv::Mat, std::vector<VelocityInfo> >
//            GetSceneFlow(std::vector<libviso2::Matcher::p_match> quad_matches, const Eigen::Matrix4d Tr, const SUN::utils::Camera &camera,
//                         float baseline, float dt, float max_velocity_ms) {
//
//
//                cv::Mat velocity_map(camera.height(), camera.width(), CV_32FC3);
//                velocity_map.setTo(cv::Vec3f(std::numeric_limits<float>::quiet_NaN(),
//                                             std::numeric_limits<float>::quiet_NaN(),
//                                             std::numeric_limits<float>::quiet_NaN()));
//
//
//                std::vector<VelocityInfo> velocity_info;
//
//                // Project matches to 3D
//                for (const auto &match:quad_matches) {
//
//                    Eigen::Vector4d p3d_c = ProjTo3D(match.u1c, match.u2c, match.v1c, camera, baseline); // p3d curr frame
//                    Eigen::Vector4d p3d_p = ProjTo3D(match.u1p, match.u2p, match.v1p, camera, baseline); // p3d prev frame
//
//                    if (std::isnan(p3d_c[0]) || std::isnan(p3d_p[0])) continue;
//
//                    const Eigen::Vector4d p3d_c_orig = p3d_c;
//                    const Eigen::Vector4d p3d_p_orig = p3d_p;
//
//                    // Project prev to curr frame using ego estimate
//                    p3d_p = Tr * p3d_p;
//
//                    // Project to ground
//                    //p3d_c.head<3>() = camera.ground_model()->ProjectPointToGround(p3d_c.head<3>());
//                    //p3d_p.head<3>() = camera.ground_model()->ProjectPointToGround(p3d_p.head<3>());
//
//                    int max_dist = 90;
//                    int max_lat = 30;
//                    if (std::fabs(p3d_c[0]) > max_lat || std::fabs(p3d_c[2]) > max_dist || std::fabs(p3d_p[0]) > max_lat || std::fabs(p3d_p[2]) > max_dist)
//                        continue;
//
//                    Eigen::Vector3d delta = (p3d_c - p3d_p).head<3>();
//
//                    if (delta.norm()*(1.0/dt) < max_velocity_ms) {
//
//                        velocity_map.at<cv::Vec3f>(match.v1c, match.u1c) = cv::Vec3f(delta[0], delta[1], delta[2]);
//
//                        VelocityInfo vi;
//                        vi.p = Eigen::Vector2i(match.u1c, match.v1c);
//                        vi.p_3d = p3d_c_orig.head<3>();
//                        vi.p_vel = delta;
//                        vi.p_prev = p3d_p_orig.head<3>();
//                        vi.p_prev_to_curr = p3d_p.head<3>();
//                        velocity_info.push_back(vi);
//                    }
//                }
//
//                return std::make_tuple(velocity_map, velocity_info);
//            }
//
//            Eigen::Matrix4d EstimateEgomotion(libviso2::VisualOdometryStereo &viso, const cv::Mat &color_left, const cv::Mat &color_right) {
//                cv::Mat grayscale_left, grayscale_right;
//                cv::cvtColor(color_left, grayscale_left, CV_BGR2GRAY);
//                cv::cvtColor(color_right, grayscale_right, CV_BGR2GRAY);
//                const int32_t width = color_left.cols;
//                const int32_t height = color_left.rows;
//
//                // Convert input images to uint8_t buffer
//                uint8_t* left_img_data  = (uint8_t*)malloc(width*height*sizeof(uint8_t));
//                uint8_t* right_img_data = (uint8_t*)malloc(width*height*sizeof(uint8_t));
//                int32_t k=0;
//                for (int32_t v=0; v<height; v++) {
//                    for (int32_t u=0; u<width; u++) {
//                        left_img_data[k]  = (uint8_t)grayscale_left.at<uchar>(v,u);
//                        right_img_data[k] = (uint8_t)grayscale_right.at<uchar>(v,u);
//                        k++;
//                    }
//                }
//
//                Eigen::Matrix<double,4,4> egomotion_eigen = Eigen::MatrixXd::Identity(4,4); // Current pose
//                int32_t dims[] = {width,height,width};
//                if (viso.process(left_img_data,right_img_data,dims)) {
//                    libviso2::Matrix frame_to_frame_motion = viso.getMotion(); //libviso2::Matrix::inv(viso->getMotion());
//                    for(size_t i=0; i<4; i++){
//                        for(size_t j=0; j<4; j++){
//                            egomotion_eigen(i,j) = frame_to_frame_motion.val[i][j];
//                        }
//                    }
//                }
//
//                free(left_img_data);
//                free(right_img_data);
//                return egomotion_eigen;
//            }
        }
    }
}
