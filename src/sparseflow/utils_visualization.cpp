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

#include "utils_visualization.h"

// std
#include <random>

// OpenCV
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

// eigen
#include <Eigen/Dense>

// Boost
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

namespace Eigen {
    namespace internal {
        template<typename Scalar>
        struct scalar_normal_dist_op
        {
            static boost::mt19937 rng;    // The uniform pseudo-random algorithm
            mutable boost::normal_distribution<Scalar> norm;  // The gaussian combinator

            EIGEN_EMPTY_STRUCT_CTOR(scalar_normal_dist_op)

            template<typename Index>
            inline const Scalar operator() (Index, Index = 0) const { return norm(rng); }
        };

        template<typename Scalar> boost::mt19937 scalar_normal_dist_op<Scalar>::rng;

        template<typename Scalar>
        struct functor_traits<scalar_normal_dist_op<Scalar> >
        { enum { Cost = 50 * NumTraits<Scalar>::MulCost, PacketAccess = false, IsRepeatable = false }; };
    } // end namespace internal
} // end namespace Eigen

namespace SUN {
    namespace utils {
        namespace visualization {

            void DrawLine(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const SUN::utils::Camera &camera, cv::Mat &ref_image, const cv::Scalar &color, int thickness, int line_type, const cv::Point2i &offset) {
                Eigen::Vector4d p1_4d, p2_4d;
                p1_4d[3] = p2_4d[3] = 1.0;
                p1_4d.head<3>() = p1;
                p2_4d.head<3>() = p2;
                Eigen::Vector3i projected_point_1 = camera.CameraToImage(p1_4d);
                Eigen::Vector3i projected_point_2 = camera.CameraToImage(p2_4d);
                auto cv_p1 = cv::Point2i(projected_point_1[0],projected_point_1[1]);
                auto cv_p2 = cv::Point2i(projected_point_2[0],projected_point_2[1]);

                bool p1_in_bounds = true;
                bool p2_in_bounds = true;
                if ((cv_p1.x < 0) && (cv_p1.y < 0) && (cv_p1.x > ref_image.cols) && (cv_p1.y > ref_image.rows) )
                    p1_in_bounds = false;

                if ((cv_p2.x < 0) && (cv_p2.y < 0) && (cv_p2.x > ref_image.cols) && (cv_p2.y > ref_image.rows) )
                    p2_in_bounds = false;

                // Draw line, but only if both end-points project into the image!
                if (p1_in_bounds || p2_in_bounds) { // This is correct. Won't draw only if both lines are out of bounds.
                    // Draw line
                    auto p1_offs = offset+cv_p1;
                    auto p2_offs = offset+cv_p2;
                    if (cv::clipLine(cv::Size(/*0, 0, */ref_image.cols, ref_image.rows), p1_offs, p2_offs)) {
                        //cv::line(ref_image, p1_offs, p2_offs, color, thickness, line_type);
                        cv::arrowedLine(ref_image, p1_offs, p2_offs, color, thickness, line_type);
                    }
                }
            }

            void DrawTransparentSquare(cv::Point center, cv::Vec3b color, int radius, double alpha, cv::Mat &ref_image) {
                for (int i=-radius; i<radius; i++) {
                    for (int j=-radius; j<radius; j++) {
                        int coord_y = center.y + i;
                        int coord_x = center.x + j;

                        if (coord_x>0 && coord_y>0 && coord_x<ref_image.cols && coord_y < ref_image.rows) {
                            ref_image.at<cv::Vec3b>(cv::Point(coord_x,coord_y)) = (1.0-alpha)*ref_image.at<cv::Vec3b>(cv::Point(coord_x,coord_y)) + alpha*color;

                        }
                    }
                }
            }


            void DrawGridBirdeye(double res_x, double res_z, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image) {

                auto color = cv::Scalar(0.0, 0.0, 0.0);
                // Draw horizontal lines
                for (double i=0; i<viz_props.birdeye_far_plane_; i+=res_z) {
                    double x_1 = viz_props.birdeye_left_plane_;
                    double y_1 = i;
                    double x_2 = viz_props.birdeye_right_plane_;
                    double y_2 = i;
                    TransformPointToScaledFrustum(x_1, y_1, viz_props);
                    TransformPointToScaledFrustum(x_2, y_2, viz_props);
                    auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
                    cv::line(ref_image, p1, p2, color);
                }

                // Draw vertical lines
                for (double i=viz_props.birdeye_left_plane_; i<viz_props.birdeye_right_plane_; i+=res_x) {
                    double x_1 = i;
                    double y_1 = 0;
                    double x_2 = i;
                    double y_2 = viz_props.birdeye_far_plane_;
                    TransformPointToScaledFrustum(x_1, y_1, viz_props);
                    TransformPointToScaledFrustum(x_2, y_2, viz_props);
                    auto p1 = cv::Point(x_1, y_1), p2=cv::Point(x_2,y_2);
                    cv::line(ref_image, p1, p2, color);
                }
            }

            void DrawSparseFlowBirdeye(
                    const std::vector<Eigen::Vector3d> &pts, const std::vector<Eigen::Vector3d> &vel,
                    const SUN::utils::Camera &camera, const BirdEyeVizProperties &viz_props, cv::Mat &ref_image) {

                // For scaling / flipping cov. matrices
                Eigen::Matrix2d flip_mat;
                flip_mat << viz_props.birdeye_scale_factor_ * 1.0, 0, 0, viz_props.birdeye_scale_factor_ * (/*-*/1.0);
                Eigen::Matrix2d world_to_cam_mat;
                const Eigen::Matrix4d &ref_to_rt_inv = camera.Rt_inv();
                world_to_cam_mat << ref_to_rt_inv(0, 0), ref_to_rt_inv(2, 0), ref_to_rt_inv(0, 2), ref_to_rt_inv(2, 2);
                flip_mat = flip_mat * world_to_cam_mat;

                // Params
                const int line_width = 2;

                ref_image = cv::Mat(viz_props.birdeye_scale_factor_*viz_props.birdeye_far_plane_,
                                    (-viz_props.birdeye_left_plane_+viz_props.birdeye_right_plane_)*viz_props.birdeye_scale_factor_, CV_32FC3);
                ref_image.setTo(cv::Scalar(1.0, 1.0, 1.0));

                DrawGridBirdeye(1.0, 1.0, viz_props, ref_image);


                for (int i=0; i<pts.size(); i++) {

                    Eigen::Vector3d p_3d = pts.at(i);
                    Eigen::Vector3d p_vel = vel.at(i);
                    Eigen::Vector3i p_proj = camera.CameraToImage(Eigen::Vector4d(p_3d[0], p_3d[1], p_3d[2], 1.0));
                    const Eigen::Vector2d velocity = Eigen::Vector2d(p_vel[0], p_vel[2]); // !!!
                    Eigen::Vector3d dir(velocity[0], 0.0, velocity[1]);

                    double x_1 = p_3d[0];
                    double z_1 = p_3d[2];

                    double x_2 = x_1 + dir[0];
                    double z_2 = z_1 + dir[2];

                    if (x_1 > viz_props.birdeye_left_plane_ && x_2 > viz_props.birdeye_left_plane_ &&
                        x_1 < viz_props.birdeye_right_plane_ && x_2 < viz_props.birdeye_right_plane_ &&
                        z_1 > 0 && z_2 > 0 &&
                        z_1 < viz_props.birdeye_far_plane_ && z_2 < viz_props.birdeye_far_plane_) {

                        TransformPointToScaledFrustum(x_1, z_1, viz_props); //velocity[0], velocity[1]);
                        TransformPointToScaledFrustum(x_2, z_2, viz_props); //velocity[0], velocity[1]);

                        cv::arrowedLine(ref_image, cv::Point(x_1, z_1), cv::Point(x_2, z_2), cv::Scalar(1.0, 0.0, 0.0), 1);
                        cv::circle(ref_image, cv::Point(x_1, z_1), 3.0, cv::Scalar(0.0, 0.0, 1.0), -1.0);
                    }
                }

                // Coord. sys.
                int arrow_len = 60;
                int offset_y = 10;
                cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                                cv::Point(ref_image.cols/2+arrow_len, offset_y),
                                cv::Scalar(1.0, 0, 0), 2);
                cv::arrowedLine(ref_image, cv::Point(ref_image.cols/2, offset_y),
                                cv::Point(ref_image.cols/2, offset_y+arrow_len),
                                cv::Scalar(0.0, 1.0, 0), 2);

                //cv::putText(ref_image, "X", cv::Point(ref_image.cols/2+arrow_len+10, offset_y+10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(1.0, 0, 0));
                //cv::putText(ref_image, "Z", cv::Point(ref_image.cols/2+10, offset_y+arrow_len), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar(0.0, 1.0, 0));

                // Flip image, because it is more intuitive to have ref. point at the bottom of the image
                cv::Mat dst;
                cv::flip(ref_image, dst, 0);
                ref_image = dst;
            }

            void TransformPointToScaledFrustum(double &pose_x, double &pose_z, const BirdEyeVizProperties &viz_props/*, const double left_plane, const double scale_factor*/) {
                pose_x += (-viz_props.birdeye_left_plane_);
                pose_x *= viz_props.birdeye_scale_factor_;
                pose_z *= viz_props.birdeye_scale_factor_;
            }

            // Adapted from Vincent Spruyt
            void DrawCovarianceMatrix2dEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image, cv::Vec3f color) {
                // Get the eigenvalues and eigenvectors
                cv::Mat eigenvalues, eigenvectors;
                cv::eigen(covmat, eigenvalues, eigenvectors);

                //Calculate the angle between the largest eigenvector and the x-axis
                double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

                // Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
                if(angle < 0)
                    angle += 6.28318530718;

                // Convert to degrees instead of radians
                angle = 180*angle/3.14159265359;

                // Calculate the size of the minor and major axes
                double half_majoraxis_size=chisquare_val*sqrt(std::abs(eigenvalues.at<double>(0)));
                double half_minoraxis_size=chisquare_val*sqrt(std::abs(eigenvalues.at<double>(1)));

                // Return the oriented ellipse
                // The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
                cv::RotatedRect rot_rect(mean, cv::Size2f(half_majoraxis_size, half_minoraxis_size), /*-*/angle);
                cv::ellipse(ref_image,  rot_rect, cv::Scalar(color[0], color[1], color[2]), 1);
            }

            // Based on: http://stackoverflow.com/questions/6142576/sample-from-multivariate-normal-gaussian-distribution-in-c
            void DrawCovarianceMatrix2dSmooth(double chisquare_val, cv::Point2f mean, cv::Mat covmat, cv::Mat &ref_image, cv::Vec3f color) {
                // Get the eigenvalues and eigenvectors
                cv::Mat eigenvalues, eigenvectors;
                cv::eigen(covmat, eigenvalues, eigenvectors);

                int size = 2; // Dimensionality (rows)
                int nn=10000; // How many samples (columns) to draw
                Eigen::internal::scalar_normal_dist_op<double> randN; // Gaussian functor
                Eigen::internal::scalar_normal_dist_op<double>::rng.seed(1); // Seed the rng

                // Define mean and covariance of the distribution
                Eigen::VectorXd meanE(size);
                Eigen::MatrixXd covar(size,size);
                covmat.at<double>(0,0);
                meanE  <<  mean.x,  mean.y;
                covar <<  covmat.at<double>(0,0), covmat.at<double>(0,1),
                        covmat.at<double>(1,0),  covmat.at<double>(1,1);

                Eigen::MatrixXd normTransform(size,size);
                Eigen::LLT<Eigen::MatrixXd> cholSolver(covar);

                // We can only use the cholesky decomposition if
                // the covariance matrix is symmetric, pos-definite.
                // But a covariance matrix might be pos-semi-definite.
                // In that case, we'll go to an EigenSolver
                if (cholSolver.info()==Eigen::Success) {
                    // Use cholesky solver
                    normTransform = cholSolver.matrixL();
                } else {
                    // Use eigen solver
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
                    Eigen::MatrixXd eigenvaluesEigen;
                    cv::cv2eigen(eigenvalues, eigenvaluesEigen);
                    normTransform = eigenSolver.eigenvectors() * eigenvaluesEigen.cwiseAbs().cwiseSqrt().asDiagonal();
                    if(!((normTransform.array() == normTransform.array())).all()) {
                        std::cout << "nan" <<std::endl;
                    }
                }

                Eigen::MatrixXd samples = (normTransform * Eigen::MatrixXd::NullaryExpr(size,nn,randN)).colwise() + meanE;
                auto TransparentCircleFloat = [](cv::Point center, cv::Vec3f color, int radius, double alpha, cv::Mat &ref_image) {
                    for (int i=-radius; i<radius; i++) {
                        for (int j=-radius; j<radius; j++) {
                            int coord_y = center.y + i;
                            int coord_x = center.x + j;
                            if (coord_x>0 && coord_y>0 && coord_x<ref_image.cols && coord_y < ref_image.rows) {
                                ref_image.at<cv::Vec3f>(cv::Point(coord_x,coord_y)) = (1.0-alpha)*ref_image.at<cv::Vec3f>(cv::Point(coord_x,coord_y)) + alpha*color;

                            }
                        }
                    }
                };

                const double alpha = 0.5; //03;
                if(covar(0,0)!=0) {
                    for(int i=0; i<nn;i++) {
                        int x = static_cast<int>(samples(1,i));
                        int y = static_cast<int>(samples(0,i));
                        if(x>0 && y>0 && x<ref_image.rows && y<ref_image.cols) {
                            // ref_image.at<cv::Vec3f>(x, y) -= color; //cv::Vec3f(0.1, 0, 0.1);
                            //ref_image.at<cv::Vec3b>(x, y) = alpha*color + (1.0-alpha)*ref_image.at<cv::Vec3b>(x, y);
                            TransparentCircleFloat(cv::Point(y,x), color, 2.0, alpha, ref_image);
                        }
                    }
                }
            }
        }
    }
}
