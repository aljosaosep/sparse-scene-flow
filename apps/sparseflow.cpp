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

// std
#include <iostream>
#include <memory>
#include <cassert>

// opencv
#include <opencv2/highgui/highgui.hpp>

// libviso
#include <libviso2/viso_stereo.h>

// utils
#include "utils_io.h"
#include "utils_visualization.h"
#include "datasets_dirty_utils.h"
#include "utils_flow.h"

// For convenience.
namespace po = boost::program_options;

namespace SparseFlowApp {

    // Application data.
    int debug_level;
    int start_frame;
    int end_frame;
    std::string output_dir;
    std::string subsequence_name;
    std::string dataset;
    Eigen::Matrix4d egomotion = Eigen::Matrix4d::Identity();

    // -------------------------------------------------------------------------------
    // +++ Command Args Parser +++
    // -------------------------------------------------------------------------------
    bool ParseCommandArguments(const int argc, const char **argv, po::variables_map &config_variables_map) {
        po::options_description cmdline_options;
        try {
            std::string config_file;

            // Declare a group of options that will be allowed only on command line
            po::options_description generic_options("Command line options:");
            generic_options.add_options()
                    ("help", "Produce help message")
                    ("config", po::value<std::string>(&config_file), "Config file path.")
                    ("debug_level", po::value<int>(&debug_level)->default_value(0), "Debug level")
                    ;

            // Declare a group of options that will be  allowed both on command line and in config file
            po::options_description config_options("Config options");
            config_options.add_options()
                    // Input
                    ("left_image_path", po::value<std::string>(), "Image (left) path")
                    ("right_image_path", po::value<std::string>(), "Image (right) path")
                    ("calib_path", po::value<std::string>(), "Camera calibration path (currently supported: kitti)")
                    ("output_dir", po::value<std::string>(&output_dir), "Output path")
                    ("start_frame", po::value<int>(&start_frame)->default_value(0), "Starting frame")
                    ("end_frame", po::value<int>(&end_frame)->default_value(10000), "Last frame")

                    ("max_velocity_threshold", po::value<double>()->default_value(40.0), "Max. velocity threshold "
                            "(exceeding this thresh. measurements are considered outliers)")

                    ("dt", po::value<double>()->default_value(0.1), "Time difference between two frames, captured by the sensor (sec.)")
                    ("dataset", po::value<std::string>(&dataset)->default_value("kitti"), "Dataset (default: KITTI)")
                    ("subsequence", po::value<std::string>(&subsequence_name), "Sub-sequence name")
                    ;

            po::options_description parameter_options("Parameters:");
            cmdline_options.add(generic_options);
            cmdline_options.add(config_options);
            cmdline_options.add(parameter_options);

            store(po::command_line_parser(argc, argv).options(cmdline_options).run(), config_variables_map);
            notify(config_variables_map);

            if (config_variables_map.count("help")) {
                std::cout << cmdline_options << std::endl;
                return false;
            }

            // "generic" config
            if (config_variables_map.count("config")) {
                std::ifstream ifs(config_file.c_str());
                if (!ifs.is_open()) {
                    std::cout << "Can not Open config file: " << config_file << "\n";
                    return false;
                } else {
                    store(parse_config_file(ifs, cmdline_options), config_variables_map);
                    notify(config_variables_map);
                }
            }
        }
        catch (std::exception& e) {
            std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
            std::cerr << cmdline_options << std::endl;
            return false;
        }
        return true;
    }
}

/*
  -------------
  Debug Levels:
  -------------
  0 - Outputs basically nothing, except relevant error messages.
  1 - Console output, logging.
  2 - Quantitative evaluation.
  3 - Most relevant visual results (per-frame, eg. segmentation, tracking results, ...).
  4 - Point clouds (per-frame), less relevant visual results.
  5 - Additional possibly relevant frame output (segmentation 3D data, integrated models, ...).
  >=6 - All possible/necessary debug stuff. Should make everything really really really slow.
  */
int main(const int argc, const char** argv) {
    std::cout << "Hello from new and awesome generic tracker!" << std::endl;

    /// Parse command args
    po::variables_map variables_map;
    if (!SparseFlowApp::ParseCommandArguments(argc, argv, variables_map)) {
        printf("Error parsing command args/configs, exiting.\r\n");
        return -1;
    }

    const int num_frames = SparseFlowApp::end_frame-SparseFlowApp::start_frame;

    // Makes sure the relevant values are correct.
    assert(num_frames>0);
    assert(SparseFlowApp::debug_level>=0);

    std::cout << "Init dataset assistant ..." << std::endl;
    SUN::utils::dirty::DatasetAssitantDirty dataset_assistant(variables_map);

    // -------------------------------------------------------------------------------
    // +++ Output (sub) directories +++
    // -------------------------------------------------------------------------------
    bool make_dir_success;
    if (SparseFlowApp::debug_level > 0) printf("[Creating output dirs in:%s] \r\n", SparseFlowApp::output_dir.c_str());
    std::string output_dir_visual_results = SparseFlowApp::output_dir + "/visual_results";
    make_dir_success = SUN::utils::IO::MakeDir(output_dir_visual_results.c_str());
    assert(make_dir_success);

    // -------------------------------------------------------------------------------
    // +++ Init visual odometry module and matcher +++
    // -------------------------------------------------------------------------------
    std::shared_ptr<libviso2::VisualOdometryStereo> vo_module = nullptr;
    auto InitVO = [](std::shared_ptr<libviso2::VisualOdometryStereo> &vo, double f, double c_u, double c_v, double baseline) {
        if (vo==nullptr) {
            libviso2::VisualOdometryStereo::parameters param;
            param.calib.f = f;
            param.calib.cu = c_u;
            param.calib.cv = c_v;
            param.base = baseline;
            vo.reset(new libviso2::VisualOdometryStereo(param));
        }
    };

    libviso2::Matrix pose = libviso2::Matrix::eye(4);
    Eigen::Matrix<double,4,4> currPose = Eigen::MatrixXd::Identity(4,4); // Current pose

    // Matcher, needed for scene-flow
    auto *matcher = SUN::utils::scene_flow::InitMatcher();

    // -------------------------------------------------------------------------------
    // +++ Per-frame containers +++
    // -------------------------------------------------------------------------------
    cv::Mat left_image;

    // -------------------------------------------------------------------------------
    // +++ Viz. threads +++
    // -------------------------------------------------------------------------------
    cv::namedWindow("visualization_2d_window");
    cv::startWindowThread();

    cv::namedWindow("visualization_3d_window");
    cv::startWindowThread();

    // -------------------------------------------------------------------------------
    // +++ MAIN_LOOP +++
    // -------------------------------------------------------------------------------
    const clock_t tracklet_gen_begin_time = clock();
    for (int current_frame=SparseFlowApp::start_frame; current_frame<=SparseFlowApp::end_frame; current_frame++) {
        const clock_t current_frame_begin_time = clock();

        if (SparseFlowApp::debug_level > 0) {
            printf("---------------------------\r\n");
            printf("| PROC FRAME %03d/%03d    |\r\n", current_frame, SparseFlowApp::end_frame);
            printf("---------------------------\r\n");

        }

        // -------------------------------------------------------------------------------
        // +++ Load data +++
        // -------------------------------------------------------------------------------
        if (SparseFlowApp::debug_level > 0) printf("[Load data ...] \r\n");
        if (!dataset_assistant.LoadData(current_frame, SparseFlowApp::dataset)) {
                printf("Oh noes, dataset assistant can no load data! Abort! \r\n");
                return -1;
        }

        if (SparseFlowApp::debug_level > 0) printf("[Load data OK!] \r\n");
        left_image = dataset_assistant.left_image_.clone();
        auto &left_camera = dataset_assistant.left_camera_;
        auto &right_camera = dataset_assistant.right_camera_;

        // -------------------------------------------------------------------------------
        // +++ Run visual odometry module => estimate egomotion +++
        // -------------------------------------------------------------------------------

        Eigen::Matrix4d ego_estimate = Eigen::Matrix4d::Identity();
        if (dataset_assistant.right_image_.data != nullptr) {
            if (SparseFlowApp::debug_level > 0) printf("[Processing VO ...] \r\n");

            InitVO(vo_module, left_camera.f_u(), left_camera.c_u(), left_camera.c_v(), dataset_assistant.stereo_baseline_);
            ego_estimate = SUN::utils::scene_flow::EstimateEgomotion(*vo_module, dataset_assistant.left_image_, dataset_assistant.right_image_);

            // Accumulated transformation
            SparseFlowApp::egomotion = SparseFlowApp::egomotion * ego_estimate.inverse();

            // Update left_camera, right_camera using estimated pose transform
            left_camera.ApplyPoseTransform(SparseFlowApp::egomotion);
            right_camera.ApplyPoseTransform(SparseFlowApp::egomotion);
        } else {
            printf("You want to compute visual odom., but got no right image. Not possible.\r\n");
            exit(EXIT_FAILURE);
        }


        // -------------------------------------------------------------------------------
        // +++ Compute Sparse Scene Flow +++
        // -------------------------------------------------------------------------------
        std::vector<SUN::utils::scene_flow::VelocityInfo> sparse_flow_info;
        cv::Mat sparse_flow_map;
        bool first_frame = current_frame <= SparseFlowApp::start_frame;
        bool use_sparse_flow = true;


        if (dataset_assistant.right_image_.data != nullptr) {
            const clock_t sceneflow_start = clock();
            if (SparseFlowApp::debug_level > 0) printf("[Computing sparse scene flow ...] \r\n");
            auto matches = SUN::utils::scene_flow::GetMatches(matcher, left_image,
                                                              dataset_assistant.right_image_,
                                                              left_camera, dataset_assistant.stereo_baseline_, first_frame);
            if (!first_frame) {
                auto flow_result = SUN::utils::scene_flow::GetSceneFlow(matches, ego_estimate, left_camera,
                                                                        dataset_assistant.stereo_baseline_,
                                                                        variables_map.at("dt").as<double>(),
                                                                        variables_map.at("max_velocity_threshold").as<double>());
                sparse_flow_info = std::get<1>(flow_result);
                sparse_flow_map = std::get<0>(flow_result);
            }
            if (SparseFlowApp::debug_level > 0) printf("[ Processing sparse scene flow %.3f s ]\r\n", float((clock() - sceneflow_start)) / CLOCKS_PER_SEC);
        } else {
            printf("You want to compute sparse flow, but got no right image. Not possible.\r\n");
            exit(EXIT_FAILURE);
        }


        // -------------------------------------------------------------------------------
        // +++ Update stats and visualizations +++
        // -------------------------------------------------------------------------------
        // Viz. sparse flow 2D
        cv::Mat left_img_sparse_flow = left_image.clone();
        cv::Mat left_img_sparse_flow_3d;
        std::vector<Eigen::Vector3d> pts_p3d, pts_vel;
        for (const auto inf:sparse_flow_info) {
            pts_p3d.push_back(inf.p_3d);
            pts_vel.push_back(inf.p_vel);
            Eigen::Vector3i p_proj = left_camera.CameraToImage(Eigen::Vector4d(inf.p_3d[0], inf.p_3d[1], inf.p_3d[2], 1.0));
            Eigen::Vector3i p_prev = left_camera.CameraToImage(Eigen::Vector4d(inf.p_prev[0], inf.p_prev[1], inf.p_prev[2], 1.0));
            Eigen::Vector3i p_prev_to_curr = left_camera.CameraToImage(
                    Eigen::Vector4d(inf.p_prev_to_curr[0], inf.p_prev_to_curr[1], inf.p_prev_to_curr[2], 1.0));
            SUN::utils::visualization::DrawTransparentSquare(cv::Point(inf.p[0], inf.p[1]), cv::Vec3b(0, 0, 255), 3.0, 0.5, left_img_sparse_flow);
            SUN::utils::visualization::DrawLine(inf.p_3d, inf.p_3d + inf.p_vel, left_camera, left_img_sparse_flow, cv::Vec3b(255, 0, 0), 2);
        }

        SUN::utils::visualization::BirdEyeVizProperties viz_props;
        viz_props.birdeye_scale_factor_ = 20.0;
        viz_props.birdeye_left_plane_ = -15.0;
        viz_props.birdeye_right_plane_ = 15.0;
        viz_props.birdeye_far_plane_ = 40.0;
        SUN::utils::visualization::DrawSparseFlowBirdeye(pts_p3d, pts_vel, left_camera, viz_props, left_img_sparse_flow_3d);

        // Write viz. to file(s)
        char output_path_buff[500];
        snprintf(output_path_buff, 500, "%s/sceneflow_sparse_%s_%06d.png", output_dir_visual_results.c_str(), SparseFlowApp::subsequence_name.c_str(), current_frame);
        cv::imwrite(output_path_buff, left_img_sparse_flow);

        snprintf(output_path_buff, 500, "%s/sceneflow_sparse_3D_%s_%06d.png", output_dir_visual_results.c_str(), SparseFlowApp::subsequence_name.c_str(), current_frame);
        cv::imwrite(output_path_buff, left_img_sparse_flow_3d*255.0);

        // Update viz. threads
        cv::imshow("visualization_2d_window", left_img_sparse_flow);
        cv::imshow("visualization_3d_window", left_img_sparse_flow_3d*255.0);

        printf("***** Processing time current frame: %.3f s*****\r\n", float( clock () - current_frame_begin_time ) /  CLOCKS_PER_SEC);
    }

    // -------------------------------------------------------------------------------
    // +++ END OF MAIN_LOOP +++
    // -------------------------------------------------------------------------------

    // Release mem for matcher
    delete matcher;
    matcher = nullptr;

    std::cout << "Finished, yay!" << std::endl;    return 0;
}
