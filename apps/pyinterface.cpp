/*
Copyright 2017. All rights reserved.
Computer Vision Group, Visual Computing Institute
Technical University Munich, Germany

This file is part of the rwth_mot framework.
Authors: Aljosa Osep (aljosa.osep -at- tum.de)

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

#include <iostream>

// libviso
#include <libviso2/viso_stereo.h>

// Eigen
#include <Eigen/Core>

// pybind
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <src/sparseflow/utils_flow.h>

int add(int i, int j) {
    return i + j;
}

// Note: images get allocated, addresses returned via references
// User needs to clear mem adter this
void pybind_to_raw(const pybind11::array_t<float> &left1, uint8_t*& left_img_1, int &rows, int &cols) {

    auto left1_buff = left1.unchecked<2>(); // x must have ndim = 3; can be non-writeable
    //auto left2_buff = left2.unchecked<2>(); // x must have ndim = 3; can be non-writeable
    //auto right1_buff = right1.unchecked<2>(); // x must have ndim = 3; can be non-writeable
    //auto right2_buff = right2.unchecked<2>(); // x must have ndim = 3; can be non-writeable

    rows = left1_buff.shape(0);
    cols = left1_buff.shape(1);

    // TODO: check that dims match
    std::cout << "Image dims: " << rows << ", " << cols << std::endl;


    // Alloc buffers for 4 images
    left_img_1  = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
    //right_img_1 = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
    //left_img_2  = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
    //right_img_2 = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));

    int32_t k=0;
    for (ssize_t i = 0; i<rows; i++) {
        for (ssize_t j = 0; j<cols; j++) {

            left_img_1[k]  = (uint8_t)(left1_buff(i, j)*255);
            //right_img_1[k] = (uint8_t)(right1_buff(i, j)*255);
            //left_img_2[k]  = (uint8_t)(left2_buff(i, j)*255);
            //right_img_2[k] = (uint8_t)(right2_buff(i, j)*255);

            k++;
        }
    }
}

pybind11::array_t<double> compute_vo(const pybind11::array_t<float> &left1, const pybind11::array_t<float> &right1,
                const pybind11::array_t<float> &left2, const pybind11::array_t<float> &right2,
                float focal_len, float cu, float cv, float baseline) {

    // Init VO
    // --------------------------------------------------------
    libviso2::VisualOdometryStereo::parameters param;
    param.calib.f = focal_len;
    param.calib.cu = cu;
    param.calib.cv = cv;
    param.base = baseline;
    libviso2::VisualOdometryStereo vo(param);
    //vo.reset(new libviso2::VisualOdometryStereo(param));
    // --------------------------------------------------------

    printf("Cam params: %f, %f, %f, %f\r\n", focal_len, cu, cv, baseline);


    uint8_t* left_img_1, *right_img_1, *left_img_2, *right_img_2;
    int rows, cols;
    pybind_to_raw(left1,left_img_1,rows, cols);
    pybind_to_raw(left2,left_img_2,rows, cols);
    pybind_to_raw(right1,right_img_1,rows, cols);
    pybind_to_raw(right2,right_img_2,rows, cols);


//    // ======= SHOULD BE A FNC =========================================
//    auto left1_buff = left1.unchecked<2>(); // x must have ndim = 3; can be non-writeable
//    auto left2_buff = left2.unchecked<2>(); // x must have ndim = 3; can be non-writeable
//    auto right1_buff = right1.unchecked<2>(); // x must have ndim = 3; can be non-writeable
//    auto right2_buff = right2.unchecked<2>(); // x must have ndim = 3; can be non-writeable
//
//    int rows = left1_buff.shape(0);
//    int cols = left1_buff.shape(1);
//
//    // TODO: check that dims match
//    std::cout << "Image dims: " << rows << ", " << cols << std::endl;
//
//    // Alloc buffers for 4 images
//    uint8_t* left_img_1  = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
//    uint8_t* right_img_1 = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
//    uint8_t* left_img_2  = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
//    uint8_t* right_img_2 = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
//
//    int32_t k=0;
//    for (ssize_t i = 0; i<rows; i++) {
//        for (ssize_t j = 0; j<cols; j++) {
//
//            left_img_1[k]  = (uint8_t)(left1_buff(i, j)*255);
//            right_img_1[k] = (uint8_t)(right1_buff(i, j)*255);
//            left_img_2[k]  = (uint8_t)(left2_buff(i, j)*255);
//            right_img_2[k] = (uint8_t)(right2_buff(i, j)*255);
//
//            k++;
//        }
//    }
//    // ======= SHOULD BE A FNC =========================================

    int32_t dims[] = {cols, rows, cols};

    // t=1
    libviso2::Matrix frame_to_frame_motion;
    vo.process(left_img_1, right_img_1, dims);
    //    frame_to_frame_motion = vo.getMotion(); //libviso2::Matrix::inv(viso->getMotion());
    // t=2
    bool f2_flag = vo.process(left_img_2, right_img_2, dims);

   // std::cout << "Flags: " << f1_flag << ", " << f2_flag << std::endl;

    if (f2_flag) {
        frame_to_frame_motion = vo.getMotion(); //libviso2::Matrix::inv(viso->getMotion());
    } else {
        std::cout << "VO error! Abort! TODO: handle this!" << std::endl;
        // TODO: handle the exception
    }


    // Convert transformation -> numpy array
    pybind11::array_t<double> result = pybind11::array_t<double>(4*4);
    auto buf3 = result.request();
    double *ptr3 = (double *)buf3.ptr;
    for (int i=0; i<4; i++) {
        for (int j=0; j<4; j++) {
            ptr3[i*4 + j] = frame_to_frame_motion.val[i][j];
        }
    }

    result.resize({4, 4});

    // Free-up the memory
    free(left_img_1);
    free(right_img_1);
    free(left_img_2);
    free(right_img_2);

    return result;
}

std::vector<libviso2::Matcher::p_match>
GetMatches(libviso2::Matcher *M, uint8_t *left_img_data, uint8_t *right_img_data, int rows, int cols,  bool only_push) {
    int32_t dims[] = {cols, rows, cols};

    // Push images
    M->pushBack(left_img_data, right_img_data, dims, false);

    std::vector<libviso2::Matcher::p_match> matches;
    if (!only_push) {
        // do matching
        M->matchFeatures(2); // 2 ... quad matching

        // Get matches
        // quad matching
        matches = M->getMatches();
    }

    return matches;
}

void compute_flow(const pybind11::array_t<float> &left1, const pybind11::array_t<float> &right1,
                  const pybind11::array_t<float> &left2, const pybind11::array_t<float> &right2,
                  float focal_len, float cu, float cv, float baseline, float dt, float velocity_thresh) {


    libviso2::Matrix pose = libviso2::Matrix::eye(4);
    Eigen::Matrix<double,4,4> currPose = Eigen::MatrixXd::Identity(4,4); // Current pose

    // Matcher, needed for scene-flow
    auto *matcher = SUN::utils::scene_flow::InitMatcher();

    uint8_t* left_img_1, *right_img_1, *left_img_2, *right_img_2;
    int rows, cols;
    pybind_to_raw(left1,left_img_1,rows, cols);
    pybind_to_raw(left2,left_img_2,rows, cols);
    pybind_to_raw(right1,right_img_1,rows, cols);
    pybind_to_raw(right2,right_img_2,rows, cols);

    auto m1 = GetMatches(matcher, left_img_1, left_img_2, rows, cols,  true);

    auto flow_result = SUN::utils::scene_flow::GetSceneFlow(matches, ego_estimate, left_camera,
                                                                        baseline,
                                                                        variables_map.at("dt").as<double>(),
                                                                        variables_map.at("max_velocity_threshold").as<double>());
    sparse_flow_info = std::get<1>(flow_result);
    sparse_flow_map = std::get<0>(flow_result);

    // TODO
    // Free memory: matcher, images

////        // -------------------------------------------------------------------------------
////        // +++ Run visual odometry module => estimate egomotion +++
////        // -------------------------------------------------------------------------------
////        Eigen::Matrix4d ego_estimate = Eigen::Matrix4d::Identity();
////        if (dataset_assistant.right_image_.data != nullptr) {
////            if (SparseFlowApp::debug_level > 0) printf("[Processing VO ...] \r\n");
////
////            InitVO(vo_module, left_camera.f_u(), left_camera.c_u(), left_camera.c_v(), dataset_assistant.stereo_baseline_);
////            ego_estimate = SUN::utils::scene_flow::EstimateEgomotion(*vo_module, dataset_assistant.left_image_, dataset_assistant.right_image_);
////
////            // Accumulated transformation
////            SparseFlowApp::egomotion = SparseFlowApp::egomotion * ego_estimate.inverse();
////
////            // Update left_camera, right_camera using estimated pose transform
////            left_camera.ApplyPoseTransform(SparseFlowApp::egomotion);
////            right_camera.ApplyPoseTransform(SparseFlowApp::egomotion);
////        } else {
////            printf("You want to compute visual odom., but got no right image. Not possible.\r\n");
////            exit(EXIT_FAILURE);
////        }
//
//
//        // -------------------------------------------------------------------------------
//        // +++ Compute Sparse Scene Flow +++
//        // -------------------------------------------------------------------------------
//        std::vector<SUN::utils::scene_flow::VelocityInfo> sparse_flow_info;
//        cv::Mat sparse_flow_map;
//        // bool first_frame = current_frame <= SparseFlowApp::start_frame;
//        // bool use_sparse_flow = true;
//
//
//        auto matches = SUN::utils::scene_flow::GetMatches(matcher, left_image, dataset_assistant.right_image_,
//                                                              left_camera, baseline,
//                                                              first_frame);
////            if (!first_frame) {
////                auto flow_result = SUN::utils::scene_flow::GetSceneFlow(matches, ego_estimate, left_camera,
////                                                                        dataset_assistant.stereo_baseline_,
////                                                                        variables_map.at("dt").as<double>(),
////                                                                        variables_map.at("max_velocity_threshold").as<double>());
////                sparse_flow_info = std::get<1>(flow_result);
////                sparse_flow_map = std::get<0>(flow_result);
////            }
}

//std::vector<seglib::ObjectProposal> compute_proposals(const pybind11::array_t<float> &x, int min_pts) {
//    auto r = x.unchecked<2>(); // x must have ndim = 3; can be non-writeable
//    const auto num_pts = r.shape(1);
//    std::cout << "Got " << num_pts << " points! Dim: " << r.ndim() << std::endl;
//    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcl(new pcl::PointCloud<pcl::PointXYZRGBA>); // dummy
//
//    for (int i=0; i<num_pts; i++) {
//        //std::cout << "x: " << r(0, i)  << ", y: " << r(1, i) << ", z: " << r(2, i) << std::endl;
//
//        pcl::PointXYZRGBA pt;
//        pt.x = static_cast<float>(r(0, i));
//        pt.y = static_cast<float>(r(1, i));
//        pt.z = static_cast<float>(r(2, i));
//        pcl->push_back(pt);
//    }
//
//    // Oversegment point cloud
//    Eigen::MatrixXd density_map;
//    Eigen::MatrixXd density_map_smooth;
//    auto props = SegmentCloud(pcl, density_map, density_map_smooth, //2.0, 2.0);
//
//            /*sigma_x*/ 2.0,
//            /*sigma_z*/ 2.0,
//            /*min_distance_ground*/ 0.3,
//            /*max_distance_ground*/ 2.2,
//            /*area_length*/ 100,
//            /*area_depth*/ 100,
//            /*disc*/ 10.0f,
//            /*min_cluster_size*/ /*5*/ min_pts,
//            /*do_qshift*/ true);
//
//    std::cout << "Proposals computed: " << props.size() << std::endl;
//    return props;
//}


namespace py = pybind11;

PYBIND11_MODULE(pyinterface, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

    m.def("compute_vo", &compute_vo, pybind11::return_value_policy::copy, R"pbdoc(
        Compute frame-to-frame egomotion
        Some other explanation goes here.
    )pbdoc");


    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers
        Some other explanation about the subtract function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}