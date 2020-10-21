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
    rows = left1_buff.shape(0);
    cols = left1_buff.shape(1);

    // TODO: check that dims match

    // Alloc buffers for 4 images
    left_img_1  = (uint8_t*)malloc(cols*rows*sizeof(uint8_t));
    int32_t k=0;
    for (ssize_t i = 0; i<rows; i++) {
        for (ssize_t j = 0; j<cols; j++) {

            left_img_1[k]  = (uint8_t)(left1_buff(i, j)*255);
            k++;
        }
    }
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

Eigen::Vector4d ProjTo3D_(float u1, float u2, float v, float f, float cu, float cv, float baseline) {

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

std::vector<SUN::utils::scene_flow::VelocityInfo> GetSceneFlow(std::vector<libviso2::Matcher::p_match> quad_matches,
        const Eigen::Matrix4d Tr,
        float f, float cu, float cv, float baseline, float dt, float max_velocity_ms) {

    std::vector<SUN::utils::scene_flow::VelocityInfo> velocity_info;

    // Project matches to 3D
    for (const auto &match:quad_matches) {

        Eigen::Vector4d p3d_c = ProjTo3D_(match.u1c, match.u2c, match.v1c, f, cu, cv, baseline); // p3d curr frame
        Eigen::Vector4d p3d_p = ProjTo3D_(match.u1p, match.u2p, match.v1p, f, cu, cv, baseline); // p3d prev frame

        if (std::isnan(p3d_c[0]) || std::isnan(p3d_p[0])) {
            continue;
        }

        const Eigen::Vector4d p3d_c_orig = p3d_c;
        const Eigen::Vector4d p3d_p_orig = p3d_p;

        // Project prev to curr frame using ego estimate
        p3d_p = Tr * p3d_p;

        // Project to ground
        //p3d_c.head<3>() = camera.ground_model()->ProjectPointToGround(p3d_c.head<3>());
        //p3d_p.head<3>() = camera.ground_model()->ProjectPointToGround(p3d_p.head<3>());

        Eigen::Vector3d delta = (p3d_c - p3d_p).head<3>();
        SUN::utils::scene_flow::VelocityInfo vi;
        vi.p = Eigen::Vector2i(match.u1c, match.v1c);
        vi.p_3d = p3d_c_orig.head<3>();
        vi.p_vel = delta;
        vi.p_prev = p3d_p_orig.head<3>();
        vi.p_prev_to_curr = p3d_p.head<3>();
        velocity_info.push_back(vi);

//        float max_dist = 20.0; //90;
//        float max_lat = 15.0; //30;
//        if (std::fabs(p3d_c[0]) > max_lat || std::fabs(p3d_c[2]) > max_dist || std::fabs(p3d_p[0]) > max_lat || std::fabs(p3d_p[2]) > max_dist) {
//            continue;
//        }
//
//        Eigen::Vector3d delta = (p3d_c - p3d_p).head<3>();
//
//        if (delta.norm()*(1.0/dt) < max_velocity_ms) {
//            //velocity_map.at<cv::Vec3f>(match.v1c, match.u1c) = cv::Vec3f(delta[0], delta[1], delta[2]);
//            SUN::utils::scene_flow::VelocityInfo vi;
//            vi.p = Eigen::Vector2i(match.u1c, match.v1c);
//            vi.p_3d = p3d_c_orig.head<3>();
//            vi.p_vel = delta;
//            vi.p_prev = p3d_p_orig.head<3>();
//            vi.p_prev_to_curr = p3d_p.head<3>();
//            velocity_info.push_back(vi);
//        }
    }

    return velocity_info; //std::make_tuple(velocity_map, velocity_info);
}


class VOEstimator {
public:
    VOEstimator() {
        //
    }
    
    ~VOEstimator() {
        if (vo_ != nullptr) {
            delete vo_;
            vo_ = nullptr;
        }

        if (sf_matcher_ != nullptr) {
            delete  sf_matcher_;
            sf_matcher_ = nullptr;
        }

        std::cout << "Cleared mem, bye!" << std::endl;
    }

    void init(const pybind11::array_t<float> &left, const pybind11::array_t<float> &right, float focal_len, float cu, float cv, float baseline,
            bool compute_scene_flow = false) {
        std::cout << "Initing stuff" << std::endl;

        //libviso2::VisualOdometryStereo::parameters param;
        param_.calib.f = focal_len;
        param_.calib.cu = cu;
        param_.calib.cv = cv;
        param_.base = baseline;

        vo_ = new libviso2::VisualOdometryStereo(param_);

        // Only push the first image pair
        uint8_t *left_img_1, *right_img_1;
        int rows, cols;
        pybind_to_raw(left, left_img_1, rows, cols);
        pybind_to_raw(right, right_img_1, rows, cols);
        int32_t dims[] = {cols, rows, cols};
        libviso2::Matrix frame_to_frame_motion;
        vo_->process(left_img_1, right_img_1, dims);

        // Init scene-flow matcher
        if (compute_scene_flow == true ){
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

            // Make sf matcher open for busisness
            this->sf_matcher_ = new libviso2::Matcher(matcher_params);
            this->sf_matcher_->pushBack(left_img_1, right_img_1, dims, false);
        }

        // Free mem
        free(left_img_1);
        free(right_img_1);
    }

    pybind11::array_t<double> compute_pose(const pybind11::array_t<float> &left, const pybind11::array_t<float> &right) {
        std::cout << "Proc stuff" << std::endl;

        uint8_t *left_img_1, *right_img_1;
        int rows, cols;
        pybind_to_raw(left, left_img_1, rows, cols);
        pybind_to_raw(right, right_img_1, rows, cols);
        int32_t dims[] = {cols, rows, cols};
        libviso2::Matrix frame_to_frame_motion;
        bool flag = vo_->process(left_img_1, right_img_1, dims);


        if (flag) {
            frame_to_frame_motion = vo_->getMotion(); //libviso2::Matrix::inv(viso->getMotion());
        } else {
            std::cout << "VO error! Abort! TODO: handle this!" << std::endl;
            // TODO: handle the exception
        }

        // Free mem
        free(left_img_1);
        free(right_img_1);

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

        return result;
    }

    pybind11::array_t<double> compute_flow(const pybind11::array_t<float> &left, const pybind11::array_t<float> &right,
            float dt, float velocity_thresh) {


        libviso2::Matrix pose = libviso2::Matrix::eye(4);
        Eigen::Matrix<double,4,4> ego = Eigen::MatrixXd::Identity(4,4); // Current pose

        uint8_t *left_img_1, *right_img_1;
        int rows, cols;
        pybind_to_raw(left, left_img_1, rows, cols);
        pybind_to_raw(right, right_img_1, rows, cols);
        int32_t dims[] = {cols, rows, cols};

        // Push images
        sf_matcher_->pushBack(left_img_1, right_img_1, dims, false);

        std::vector<libviso2::Matcher::p_match> matches;

        // Get quad matches
        sf_matcher_->matchFeatures(2); // 2 ... quad matching
        auto quad_matches = sf_matcher_->getMatches();

        auto sparse_flow_info =  GetSceneFlow(quad_matches, ego, param_.calib.f, param_.calib.cu, param_.calib.cv,  param_.base,
                dt, velocity_thresh);

        // Free mem
        free(left_img_1);
        free(right_img_1);

        int row_len = 2*3; //6+2;
        pybind11::array_t<double> result = pybind11::array_t<double>(sparse_flow_info.size()*row_len);
        auto buf3 = result.request();
        double *ptr3 = (double *)buf3.ptr;
        for (int i=0; i<sparse_flow_info.size(); i++) {
            auto info = sparse_flow_info.at(i);
            Eigen::Vector3d p = info.p_3d;
            Eigen::Vector3d p_prev = info.p_prev;

            ptr3[i*row_len + 0] = p[0];
            ptr3[i*row_len + 1] = p[1];
            ptr3[i*row_len + 2] = p[2];

            ptr3[i*6 + 3] = p_prev[0];
            ptr3[i*6 + 4] = p_prev[1];
            ptr3[i*6 + 5] = p_prev[2];

//            ptr3[i*6 + 6] = info.p[0];
//            ptr3[i*6 + 7] = info.p[1];
//            ptr3[i*6 + 8] = p_prev[2];
//            ptr3[i*6 + 9] = p_prev[2];

        }
        result.resize({static_cast<int>(sparse_flow_info.size()), row_len});

        return result;
    }

    // Returns: Nx4 matrix, rows correspond to indices of matched points (curr, prev
    pybind11::array_t<int> get_matches_indices() {
//        int32_t i1p;     // feature index (for tracking)
//        int32_t i2p;     // feature index (for tracking)
//        int32_t i1c;     // feature index (for tracking)
//        int32_t i2c;     // feature index (for tracking)

        const auto &quad_matches = sf_matcher_->getMatches();
        const int num_matches = quad_matches.size();

        std::cout << "get_matches_indices num_matches: " << num_matches << std::endl;

        const int row_len = 4;
        pybind11::array_t<int> result = pybind11::array_t<int>(num_matches*row_len);
        auto buf = result.request();
        int *ptr = (int*)buf.ptr;

        for (int i=0; i<num_matches; i++) {
            const auto match = quad_matches.at(i);
            // Curr point
            ptr[i*row_len + 0] = match.i1c;
            ptr[i*row_len + 1] = match.i2c;

            // Prev point
            ptr[i*row_len + 2] = match.i1p;
            ptr[i*row_len + 3] = match.i2p;
        }

        result.resize({static_cast<int>(num_matches), row_len});
        return result;
    }

    // Returns: Nx8 matrix, rows correspond to coords of matched points (curr (u, v), prev (u, v))
    pybind11::array_t<float> get_matches_coords() {
//        float u1p, v1p; // u,v-coordinates in previous left  image
//        float u2p, v2p; // u,v-coordinates in previous right image
//        float u1c, v1c; // u,v-coordinates in current  left  image
//        float u2c, v2c; // u,v-coordinates in current  right image

        const auto &quad_matches = sf_matcher_->getMatches();
        const int num_matches = quad_matches.size();
        const int row_len = 8;
        pybind11::array_t<float> result = pybind11::array_t<float>(num_matches*row_len);
        auto buf = result.request();
        float *ptr = (float*)buf.ptr;

        for (int i=0; i<num_matches; i++) {
            const auto match = quad_matches.at(i);
            // Curr point
            ptr[i*row_len + 0] = match.u1c;
            ptr[i*row_len + 1] = match.v1c;
            ptr[i*row_len + 2] = match.u2c;
            ptr[i*row_len + 3] = match.v2c;

            // Prev point
            ptr[i*row_len + 4] = match.u1p;
            ptr[i*row_len + 5] = match.v1p;
            ptr[i*row_len + 6] = match.u2p;
            ptr[i*row_len + 7] = match.v2p;
        }

        result.resize({static_cast<int>(num_matches), row_len});
        return result;
    }

private:
    libviso2::VisualOdometryStereo *vo_ = nullptr;
    libviso2::Matcher *sf_matcher_ = nullptr;
    libviso2::VisualOdometryStereo::parameters param_;
};

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

    py::class_<VOEstimator>(m, "VOEstimator")
            .def(py::init<>()).def("init", &VOEstimator::init).def("compute_pose", &VOEstimator::compute_pose).def("compute_flow", &VOEstimator::compute_flow).def("get_matches_indices", &VOEstimator::get_matches_indices).def("get_matches_coords", &VOEstimator::get_matches_coords);

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