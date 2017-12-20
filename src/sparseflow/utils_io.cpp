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

#include "utils_io.h"

// Boost
#include <boost/filesystem.hpp>

namespace SUN {
    namespace utils {
        namespace IO {

            bool ReadEigenMatrixFromTXT(const char *filename, Eigen::MatrixXd &mat_out) {

                // General structure
                // 1. Read file contents into vector<double> and count number of lines
                // 2. Initialize matrix
                // 3. Put data in vector<double> into matrix

                std::ifstream input(filename);
                if (input.fail()) {
                    std::cerr << "ReadEigenMatrixFromTXT::Error: Can't Open file:'" << filename << "'." << std::endl;
                    mat_out = Eigen::MatrixXd(0,0);
                    return false;
                }
                std::string line;
                double d;

                std::vector<double> v;
                int n_rows = 0;
                while (getline(input, line)) {
                    ++n_rows;
                    std::stringstream input_line(line);
                    while (!input_line.eof()) {
                        input_line >> d;
                        v.push_back(d);
                    }
                }
                input.close();

                int n_cols = v.size()/n_rows;
                mat_out = Eigen::MatrixXd(n_rows,n_cols);

                for (int i=0; i<n_rows; i++)
                    for (int j=0; j<n_cols; j++)
                        mat_out(i,j) = v[i*n_cols + j];

                return true;
            }

            bool MakeDir(const char *path) {
                if (!path) {
                    return false;
                }

                boost::filesystem::path fpath(path);
                if (!boost::filesystem::exists(fpath)) {
                    boost::filesystem::path dir(fpath);
                    try {
                        boost::filesystem::create_directories(dir);
                    }
                    catch (boost::filesystem::filesystem_error e) {
                        std::cout << "Error: " << std::endl << e.what() << std::endl;
                        return false;
                    }
                }
                return true;
            }
        }
    }
}



