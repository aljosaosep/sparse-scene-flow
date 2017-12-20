# Sparse Scene Flow

This repository contains code for sparse scene flow method, used in P. Lenz etal.: Sparse Scene Flow Segmentation for Moving Object Detection in Urban Environments, Intelligent Vehicles Symposium (IV), 2011
 
## Demo  Video
TODO

## Prerequisite

In order to run the code, your setup has to meet the following minimum requirements (tested versions in parentheses. Other versions might work, too):

* GCC 4.8.4
  * Eigen (3.x)
  * Boost (1.55 or later)
  * OpenCV (3.2.0 + OpenCV contrib)

## Install

### Compiling the code using CMake
0.  `mkdir build`
0.  `cmake ..`
0.  `make all`

### Running the tracker
0.  Edit the config `%PROJ_DIR%/data/kitti_sample.cfg`, set all the paths.
0.  Run the tracker eg. `CIWTApp --config %PROJ_DIR%/data/kitti_sample.cfg --start_frame 0 --end_frame 15 --show_visualization_2d --show_visualization_3d`
0.  Find a small sample of KITTI tracking dataset in `%PROJ_DIR%/data/kitti_sample` (left/right camera images, Regionlets detections, calibration files).

## Remarks


* External libraries
    * The tracker ships the following external modules:
        * **libviso2** - egomotion estimation (http://www.cvlibs.net/software/libviso/)

* For optimal performance, run the sf-estimator in `release` mode.

If you have any issues or questions about the code, please contact me https://www.vision.rwth-aachen.de/person/13/

## Citing

If you find the tracker useful in your research, please consider citing:

	@inproceedings{Lenz2011IV,
	  author = {Philip Lenz and Julius Ziegler and Andreas Geiger and Martin Roser},
	  title = {Sparse Scene Flow Segmentation for Moving Object Detection in Urban Environments},
	  booktitle = {Intelligent Vehicles Symposium (IV)},
	  year = {2011}
	}

## License

GNU General Public License (http://www.gnu.org/licenses/gpl.html)

Copyright (c) 2017 Aljosa Osep
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
