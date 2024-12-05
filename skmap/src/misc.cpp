#ifndef MISC_CPP
#define MISC_CPP

#define _USE_MATH_DEFINES
#include <iostream>
#define EIGEN_VECTORIZE_AVX512
#include <Eigen/Dense>
#include <gdal/gdal.h>
#include <gdal/gdalwarper.h>
#include <gdal/gdal_priv.h>
#include <omp.h>
#include <functional>
#include <cassert>
#include <unordered_map>
#include <cmath>
#include <optional>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <opencv2/opencv.hpp>

//#include <cpl_conv.h>
//#include <ogr_spatialref.h>

namespace skmap {

template<typename T>
inline void skmapPrint(T message)
{
    std::cout << message << std::endl;
}

inline void skmapAssertIfTrue(bool cond, std::string message)
{
    if (cond)
    {
        std::cerr << message << std::endl;
        std::cout << message << std::endl;
        assert(false);
    }
}

inline void runBashCommand(std::string command)
{
    command += " > /dev/null 2>&1";
    int result = system(command.c_str());
    skmapAssertIfTrue(result != 0, "scikit-map ERROR 10: issues running the command " + command);
}

inline float_t signFunc(float_t x)
{
    if (x > 0.)
        return +1.;
    else if (x == 0.)
        return 0.;
    else
        return -1.;
}

inline float_t standardNormalCdf(float_t x) {
    return 0.5 * (1. + erf(x / sqrt(2.)));
}

using uint_t = long unsigned int;
using float_t = float;
using byte_t = unsigned char;
using int16_t = short;
using uint16_t = std::uint16_t;
inline float_t nan_v = std::numeric_limits<float_t>::quiet_NaN();
inline float_t inf_v = std::numeric_limits<float_t>::infinity();
using dict_t = std::unordered_map<std::string, std::string>;
using map_t = std::map<std::string, std::vector<uint_t>>;
// RowMajor -> C order ,  ColMajor -> F order 
// C order is default in Numpy and Eigen pybind11 require it to get this input
using MatFloat = Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatBool = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatByte = Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VecFloat = Eigen::Vector<float_t, Eigen::Dynamic>;
using VecUint = Eigen::Vector<uint_t, Eigen::Dynamic>;
using VecBool = Eigen::Vector<bool, Eigen::Dynamic>;


}


#endif
