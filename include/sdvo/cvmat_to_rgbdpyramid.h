#ifndef CVMAT_TO_RGBDPYRAMID_H
#define CVMAT_TO_RGBDPYRAMID_H
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dvo/core/rgbd_image.h>
namespace sdvo{
class cvmat_to_rhbdpyramid
{
public:
    dvo::core::RgbdImagePyramid operator ()(const cv::Mat & rgb,const cv::Mat & depth)
    {
        cv::Mat intensity;
        cv::Mat depth_float;
        cv::cvtColor(rgb,intensity,cv::COLOR_RGB2GRAY);
        depth.convertTo(depth_float,CV_32FC1);
        intensity.convertTo(intensity,CV_32FC1);
        return dvo::core::RgbdImagePyramid(intensity,depth_float);
    }
};
}
#endif // CVMAT_TO_RGBDPYRAMID_H
