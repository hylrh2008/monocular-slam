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
        depth.convertTo(depth_float,CV_32FC1,1/5000.);
        intensity.convertTo(intensity,CV_32FC1);
        for (cv::MatIterator_<float> it = depth_float.begin<float>();  it != depth_float.end<float>(); ++it) {
            if(*it > 3 || *it == 0)
              *it = std::numeric_limits<float>::quiet_NaN();
        }

        dvo::core::RgbdImagePyramid p(intensity,depth_float);
        p.level(0).calculateIntensityDerivatives();
        return p;
    }
};
}
#endif // CVMAT_TO_RGBDPYRAMID_H
