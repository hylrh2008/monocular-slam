#include <sdvo/image_depth_initializer.h>

#include <opencv2/highgui/highgui.hpp>


namespace sdvo
{


image_depth_initializer::image_depth_initializer(
    const std::string& rbg_filename,
    const std::string& depth_filename,
    float scale_factor)
  : _rgb(rbg_filename)
  , _depth(depth_filename)
  , _scale_factor(scale_factor) {}


std::pair<cv::Mat, cv::Mat> image_depth_initializer::get_init()
{
  std::pair<cv::Mat, cv::Mat> result;

  result.first = cv::imread(_rgb, cv::IMREAD_COLOR);
  cv::Mat tmp = cv::imread(_depth, cv::IMREAD_GRAYSCALE);
  tmp.convertTo(result.second, CV_32FC1, _scale_factor);
}


} // sdvo
