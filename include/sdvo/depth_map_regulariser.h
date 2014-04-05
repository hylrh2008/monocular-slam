#ifndef DEPTH_MAP_REGULARISER_H
#define DEPTH_MAP_REGULARISER_H
#include <opencv2/opencv.hpp>
#include <sdvo/depth_hypothesis.h>

namespace sdvo{

class depth_map_regulariser
{
public:
  depth_map_regulariser(depth_hypothesis * _H);
private:
  depth_hypothesis & H;

  cv::Mat1f tmp;
  cv::Mat1f tmp_var;
  cv::Mat1b tmp_age;

  void smooth_map();
  void fill_holes();
};
} // End SDVO
#endif // DEPTH_MAP_REGULARISER_H
