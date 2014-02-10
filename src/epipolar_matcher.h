#ifndef EPIPOLAR_MATCHER_H
#define EPIPOLAR_MATCHER_H
#include <boost/circular_buffer.hpp>
#include <dvo/core/rgbd_image.h>

class epipolar_matcher
{
public:
  epipolar_matcher();
  bool push_new_data_in_buffer();
  bool compute_new_observation();
private:
  boost::circular_buffer<std::pair<dvo::core::RgbdImagePyramid>,Eigen::Affine> last_images_buffer;

};

#endif // EPIPOLAR_MATCHER_H
