#ifndef _SDVO_IMAGE_DEPTH_INITIALIZER_H_
#define _SDVO_IMAGE_DEPTH_INITIALIZER_H_

#include <sdvo/initializer.h>


namespace sdvo
{

class image_depth_initializer
    : public initializer
{
public:

  image_depth_initializer(
      std::string const& rbg_filename,
      std::string const& depth_filename,
      float scale_factor = 0.0002);

  std::pair<cv::Mat, cv::Mat> get_init(void);

private:

  std::string _rgb;
  std::string _depth;
  float _scale_factor;
};

} // sdvo

#endif // _SDVO_IMAGE_DEPTH_INITIALIZER_H_
