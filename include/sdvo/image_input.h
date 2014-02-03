#ifndef _SDVO_IMAGE_INPUT_H_
#define _SDVO_IMAGE_INPUT_H_

#include <opencv2/core/core.hpp>


namespace sdvo
{

class image_input
{
public:

  virtual cv::Mat get_next_image(void) = 0;
};

} // sdvo

#endif // _SDVO_IMAGE_INPUT_H_
