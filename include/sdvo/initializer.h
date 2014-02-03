#ifndef _SDVO_INITIALIZER_H_
#define _SDVO_INITIALIZER_H_

#include <utility>

#include <opencv2/core/core.hpp>


namespace sdvo
{

class initializer
{
public:

  virtual std::pair<cv::Mat, cv::Mat> get_init(void) = 0;
};

} // sdvo

#endif // _SDVO_INITIALIZER_H_
