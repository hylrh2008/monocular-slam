#include <sdvo/file_stream_input_image.h>

namespace sdvo
{


file_stream_input_image::file_stream_input_image(
    const std::string& directory_name)
  : _files_iterator(directory_name) {}


cv::Mat file_stream_input_image::get_next_image()
{
  return cv::Mat();
}


} // sdvo
