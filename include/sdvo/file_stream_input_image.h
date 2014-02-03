#ifndef _SDVO_FILE_STREAM_INPUT_IMAGE_H_
#define _SDVO_FILE_STREAM_INPUT_IMAGE_H_

#include <string>
#include <boost/filesystem.hpp>

#include <sdvo/image_input.h>


namespace sdvo
{

class file_stream_input_image
{
public:

  file_stream_input_image(std::string const& directory_name);

  cv::Mat get_next_image(void);

private:

  boost::filesystem::directory_iterator _files_iterator;
};

} // sdvo

#endif // _SDVO_FILE_STREAM_INPUT_IMAGE_H_
