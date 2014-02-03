#ifndef _SDVO_FILE_STREAM_INPUT_IMAGE_H_
#define _SDVO_FILE_STREAM_INPUT_IMAGE_H_

#include <string>
#include <set>

#include <sdvo/image_input.h>


namespace sdvo
{

class file_stream_input_image
{
public:

  file_stream_input_image(
      std::string const& directory_name,
      std::string const& extension);

  cv::Mat get_next_image(void);

  void print_remaining_files(std::ostream& stream) const;
  std::string get_current_file_name(void) const;

private:

  std::set<std::string> _files;
  std::set<std::string>::const_iterator _current_file;
  std::set<std::string>::const_iterator _last_file;
};

} // sdvo

#endif // _SDVO_FILE_STREAM_INPUT_IMAGE_H_
