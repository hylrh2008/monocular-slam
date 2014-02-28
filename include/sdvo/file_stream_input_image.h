#ifndef _SDVO_FILE_STREAM_INPUT_IMAGE_H_
#define _SDVO_FILE_STREAM_INPUT_IMAGE_H_

#include <string>
#include <set>
#include <map>
#include <sdvo/image_input.h>


namespace sdvo
{

class file_stream_input_image
    : public image_input
{
public:

  file_stream_input_image(
      std::string const& directory_name,
      std::string const& image_base_name,
      std::string const& extension,
      int cv_load_code);

  cv::Mat get_next_image(void);

  void print_remaining_files(std::ostream& stream) const;
  std::string get_current_file_name(void) const;
  double get_current_time_stamp(void) const
  { return _current_time_stamp; }

private:

  typedef std::map<double, std::string> file_container;
  file_container _files;
  file_container::const_iterator _current_file;
  file_container::const_iterator _last_file;
  double _current_time_stamp;
  int _cv_load_code;
};

} // sdvo

#endif // _SDVO_FILE_STREAM_INPUT_IMAGE_H_
