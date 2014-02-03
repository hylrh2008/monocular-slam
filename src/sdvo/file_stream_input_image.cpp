#include <sdvo/file_stream_input_image.h>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>


namespace sdvo
{


file_stream_input_image::file_stream_input_image(
    const std::string& directory_name,
    std::string const& extension,
    int cv_load_code)
  : _cv_load_code(cv_load_code)
{
  boost::filesystem::directory_iterator it(directory_name), itEnd;

  for (; it != itEnd; ++it)
  {
    boost::filesystem::directory_entry entry(*it);
    if (entry.path().extension() == extension)
      _files.insert(entry.path().c_str());
#ifndef NDEBUG
    else
      std::cerr << "warning: unused file in directory: " << entry.path() << std::endl;
#endif // NDEBUG
  }

  _current_file = _files.begin();
  _last_file = _files.end();
}


cv::Mat
file_stream_input_image::get_next_image(void)
{
  if (_current_file != _last_file)
  {
    cv::Mat result = cv::imread(*_current_file, _cv_load_code);
    ++_current_file;
    return result;
  }

  return cv::Mat();
}


void file_stream_input_image::print_remaining_files(std::ostream& stream) const
{
  std::set<std::string>::const_iterator
      it(_current_file);

  for (; it != _last_file; ++it)
    stream << (*it) << std::endl;
}


std::string file_stream_input_image::get_current_file_name() const
{ return (_current_file != _last_file) ? *_current_file : std::string(); }


} // sdvo
