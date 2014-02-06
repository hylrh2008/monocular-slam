#include <sdvo/file_stream_input_image.h>

#include <boost/regex.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/highgui/highgui.hpp>


namespace sdvo
{


file_stream_input_image::file_stream_input_image(
    const std::string& directory_name,
    const std::string& image_base_name,
    std::string const& extension,
    int cv_load_code)
  : _cv_load_code(cv_load_code)
{
  boost::filesystem::directory_iterator it(directory_name), itEnd;

  std::string rgx = image_base_name + "([0-9.]*)\\." + extension;
  boost::regex file_match(rgx);

  for (; it != itEnd; ++it)
  {
    boost::filesystem::directory_entry entry(*it);
    std::string current_filename(entry.path().filename().c_str());
    boost::smatch sm;
    boost::regex_match(current_filename, sm, file_match);
    if (sm.size() > 0)
    {
      std::string result;
      boost::regex_replace(
            std::back_inserter(result),
            current_filename.begin(),
            current_filename.end(),
            file_match,
            std::string("$1"));

      _files.insert(
            file_container::value_type(
              std::atof(result.c_str()), directory_name + '/' + current_filename));
    }
#ifndef NDEBUG
    else
      std::cerr << "warning: unused file in directory: " << entry.path() << std::endl;
#endif // NDEBUG
  }

  _current_file = _files.begin();
  _current_time_stamp = _current_file->first;
  _last_file = _files.end();
}


cv::Mat
file_stream_input_image::get_next_image(void)
{
  if (_current_file != _last_file)
  {
    cv::Mat result = cv::imread(_current_file->second, _cv_load_code);
    _current_time_stamp = _current_file->first;
    ++_current_file;
    return result;
  }

  return cv::Mat();
}


void file_stream_input_image::print_remaining_files(std::ostream& stream) const
{
  file_container::const_iterator
      it(_current_file);

  for (; it != _last_file; ++it)
    stream << it->second << std::endl;
}


std::string file_stream_input_image::get_current_file_name() const
{ return (_current_file != _last_file) ? _current_file->second : std::string(); }


} // sdvo
