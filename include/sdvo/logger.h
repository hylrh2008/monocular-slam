#ifndef _SDVO_LOGGER_H_
#define _SDVO_LOGGER_H_

#include <iostream>
#include <Eigen/Eigen>
#include <iomanip>
namespace sdvo
{

class logger
{
  std::ostream& _output;
  double _timestamp;


public:

  logger(std::ostream& output)
    : _output(output) {}

  void set_current_time_stamp(double timestamp)
  { _timestamp = timestamp; }

  void set_current_time_stamp(timeval timestamp)
  { _timestamp = timestamp.tv_sec + timestamp.tv_usec * 0.000001; }

  void log(Eigen::Affine3d const& transform)
  {
    Eigen::Quaterniond q = static_cast<Eigen::Quaterniond> (transform.linear());

    _output
        << std::setprecision(16) << _timestamp << ' '
        << transform.translation()[0] << ' '
        << transform.translation()[1] << ' '
        << transform.translation()[2] << ' '
        << q.x() << ' '
        << q.y() << ' '
        << q.z() << ' '
        << q.w() << std::endl;
  }
};

} // sdvo

#endif // _SDVO_LOGGER_H_
