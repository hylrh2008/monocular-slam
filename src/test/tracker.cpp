#include <sdvo/file_stream_input_image.h>
#include <sdvo/cvmat_to_rgbdpyramid.h>
#include <dvo/dense_tracking.h>
#include <sys/time.h>
#include <sdvo/logger.h>
#include <fstream>
#include <iostream>

class tracker_with_depth
{
public:
  tracker_with_depth():
    rgb_source("../dataset/rgb/", "", ".png",CV_LOAD_IMAGE_COLOR),
    depth_source("../dataset/depth", "", ".png",CV_LOAD_IMAGE_ANYDEPTH),
    create_rgbdpyramid()
  {
  }
  sdvo::file_stream_input_image rgb_source;
  sdvo::file_stream_input_image depth_source;
  sdvo::cvmat_to_rhbdpyramid create_rgbdpyramid;

  void run(void)
  {
    std::string logfilename("./pose.txt");
    std::ofstream log(logfilename);
    if (!log.is_open())
    {
      std::cerr << "unable to open: " << "./pose.txt" << std::endl;
      return;
    }
    sdvo::logger logger(log);

    dvo::core::IntrinsicMatrix intrinsic = dvo::core::IntrinsicMatrix::create(525,525,319.5,139.5);
    dvo::DenseTracker::Config cfg = dvo::DenseTracker::getDefaultConfig();
    cfg.Lambda = 0;
    cfg.MaxIterationsPerLevel=100;
    cfg.UseWeighting=true;
    cfg.Precision=1E-7;

    dvo::DenseTracker tracker(intrinsic,cfg);
    cv::Mat rgb;
    cv::Mat depth;
    std::string rgb_filename = rgb_source.get_current_file_name();
    std::string depth_filename = depth_source.get_current_file_name();
    cv::Mat rgbnew = rgb_source.get_next_image();
    cv::Mat depthnew = depth_source.get_next_image();
    dvo::core::RgbdImagePyramid pyramid = create_rgbdpyramid(rgbnew,depthnew);
    dvo::core::RgbdImagePyramid pyramidnew = create_rgbdpyramid(rgbnew,depthnew);
    timeval start;
    timeval end;
    Eigen::Affine3d cumulated_transform(Eigen::Affine3d::Identity());

    while(true)
    {
      std::swap(rgb, rgbnew);
      std::swap(depth, depthnew);
      std::swap(pyramidnew, pyramid);

      rgb_filename = rgb_source.get_current_file_name();
      depth_filename = depth_source.get_current_file_name();

      logger.set_current_time_stamp(rgb_source.get_current_time_stamp());

      depthnew = depth_source.get_next_image();
      rgbnew = rgb_source.get_next_image();

      pyramidnew  = create_rgbdpyramid(rgbnew,depthnew);
      Eigen::Affine3d transform;

      gettimeofday(&start,NULL);
      tracker.match(pyramid,pyramidnew,transform);
      logger.log(transform);
      gettimeofday(&end,NULL);

      cumulated_transform = cumulated_transform*transform;

      std::cerr<<start.tv_sec<<" "<<start.tv_usec<<std::endl;
      std::cerr<<end.tv_sec<<" "<<end.tv_usec<<std::endl<<std::endl;
      std::cerr<<transform.matrix()<<std::endl;

      cv::imshow("Nouvelle",pyramidnew.level(0).intensity/255);
      cv::imshow("Ancienne",pyramid.level(0).intensity/255);
      cv::imshow("NouvelleD",pyramidnew.level(0).depth);
      cv::imshow("AncienneD",pyramid.level(0).depth);

      std::cerr<<"rgb_filename :"<<rgb_filename<<std::endl;
      std::cerr<<"depth_filename :"<<depth_filename<<std::endl;

      cv::waitKey(10);
    }
  }
};

int main(int argc, char *argv[])
{
  tracker_with_depth tracker;
  tracker.run();
  return 0;
}
