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
  tracker_with_depth(const std::string & dataset_folder):
    rgb_source(dataset_folder + "/rgb/", "", ".png",CV_LOAD_IMAGE_COLOR),
    depth_source(dataset_folder + "/depth/", "", ".png",CV_LOAD_IMAGE_ANYDEPTH),
    create_rgbdpyramid()
  {

  }
  sdvo::file_stream_input_image rgb_source;
  sdvo::file_stream_input_image depth_source;
  sdvo::cvmat_to_rhbdpyramid create_rgbdpyramid;

  void run(void)
  {
    std::string logfilename_relative("./relativ_pose.txt");
    std::string logfilename("./pose.txt");

    std::ofstream log(logfilename);
    if (!log.is_open())
    {
      std::cerr << "unable to open: " << "./pose.txt" << std::endl;
      return;
    }
    std::ofstream log_relative(logfilename_relative);
    if (!log.is_open())
    {
      std::cerr << "unable to open: " << "./relativ_pose.txt" << std::endl;
      return;
    }
    sdvo::logger logger(log);
    sdvo::logger logger_relative(log_relative);

    dvo::core::IntrinsicMatrix intrinsic =
        dvo::core::IntrinsicMatrix::create(517.3,516.5,318.6,255.3);
        //dvo::core::IntrinsicMatrix::create(535.4,	 539.2,	 320.1,	 247.6);
    dvo::DenseTracker::Config cfg = dvo::DenseTracker::getDefaultConfig();
    cfg.Lambda = 5E-3;
    cfg.MaxIterationsPerLevel=100;
    cfg.UseWeighting=true;
    cfg.Precision=1E-7;
    cfg.UseInitialEstimate=false;




    dvo::DenseTracker tracker(intrinsic,cfg);
    cv::Mat rgb;
    cv::Mat depth;
    std::string rgb_filename = rgb_source.get_current_file_name();
    std::string depth_filename = depth_source.get_current_file_name();
    logger.set_current_time_stamp(rgb_source.get_current_time_stamp());
    cv::Mat rgbnew = rgb_source.get_next_image();
    cv::Mat depthnew = depth_source.get_next_image();
    dvo::core::RgbdImagePyramid pyramid = create_rgbdpyramid(rgbnew,depthnew);
    dvo::core::RgbdImagePyramid pyramidnew = create_rgbdpyramid(rgbnew,depthnew);
    timeval start;
    timeval end;
    Eigen::Affine3d cumulated_transform(Eigen::Affine3d::Identity());
    cumulated_transform = static_cast<Eigen::Affine3d>(
          Eigen::Quaterniond(-0.3248,0.6574,0.6126,-0.2949)
          //Eigen::Quaterniond( -0.0145,0.0003, 0.8617, -0.5072)
          );
    Eigen::Affine3d last_transform=cumulated_transform;

    while(true)
    {
      std::swap(rgb, rgbnew);
      std::swap(depth, depthnew);
      std::swap(pyramidnew, pyramid);

      rgb_filename = rgb_source.get_current_file_name();
      depth_filename = depth_source.get_current_file_name();

      logger.set_current_time_stamp(rgb_source.get_current_time_stamp());
      logger_relative.set_current_time_stamp(rgb_source.get_current_time_stamp());

      rgbnew = rgb_source.get_next_image();

      while (depth_source.get_current_time_stamp() < rgb_source.get_current_time_stamp())
        depthnew = depth_source.get_next_image();

      pyramidnew  = create_rgbdpyramid(rgbnew,depthnew);
      Eigen::Affine3d transform = Eigen::Affine3d::Identity();
      gettimeofday(&start,NULL);
      tracker.match(pyramid,pyramidnew,transform);
      if(transform.translation()(0)==0 && transform.translation()(1)==0 && transform.translation()(2)==0 ){
        Eigen::Affine3d id= Eigen::Affine3d::Identity();
        tracker.updateLastTransform(last_transform);
        transform=last_transform;
      }
      else{
        last_transform = transform;
      }

      logger_relative.log(transform);
      gettimeofday(&end,NULL);

      cumulated_transform = cumulated_transform * transform;
      logger.log(cumulated_transform);

      std::cerr<<(end.tv_usec - start.tv_usec)/1000<<std::endl;
      std::cerr<<transform.matrix()<<std::endl;

      cv::imshow("Nouvelle",pyramidnew.level(0).intensity/255);
//      cv::imshow("Ancienne",pyramid.level(0).intensity/255);
//      cv::imshow("AncienneD",pyramid.level(0).depth);

      cv::imshow("NouvelleD",pyramidnew.level(0).depth);

      std::cerr<<"rgb_filename :"<<rgb_filename<<std::endl;
      std::cerr<<"depth_filename :"<<depth_filename<<std::endl;

      cv::waitKey(10);
    }
  }
};

int main(int argc, char *argv[])
{
  if(argc < 2){
    std::cerr<<"No Dataset"<<std::endl;
    return 1;
  }

  tracker_with_depth tracker(argv[1]);
  tracker.run();
  return 0;
}
