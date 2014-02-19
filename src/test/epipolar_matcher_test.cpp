#include <opencv2/highgui/highgui.hpp>
#include <sdvo/epipolar_matcher.h>
#include <sdvo/file_stream_input_image.h>
#include <sdvo/cvmat_to_rgbdpyramid.h>
#include <dvo/dense_tracking.h>
#include <opencv2/highgui/highgui.hpp>
#include <sdvo/depth_ma_fusionner.h>
//#include <pcl/visualization/cloud_viewer.h>

//pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
//void displayPointCloud (cv::Mat1f depth,Eigen::Matrix3d intrinsics)
//{
//  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//  for (int x = 0; x < depth.rows; ++x) {
//    for (int y = 0; y < depth.cols; ++y) {
//      Eigen::Vector3d v(x,y,1);
//      v[0]/=intrinsics(0,0);
//      v[1]/=intrinsics(1,1);
//      cloud->push_back(pcl::PointXYZ(-depth(x,y)*v[0],depth(x,y)*v[1],depth(x,y)));
//    }
//  }
//  viewer.showCloud(cloud);
//}
std::string test_directory;
std::string data_path;
using namespace sdvo;

void warp_forward(cv::Mat1f & to_warp,const cv::Mat1f & depth_mat, const Eigen::Affine3d& transformationx, const Eigen::Matrix3d& intrinsics)
{
  Eigen::Affine3d transformation = transformationx;

  cv::Mat warped_mat = cv::Mat::zeros(to_warp.size(), to_warp.type());
  warped_mat.setTo(std::numeric_limits<float>::quiet_NaN());

  float ox = intrinsics(0,2);
  float oy = intrinsics(1,2);

  const float* depth_ptr = depth_mat.ptr<float>();
  int outliers = 0;
  int total = 0;

  for(size_t y = 0; y < to_warp.rows; ++y)
  {
    for(size_t x = 0; x < to_warp.cols; ++x, ++depth_ptr)
    {
      if(!std::isfinite(*depth_ptr))
      {
        continue;
      }

      float depth = *depth_ptr;
      Eigen::Vector3d p3d((x - ox) * depth / intrinsics(0,0), (y - oy) * depth / intrinsics(1,1), depth);
      Eigen::Vector3d p3d_transformed = transformation * p3d;

      float x_projected = (float) (p3d_transformed(0) * intrinsics(0,0) / p3d_transformed(2) + ox);
      float y_projected = (float) (p3d_transformed(1) * intrinsics(1,1) / p3d_transformed(2) + oy);

      if(0<x_projected && x_projected<to_warp.cols &&
         0<y_projected && y_projected<to_warp.rows)
      {
        int yi = (int) y_projected, xi = (int) x_projected;

        if(!std::isfinite(warped_mat.at<float>(yi, xi)) || (warped_mat.at<float>(yi, xi) - 0.05) > depth)
          warped_mat.at<float>(yi, xi) = to_warp.at<float>(y,x);
      }

      p3d = p3d_transformed;

      total++;
    }
  }
  to_warp = warped_mat;
}
int main(int argc, char** argv)
{
  cv::namedWindow("Observation");
  cv::namedWindow("Prior");
  cv::namedWindow("Variance");
  sdvo::file_stream_input_image rgb_source("../dataset/rgbd_dataset_freiburg3_long_office_household/rgb/","",".png",
                                          CV_LOAD_IMAGE_COLOR);
  sdvo::file_stream_input_image depth_source("../dataset/rgbd_dataset_freiburg3_long_office_household/depth/","",".png",
                                             CV_LOAD_IMAGE_ANYDEPTH);
  cv::Mat rgb = rgb_source.get_next_image();
  cv::Mat depth = depth_source.get_next_image();

  cv::Mat rgbnew = rgb_source.get_next_image();
  cv::Mat depthnew = depth_source.get_next_image();

  sdvo::cvmat_to_rhbdpyramid create_rgbdpyramid;
  dvo::core::RgbdImagePyramid pyramid = create_rgbdpyramid(rgb,depth);
  dvo::core::RgbdImagePyramid pyramidnew = create_rgbdpyramid(rgbnew,depthnew);
  dvo::core::RgbdImagePyramid pyramidnew2 = pyramidnew;

  dvo::core::IntrinsicMatrix i =
      //dvo::core::IntrinsicMatrix::create(517.3,516.5,318.6,255.3); //fr1
      dvo::core::IntrinsicMatrix::create(535.4,539.2,320.1,247.6); //fr3
  dvo::DenseTracker tracker(i);
  epipolar_matcher stereo_matcher(i.data.cast<double>());
  cv::setMouseCallback("Observation",epipolar_matcher_utils::mouseHandler,(void*) &stereo_matcher);
  cv::setMouseCallback("Prior",epipolar_matcher_utils::mouseHandlerPrior,(void*) &stereo_matcher);
  cv::setMouseCallback("Variance",epipolar_matcher_utils::mouseHandlerVariance,(void*) &stereo_matcher);

  cv::Mat1f crtPrior = pyramidnew2.level(0).depth.clone();
  cv::Mat1f crtPrior_variance = 0.10 * (crtPrior.mul(crtPrior));
  cv::Mat1f crtDepth_variance = crtPrior_variance;
  cv::Mat1f crtDepth = crtPrior;

  Eigen::Affine3d cumul_t = Eigen::Affine3d::Identity();
  char k='\0';
while(true){

    //----------------------
    // Retrieve next image
    //----------------------
    std::swap(pyramidnew2, pyramid);
    rgbnew = rgb_source.get_next_image();
    depthnew = depth_source.get_next_image();

    if(depth_source.get_current_time_stamp() < rgb_source.get_current_time_stamp() - 0.015 ){
      depthnew = depth_source.get_next_image();
    }
    else if(depth_source.get_current_time_stamp() > rgb_source.get_current_time_stamp() + 0.015 ){
      rgbnew = rgb_source.get_next_image();
    }

    pyramidnew  = create_rgbdpyramid(rgbnew,depthnew);
    pyramidnew2 = pyramidnew;

    //-----------
    // TRACKING
    //-----------

    Eigen::Affine3d t = Eigen::Affine3d::Identity();
    tracker.match(pyramid,pyramidnew,t);

    //----------------
    // STEREO_MATCHING
    //----------------
    warp_forward(crtDepth,crtDepth,t.inverse(), i.data.cast<double>());
    warp_forward(crtDepth_variance,crtDepth,t.inverse(), i.data.cast<double>());

    cv::imshow("firstWarpedDepth",crtDepth);
    cv::imshow("firstWarpedCovar",crtDepth_variance);
    cv::waitKey(0);
    stereo_matcher.set_depth_prior(crtDepth);
    stereo_matcher.push_new_data_in_buffer(std::move(pyramid),std::move(cumul_t));

    cumul_t = cumul_t * t;
    stereo_matcher.compute_new_observation();


    cv::Mat obs(stereo_matcher.getObserved_depth_crt());
    cv::Mat obs_var(stereo_matcher.get_observed_variance());
    cv::Mat prior(stereo_matcher.getObserved_depth_prior());
    depth_ma_fusionner fusion(obs,obs_var,crtDepth,crtDepth_variance);
    cv::Mat oldVar=crtDepth_variance;
    crtDepth = fusion.get_inverse_depth_posterior();
    crtDepth_variance = fusion.get_inverse_depth_posterior_variance();

    if(k == 'k'){
      cv::imshow("Prior",prior/5);
//      cv::imshow("PriorVariance",crtDepth_variance*10);
      //--------
      // FUSION
      //--------
      cv::imshow("Observation",obs/5);
      cv::imshow("ObservationVariance",obs_var*10);
      cv::imshow("depth",crtDepth/5);
    }
    else{
      cv::Mat PriorNan(prior!=prior);
      cv::Mat ObserNan(obs!=obs);
      cv::Mat DepthNan(crtDepth!=crtDepth);
      cv::Mat ObsVarNan(obs_var!=obs_var);
      cv::Mat PriorVar(oldVar!=oldVar);
      //--------
      // FUSION
      //--------
      cv::imshow("Prior",PriorNan);
      cv::imshow("Observation",ObserNan);
      cv::imshow("ObservationVariance",ObsVarNan);
      cv::imshow("PriorVariance",PriorVar);
      cv::imshow("depth",DepthNan);
    }
    k=cv::waitKey(0);
  }
}
