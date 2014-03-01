#include <opencv2/highgui/highgui.hpp>
#include <sdvo/epipolar_matcher.h>
#include <sdvo/file_stream_input_image.h>
#include <sdvo/cvmat_to_rgbdpyramid.h>
#include <dvo/dense_tracking.h>
#include <opencv2/highgui/highgui.hpp>
#include <sdvo/depth_ma_fusionner.h>
#include <sdvo/depth_map_regulariser.h>
#include <sdvo/depth_hypothesis.h>
#include <pcl/visualization/cloud_viewer.h>

pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
void displayPointCloud (cv::Mat1f depth,Eigen::Matrix3d intrinsics,cv::Mat color)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int x = 0; x < depth.rows; ++x) {
    for (int y = 0; y < depth.cols; ++y) {
      if(depth(x,y) != 0){
        Eigen::Vector3d v(x,y,1);

        v[0]-=intrinsics(0,2);
        v[0]/=intrinsics(0,0);

        v[1]-=intrinsics(1,2);
        v[1]/=intrinsics(1,1);

        pcl::PointXYZRGB pt(color.at<cv::Vec3b>(x,y)[2],color.at<cv::Vec3b>(x,y)[1],color.at<cv::Vec3b>(x,y)[0]);
        pt.x = -depth(x,y) * v[0];
        pt.y =  depth(x,y) * v[1];
        pt.z =  depth(x,y);
        cloud->push_back(pt);
      }
    }
  }

  viewer.showCloud(cloud);
}

std::string test_directory;
std::string data_path;
using namespace sdvo;
using namespace std;
using namespace cv;

Mat colorised_depth(const cv::Mat1f & depth)
{
  cv::Mat coloredDepth,scaledDepth;
  depth.convertTo(scaledDepth,CV_8U,1./5.*255.);
  cv::applyColorMap(scaledDepth,coloredDepth,cv::COLORMAP_JET);

  return coloredDepth;
}

Mat colorised_variance(const cv::Mat1f & crt_inverse_depth_variance)
{
  cv::Mat coloredDepthVariance,scaledDepthVariance;
  crt_inverse_depth_variance.convertTo(scaledDepthVariance,CV_8U,1./0.01*255);
  cv::applyColorMap(scaledDepthVariance,coloredDepthVariance,cv::COLORMAP_JET);

  return coloredDepthVariance;
}

int main(int argc, char** argv)
{
  cv::namedWindow("Observation");
  cv::namedWindow("Prior");
  cv::namedWindow("Variance");
  sdvo::file_stream_input_image rgb_source("../dataset/rgbd_dataset_freiburg3_long_office_household/rgb/","",".png",
                                           cv::IMREAD_ANYCOLOR);
  sdvo::file_stream_input_image depth_source("../dataset/rgbd_dataset_freiburg3_long_office_household/depth/","",".png",
                                             cv::IMREAD_ANYDEPTH);
  cv::Mat rgb = rgb_source.get_next_image();
  cv::Mat depth = depth_source.get_next_image();

  cv::Mat rgbnew = rgb_source.get_next_image();
  cv::Mat depthground = depth_source.get_next_image();
  cv::Mat depthnew = depthground.clone();

  sdvo::cvmat_to_rhbdpyramid create_rgbdpyramid;
  dvo::core::RgbdImagePyramid pyramid = create_rgbdpyramid(rgb,depth,cvmat_to_rhbdpyramid::TUMDATASET);
  dvo::core::RgbdImagePyramid pyramidnew = create_rgbdpyramid(rgbnew,depthground,cvmat_to_rhbdpyramid::TUMDATASET);
  dvo::core::RgbdImagePyramid pyramidnew2 = pyramidnew;

  dvo::core::IntrinsicMatrix i =
      //dvo::core::IntrinsicMatrix::create(517.3,516.5,318.6,255.3); //fr1
      dvo::core::IntrinsicMatrix::create(535.4,539.2,320.1,247.6); //fr3
  dvo::DenseTracker::Config cfg = dvo::DenseTracker::getDefaultConfig();
  cfg.Lambda = 5E-2;
  cfg.MaxIterationsPerLevel=100;
  cfg.UseWeighting=true;
  cfg.Precision=1E-7;
  cfg.UseInitialEstimate=false;
  dvo::DenseTracker tracker(i,cfg);

  epipolar_matcher stereo_matcher(i.data.cast<double>());
  cv::setMouseCallback("Observation",epipolar_matcher_utils::mouseHandler,(void*) &stereo_matcher);
  cv::setMouseCallback("Prior",epipolar_matcher_utils::mouseHandlerPrior,(void*) &stereo_matcher);
  cv::setMouseCallback("Variance",epipolar_matcher_utils::mouseHandlerVariance,(void*) &stereo_matcher);


  sdvo::depth_hypothesis H(pyramidnew2.level(0).depth,
                           cv::Mat1f::ones(pyramidnew2.level(0).depth.size()) * 0.00001,
                           pyramidnew2.level(0).intensity,
                           i.fx(),
                           i.fy(),
                           i.ox(),
                           i.oy());

  Eigen::Affine3d cumul_t = Eigen::Affine3d::Identity();
  Eigen::Affine3d cumul_t_since_last = Eigen::Affine3d::Identity();

  char k='\0';
  for(int j=0; ;j++){

    //----------------------
    // Retrieve next image
    //----------------------
    std::swap(pyramidnew2, pyramid);
    rgbnew = rgb_source.get_next_image();
    depthground = depth_source.get_next_image();

    if(depth_source.get_current_time_stamp() <
       rgb_source.get_current_time_stamp() - 0.015 ){
      depthground = depth_source.get_next_image();
    }
    else if(depth_source.get_current_time_stamp() >
            rgb_source.get_current_time_stamp() + 0.015 ){
      rgbnew = rgb_source.get_next_image();
    }

    if(0){
      depthnew = depthground;
      pyramidnew  = create_rgbdpyramid(rgbnew,depthnew,
                                       sdvo::cvmat_to_rhbdpyramid::TUMDATASET);
    }
    else{
      depthnew = H.d;
      pyramidnew  = create_rgbdpyramid(rgbnew,depthnew,
                                       sdvo::cvmat_to_rhbdpyramid::FLOAT_MAP);
    }
    pyramidnew2 = pyramidnew;

    //-----------
    // TRACKING
    //-----------

    Eigen::Affine3d t = Eigen::Affine3d::Identity();
    tracker.match(pyramid,pyramidnew,t);

    //----------------
    // STEREO_MATCHING
    //----------------
    H.update_hypothesis(t.inverse(),pyramidnew2.level(0).intensity);

    cumul_t_since_last = cumul_t_since_last * t;
    cumul_t = cumul_t * t;

    double norm = sqrt(cumul_t_since_last.matrix()(0,3)
                       * cumul_t_since_last.matrix()(0,3)
                       + cumul_t_since_last.matrix()(1,3)
                       * cumul_t_since_last.matrix()(1,3)
                       + cumul_t_since_last.matrix()(2,3)
                       * cumul_t_since_last.matrix()(2,3));


    if(norm < 0.02){
      H.regularise_hypothesis();
    }
    else
    {
      stereo_matcher.set_depth_prior(H.d);
      stereo_matcher.set_depth_prior_variance(H.variance);

      stereo_matcher.push_new_data_in_buffer(std::move(pyramid),
                                             std::move(cumul_t));
      stereo_matcher.compute_new_observation();
      cumul_t_since_last = Eigen::Affine3d::Identity();

      cv::Mat obs(stereo_matcher.getObserved_depth());
      cv::Mat obs_var(stereo_matcher.getObserved_variance());
      cv::Mat prior(stereo_matcher.getObserved_depth_prior());
      cv::Mat oldVar = H.variance.clone();

      //--------
      // FUSION
      //--------
      for (int x = 0; x < H.d.cols; ++x) {
        for (int y = 0; y < H.d.rows; ++y) {
          if(
             pyramidnew2.level(0).intensity_dx.at<float>(y,x) * pyramidnew2.level(0).intensity_dx.at<float>(y,x)
             +
             pyramidnew2.level(0).intensity_dy.at<float>(y,x) * pyramidnew2.level(0).intensity_dy.at<float>(y,x)
             < 10){
            H.d(y,x) = 0;
            H.variance(y,x) = 0;
            H.precise_position(y,x).zeros();
          }
          assert(H.d(y,x)>=0);
          assert(H.variance(y,x)>=0);
        }
      }
      H.check_gradient_norm();
      H.add_observation_to_hypothesis(obs,obs_var);


      //--------
      // DISPLAY
      //--------

      if(k != 'k'){
        cv::Mat groundTruthDepth = colorised_depth(depthground/5000);
        cv::imshow("GroundTruth",groundTruthDepth);

        cv::Mat coloredPrior = colorised_depth(prior);
        cv::imshow("Prior",coloredPrior);

        cv::Mat coloredOldVar = colorised_variance(oldVar);
        cv::imshow("P riorVariance",coloredOldVar);

        cv::Mat coloredObs = colorised_depth(obs);
        cv::imshow("Observation",coloredObs);

        cv::Mat coloredObsv = colorised_variance(obs_var);
        cv::imshow("ObservationVariance",coloredObsv);

      }
      else{
        cv::Mat PriorNan(prior==0);
        cv::Mat ObserNan(obs==0);
        cv::Mat DepthNan(H.d==0);
        cv::Mat ObsVarNan(obs_var==0);
        cv::Mat PriorVar(oldVar==0);
        //--------
        // FUSION
        //--------
        cv::imshow("Prior",PriorNan);
        cv::imshow("PriorVariance",PriorVar);
        cv::imshow("Observation",ObserNan);
        cv::imshow("ObservationVariance",ObsVarNan);
        cv::imshow("depth",DepthNan);
      }
    }
    cv::Mat coloredDepthVariance = colorised_variance(H.variance);
    cv::Mat coloredDepth = colorised_depth(H.d);
    cv::Mat mask(H.d!=0);
    cv::Mat id(rgbnew);

    id.setTo(cv::Vec3d::zeros(),mask);
    coloredDepth.setTo(cv::Vec3d::zeros(),~mask);
    coloredDepthVariance.setTo(cv::Vec3d::zeros(),~mask);

    cv::imshow("depth",coloredDepth+id);

    cv::imshow("depthVariance",coloredDepthVariance);
    displayPointCloud (H.d,i.data.cast<double>(),coloredDepthVariance);

    k = cv::waitKey(2);
  }
}
