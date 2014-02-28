#include <opencv2/highgui/highgui.hpp>
#include <sdvo/epipolar_matcher.h>
#include <sdvo/file_stream_input_image.h>
#include <sdvo/cvmat_to_rgbdpyramid.h>
#include <dvo/dense_tracking.h>
#include <opencv2/highgui/highgui.hpp>
#include <sdvo/depth_ma_fusionner.h>
#include <sdvo/depth_map_regulariser.h>
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

void warp_variance_forward(cv::Mat1f & to_wrap, const cv::Mat1f & depth,
                           const Eigen::Affine3d& transformationx,
                           const Eigen::Matrix3d& intrinsics,
                           const cv::Mat2f & precise_pos)
{
  Eigen::Affine3d transformation = transformationx.cast<double>();

  bool identity = transformation.affine().isIdentity(1e-6);

  cv::Mat1f warped_image(cv::Mat::zeros(to_wrap.size(), to_wrap.type()));

  float ox = intrinsics(0,2);
  float oy = intrinsics(1,2);
  float fx = intrinsics(0,0);
  float fy = intrinsics(1,1);

  int width= to_wrap.cols;
  int height= to_wrap.rows;

  const float* depth_ptr = depth.ptr<float>();

  float x_precise;
  float y_precise;

  for(size_t y = 0; y < height; ++y)
  {
    for(size_t x = 0; x < width; ++x, ++depth_ptr)
    {
      if(*depth_ptr <= 1e-6f) continue;


      if(precise_pos(y,x)[0]!=0 && precise_pos(y,x)[1]!=0)
      {
        x_precise = precise_pos(y,x)[0];
        y_precise = precise_pos(y,x)[1];
      }
      else{
        x_precise = x;
        y_precise = y;
      }
      float depth = *depth_ptr;
      Eigen::Vector3d p3d((x_precise - ox) * depth / fx, (y_precise - oy) * depth / fy, depth);

      if(!identity)
      {
        Eigen::Vector3d p3d_transformed = transformation * p3d;
        if(p3d_transformed(2)<0) continue;

        float x_projected = (float) (p3d_transformed(0) * fx / p3d_transformed(2) + ox);
        float y_projected = (float) (p3d_transformed(1) * fy / p3d_transformed(2) + oy);

        if(0 <= x_projected && x_projected<width &&
           0 <= y_projected && y_projected<height)
        {
          int xp, yp;
          xp = (int) std::floor(x_projected);
          yp = (int) std::floor(y_projected);

          warped_image.at<float>(yp, xp) = std::pow(p3d_transformed(2)/p3d(2),int(4)) * to_wrap.at<float>(y, x) + 0.00001;
        }

        p3d = p3d_transformed;
      }
    }
  }

  if(identity)
  {
    std::swap(warped_image,to_wrap);
  }
  std::swap(to_wrap,warped_image);
}

void warp_depth_forward(cv::Mat1f & depth_mat,
                        const Eigen::Affine3d& transformationx,
                        const Eigen::Matrix3d& intrinsics,
                        cv::Mat2f & precise_pos,
                        const cv::Mat1f & old_intensity,
                        const cv::Mat1f & new_intensity)
{
  // TOTO NEXT: Attention!!!!
  // Cette fonction ne fait pas ce qu'on veut du tout!!!!
  Eigen::Affine3d transformation = transformationx;

  cv::Mat1f warped_mat(cv::Mat::zeros(depth_mat.size(), depth_mat.type()));
  cv::Mat2f next_precise_pos(cv::Mat::zeros(depth_mat.size(), depth_mat.type()));

  warped_mat.setTo(0);

  float ox = intrinsics(0,2);
  float oy = intrinsics(1,2);

  const float* depth_ptr = depth_mat.ptr<float>();
  int total = 0;
  float x_precise;
  float y_precise;
  for(size_t y = 0; y < depth_mat.rows; ++y)
  {
    for(size_t x = 0; x < depth_mat.cols; ++x, ++depth_ptr)
    {
      if(*depth_ptr==0)
      {
        continue;
      }

      if(precise_pos(y,x)[0]!=0 && precise_pos(y,x)[1]!=0)
      {
        x_precise = precise_pos(y,x)[0];
        y_precise = precise_pos(y,x)[1];
      }
      else{
        x_precise = x;
        y_precise = y;
      }
      float depth = *depth_ptr;

      Eigen::Vector3d p3d((x_precise - ox) * depth / intrinsics(0,0), (y_precise - oy) * depth / intrinsics(1,1), depth);
      Eigen::Vector3d p3d_transformed = transformation * p3d;
      if(p3d_transformed(2)<0) continue;
      float x_projected = (float) (p3d_transformed(0) * intrinsics(0,0) / p3d_transformed(2) + ox);
      float y_projected = (float) (p3d_transformed(1) * intrinsics(1,1) / p3d_transformed(2) + oy);

      if(0<x_projected && x_projected<depth_mat.cols &&
         0<y_projected && y_projected<depth_mat.rows)
      {
        int yi = (int) y_projected, xi = (int) x_projected;
        if(warped_mat.at<float>(yi, xi) == 0 || warped_mat.at<float>(yi, xi) > depth + 0.05){

          next_precise_pos(yi,xi) = cv::Vec2f(x_projected,y_projected);

          if(abs(old_intensity(y,x)-new_intensity(yi,xi))>50){
            warped_mat.at<float>(yi, xi) = 0;
          }
          else
          {
            warped_mat.at<float>(yi, xi) = p3d_transformed(2);
          }
        }
      }

      p3d = p3d_transformed;

      total++;
    }
  }
  std::swap(depth_mat,warped_mat);
  std::swap(precise_pos,next_precise_pos);
}
Mat colorised_depth(const cv::Mat1f & crt_depth)
{
  cv::Mat coloredDepth,scaledDepth;
  crt_depth.convertTo(scaledDepth,CV_8U,1./5.*255.);
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
  cv::Mat depthnew = depth_source.get_next_image();

  sdvo::cvmat_to_rhbdpyramid create_rgbdpyramid;
  dvo::core::RgbdImagePyramid pyramid = create_rgbdpyramid(rgb,depth);
  dvo::core::RgbdImagePyramid pyramidnew = create_rgbdpyramid(rgbnew,depthnew);
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


  cv::Mat1f crt_depth = pyramidnew2.level(0).depth.clone();
  cv::Mat1f crt_inverse_depth_variance = cv::Mat1f::ones(crt_depth.size()) * 0.00001;
  cv::Mat2f subpixel_hypothesis_pos = cv::Mat2f::zeros(crt_depth.size());


  Eigen::Affine3d cumul_t = Eigen::Affine3d::Identity();
  Eigen::Affine3d cumul_t_since_last = Eigen::Affine3d::Identity();

  char k='\0';
  for(int j=0; ;j++){

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
    warp_variance_forward(crt_inverse_depth_variance,
                          crt_depth,t.inverse(),
                          i.data.cast<double>(),
                          subpixel_hypothesis_pos
                          );

    warp_depth_forward(crt_depth,t.inverse(),
                       i.data.cast<double>(),
                       subpixel_hypothesis_pos,
                       pyramid.level(0).intensity,
                       pyramidnew2.level(0).intensity
                       );

    double norm = sqrt(cumul_t_since_last.matrix()(0,3)
                       * cumul_t_since_last.matrix()(0,3)
                       + cumul_t_since_last.matrix()(1,3)
                       * cumul_t_since_last.matrix()(1,3)
                       + cumul_t_since_last.matrix()(2,3)
                       * cumul_t_since_last.matrix()(2,3));

    cumul_t_since_last = cumul_t_since_last * t;
    cumul_t = cumul_t * t;

    if(norm < 0.03){
      depth_map_regulariser
          regularise(1./crt_depth,
                     crt_inverse_depth_variance);

      crt_depth = 1./regularise.get_inverse_depth_regularised();
      crt_inverse_depth_variance = regularise.get_inverse_depth_regularised_variance();

    }
    else
    {
      stereo_matcher.set_depth_prior(crt_depth);
      stereo_matcher.set_depth_prior_variance(crt_inverse_depth_variance);

      stereo_matcher.push_new_data_in_buffer(std::move(pyramid),
                                             std::move(cumul_t));
      stereo_matcher.compute_new_observation();
      cumul_t_since_last = Eigen::Affine3d::Identity();

      cv::Mat obs(stereo_matcher.getObserved_depth());
      cv::Mat obs_var(stereo_matcher.getObserved_variance());
      cv::Mat prior(stereo_matcher.getObserved_depth_prior());
      cv::Mat oldVar = crt_inverse_depth_variance.clone();

      //--------
      // FUSION
      //--------
      for (int x = 0; x < crt_depth.cols; ++x) {
        for (int y = 0; y < crt_depth.rows; ++y) {
          if(
             pyramidnew2.level(0).intensity_dx.at<float>(y,x) * pyramidnew2.level(0).intensity_dx.at<float>(y,x)
             +
             pyramidnew2.level(0).intensity_dy.at<float>(y,x) * pyramidnew2.level(0).intensity_dy.at<float>(y,x)
             < 10){
            crt_depth(y,x) = 0;
            crt_inverse_depth_variance(y,x) = 0;
            subpixel_hypothesis_pos(y,x).zeros();
          }
          assert(crt_depth(y,x)>=0);
          assert(crt_inverse_depth_variance(y,x)>=0);
        }
      }
      depth_ma_fusionner
          fusion(1./obs,
                 obs_var,
                 1./crt_depth,
                 crt_inverse_depth_variance);



      depth_map_regulariser
          regularise(fusion.get_inverse_depth_posterior(),
                     fusion.get_inverse_depth_posterior_variance());

      crt_depth = 1./regularise.get_inverse_depth_regularised();
      crt_inverse_depth_variance = regularise.get_inverse_depth_regularised_variance();



      //--------
      // DISPLAY
      //--------

      if(k != 'k'){
              cv::Mat groundTruthDepth = colorised_depth(pyramidnew2.level(0).depth);
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
        cv::Mat DepthNan(crt_depth==0);
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
    cv::Mat coloredDepthVariance = colorised_variance(crt_inverse_depth_variance);
    cv::Mat coloredDepth = colorised_depth(crt_depth);
    cv::imshow("depth",coloredDepth);
    cv::imshow("depthVariance",coloredDepthVariance);

    displayPointCloud (crt_depth,i.data.cast<double>(),coloredDepthVariance);

    k = cv::waitKey(2);
  }
}
