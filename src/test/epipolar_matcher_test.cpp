#include <opencv2/highgui/highgui.hpp>
#include <sdvo/epipolar_matcher.h>
#include <sdvo/file_stream_input_image.h>
#include <sdvo/cvmat_to_rgbdpyramid.h>
#include <dvo/dense_tracking.h>
#include <opencv2/highgui/highgui.hpp>
#include <sdvo/depth_ma_fusionner.h>
#include <sdvo/depth_map_regulariser.h>
#include <sdvo/depth_hypothesis.h>

#define OPENCV_DRAW
#ifdef _ENABLE_PCL
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>

#define GT_TRAJ 0
#define GT_OBS 0
//pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

void displayPointCloud (const cv::Mat1f & depth, const Eigen::Matrix3d & intrinsics,const cv::Mat3b & color,const cv::Mat1f & var)
{
  viewer->setBackgroundColor (0, 0, 0);
  viewer->resetStoppedFlag();
  viewer->removePointCloud("sample cloud");
  viewer->removeAllShapes();
  std::ostringstream s;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int x = 0; x < depth.rows; ++x) {
    for (int y = 0; y < depth.cols; ++y) {
      if(depth(x,y) != 0 && var(x,y) < 0.006){
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

        if( 0 && x % 4 == 0 && y % 4 == 0){

          float d_min = 1./ (1./depth(x,y) + 2 * var(x,y));
          float d_max = 1./ (1./depth(x,y) - 2 * var(x,y));
          if(d_max<0 || d_max>100) d_max = 100;

          if(var(x,y) > 0.005){
            pcl::PointXYZRGB pt1(128,128,128);
            pt1.x = -d_min * v[0];
            pt1.y =  d_min * v[1];
            pt1.z =  d_min;
            pcl::PointXYZRGB pt2(128,128,128);
            pt2.x = -d_max * v[0];
            pt2.y =  d_max * v[1];
            pt2.z =  d_max;
            s.str("");
            s<<x<<" "<<y;
            viewer->addLine<pcl::PointXYZRGB> (pt1,pt2,s.str());
          }
        }
      }
    }
  }
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);

  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->addCoordinateSystem(0.1);
  //  viewer->initCameraParameters();
  viewer->spinOnce(100);

  /*while(!viewer->wasStopped())
  {
    viewer->spinOnce (10);
    boost::this_thread::sleep (boost::posix_time::microseconds (1000));
  }
*/

}
#endif
std::string test_directory;
std::string data_path;
using namespace sdvo;
using namespace std;
using namespace cv;
Mat colorised_inverse_depth(const cv::Mat1f & depth)
{
  cv::Mat coloredDepth,scaledDepth,inverse_depth;
  inverse_depth=1/depth;
  inverse_depth.convertTo(scaledDepth,CV_8U,0.75*255.,5/255.*0.1);
  cv::Mat mask(depth == 0);
  cv::applyColorMap(scaledDepth,coloredDepth,cv::COLORMAP_JET);
  coloredDepth.setTo(0,mask);

  return coloredDepth;
}

Mat colorised_depth(const cv::Mat1f & depth)
{
  cv::Mat coloredDepth,scaledDepth;
  depth.convertTo(scaledDepth,CV_8U,1./5.*255.);
  cv::Mat mask(depth == 0);
  cv::applyColorMap(scaledDepth,coloredDepth,cv::COLORMAP_JET);
  coloredDepth.setTo(0,mask);

  return coloredDepth;
}

Mat colorised_variance(const cv::Mat1f & crt_inverse_depth_variance)
{
  cv::Mat coloredDepthVariance,scaledDepthVariance;
  crt_inverse_depth_variance.convertTo(scaledDepthVariance,CV_8U,1./0.01*255);
  cv::Mat mask(crt_inverse_depth_variance == 0);
  cv::applyColorMap(scaledDepthVariance,coloredDepthVariance,cv::COLORMAP_JET);
  coloredDepthVariance.setTo(0,mask);
  return coloredDepthVariance;
}
Mat colorised_outliers_proba(sdvo::depth_hypothesis H)
{
  cv::Mat coloredOutlierProba;
  cv::Mat scaleProba;
  H.outlier_probability.convertTo(scaleProba,CV_8U,1.*255.);
  cv::applyColorMap(scaleProba,coloredOutlierProba,cv::COLORMAP_JET);
  cv::Mat maskProba(H.var == 0);
  coloredOutlierProba.setTo(0,maskProba);

  return coloredOutlierProba;
}

Mat colorised_age(sdvo::depth_hypothesis H)
{
  cv::Mat coloredAge;
  cv::Mat maskAge(H.age != 0);
  cv::applyColorMap((H.age+1)/50*255,coloredAge,cv::COLORMAP_JET);
  coloredAge.setTo(0,~maskAge);

  return coloredAge;
}

Eigen::Affine3d parse_next_line(std::istream & stream){
  Eigen::Affine3d t;
  stream >> t(0,0) >> t(0,1) >> t(0,2) >> t(0,3)
         >> t(1,0) >> t(1,1) >> t(1,2) >> t(1,3)
         >> t(2,0) >> t(2,1) >> t(2,2) >> t(2,3);

  return t;
}

int main(int argc, char** argv)
{
  int code;
  char* cc = reinterpret_cast<char*>(&code);
  cc[0] = 'D';
  cc[1] = 'I';
  cc[2] = 'V';
  cc[3] = '4';

  cv::VideoWriter avi("output.avi",code,30,cv::Size(640,480));
  cv::namedWindow("Observation");
  cv::namedWindow("Prior");
  cv::namedWindow("Variance");

  //-------------------------------
  // Configure and open input files
  //-------------------------------
  std::ifstream traj("../dataset/200fps_images_archieve/se3.csv");
  sdvo::file_stream_input_image
      rgb_source(
        //"../dataset/rgb/","",".png",
        "../dataset/rgbd_dataset_freiburg3_long_office_household/rgb/","","png",
        //"../dataset/rgbd_dataset_freiburg2_large_no_loop/rgb/","","png",
        //"../dataset/200fps_images_archieve/rgb/","scene_00_","png",
        cv::IMREAD_ANYCOLOR);

  sdvo::file_stream_input_image
      depth_source(
        //"../dataset/depth/","","png",
        "../dataset/rgbd_dataset_freiburg3_long_office_household/depth/","","png",
        //"../dataset/rgbd_dataset_freiburg2_large_no_loop/depth/","","png",
        //"../dataset/200fps_images_archieve/depth/","scene_00_","png",
        cv::IMREAD_ANYDEPTH);
  dvo::core::IntrinsicMatrix i =
      //dvo::core::IntrinsicMatrix::create(517.3,516.5,318.6,255.3); //fr1
      //dvo::core::IntrinsicMatrix::create(520.9,521.0,325.1,249.7); //fr2
      dvo::core::IntrinsicMatrix::create(535.4,539.2,320.1,247.6); //fr3
      //dvo::core::IntrinsicMatrix::create(480,-481,320.5,240.5); //


  //------------------------------
  // Create Dense Tracker Instance
  //------------------------------
  dvo::DenseTracker::Config cfg = dvo::DenseTracker::getDefaultConfig();
  cfg.Lambda = 5E-3;
  cfg.MaxIterationsPerLevel = 100;
  cfg.UseWeighting = true;
  cfg.Precision = 1E-7;
  cfg.UseInitialEstimate = true;
  dvo::DenseTracker tracker(i,cfg);

  //------------------------------
  // Create Stereo Matcher Instance
  //------------------------------

  epipolar_matcher stereo_matcher(i.data.cast<double>());

  cv::setMouseCallback("Observation",epipolar_matcher_utils::mouseHandler,(void*) &stereo_matcher);
  cv::setMouseCallback("Prior",epipolar_matcher_utils::mouseHandlerPrior,(void*) &stereo_matcher);
  cv::setMouseCallback("Variance",epipolar_matcher_utils::mouseHandlerVariance,(void*) &stereo_matcher);

  //------------------------------
  // Initial hypothesis
  //------------------------------
  cv::Mat rgb = rgb_source.get_next_image();
  cv::Mat depth = depth_source.get_next_image();
  cv::Mat rgbnew = rgb_source.get_next_image();
  cv::Mat1f depthground = depth_source.get_next_image();

  cv::Mat1f depthnew = depthground.clone();

  sdvo::cvmat_to_rhbdpyramid create_rgbdpyramid;

  dvo::core::RgbdImagePyramid pyramid = create_rgbdpyramid(rgb,depth,cvmat_to_rhbdpyramid::TUMDATASET);
  dvo::core::RgbdImagePyramid pyramidnew = create_rgbdpyramid(rgbnew,depthnew,cvmat_to_rhbdpyramid::TUMDATASET);

  sdvo::depth_hypothesis H(pyramidnew.level(0).depth,
                           cv::Mat1f::ones(pyramidnew.level(0).depth.size()) * 0.01,
                           pyramidnew.level(0).intensity,
                           i.fx(),
                           i.fy(),
                           i.ox(),
                           i.oy());

  Eigen::Affine3d cumul_t = Eigen::Affine3d::Identity();
  Eigen::Affine3d cumul_t_since_last = Eigen::Affine3d::Identity();

  if(GT_TRAJ){
    Eigen::Affine3d ts;
    ts = parse_next_line(traj);
    cumul_t_since_last = cumul_t_since_last * ts;
    cumul_t = cumul_t * ts;
  }
  char k='\0';

  for(int j=0; ;j++){

    //----------------------
    // Retrieve next image
    //----------------------
    std::swap(pyramidnew, pyramid);

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
      pyramidnew  = create_rgbdpyramid(rgbnew,
                                       depthnew,
                                       sdvo::cvmat_to_rhbdpyramid::TUMDATASET);
    }
    else{
      depthnew = H.d;
      pyramidnew  = create_rgbdpyramid(rgbnew,depthnew,
                                       sdvo::cvmat_to_rhbdpyramid::FLOAT_MAP);
    }

    //-----------
    // TRACKING
    //-----------

    Eigen::Affine3d t = Eigen::Affine3d::Identity();
    if(GT_TRAJ){
      t = parse_next_line(traj);
      Eigen::Affine3d t_match;
      tracker.match(pyramid,pyramidnew,t_match);
      cerr<< t.matrix() <<"\n"<<t_match.matrix()<<"\n"<<std::endl;
    }
    else{
      tracker.match(pyramid,pyramidnew,t);
      Eigen::Affine3d t_gt;
      t_gt = parse_next_line(traj);
      cerr<< t.matrix() <<"\n" << t_gt.matrix() << "\n"<<std::endl;

    }

    //----------------------------------
    // UPDATING HYPOTHESIS (PREDICTION)
    //----------------------------------
    H.update_hypothesis(t.inverse(),pyramidnew.level(0).intensity);
    cumul_t_since_last = cumul_t_since_last * t;
    cumul_t = cumul_t * t;

    double norm = sqrt(cumul_t_since_last.matrix()(0,3)
                       * cumul_t_since_last.matrix()(0,3)
                       + cumul_t_since_last.matrix()(1,3)
                       * cumul_t_since_last.matrix()(1,3)
                       + cumul_t_since_last.matrix()(2,3)
                       * cumul_t_since_last.matrix()(2,3));


    cv::Mat gradient_norm2(pyramidnew.level(0).intensity_dx.mul(pyramidnew.level(0).intensity_dx) +
                           pyramidnew.level(0).intensity_dy.mul(pyramidnew.level(0).intensity_dy));
    //----------------
    // STEREO_MATCHING
    //----------------

    if(norm < 0){
      H.check_gradient_norm(gradient_norm2);
      H.regularise_hypothesis();
    }
    else
    {
      stereo_matcher.set_depth_prior(H.d);
      stereo_matcher.set_depth_prior_variance(H.var);
      stereo_matcher.set_pixel_age(H.age);

      stereo_matcher.push_new_data_in_buffer(pyramidnew, cumul_t);
      stereo_matcher.compute_new_observation();
      cumul_t_since_last = Eigen::Affine3d::Identity();

      cv::Mat obs(stereo_matcher.get_observed_depth());
      cv::Mat obs_var(stereo_matcher.get_observed_variance());
      cv::Mat prior(stereo_matcher.get_depth_prior());
      cv::Mat oldVar = H.var.clone();

      if(GT_OBS){
        obs = cv::Mat1f((cv::Mat1b(obs!=0) & 1)).mul(depthground/5000);
      }

      //--------
      // FUSION
      //--------

      H.outlier_probability -= 0.1 * cv::Mat1f((cv::Mat1b(obs!=0) & cv::Mat1b(obs_var) & 1));
      H.outlier_probability += 0.1 * cv::Mat1f((cv::Mat1b(obs==0) & cv::Mat1b(obs_var) & 1));

      H.check_gradient_norm(gradient_norm2);
      H.add_observation_to_hypothesis(obs,obs_var);
      H.regularise_hypothesis();


      //--------
      // DISPLAY
      //--------
#ifdef OPENCV_DRAW
      if(k != 'k'){
        //        cv::Mat groundTruthDepth = colorised_depth(depthground/5000);
        //        cv::imshow("GroundTruth",groundTruthDepth);

        cv::Mat groundTruthDepthInverse = colorised_inverse_depth(depthground/5000);
        cv::imshow("GroundTruthInverse",groundTruthDepthInverse);

        cv::Mat coloredPrior = colorised_inverse_depth(prior);
        cv::imshow("Prior",coloredPrior);

        cv::Mat coloredOldVar = colorised_variance(oldVar);
        cv::imshow("PriorVariance",coloredOldVar);

        cv::Mat coloredObs = colorised_inverse_depth(obs);
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
#endif

    }
    cv::Mat coloredDepthVariance = colorised_variance(H.var);
    //cv::Mat coloredDepth = colorised_depth(H.d);
    cv::Mat coloredInverseDepth = colorised_inverse_depth(H.d);
    cv::Mat coloredAge = colorised_age(H);
    cv::Mat coloredOutlierProba = colorised_outliers_proba(H);

    cv::Mat mask(H.d != 0);
    cv::Mat id(rgbnew);
    id.setTo(cv::Vec3d::zeros(),mask);
    coloredInverseDepth.setTo(cv::Vec3d::zeros(),~mask);
    coloredDepthVariance.setTo(cv::Vec3d::zeros(),~mask);
#ifdef OPENCV_DRAW

    //cv::imshow("depth",coloredDepth+id);
    cv::imshow("depthInverse",coloredInverseDepth+id);
    cv::imshow("depthVariance",coloredDepthVariance);
    cv::imshow("age",coloredAge);
    cv::imshow("outlier",coloredOutlierProba);
    avi << coloredInverseDepth+id;
    k = cv::waitKey(2);
#endif
#ifdef _ENABLE_PCL
    displayPointCloud (H.d,i.data.cast<double>(),coloredDepthVariance,H.var);
#endif
  }
}
