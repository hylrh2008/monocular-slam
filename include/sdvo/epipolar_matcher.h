#ifndef EPIPOLAR_MATCHER_H
#define EPIPOLAR_MATCHER_H
#include <boost/circular_buffer.hpp>
#include <dvo/core/rgbd_image.h>
namespace sdvo{

class epipolar_matcher
{
  friend class epipolar_matcher_utils;
public:
  epipolar_matcher(const Eigen::Matrix3d & intrinsics_matrix);

  bool push_new_data_in_buffer(dvo::core::RgbdImagePyramid && pyr, Eigen::Affine3d && transform_from_start);

  bool compute_new_observation();

  cv::Mat get_observed_depth() const;

  cv::Mat get_observed_variance() const;

  void set_depth_prior(cv::Mat1f depth);

  void warp_depth_prior(Eigen::Affine3d t);

  cv::Mat1f getObserved_depth_crt() const;

  cv::Mat1f getObserved_depth_prior() const;

  cv::Mat1f getObserved_depth_prior_variance() const;
  void set_depth_prior_variance(cv::Mat1f depth_variance);
private:
  boost::circular_buffer< std::pair<dvo::core::RgbdImagePyramid,Eigen::Affine3d> >  last_images_buffer;

  cv::Mat1f observed_depth_crt;
  cv::Mat1f depth_prior;
  cv::Mat1f depth_prior_variance;

  cv::Mat1f observed_inverse_depth_variance;

  void init_matrices(cv::Size size);

  double compute_error(const cv::Point2d & point,
                       const cv::Vec2d & epipole_direction,
                       double sigma_l, double sigma2_i,double alpha,
                        dvo::core::RgbdImagePyramid &img);

  Eigen::Vector3d ProjectInZEqualOne(const Eigen::Vector4d & point);

  Eigen::Vector4d UnProject(const Eigen::Vector3d & pt_3D);

  static cv::Mat_<double> LinearLSTriangulation(cv::Point3d u,
                                                cv::Matx34d P,
                                                cv::Point3d u1,
                                                cv::Matx34d P1);

  void warp_prior_forward(const Eigen::Affine3d &transformationx,
                          const Eigen::Matrix3d &intrinsics);

  float getFloatSubpix(const cv::Mat1f &img,
                       const cv::Point2d & pt);

  cv::Point2d project_from_image_to_image(const cv::Point2d &p_in_1,
                                          const Eigen::Affine3d &se3_2_from_1,
                                          float distance);

  void triangulate_and_populate_observation(const cv::Point2d & p,
                                            const cv::Point2d & match,
                                            const Eigen::Affine3d & se3_ref_from_crt);
  bool b_matrices_inited;
  Eigen::Matrix3d intrinsics_matrix;

};

class epipolar_matcher_utils{
public:
  static void mouseHandler(int event, int x, int y, int flags, void* t)
  {
      std::cerr << x <<" "<< y <<" "<< ((epipolar_matcher*)t)->get_observed_depth().at<float>(y,x)<<std::endl;
  }
  static void mouseHandlerPrior(int event, int x, int y, int flags, void* t)
  {
      std::cerr << x <<" "<< y <<" "<< ((epipolar_matcher*)t)->getObserved_depth_prior().at<float>(y,x)<<std::endl;
  }
  static void mouseHandlerVariance(int event, int x, int y, int flags, void* t)
  {
      std::cerr << x <<" "<< y <<" "<<((epipolar_matcher*)t)->get_observed_variance().at<float>(y,x)<<std::endl;
  }
};
}


#endif // EPIPOLAR_MATCHER_H
