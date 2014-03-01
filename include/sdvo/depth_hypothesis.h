#ifndef DEPTH_HYPOTHESIS_H
#define DEPTH_HYPOTHESIS_H
#include <opencv2/core/core.hpp>
#include <Eigen/Geometry>
#include <sdvo/depth_map_regulariser.h>
#include <sdvo/depth_ma_fusionner.h>

namespace sdvo{
struct depth_hypothesis{

  depth_hypothesis(const cv::Mat1f &depth_init, const cv::Mat1f &variance_init,const cv::Mat1f & intensity_img, float fx, float fy, float ox, float oy);

  cv::Mat1f d;
  cv::Mat1f var; // of INVERSE depth!
  cv::Mat2f precise_position; //  allow to give floating coordinate to hypothesis in image
  cv::Mat1f outlier_probability; // Keep track of validity of hypothesis
  cv::Mat1f intensity_img;
  cv::Mat1b age;


  float height;
  float width;
  float fx;
  float fy;
  float ox;
  float oy;

  void update_hypothesis(const Eigen::Affine3d &transformationx, const cv::Mat1f &new_intensity);
  void regularise_hypothesis();
  void add_observation_to_hypothesis(const cv::Mat1f depth_obs,const cv::Mat1f var_obs);
  void check_gradient_norm();
  void remove_pixel_hypothesis(int r,int c);
private:
  void warp_hypothesis(const Eigen::Affine3d &transformationx, const cv::Mat1f &new_intensity);
  void warp_maps_forward(const Eigen::Affine3d &transformationx, const cv::Mat1f &old_intensity, const cv::Mat1f &new_intensity);
};
} //namespace sdvo
#endif // DEPTH_HYPOTHESIS_H
