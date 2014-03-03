#ifndef DEPTH_MAP_REGULARISER_H
#define DEPTH_MAP_REGULARISER_H
#include <opencv2/opencv.hpp>
class depth_map_regulariser
{
public:
  depth_map_regulariser(const cv::Mat & in, const cv::Mat & in_var,const cv::Mat1f & outliersProba);
  cv::Mat1f get_inverse_depth_regularised(){return out;}
  cv::Mat1f get_inverse_depth_regularised_variance(){return out_var;}

private:
  cv::Mat1f tmp;
  cv::Mat1f tmp_var;

  cv::Mat1f out;
  cv::Mat1f out_var;

  cv::Mat1f outlier_proba;
  void smooth_map(const cv::Mat1f &in_var, const cv::Mat1f &in);
  void fill_holes();
};

#endif // DEPTH_MAP_REGULARISER_H
