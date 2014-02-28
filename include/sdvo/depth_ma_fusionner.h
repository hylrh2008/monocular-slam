#ifndef DEPTH_MA_FUSIONNER_H
#define DEPTH_MA_FUSIONNER_H
#include <opencv2/opencv.hpp>
class depth_ma_fusionner
{
public:
  depth_ma_fusionner(const cv::Mat & _inverse_depth_observation,
                     const cv::Mat & _inverse_depth_observation_variance,
                     const cv::Mat & _inverse_depth_prior,
                     const cv::Mat & _inverse_depth_prior_variance);

  cv::Mat1f get_inverse_depth_posterior(){return inverse_depth_posterior;}
  cv::Mat1f get_inverse_depth_posterior_variance(){return inverse_depth_posterior_variance;}

private:
  cv::Mat1f  inverse_depth_observation;
  cv::Mat1f  inverse_depth_observation_variance;
  cv::Mat1f  inverse_depth_prior;
  cv::Mat1f  inverse_depth_prior_variance;

  cv::Mat1f inverse_depth_posterior;
  cv::Mat1f inverse_depth_posterior_variance;

};

#endif // DEPTH_MA_FUSIONNER_H
