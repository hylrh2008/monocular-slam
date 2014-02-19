#include <sdvo/depth_ma_fusionner.h>

depth_ma_fusionner::depth_ma_fusionner(const cv::Mat & _inverse_depth_observation,
                                       const cv::Mat & _inverse_depth_observation_variance,
                                       const cv::Mat & _inverse_depth_prior,
                                       const cv::Mat & _inverse_depth_prior_variance):
  inverse_depth_observation(_inverse_depth_observation.clone()),
  inverse_depth_observation_variance(_inverse_depth_observation_variance.clone()),
  inverse_depth_prior(_inverse_depth_prior.clone()),
  inverse_depth_prior_variance(_inverse_depth_prior_variance.clone())
{
  cv::Mat mask_no_prior = cv::Mat((inverse_depth_prior == 0)
                                  | (inverse_depth_prior_variance == 0)) & 1 ;
  cv::Mat mask_prior = (~mask_no_prior & 1);
  cv::Mat mask_no_observation = cv::Mat((inverse_depth_observation == 0 )
                                  | (inverse_depth_observation_variance == 0)) & 1 ;

  cv::Mat mask_observation = ~mask_no_observation & 1;

  inverse_depth_prior.setTo(cv::Scalar(0),mask_no_prior);
  inverse_depth_prior_variance.setTo(cv::Scalar(0),mask_no_prior);

  inverse_depth_observation.setTo(cv::Scalar(0),mask_no_observation);
  inverse_depth_observation_variance.setTo(cv::Scalar(0),mask_no_observation);

  inverse_depth_observation.setTo(cv::Scalar(0),mask_no_observation & mask_no_prior);
  inverse_depth_observation_variance.setTo(cv::Scalar(0),mask_no_observation & mask_no_prior);

  cv::Mat tmp =
      inverse_depth_prior_variance.mul(inverse_depth_observation) +
      inverse_depth_observation_variance.mul(inverse_depth_prior);

  inverse_depth_posterior
      =  tmp/
      (inverse_depth_observation_variance+inverse_depth_prior_variance);
  //std::cerr<<(inverse_depth_observation_variance+inverse_depth_prior_variance)<<std::endl;

  inverse_depth_posterior_variance = inverse_depth_prior_variance.mul(inverse_depth_observation_variance) /
      (inverse_depth_observation_variance+inverse_depth_prior_variance);

  // Si on avait pas de prior on garde juste l'observation

  inverse_depth_posterior = cv::Mat1f(mask_no_prior & mask_observation).mul(inverse_depth_observation)
      + cv::Mat1f(mask_prior).mul(inverse_depth_posterior);

  inverse_depth_posterior_variance = cv::Mat1f(mask_no_prior & mask_observation).mul(inverse_depth_observation_variance)
      + cv::Mat1f(mask_prior).mul(inverse_depth_posterior_variance);

  // Si on avait pas de observation on garde juste le prior

  inverse_depth_posterior = cv::Mat1f(mask_no_observation & mask_prior).mul(inverse_depth_prior)
      + cv::Mat1f(mask_observation).mul(inverse_depth_posterior);

  inverse_depth_posterior_variance = cv::Mat1f(mask_no_observation & mask_prior).mul(inverse_depth_prior_variance)
      + cv::Mat1f(mask_observation).mul(inverse_depth_posterior_variance);

  cv::Mat mask_Nan = cv::Mat((inverse_depth_posterior!=inverse_depth_posterior));
  inverse_depth_posterior.setTo(cv::Scalar(0),mask_Nan);
  inverse_depth_posterior_variance.setTo(cv::Scalar(0),mask_Nan);

}
