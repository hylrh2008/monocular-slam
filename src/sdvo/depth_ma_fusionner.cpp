#include <sdvo/depth_ma_fusionner.h>

depth_ma_fusionner::depth_ma_fusionner(const cv::Mat & _inverse_depth_observation,
                                       const cv::Mat & _inverse_depth_observation_variance,
                                       const cv::Mat & _inverse_depth_prior,
                                       const cv::Mat & _inverse_depth_prior_variance):
  inverse_depth_observation(_inverse_depth_observation.clone()),
  inverse_depth_observation_variance(_inverse_depth_observation_variance.clone()),
  inverse_depth_prior(_inverse_depth_prior.clone()),
  inverse_depth_prior_variance(_inverse_depth_prior_variance.clone()),
  inverse_depth_posterior(cv::Mat1f::zeros(_inverse_depth_observation.rows,_inverse_depth_observation.cols)),
  inverse_depth_posterior_variance(cv::Mat1f::zeros(_inverse_depth_observation.rows,_inverse_depth_observation.cols))
{
  /*cv::Mat mask_no_prior = cv::Mat((inverse_depth_prior == 0)
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
  inverse_depth_posterior_variance.setTo(cv::Scalar(0),mask_Nan);*/

  for (int r = 0; r < inverse_depth_observation.rows; ++r) {
    for (int c = 0; c < inverse_depth_observation.cols; ++c) {
      assert(inverse_depth_observation(r,c) >= 0);
      assert(inverse_depth_observation_variance(r,c) >= 0);
      assert(inverse_depth_prior(r,c) >= 0);
      assert(inverse_depth_prior_variance(r,c) >= 0);

      float obs=inverse_depth_observation(r,c);
      float obsv=inverse_depth_observation_variance(r,c);

      float prio = inverse_depth_prior(r,c);
      float priov = inverse_depth_prior_variance(r,c);

      bool observ_exist =  obs != 0 && obsv != 0  && std::isfinite(obs) && std::isfinite(obsv);
      bool prior_exist = prio != 0 && priov != 0 && std::isfinite(prio) && std::isfinite(priov);

      if(!observ_exist && prior_exist){
        inverse_depth_posterior(r,c) = prio;
        inverse_depth_posterior_variance(r,c) = priov;
      }
      else if(observ_exist && !prior_exist){
        inverse_depth_posterior(r,c) = obs;
        inverse_depth_posterior_variance(r,c) = obsv;
      }

      else if(!observ_exist && !prior_exist){
        inverse_depth_posterior(r,c) = 0;
        inverse_depth_posterior_variance(r,c) = 0;
      }
      else if(observ_exist && prior_exist){
        inverse_depth_posterior(r,c) = (priov * obs + obsv * prio)/(obsv + priov);
        inverse_depth_posterior_variance(r,c) = (priov * obsv)/(obsv+priov);
      }
      assert(inverse_depth_posterior(r,c) >= 0);
      assert(inverse_depth_posterior_variance(r,c) >= 0);
    }
  }
}
