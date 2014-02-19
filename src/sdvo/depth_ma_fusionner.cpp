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

  for (int c = 0; c < inverse_depth_observation.rows; ++c) {
    for (int r = 0; r < inverse_depth_observation.cols; ++r) {
      float obs=inverse_depth_observation(c,r);
      float obsv=inverse_depth_observation_variance(c,r);

      float prio = inverse_depth_prior(c,r);
      float priov = inverse_depth_prior_variance(c,r);
      bool observ_exist = obs != 0 && obsv != 0;
      bool prior_exist = prio != 0 && priov != 0;

      if(!observ_exist && prior_exist){
        inverse_depth_posterior(c,r) = prio;
        inverse_depth_posterior_variance(c,r) = priov;
      }
      else if(observ_exist && !prior_exist){
        inverse_depth_posterior(c,r) = obs;
        inverse_depth_posterior_variance(c,r) = obsv;
      }

      else if(!observ_exist && !prior_exist){
        inverse_depth_posterior(c,r) = 0;
        inverse_depth_posterior_variance(c,r) = 0;
      }
      else if(observ_exist && prior_exist){
        inverse_depth_posterior(c,r) = (priov * obs + obsv * prio)/(obsv + priov);
        inverse_depth_posterior_variance(c,r) = (priov * obsv)/(obsv+priov);
      }
    }
  }
  cv::Mat1f tmp=inverse_depth_posterior.clone();
  cv::Mat1f tmpv=inverse_depth_posterior_variance.clone();

  for (int c = 1; c < inverse_depth_observation.rows-1; ++c) {
    for (int r = 1; r < inverse_depth_observation.cols-1; ++r) {
      float depth = tmp(c,r);
      float depthv = tmpv(c,r);

      if(abs(tmp(c-1,r+0)-depth) < 2*depthv && tmp(c-1,r+0)!=0 &&
         abs(tmp(c+1,r+0)-depth) < 2*depthv && tmp(c+1,r+0)!=0 &&
         abs(tmp(c+0,r+1)-depth) < 2*depthv && tmp(c-0,r+1)!=0 &&
         abs(tmp(c+0,r-1)-depth) < 2*depthv && tmp(c-0,r-1)!=0)
      {
         inverse_depth_posterior(c,r) =
             (1.f/tmpv(c-1,r+0)*tmp(c-1,r+0) + 1.f/tmpv(c+1,r+0)*tmp(c+1,r+0) +
             1.f/tmpv(c-0,r+1)*tmp(c-0,r+1) +1.f/tmpv(c-0,r-1)*tmp(c-0,r-1))
             /
            (1.f/tmpv(c-1,r+0) + 1.f/tmpv(c+1,r+0) + 1.f/tmpv(c-0,r+1) + 1.f/tmpv(c-0,r-1));

         inverse_depth_posterior_variance(c,r) =
             1./(1.f/tmpv(c-1,r+0) + 1.f/tmpv(c+1,r+0) + 1.f/tmpv(c-0,r+1) + 1.f/tmpv(c-0,r-1));


      }
    }
  }
}
