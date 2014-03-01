#include <sdvo/depth_hypothesis.h>
namespace sdvo{
depth_hypothesis::depth_hypothesis(const cv::Mat1f & depth_init,
                                   const cv::Mat1f & variance_init,
                                   const cv::Mat1f & intensity_img,
                                   float fx,float fy,float ox, float oy):
  d(depth_init.clone()),
  variance(variance_init.clone()),
  intensity_img(intensity_img.clone()),
  fx(fx),
  fy(fy),
  ox(ox),
  oy(oy),
  precise_position(cv::Mat2f::zeros(depth_init.size())),
  outlier_probability(cv::Mat1f::zeros(depth_init.size())),
  height(depth_init.rows),
  width(depth_init.cols){}

void depth_hypothesis::update_hypothesis(const Eigen::Affine3d &transformationx,
                                         const cv::Mat1f & new_intensity)
{
  warp_hypothesis(transformationx,new_intensity);
  intensity_img=new_intensity;
}

void
depth_hypothesis::regularise_hypothesis()
{
  depth_map_regulariser regularise(1./d,variance);
  d = 1./regularise.get_inverse_depth_regularised();
  variance = regularise.get_inverse_depth_regularised_variance();

}
void depth_hypothesis::check_gradient_norm(){
  //TODO;
}

void
depth_hypothesis::add_observation_to_hypothesis(const cv::Mat1f depth_obs,
                                                     const cv::Mat1f var_obs)
{
  depth_ma_fusionner fusion(1./depth_obs,var_obs,1./d,variance);
  depth_map_regulariser
      regularise(fusion.get_inverse_depth_posterior(),
                 fusion.get_inverse_depth_posterior_variance());
  d = 1./regularise.get_inverse_depth_regularised();
  variance = regularise.get_inverse_depth_regularised_variance();

}

void depth_hypothesis::warp_hypothesis(const Eigen::Affine3d &transformationx,
                                       const cv::Mat1f & new_intensity)
{
  warp_maps_forward(transformationx);
  warp_depth_forward(transformationx,intensity_img,new_intensity);
}


void
depth_hypothesis::warp_maps_forward(const Eigen::Affine3d& transformationx)
{

  bool identity = transformationx.affine().isIdentity(1e-6);

  cv::Mat1f warped_variance(cv::Mat1f::zeros(variance.size()));
  cv::Mat1f warped_outliers_proba(cv::Mat1f::zeros(outlier_probability.size()));

  const float* depth_ptr = d.ptr<float>();

  float x_precise;
  float y_precise;

  for(size_t y = 0; y < height; ++y)
  {
    for(size_t x = 0; x < width; ++x, ++depth_ptr)
    {
      if(*depth_ptr <= 1e-6f) continue;


      if(precise_position(y,x)[0]!=0 && precise_position(y,x)[1]!=0)
      {
        x_precise = precise_position(y,x)[0];
        y_precise = precise_position(y,x)[1];
      }
      else{
        x_precise = x;
        y_precise = y;
      }
      float depth = *depth_ptr;
      Eigen::Vector3d p3d((x_precise - ox) * depth / fx,
                          (y_precise - oy) * depth / fy,
                          depth);

      if(!identity)
      {
        Eigen::Vector3d p3d_transformed = transformationx * p3d;
        if(p3d_transformed(2) < 0) continue;

        float x_projected =
            (float) (p3d_transformed(0) * fx / p3d_transformed(2) + ox);

        float y_projected =
            (float) (p3d_transformed(1) * fy / p3d_transformed(2) + oy);

        if(0 <= x_projected && x_projected<width &&
           0 <= y_projected && y_projected<height)
        {
          int xp, yp;
          xp = (int) std::floor(x_projected);
          yp = (int) std::floor(y_projected);

          warped_variance(yp, xp) = std::pow(p3d_transformed(2)/p3d(2),int(4)) * variance.at<float>(y, x) + 0.00001;
          warped_outliers_proba(yp,xp) = outlier_probability.at<float>(y,x);
        }

        p3d = p3d_transformed;
      }
    }
  }

  if(identity)
  {
    std::swap(variance,warped_variance);
    std::swap(outlier_probability,warped_outliers_proba);

  }

  std::swap(variance,warped_variance);
  std::swap(outlier_probability,warped_outliers_proba);

}

void
depth_hypothesis::warp_depth_forward(
    const Eigen::Affine3d& transformationx,
    const cv::Mat1f & old_intensity,
    const cv::Mat1f & new_intensity)
{
  cv::Mat1f warped_mat(cv::Mat::zeros(d.size(), d.type()));
  cv::Mat2f next_precise_pos(cv::Mat::zeros(d.size(), d.type()));

  const float* depth_ptr = d.ptr<float>();
  float x_precise;
  float y_precise;
  for(size_t y = 0; y < d.rows; ++y)
  {
    for(size_t x = 0; x < d.cols; ++x, ++depth_ptr)
    {
      if(*depth_ptr==0)
      {
        continue;
      }

      if(precise_position(y,x)[0]!=0 && precise_position(y,x)[1]!=0)
      {
        x_precise = precise_position(y,x)[0];
        y_precise = precise_position(y,x)[1];
      }
      else{
        x_precise = x;
        y_precise = y;
      }
      float depth = *depth_ptr;

      Eigen::Vector3d p3d((x_precise - ox) * depth / fx, (y_precise - oy) * depth / fy, depth);
      Eigen::Vector3d p3d_transformed = transformationx * p3d;
      if(p3d_transformed(2)<0) continue;
      float x_projected = (float) (p3d_transformed(0) * fx / p3d_transformed(2) + ox);
      float y_projected = (float) (p3d_transformed(1) * fy / p3d_transformed(2) + oy);

      if(0<x_projected && x_projected<d.cols &&
         0<y_projected && y_projected<d.rows)
      {
        int yi = (int) y_projected, xi = (int) x_projected;
        if(warped_mat.at<float>(yi, xi) == 0 || warped_mat.at<float>(yi, xi) > depth + 0.05){

          next_precise_pos(yi,xi) = cv::Vec2f(x_projected,y_projected);

          if(abs(old_intensity(y,x)-new_intensity(yi,xi))>50){
            outlier_probability.at<float>(yi, xi)++;
          }
            warped_mat.at<float>(yi, xi) = p3d_transformed(2);
        }
      }
      p3d = p3d_transformed;
    }
  }
  std::swap(d,warped_mat);
  std::swap(precise_position,next_precise_pos);
}
} //namespace sdvo
