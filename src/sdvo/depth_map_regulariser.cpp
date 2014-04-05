#include <sdvo/depth_map_regulariser.h>
#undef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
#define TRESHOLD_OUTLIER_FOR_REMOVE 0.5
#define TRESHOLD_OUTLIER_FOR_CREATE 0.2

namespace sdvo {
void
depth_map_regulariser::smooth_map()
{  
  std::vector<float> contributing_data;
  contributing_data.reserve(9);
  std::vector<float> contributing_variance;
  contributing_variance.reserve(9);
  std::vector<unsigned char> contributing_age;
  contributing_age.reserve(9);
  std::vector<float> contributing_outlier;
  contributing_outlier.reserve(9);

  for (int r = 0; r < H.d.rows; ++r) {
    for (int c = 0; c < H.d.cols; ++c) {

      if(r<2 || r >= H.d.rows-2 || c<2 || c >= H.d.cols-2){
        tmp(r,c) = (H.d(r,c) != 0)? 1./H.d(r,c) : 0 ;
        tmp_var(r,c) =  H.var(r,c);
        continue;
      }

      float in_inverse_d = H.d(r,c);
      float in_d = (in_inverse_d!=0)? 1./in_inverse_d : 0;
      float in_v = H.var(r,c);
      int in_a = H.age(r,c);
      if(in_d == 0 || in_v == 0) continue;

      contributing_data.clear();
      contributing_variance.clear();
      contributing_age.clear();
      contributing_outlier.clear();

      for (int y = c-1; y < c+2; ++y) {
        for (int x = r-1; x < r+2; ++x) {
          float in_other_inverse_d = H.d(x,y);
          float in_other_d =  (in_other_inverse_d!=0)? 1./in_other_inverse_d : 0 ;
          float in_other_v = H.var(x,y);
          unsigned char in_other_a = H.age(x,y);
          float in_other_o = H.outlier_probability(x,y);

          float diff = std::abs(in_d-in_other_d);

          if(diff < 2. * std::sqrt(std::min(in_v,in_other_v)) &&
             (in_other_d != 0 || in_other_v != 0)){
            contributing_data.push_back(in_other_d);
            contributing_variance.push_back(in_other_v);
            contributing_age.push_back(in_other_a);
            contributing_outlier.push_back(in_other_o);
          }
        }
      }

      float product_proba_outlier=1;
      for(int i=0; i < contributing_outlier.size(); i++){
        product_proba_outlier *= contributing_outlier[i];
      }
      if(contributing_data.size() >= 3 &&
         product_proba_outlier < TRESHOLD_OUTLIER_FOR_REMOVE)
      {
        float ponderate_sum = 0;
        float inverse_variance_sum = 0;
        double min_var = INFINITY;
        unsigned char min_age = INFINITY;
        for (int i = 0; i < contributing_data.size(); ++i) {
          assert(contributing_data[i] != 0);
          ponderate_sum +=
              contributing_data[i] / contributing_variance[i];

          inverse_variance_sum +=
              1/contributing_variance[i];

          if(contributing_variance[i]<min_var)
            min_var = contributing_variance[i];
          if(contributing_age[i]<min_age && contributing_age[i]>0)
            min_age = contributing_age[i];
        }

        tmp(r,c) = 1./(ponderate_sum / inverse_variance_sum);
        if(isinf(tmp(r,c))) tmp(r,c)=0;
        tmp_var(r,c) = H.var(r,c);// On garde la variance originale  //min_var;
        tmp_age(r,c) = H.age(r,c);
        assert(tmp(r,c) >= 0);
        assert(tmp_var(r,c) >= 0);
      }
      else
      {
        H.outlier_probability(r,c)++;
      }
    }
  }
  swap(tmp,H.d);
  swap(tmp_var,H.var);
  swap(tmp_age,H.age);
}

void depth_map_regulariser::fill_holes()
{
  std::vector<float> contributing_data;
  contributing_data.reserve(9);
  std::vector<float> contributing_variance;
  contributing_variance.reserve(9);
  std::vector<unsigned char> contributing_age;
  contributing_age.reserve(9);

  std::vector<float> contributing_data_for_pixel;
  contributing_data_for_pixel.reserve(9);
  std::vector<float> contributing_variance_for_pixel;
  contributing_variance_for_pixel.reserve(9);
  std::vector<unsigned char> contributing_age_for_pixel;
  contributing_age_for_pixel.reserve(9);

  for (int r = 0; r < tmp.rows; ++r) {
    for (int c = 0; c < tmp.cols; ++c) {

      if(r<2 || r > H.d.rows-2 || c<2 || c>H.d.cols-2){
        tmp(r,c) = (H.d(r,c) != 0)? 1./H.d(r,c) : 0 ;
        tmp_var(r,c) =  H.var(r,c);
        continue;
      }

      if(H.d(r,c)==0 || H.var(r,c)==0){
        contributing_data.clear();
        contributing_variance.clear();
        contributing_age.clear();


        float product_proba_outlier=1;


        for (int y = c-1; y < c+2; ++y) {
          for (int x = r-1; x < r+2; ++x) {

            float crt_depth_inverse = H.d(x,y);
            float crt_depth =  (crt_depth_inverse!=0)? 1./crt_depth_inverse : 0 ;

            float crt_depth_var = H.var(x,y);
            if(crt_depth==0 || crt_depth_var==0) continue;

            contributing_data_for_pixel.clear();
            contributing_variance_for_pixel.clear();
            contributing_age_for_pixel.clear();

            float outlier_proba_neighbour_for_pixel = 1;

            for (int y2 = c-1; y2 < c+2; ++y2) {
              for (int x2 = r-1; x2 < r+2; ++x2) {

                float other_depth_inverse = H.d(x2,y2);
                float other_depth =  (other_depth_inverse!=0)? 1./other_depth_inverse : 0 ;

                float other_depth_var = H.var(x2,y2);
                unsigned char other_age = H.age(x2,y2);
                if(other_depth!=0 && other_depth_var!=0){
                  float diff = std::abs(other_depth-crt_depth);

                  if(diff < 2. * std::sqrt(std::min(other_depth_var,crt_depth_var))){
                    {
                      contributing_data_for_pixel.push_back(other_depth);
                      contributing_variance_for_pixel.push_back(other_depth_var);
                      contributing_age_for_pixel.push_back(other_age);
                      outlier_proba_neighbour_for_pixel *= H.outlier_probability(x2,y2);
                    }
                  }
                }
              }
              if(contributing_data.size() < contributing_data_for_pixel.size()){
                contributing_data.swap(contributing_data_for_pixel);
                contributing_variance.swap(contributing_variance_for_pixel);
                contributing_age.swap(contributing_age_for_pixel);
                product_proba_outlier = outlier_proba_neighbour_for_pixel;
              }
            }
          }
        }

        if(contributing_data.size() > 4 &&
           product_proba_outlier < TRESHOLD_OUTLIER_FOR_CREATE ){
          float ponderate_sum = 0;
          float inverse_variance_sum = 0;
          float min_var = INFINITY;
          unsigned char min_age = INFINITY;

          for (int i = 0; i < contributing_data.size(); ++i) {
            assert(contributing_data[i] > 0);
            assert(contributing_variance[i] > 0);
            ponderate_sum += contributing_data[i] / contributing_variance[i];
            inverse_variance_sum += 1/contributing_variance[i];
            if(contributing_age[i] < min_age && contributing_age[i] > 0)
              min_age = contributing_age[i];
            if(contributing_variance[i] < min_var)
              min_var = contributing_variance[i];
          }
          tmp(r,c) = 1./(ponderate_sum / inverse_variance_sum);
          if(isinf(tmp(r,c))) tmp(r,c)=0;
          tmp_var(r,c) = min_var;
          tmp_age(r,c) = min_age;
        }

      }
      else
      {
        tmp(r,c) = H.d(r,c);
        tmp_var(r,c) = H.var(r,c);
        tmp_age(r,c) = H.age(r,c);
      }
      assert(tmp(r,c) >= 0);
      assert(tmp_var(r,c) >= 0);

    }
  }
  swap(tmp,H.d);
  swap(tmp_var,H.var);
  swap(tmp_age,H.age);

}

depth_map_regulariser::depth_map_regulariser(depth_hypothesis * _H):
  H(reinterpret_cast<depth_hypothesis&>(*_H)),
  tmp(cv::Mat1f::zeros(_H->d.size())),
  tmp_var(cv::Mat1f::zeros(_H->d.size())),
  tmp_age(cv::Mat1b::zeros(_H->d.size()))
{
  smooth_map();
  fill_holes();
}
}
