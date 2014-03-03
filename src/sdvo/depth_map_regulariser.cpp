#include <sdvo/depth_map_regulariser.h>
#undef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif
void
depth_map_regulariser::smooth_map(const cv::Mat1f& in_var, const cv::Mat1f& in)
{
  for (int r = 0; r < in.rows; ++r) {
    for (int c = 0; c < in.cols; ++c) {

      if(r<2 || r >= in.rows-2 || c<2 || c >= in.cols-2){
        out(r,c) = in(r,c);
        out_var(r,c) =  in_var(r,c);
        continue;
      }
      float in_d = in(r,c);
      float in_v = in_var(r,c);

      if(in_d == 0 || in_v == 0) continue;

      std::vector<float> contributing_data;
      contributing_data.reserve(9);
      std::vector<float> contributing_variance;
      contributing_variance.reserve(9);

      for (int y = c-1; y < c+2; ++y) {
        for (int x = r-1; x < r+2; ++x) {
          float in_other_d = in(x,y);
          float in_other_v = in_var(x,y);
          float diff = std::abs(in_d-in_other_d);

          if(diff < 2. * std::sqrt(in_v) &&
             (in_other_d != 0 || in_other_v != 0)){
            contributing_data.push_back(in_other_d);
            contributing_variance.push_back(in_other_v);
          }
        }
      }
      if(contributing_data.size()>=2){
        float ponderate_sum = 0;
        float inverse_variance_sum = 0;
        double min_var = INFINITY;

        for (int i = 0; i < contributing_data.size(); ++i) {
          assert(contributing_data[i] != 0);
          ponderate_sum +=
              contributing_data[i] / contributing_variance[i];
          inverse_variance_sum += 1/contributing_variance[i];
          if(contributing_variance[i]<min_var)
            min_var = contributing_variance[i];
        }

        out(r,c) = ponderate_sum / inverse_variance_sum;
        out_var(r,c) =  min_var;
        assert(out(r,c) >= 0);
        assert(out_var(r,c) >= 0);
      }
      else{
        out(r,c) = 0;
        out_var(r,c) = 0;
      }
    }
  }
}

void depth_map_regulariser::fill_holes()
{
  for (int r = 0; r < tmp.rows; ++r) {
    for (int c = 0; c < tmp.cols; ++c) {

      if(r<2 || r > tmp.rows-2 || c<2 || c>tmp.cols-2){
        out(r,c) = tmp(r,c);
        out_var(r,c) =  tmp_var(r,c);
        continue;
      }

      if(tmp(r,c)==0 || tmp_var(r,c)==0){
        std::vector<float> contributing_data;
        contributing_data.reserve(9);
        std::vector<float> contributing_variance;
        contributing_variance.reserve(9);
        float outlier_proba_neighbour;
        float min_outlier_proba;
        for (int y = c-1; y < c+2; ++y) {
          for (int x = r-1; x < r+2; ++x) {
            float crt_depth = tmp(x,y);
            float crt_depth_var = tmp_var(x,y);
            if(crt_depth==0 || crt_depth_var==0) continue;

            std::vector<float> contributing_data_for_pixel;
            contributing_data_for_pixel.reserve(9);
            std::vector<float> contributing_variance_for_pixel;
            contributing_variance_for_pixel.reserve(9);
            float outlier_proba_neighbour_for_pixel=0;
            float min_outlier_proba_for_pixel;

            for (int y2 = c-1; y2 < c+2; ++y2) {
              for (int x2 = r-1; x2 < r+2; ++x2) {

                float other_depth = tmp(x2,y2);
                float other_depth_var = tmp_var(x2,y2);

                if(other_depth!=0 && other_depth_var!=0){
                  float diff = std::abs(other_depth-crt_depth);

                  if(diff < 2. * std::sqrt(crt_depth_var)){
                    {
                      contributing_data_for_pixel.push_back(other_depth);
                      contributing_variance_for_pixel.push_back(other_depth_var);
                      outlier_proba_neighbour_for_pixel += outlier_proba(x2,y2);
                      if(outlier_proba(x2,y2) < outlier_proba_neighbour_for_pixel)
                        min_outlier_proba = outlier_proba(x2,y2);
                    }
                  }
                }
              }
              if(contributing_data.size() < contributing_data_for_pixel.size()){
                contributing_data.swap(contributing_data_for_pixel);
                contributing_variance.swap(contributing_variance_for_pixel);
                outlier_proba_neighbour = outlier_proba_neighbour_for_pixel;
                min_outlier_proba = min_outlier_proba_for_pixel;
              }
            }
          }
        }
        if(contributing_data.size() >= 4 && outlier_proba_neighbour/contributing_data.size() < 5 ){
          float ponderate_sum = 0;
          float inverse_variance_sum = 0;
          float min_var = INFINITY;

          for (int i = 0; i < contributing_data.size(); ++i) {
            assert(contributing_data[i] > 0);
            assert(contributing_variance[i] > 0);
            ponderate_sum += contributing_data[i] / contributing_variance[i];
            inverse_variance_sum += 1/contributing_variance[i];
            if(contributing_variance[i] < min_var)
              min_var = contributing_variance[i];
          }
          out(r,c) = ponderate_sum / inverse_variance_sum;
          out_var(r,c) = min_var;
          outlier_proba(r,c) = min_outlier_proba;
        }

      }
      else
      {
        out(r,c) = tmp(r,c);
        out_var(r,c) = tmp_var(r,c);
      }
      assert(out(r,c) >= 0);
      assert(out_var(r,c) >= 0);

    }
  }
}

depth_map_regulariser::depth_map_regulariser(const cv::Mat & in,
                                             const cv::Mat & in_var,
                                             const cv::Mat1f & _outlier_proba):

  tmp(cv::Mat1f::zeros(in.size())),
  tmp_var(cv::Mat1f::zeros(in.size())),
  out(cv::Mat1f::zeros(in.size())),
  out_var(cv::Mat1f::zeros(in.size())),
  outlier_proba(_outlier_proba)
{
  smooth_map(in_var, in);
  std::swap(tmp,out);
  std::swap(tmp_var,out_var);
  fill_holes();
}



