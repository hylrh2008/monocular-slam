#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sdvo/depth_ma_fusionner.h>

namespace sdvo
{
TEST(depth_map_fusionner_test, opencvTests){
  float m1[25] = {4, 4, 4, 4, 4,
    4, 4, 4, 4, 4,
    4, 4, 0, 4, 4,
    4, 4, 4, 4, 4,
    4, 4, 4, 4, 4};
  float m2[25] = {4, 4, 4, 4, 4,
      4, 4, 4, 4, 4,
      4, 4, 0, 4, 4,
      4, 4, 4, 4, 4,
      4, 4, 4, 4, 4};
      cv::Mat1f M1_ = cv::Mat1f(5,5,m1);
      cv::Mat1f M2_ = cv::Mat1f(5,5,m2);

      cv::Mat1f M3 = M1_/M2_;
      std::cerr<<M3<<std::endl;

       std::cerr<<"TOTO"<<(0*NAN)<<std::endl;
}

TEST(depth_map_fusionner_test, depth_map_fusionner_test_no_observation_variance)
{
  float data_inverse_depth_prior[2] = {1,1};
  float data_variance_prior[2] = {2,2};

  float data_inverse_depth_obs[2] = {1,1};
  float data_variance2_obs[2] = {NAN,2};

  cv::Mat1f inverse_depth_prior = cv::Mat1f(2,1,data_inverse_depth_prior);
  cv::Mat1f variance_prior = cv::Mat1f(2,1,data_variance_prior);
  cv::Mat1f inverse_depth_obs = cv::Mat1f(2,1,data_inverse_depth_obs);
  cv::Mat1f variance_obs = cv::Mat1f(2,1,data_variance2_obs) ;

  depth_ma_fusionner fusion(inverse_depth_obs,variance_obs,inverse_depth_prior,variance_prior);
  cv::Mat1f posterior = fusion.get_inverse_depth_posterior();
  cv::Mat1f variance = fusion.get_inverse_depth_posterior_variance();
  std::cerr<<posterior<<std::endl;
  std::cerr<<variance<<std::endl;

  EXPECT_EQ(posterior(0,0),1);
  EXPECT_EQ(variance(0,0),2);
  EXPECT_EQ(posterior(0,1),1);
  EXPECT_EQ(variance(0,1),1);
}

TEST(depth_map_fusionner_test, depth_map_fusionner_test_no_observation_no_prior_variance)
{
  float data_inverse_depth_prior[25] = {1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1};
  float data_variance_prior[25] = {2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,0,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2};
  float data_inverse_depth_obs[25] = {1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1};
  float data_variance2_obs[25] = {2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2};

  cv::Mat1f inverse_depth_prior = cv::Mat1f(5,5,data_inverse_depth_prior);
  cv::Mat1f variance_prior = cv::Mat1f(5,5,data_variance_prior);
  cv::Mat1f inverse_depth_obs = cv::Mat1f(5,5,data_inverse_depth_obs);
  cv::Mat1f variance_obs = cv::Mat1f(5,5,data_variance2_obs) ;

  depth_ma_fusionner fusion(inverse_depth_obs,variance_obs,inverse_depth_prior,variance_prior);
  cv::Mat1f posterior = fusion.get_inverse_depth_posterior();
  cv::Mat1f variance = fusion.get_inverse_depth_posterior_variance();
  //std::cerr<<posterior<<std::endl;
  //std::cerr<<variance<<std::endl;

  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      if(x==2 && y==2){
        EXPECT_TRUE(posterior(x,y)==1);
        EXPECT_TRUE(variance(x,y)==2);
      }
      else{
        EXPECT_TRUE(posterior(x,y)==1);
        EXPECT_TRUE(variance(x,y)==1);
      }
    }
  }

}

TEST(depth_map_fusionner_test, depth_map_fusionner_test_no_observation_no_prior)
{
  float data_inverse_depth_prior[25] = {1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,0,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1};
  float data_variance_prior[25] = {2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2};
  float data_inverse_depth_obs[25] = {1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,0,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1};
  float data_variance2_obs[25] = {2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2};

  cv::Mat1f inverse_depth_prior = cv::Mat1f(5,5,data_inverse_depth_prior);
  cv::Mat1f variance_prior = cv::Mat1f(5,5,data_variance_prior);
  cv::Mat1f inverse_depth_obs = cv::Mat1f(5,5,data_inverse_depth_obs);
  cv::Mat1f variance_obs = cv::Mat1f(5,5,data_variance2_obs) ;

  depth_ma_fusionner fusion(inverse_depth_obs,variance_obs,inverse_depth_prior,variance_prior);
  cv::Mat1f posterior = fusion.get_inverse_depth_posterior();
  cv::Mat1f variance = fusion.get_inverse_depth_posterior_variance();
  //std::cerr<<posterior<<std::endl;
  //std::cerr<<variance<<std::endl;

  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      if(x==2 && y==2){
        EXPECT_EQ(posterior(x,y),0);
        EXPECT_EQ(variance(x,y),0);
      }
      else{
        EXPECT_EQ(posterior(x,y),1);
        EXPECT_EQ(variance(x,y),1);
      }
    }
  }
}

TEST(depth_map_fusionner_test, depth_map_fusionner_test_no_observation)
{
  float data_inverse_depth_prior[25] = {1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1};
  float data_variance_prior[25] = {2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2};
  float data_inverse_depth_obs[25] = {1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,0,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1};
  float data_variance2_obs[25] = {2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2};

  cv::Mat1f inverse_depth_prior = cv::Mat1f(5,5,data_inverse_depth_prior);
  cv::Mat1f variance_prior = cv::Mat1f(5,5,data_variance_prior);
  cv::Mat1f inverse_depth_obs = cv::Mat1f(5,5,data_inverse_depth_obs);
  cv::Mat1f variance_obs = cv::Mat1f(5,5,data_variance2_obs) ;

  depth_ma_fusionner fusion(inverse_depth_obs,variance_obs,inverse_depth_prior,variance_prior);
  cv::Mat1f posterior = fusion.get_inverse_depth_posterior();
  cv::Mat1f variance = fusion.get_inverse_depth_posterior_variance();
  //std::cerr<<posterior<<std::endl;
  //std::cerr<<variance<<std::endl;

  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      if(x==2 && y==2){
        EXPECT_EQ(posterior(x,y),1);
        EXPECT_EQ(variance(x,y),2);
      }
      else{
        EXPECT_EQ(posterior(x,y),1);
        EXPECT_EQ(variance(x,y),1);
      }
    }
  }
}

TEST(depth_map_fusionner_test, depth_map_fusionner_test_no_prior)
{
  float data_inverse_depth_prior[25] = {1,1,1,1,1,
                                       1,1,1,1,1,
                                       1,1,0,1,1,
                                       1,1,1,1,1,
                                       1,1,1,1,1};
  float data_variance_prior[25] = {2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2,
                                  2,2,2,2,2};
  float data_inverse_depth_obs[25] = {1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1,
                                     1,1,1,1,1};
  float data_variance2_obs[25] = {2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2,
                                 2,2,2,2,2};

  cv::Mat1f inverse_depth_prior = cv::Mat1f(5,5,data_inverse_depth_prior);
  cv::Mat1f variance_prior = cv::Mat1f(5,5,data_variance_prior);
  cv::Mat1f inverse_depth_obs = cv::Mat1f(5,5,data_inverse_depth_obs);
  cv::Mat1f variance_obs = cv::Mat1f(5,5,data_variance2_obs) ;

  depth_ma_fusionner fusion(inverse_depth_obs,variance_obs,inverse_depth_prior,variance_prior);
  cv::Mat1f posterior = fusion.get_inverse_depth_posterior();
  cv::Mat1f variance = fusion.get_inverse_depth_posterior_variance();
  //std::cerr<<posterior<<std::endl;
  //std::cerr<<variance<<std::endl;

  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      if(x==2 && y==2){
        EXPECT_EQ(posterior(x,y),1);
        EXPECT_EQ(variance(x,y),2);
      }
      else{
        EXPECT_EQ(posterior(x,y),1);
        EXPECT_EQ(variance(x,y),1);
      }
    }
  }
}
} //sdvo
int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc,argv);
  return RUN_ALL_TESTS();
}
