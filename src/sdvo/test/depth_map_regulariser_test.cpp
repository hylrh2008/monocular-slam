#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sdvo/depth_map_regulariser.h>

using namespace std;
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
}
TEST(depth_map_regulariser_test, depth_map_regulariser_test_nominal)
{
  float m[25] = {1,1,1,1,1,
                 1,1,1,1,1,
                 1,1,0,1,1,
                 1,1,1,1,1,
                 1,1,1,1,1};
  float d[25] = {2,2,2,2,2,
                 2,2,2,2,2,
                 2,2,0,2,2,
                 2,2,2,2,2,
                 2,2,2,2,2};


  cv::Mat1f M = cv::Mat1f(5,5,m);
  cv::Mat1f D = cv::Mat1f(5,5,d);
  cv::Mat1f P = cv::Mat1f::zeros(5,5);
  depth_map_regulariser regularise(M,D,P);

  cv::Mat1f rd = regularise.get_inverse_depth_regularised();
  cv::Mat1f rv = regularise.get_inverse_depth_regularised_variance();

   for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      if(x==2 && y==2){
        EXPECT_EQ(rd(x,y),1);
        EXPECT_EQ(rv(x,y),2);
      }
      else{
        EXPECT_EQ(rd(x,y),1);
        EXPECT_EQ(rv(x,y),2);
      }
    }
  }
}

TEST(depth_map_regulariser_test, depth_map_regulariser_test_one_outlier)
{
  float m[25] = {1,1,1,1,1,
                 1,1,1,1,1,
                 1,4,0,1,1,
                 1,1,1,1,1,
                 1,1,1,1,1};
  float d[25] = {2,2,2,2,2,
                 2,2,2,2,2,
                 2,1,0,2,2,
                 2,2,2,2,2,
                 2,2,2,2,2};


  cv::Mat1f M = cv::Mat1f(5,5,m);
  cv::Mat1f D = cv::Mat1f(5,5,d);
  cv::Mat1f P = cv::Mat1f::zeros(5,5);
  depth_map_regulariser regularise(M,D,P);

  cv::Mat1f rd = regularise.get_inverse_depth_regularised();
  cv::Mat1f rv = regularise.get_inverse_depth_regularised_variance();

  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      if(x==2 && y==1){
        EXPECT_EQ(rd(x,y),4);
        EXPECT_EQ(rv(x,y),1);
      }
      else{
        EXPECT_EQ(rd(x,y),1);
        EXPECT_EQ(rv(x,y),2);
      }
    }
  }
}

TEST(depth_map_regulariser_test, depth_map_regulariser_test_two_outlier)
{
  float m[25] = {1,1,1,1,1,
                 1,1,1,1,1,
                 1,4,0,4,1,
                 1,1,1,1,1,
                 1,1,1,1,1};
  float d[25] = {2,2,2,2,2,
                 2,2,2,2,2,
                 2,1,0,1,2,
                 2,2,2,2,2,
                 2,2,2,2,2};


  cv::Mat1f M = cv::Mat1f(5,5,m);
  cv::Mat1f D = cv::Mat1f(5,5,d);
  cv::Mat1f P = cv::Mat1f::zeros(5,5);
  depth_map_regulariser regularise(M,D,P);

  cv::Mat1f rd = regularise.get_inverse_depth_regularised();
  cv::Mat1f rv = regularise.get_inverse_depth_regularised_variance();

  cerr<<rd<<endl;
  cerr<<rv<<endl;
  for (int x = 0; x < 5; ++x) {
    for (int y = 0; y < 5; ++y) {
      if(x==2 && (y==1 || y==3)){
        EXPECT_EQ(rd(x,y),4);
        EXPECT_EQ(rv(x,y),1);
      }
      else{
        EXPECT_EQ(rd(x,y),1);
        EXPECT_EQ(rv(x,y),2);
      }
    }
  }
}
TEST(depth_map_regulariser_test, depth_map_regulariser_test_two_outlier_big_mat)
{
  float m[49] = {1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,
                 1,1,4,0,4,1,1,
                 1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1};
  float d[49] = {2,2,2,2,2,2,2,
                 2,2,2,2,2,2,2,
                 2,2,2,2,2,2,2,
                 2,2,1,0,1,2,2,
                 2,2,2,2,2,2,2,
                 2,2,2,2,2,2,2,
                 2,2,2,2,2,2,2};


  cv::Mat1f M = cv::Mat1f(7,7,m);
  cv::Mat1f D = cv::Mat1f(7,7,d);

  cv::Mat1f P = cv::Mat1f::zeros(7,7);
  depth_map_regulariser regularise(M,D,P);

  cv::Mat1f rd = regularise.get_inverse_depth_regularised();
  cv::Mat1f rv = regularise.get_inverse_depth_regularised_variance();

  for (int x = 0; x < 7; ++x) {
    for (int y = 0; y < 7; ++y) {
      if(x==3 && (y==2 || y==4)){
        EXPECT_EQ(rd(x,y),4);
        EXPECT_EQ(rv(x,y),1);
      }
      else{
        EXPECT_EQ(rd(x,y),1);
        EXPECT_EQ(rv(x,y),2);
      }
    }
  }
}
TEST(depth_map_regulariser_test, depth_map_regulariser_test_try)
{
  float m[49] = {1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1,
                 1,1,1,4,1,1,1,
                 1,1,4,0,4,1,1,
                 1,1,4,4,1,1,1,
                 1,1,1,1,1,1,1,
                 1,1,1,1,1,1,1};
  float d[49] = {2,2,2,2,2,2,2,
                 2,2,2,2,2,2,2,
                 2,2,1,2,2,2,2,
                 2,2,1,0,1,2,2,
                 2,2,1,1,2,2,2,
                 2,2,2,2,2,2,2,
                 2,2,2,2,2,2,2};


  cv::Mat1f M = cv::Mat1f(7,7,m);
  cv::Mat1f D = cv::Mat1f(7,7,d);
  cv::Mat1f P = cv::Mat1f::zeros(7,7);
  depth_map_regulariser regularise(M,D,P);

  cv::Mat1f rd = regularise.get_inverse_depth_regularised();
  cv::Mat1f rv = regularise.get_inverse_depth_regularised_variance();

  cerr<<rd<<endl;
  cerr<<rv<<endl;
}
} //sdvo
int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc,argv);
  return RUN_ALL_TESTS();
}

