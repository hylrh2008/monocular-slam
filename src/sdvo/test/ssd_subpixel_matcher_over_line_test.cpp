#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <sdvo/ssd_subpixel_matcher_over_line.h>


std::string test_directory;
std::string data_path;


namespace sdvo
{

TEST(ssd_subpixel_matcher_over_line, ssd_subpixel_matcher_over_line_test_diag)
{
  cv::Mat1f m1 = cv::Mat1f::zeros(11,11) ;
  cv::Mat1f m2 = cv::Mat1f::zeros(11,11);

  m1(4,4)=2;
  m1(5,5)=1;
  m1(6,6)=2;

  m2(4,4)=2;
  m2(5,5)=1;
  m1(6,6)=2;

  SSD_Subpixel_Matcher_Over_Line ssd_matcher(m1,m2,cv::Point2d(5,5),
                                             cv::Point2d(0,0),cv::Point2d(10,10),
                                             cv::Vec2d(sqrt(2)/2.,sqrt(2)/2.),
                                             cv::Vec2d(sqrt(2)/2.,sqrt(2)/2.),
                                             0.1);
  double error = ssd_matcher.get_error();
  cv::Point2d match = ssd_matcher.getMatch_point();
  std::cout<<error<<" "<<match<<std::endl;
  ASSERT_TRUE(abs(match.y-5)<2 && abs(match.x-5)<2);
}

TEST(ssd_subpixel_matcher_over_line, ssd_subpixel_matcher_over_line_test_vertical)
{

  float m1_data[9] = {0,0,0,0,1,0,0,0,0};
  float m2_data[9] = {0,0,0,0,0,1,0,0,0};
  cv::Mat1f m1(9,1,m1_data);
  cv::Mat1f m2(9,1,m2_data);

  SSD_Subpixel_Matcher_Over_Line ssd_matcher(m1,m2,
                                             cv::Point2d(0,4),
                                             cv::Point2d(0,0),cv::Point2d(0,8),
                                             cv::Vec2d(0,1),cv::Vec2d(0,1));
  double error = ssd_matcher.get_error();
  cv::Point2d match = ssd_matcher.getMatch_point();
  std::cout<<error<<" "<<match<<std::endl;
  ASSERT_TRUE(abs(match.y-5)<1 && abs(match.x)<1);
}

TEST(ssd_subpixel_matcher_over_line, ssd_subpixel_matcher_over_line_test_horizontal)
{

  float m1_data[9] = {0,0,0,0,1,0,0,0,0};
  float m2_data[9] = {0,0,0,0,0,1,0,0,0};
  cv::Mat1f m1(1,9,m1_data);
  cv::Mat1f m2(1,9,m2_data);

  SSD_Subpixel_Matcher_Over_Line ssd_matcher(m1,m2,
                                             cv::Point2d(4,0),
                                             cv::Point2d(0,0),cv::Point2d(8,0),
                                             cv::Vec2d(1,0),cv::Vec2d(1,0),
                                             0.2);
  double error = ssd_matcher.get_error();
  cv::Point2d match = ssd_matcher.getMatch_point();
  std::cout<<error<<" "<<match<<std::endl;
  ASSERT_TRUE(abs(match.y)<1 && abs(match.x-5)<1);
}
} // sdvo

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
