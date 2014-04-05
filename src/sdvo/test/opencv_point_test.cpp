#include <gtest/gtest.h>

#include <opencv2/highgui/highgui.hpp>

#include <sdvo/file_stream_input_image.h>
#include <opencv2/opencv.hpp>

std::string test_directory;
std::string data_path;


namespace sdvo
{


TEST(file_stream_input_image, browse_files_in_lexical_order)
{
  cv::Mat1f M1_ = cv::Mat1f(5,5);
  int i=0;
  for (int r = 0; r < M1_.rows; ++r) {
    for (int c = 0; c < M1_.cols; ++c,i++) {
      M1_(r,c) = i;
    }
  }
  EXPECT_EQ(M1_(3,4),M1_(cv::Point2d(4,3)));
  EXPECT_NE(M1_(3,4),M1_(cv::Point2d(3,4)));

}


} // sdvo


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
