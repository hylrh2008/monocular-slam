#include <gtest/gtest.h>

#include <opencv2/highgui/highgui.hpp>

#include <sdvo/file_stream_input_image.h>


std::string test_directory;
std::string data_path;


namespace sdvo
{


TEST(file_stream_input_image, browse_files_in_lexical_order)
{
  ASSERT_FALSE(test_directory.empty());
  std::cout << "test_directory: " << test_directory << std::endl;
  file_stream_input_image input(test_directory, ".png");
  std::stringstream ss_input, ss_ref;
  input.print_remaining_files(ss_input);

  std::string files_in_order[] =
  {
    "00",
    "001",
    "0a",
    "aa",
    "ab",
    "ba"
  };

  for (auto const& str : files_in_order)
    ss_ref << test_directory << "/" << str << ".png" << std::endl;

  ASSERT_EQ(ss_input.str(), ss_ref.str());
}


TEST(file_stream_input_image, display_test)
{
  if (data_path.empty()) return;
  std::cout << "data_path: " << data_path << std::endl;

  file_stream_input_image input(data_path, ".png");

  while (true)
  {
    std::cout << "current image: " << input.get_current_file_name() << std::endl;
    cv::Mat current = input.get_next_image();
    if (!current.data) break;

    cv::imshow("display_test", current);
    cv::waitKey(1);
  }
}


} // sdvo


int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  if (argc < 2)
  {
    std::cout << "Usage: " << argv[0] << " directory" << std::endl;
    return 1;
  }

  test_directory = argv[1];

  if (argc > 2) data_path = argv[2];

  return RUN_ALL_TESTS();
}
