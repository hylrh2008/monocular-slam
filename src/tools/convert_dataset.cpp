#include <fstream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{
  float u0 = 240.5;
  float v0 = 320.5;
  float focal_x = 480;
  float focal_y = -481;
  if (argc < 5)
  {
    std::cerr << "usage: " << argv[0]
              << " <input_file:txt> <size.x> <size.y> <output_file:png>"
              << std::endl;
    return 1;
  }

  std::ifstream input(argv[1]);
  {
    if (!input.is_open())
    {
      std::cerr << "unable to open " << argv[1] << std::endl;
      return 2;
    }
  }

  int size_x = std::atoi(argv[2]);
  {
    if (size_x <= 0)
    {
      std::cerr << "size_x must be > 0" << std::endl;
      return 3;
    }
  }

  int size_y = std::atoi(argv[3]);
  {
    if (size_y <= 0)
    {
      std::cerr << "size_x must be > 0" << std::endl;
      return 3;
    }
  }

  cv::Mat output(size_y, size_x, CV_32F);

  for (int i = 0; i < size_y; ++i)
  {
    for (int j = 0; j < size_x; ++j)
    {
      float val;
      input >> val;
      output.at<float>(i,j) = val;
    }
  }

#ifndef NDEBUG
  cv::imshow("convert", output);
  cv::waitKey(0);
#endif
  // Convert to planar depth
  cv::Mat out(size_y, size_x, CV_16UC1);

  for(int v = 0 ; v < output.cols ; v++)
     {
         for(int u = 0 ; u < output.rows ; u++)
         {
             float u_u0_by_fx = (u-u0)/focal_x;
             float v_v0_by_fy = (v-v0)/focal_y;

             out.at<u_int16_t>(u,v) =  50 * output.at<float>(u,v) / std::sqrt(u_u0_by_fx*u_u0_by_fx +
                                                                     v_v0_by_fy*v_v0_by_fy + 1 ) ;

         }
     }

  cv::imwrite(argv[4], out);

  return 0;
}
