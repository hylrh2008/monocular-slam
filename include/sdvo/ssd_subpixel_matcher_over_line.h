#ifndef SSD_SUBPIXEL_MATCHER_OVER_LINE_H
#define SSD_SUBPIXEL_MATCHER_OVER_LINE_H
#include <opencv2/opencv.hpp>
#include <boost/circular_buffer.hpp>

class SSD_Subpixel_Matcher_Over_Line
{
public:
  SSD_Subpixel_Matcher_Over_Line(const cv::Mat & _img1,
                                 const cv::Mat & _img_to_search,
                                 cv::Point2d _center_of_patch_in_img,
                                 cv::Point2d startPointIn2,
                                 cv::Point2d endPointIn2,
                                 cv::Vec2d _direction_in_img_to_search,
                                 cv::Vec2d _direction_in_img,
                                  float step=1,int size = 5);
  double match();
  cv::Point2d getMatch_point() const;
private:
  int ssd_window_size;
  float step;

  cv::Mat img;
  cv::Vec2d center_from_patch_in_img;
  cv::Vec2d direction_in_img;

  cv::Mat img_to_search;
  cv::Vec2d direction_in_img_to_search;
  cv::Vec2d startPoint;
  cv::Vec2d endPoint;
  cv::Vec2d match_point;

  ;
  std::vector<float> data_fixed;
  std::vector<float> data_moving;

  int crt_write_index;
  double computeSSD(const float *vec1, const float *vec2, int len);
  float getFloatSubpix(const cv::Mat1f &img, cv::Point2d pt);
};

#endif // SSD_SUBPIXEL_MATCHER_OVER_LINE_H
