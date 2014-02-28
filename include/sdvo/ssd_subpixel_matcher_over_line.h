#ifndef SSD_SUBPIXEL_MATCHER_OVER_LINE_H
#define SSD_SUBPIXEL_MATCHER_OVER_LINE_H
#include <opencv2/opencv.hpp>
#include <boost/circular_buffer.hpp>

class SSD_Subpixel_Matcher_Over_Line
{
public:
  SSD_Subpixel_Matcher_Over_Line(const cv::Mat & _img1,
                                 const cv::Mat & _img_to_search,
                                 const cv::Point2f &_center_of_patch_in_img,
                                 const cv::Point2f &startPointIn2,
                                 const cv::Point2f &endPointIn2,
                                 const cv::Vec2f &_direction_in_img_to_search,
                                 const cv::Vec2f &_direction_in_img,
                                 float step=1, int size = 5);
  cv::Point2f getMatch_point() const;
  float get_error(){ return error;}
private:
  int ssd_window_size;
  float step;
  float error;

  const cv::Mat & img;

  const cv::Vec2f & center_from_patch_in_img;
  const cv::Vec2f & direction_in_img;

  const cv::Mat   & img_to_search;
  const cv::Vec2f & direction_in_img_to_search;
  const cv::Vec2f & startPoint;
  const cv::Vec2f & endPoint;
  cv::Vec2f match_point;

  std::vector<float> data_fixed;
  std::vector<float> data_moving;

  int crt_write_index;
  float computeSSD(const float *vec1, const float *vec2, int len);
  float getSubpix(const cv::Mat1f &img, const cv::Point2f &pt);
  float match();
  float getSubpixFixedPoint(const cv::Mat1f &img, const cv::Point2f &pt);
  int borderInterpolate(int p, int len);
};

#endif // SSD_SUBPIXEL_MATCHER_OVER_LINE_H
