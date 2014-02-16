#include <sdvo/ssd_subpixel_matcher_over_line.h>

float SSD_Subpixel_Matcher_Over_Line::getFloatSubpix(const cv::Mat1f& img, cv::Point2d pt)
{
    assert(!img.empty());


    int x = (int)pt.x;
    int y = (int)pt.y;

    if (x<0 || x > img.cols) return 0;
    if (y<0 || y > img.rows) return 0;

    int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
    int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
    int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);
    int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

    float a = pt.x - (float)x;
    float c = pt.y - (float)y;

    float b = (img.at<float>(y0, x0) * (1.f - a)  + img.at<float>(y0, x1) * a) * (1.f - c)
                +
              (img.at<float>(y1, x0) * (1.f - a)  + img.at<float>(y1, x1) * a) * c;

    return b;
}

/* From Opencv*/
double
SSD_Subpixel_Matcher_Over_Line::computeSSD( const float *vec1, const float *vec2, int len )
{
  double sum = 0;
  int i;

  for( i = 0; i <= len - 4; i += 4 )
  {
    double v0 = vec1[i] - vec2[i];
    double v1 = vec1[i + 1] - vec2[i + 1];
    double v2 = vec1[i + 2] - vec2[i + 2];
    double v3 = vec1[i + 3] - vec2[i + 3];

    sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
  }
  for( ; i < len; i++ )
  {
    double v = vec1[i] - vec2[i];

    sum += v * v;
  }
  return sum;
}

SSD_Subpixel_Matcher_Over_Line::SSD_Subpixel_Matcher_Over_Line(const cv::Mat & _img,
                                                               const cv::Mat & _img_to_search,
                                                               cv::Point2d _center_from_patch_in_img,
                                                               cv::Point2d _startPoint,
                                                               cv::Point2d _endPoint,
                                                               cv::Vec2d _direction_in_img_to_search,
                                                               cv::Vec2d _direction_in_img,
                                                               float _step,
                                                               int size):
  ssd_window_size(size),
  step(_step),
  img(_img),
  img_to_search(_img_to_search),
  center_from_patch_in_img(_center_from_patch_in_img),
  data_fixed(2*ssd_window_size),
  data_moving(ssd_window_size),
  startPoint(_startPoint),
  endPoint(_endPoint),
  direction_in_img_to_search(_direction_in_img_to_search),
  direction_in_img(_direction_in_img),
  match_point(startPoint)
{
  for(int i = 0;i< ssd_window_size;i++){
    data_fixed.at(i) = getFloatSubpix(img,_center_from_patch_in_img + (i-size/2)*step*cv::Point2d(direction_in_img));
    data_fixed.at(i + ssd_window_size) = getFloatSubpix(img,_center_from_patch_in_img + (i-size/2)*step*cv::Point2d(direction_in_img));
  }
  for(int i = 0;i< ssd_window_size;i++){
    data_moving.at(i) = getFloatSubpix(img_to_search,_startPoint + (i-size/2) * step * cv::Point2d(direction_in_img_to_search));
  }
  crt_write_index = ssd_window_size;
}

double SSD_Subpixel_Matcher_Over_Line::match(){
  cv::Vec2d crtPoint = startPoint;
  double error=INFINITY;
  while(norm(crtPoint-endPoint) >= 1){

    float* data_fixed_ordered = &data_fixed.data()[(ssd_window_size - crt_write_index) % ssd_window_size];

//    //--------------------------------------
//    std::cerr<<"crtPoint "<<crtPoint<<std::endl;
//    std::cerr<<"New Data : "<<getFloatSubpix(img_to_search,crtPoint)<<std::endl;
//    std::cerr<<"data_initial : ";
//    for (int i = 0; i < ssd_window_size; ++i) {
//      std::cerr<<data_fixed_ordered[(crt_write_index + i) % ssd_window_size]<<" ; ";
//    }
//    std::cerr<<std::endl;
//    std::cerr<<"data_courant : ";
//    for (int i = 0; i < ssd_window_size; ++i) {
//      std::cerr<<data_moving[(crt_write_index + i)%ssd_window_size]<<" ; ";
//    }
//    std::cerr<<std::endl;
//    std::cerr<<"error : "<<computeSSD(data_fixed_ordered,data_moving.data(),5)<<std::endl;
//    std::cerr<<std::endl;
//    //---------------------------------------

    if(computeSSD(data_fixed_ordered,data_moving.data(),5) < error){
      error = computeSSD(data_fixed_ordered,data_moving.data(),ssd_window_size);
      match_point=crtPoint;
    }
    data_moving.at(crt_write_index % ssd_window_size) = getFloatSubpix(img_to_search,crtPoint + (ssd_window_size/2+1)*step*direction_in_img_to_search);
    crt_write_index = (crt_write_index + 1) % ssd_window_size;
    crtPoint = crtPoint + step*direction_in_img_to_search;
  }
  return error;
}
cv::Point2d SSD_Subpixel_Matcher_Over_Line::getMatch_point() const
{
  return cv::Point2d(match_point);
}
















