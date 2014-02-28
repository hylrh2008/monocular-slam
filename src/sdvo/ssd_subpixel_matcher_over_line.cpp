#include <sdvo/ssd_subpixel_matcher_over_line.h>

int SSD_Subpixel_Matcher_Over_Line::borderInterpolate( int p, int len )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else
        p = p < 0 ? 0 : len - 1;
    return p;
}

float SSD_Subpixel_Matcher_Over_Line::getSubpixFixedPoint(const cv::Mat1f& img, const cv::Point2f & pt) {

  int x = (int)pt.x;
  int y = (int)pt.y;


  unsigned int x0 = borderInterpolate(x,   img.cols);
  unsigned int x1 = borderInterpolate(x+1, img.cols);
  unsigned int y0 = borderInterpolate(y,   img.rows);
  unsigned int y1 = borderInterpolate(y+1, img.rows);

  const unsigned int shift = 8; // shift can have values 8 to 16
  const unsigned int fixed = 1<<shift;

  unsigned int Fx = (unsigned int) (pt.x * fixed); // convert to Fixed
  unsigned int Fy = (unsigned int) (pt.y * fixed); // convert to Fixed

  unsigned int py = (Fy & -fixed) >> shift; // Take integer part
  unsigned int px = (Fx & -fixed) >> shift; // Take integer part


  const float p1 = img(y0,x0);
  const float p2 = img(y0,x1);
  const float p3 = img(y1,x0);
  const float p4 = img(y1,x1);

  unsigned int Fp1 = (unsigned int) (p1 * fixed);
  unsigned int Fp2 = (unsigned int) (p2 * fixed);
  unsigned int Fp3 = (unsigned int) (p3 * fixed);
  unsigned int Fp4 = (unsigned int) (p4 * fixed);

  unsigned int fx = Fx & (fixed-1);
  unsigned int fy = Fy & (fixed-1);
  unsigned int fx1 = fixed - fx;
  unsigned int fy1 = fixed - fy;

  unsigned int w1 = (fx1 * fy1) >> shift;
  unsigned int w2 = (fx * fy1) >> shift;
  unsigned int w3 = (fx1 * fy ) >> shift;
  unsigned int w4 = (fx * fy ) >> shift;

  // Calculate the weighted sum of pixels (for each color channel)
  unsigned int out = unsigned((Fp1 * w1 + Fp2 * w2 + Fp3 * w3 + Fp4 * w4) >> shift);

  return float(out)/fixed;
}
inline float SSD_Subpixel_Matcher_Over_Line::getSubpix(const cv::Mat1f& img, const cv::Point2f & pt)
{
  assert(!img.empty());

  int x = (int)pt.x;
  int y = (int)pt.y;


  int x0 = borderInterpolate(x,   img.cols);
  int x1 = borderInterpolate(x+1, img.cols);
  int y0 = borderInterpolate(y,   img.rows);
  int y1 = borderInterpolate(y+1, img.rows);

  float a = pt.x - (float)x;
  float c = pt.y - (float)y;


  const float p1 = img(y0,x0);
  const float p2 = img(y0,x1);
  const float p3 = img(y1,x0);
  const float p4 = img(y1,x1);

  float w1 =  (1.f - a)*(1.f - c);
  float w2 = a * (1.f - c);
  float w3 = (1.f - a)*c;
  float w4 = a * c;

  float out = p1 * w1  + p2 * w2 + p3 * w3 + p4 * w4;

  return out  ;
}

/* From Opencv*/
float SSD_Subpixel_Matcher_Over_Line::computeSSD( const float *vec1, const float *vec2, int len )
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
                                                               const cv::Point2f &_center_from_patch_in_img,
                                                               const cv::Point2f &_startPoint,
                                                               const cv::Point2f &_endPoint,
                                                               const cv::Vec2f &_direction_in_img_to_search,
                                                               const cv::Vec2f &_direction_in_img,
                                                               float _step,
                                                               int size):
  ssd_window_size(size),
  step(_step),
  img(_img),
  img_to_search(_img_to_search),
  center_from_patch_in_img(_center_from_patch_in_img),
  data_fixed( 2 * ssd_window_size),
  data_moving(ssd_window_size),
  startPoint(_startPoint),
  endPoint(_endPoint),
  direction_in_img_to_search(_direction_in_img_to_search),
  direction_in_img(_direction_in_img),
  match_point(startPoint)
{
  for(int i = 0;i< ssd_window_size;i++){
    data_fixed.at(i) = getSubpix(img,_center_from_patch_in_img + (i-size/2)*step*cv::Point2f(direction_in_img));
    data_fixed.at(i + ssd_window_size) = getSubpix(img,_center_from_patch_in_img + (i-size/2)*step*cv::Point2f(direction_in_img));
  }
  for(int i = 0;i< ssd_window_size;i++){
    data_moving.at(i) = getSubpix(img_to_search,_startPoint + (i-size/2) * step * cv::Point2f(direction_in_img_to_search));
  }
  crt_write_index = ssd_window_size;

  error = match();
}

float SSD_Subpixel_Matcher_Over_Line::match(){
  cv::Vec2f crtPoint = startPoint;
  double error_crt=INFINITY;

  while(norm(crtPoint-endPoint) >= step){
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

    float SSD = computeSSD(data_fixed_ordered,data_moving.data(),ssd_window_size);
    if(SSD < error_crt){
      error_crt = SSD;
      match_point = crtPoint;
    }
    data_moving[crt_write_index % ssd_window_size] =
        getSubpix(img_to_search,
                       crtPoint + (ssd_window_size/2+1) * step * direction_in_img_to_search);
      crt_write_index++;
    crtPoint += step * direction_in_img_to_search;
  }
  return (error_crt)/float(ssd_window_size);
}
cv::Point2f SSD_Subpixel_Matcher_Over_Line::getMatch_point() const
{
  return cv::Point2d(match_point);
}
















