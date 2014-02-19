#include <sdvo/epipolar_matcher.h>
#include <utility>
#include <opencv2/core/eigen.hpp>
#include <sdvo/ssd_subpixel_matcher_over_line.h>
using namespace std;
using namespace cv;
#define CLOSE_INVERSE_DISTANCE 3
#define FAR_INVERSE_DISTANCE 0.0001
#define VARIANCE_MAX 0.05
#define SEUIL_ERROR_SSD 5
#define DRAW
namespace sdvo{

float epipolar_matcher::getFloatSubpix(const cv::Mat1f& img, const Point2d &pt)
{
    assert(!img.empty());


    int x = (int)pt.x;
    int y = (int)pt.y;

    if (x<0 || x>img.cols) return 0;
    if (y<0 || y>img.rows) return 0;

    int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REPLICATE);
    int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REPLICATE);
    int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REPLICATE);
    int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REPLICATE);

    float a = pt.x - (float)x;
    float c = pt.y - (float)y;

    float b = (float)(img.at<float>(y0, x0) * (1.f - a) + img.at<float>(y0, x1) * a) * (1.f - c)
                           + (img.at<float>(y1, x0) * (1.f - a) + img.at<float>(y1, x1) * a) * c;

    return b;
}

/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> epipolar_matcher::LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
                                                     Matx34d P,       //camera 1 matrix
                                                     Point3d u1,      //homogenous image point in 2nd camera
                                                     Matx34d P1       //camera 2 matrix
                                                     )
{
  //build matrix A for homogenous equation system Ax = 0
  //assume X = (x,y,z,1), for Linear-LS method
  //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
  Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
            u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
            u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
            u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
            );
  Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
                    -(u.y*P(2,3)  -P(1,3)),
                    -(u1.x*P1(2,3)    -P1(0,3)),
                    -(u1.y*P1(2,3)    -P1(1,3)));

  Mat_<double> X;
  solve(A,B,X,DECOMP_SVD);

  return X;
}

epipolar_matcher::epipolar_matcher(const Eigen::Matrix3d &_intrinsics_matrix):
  last_images_buffer(10),
  b_matrices_inited(false),
  intrinsics_matrix(_intrinsics_matrix)
{
}

void epipolar_matcher::set_depth_prior(cv::Mat1f depth){
  depth_prior = depth;
}
void epipolar_matcher::set_depth_prior_variance(cv::Mat1f depth_variance){
  depth_prior_variance = depth_variance;
}
cv::Mat1f epipolar_matcher::getObserved_depth_crt() const
{
  return observed_depth_crt;
}

cv::Mat1f epipolar_matcher::getObserved_depth_prior() const
{
  return depth_prior;
}

cv::Mat1f epipolar_matcher::getObserved_depth_prior_variance() const
{
  return depth_prior_variance;
}

void epipolar_matcher::init_matrices(cv::Size size){
  observed_inverse_depth_variance = cv::Mat1f::zeros(size);
  observed_depth_crt = cv::Mat1f::zeros(size);
  b_matrices_inited=true;
}

double epipolar_matcher::compute_error(const cv::Point2d & point,
                                        const cv::Vec2d & epipole_direction,
                                        double sigma2_l,
                                        double sigma2_i,
                                        double alpha2,
                                        dvo::core::RgbdImagePyramid & img
                                        ){
  const cv::Mat1f & intensity_dx = img.level(0).intensity_dx;
  const cv::Mat1f & intensity_dy = img.level(0).intensity_dy;

  // On rejette l'endroit si ya pas de gradient du tout
  if(intensity_dx(point)*intensity_dx(point) + intensity_dy(point) * intensity_dy(point) < 100.){
    return INFINITY;
  }
  double geometricError2 = INFINITY;
  double photometricError2 = INFINITY;

  for (int i = -2; i < 2; ++i) {
    cv::Point2d pt_i(point + i*cv::Point2d(epipole_direction));

    pt_i.x = std::max(0.,pt_i.x);
    pt_i.x = std::min(double(intensity_dx.cols-1),pt_i.x);
    pt_i.y = std::max(0.,pt_i.y);
    pt_i.y = std::min(double(intensity_dx.rows-1),pt_i.y);

    cv::Vec2d g = cv::Vec2d(getFloatSubpix(intensity_dx,pt_i),
                            getFloatSubpix(intensity_dy,pt_i));

    cv::Vec2d g_normalized = cv::normalize(g);

    double geometricError2_i = sigma2_l/std::pow(g_normalized.dot(epipole_direction),2);
    if(geometricError2_i < geometricError2)
      geometricError2=geometricError2_i;

    double photometricError2_i = 2 * sigma2_i/std::pow(g.dot(epipole_direction),2);
    if(photometricError2_i < photometricError2)
      photometricError2 = photometricError2_i;
  }
  return alpha2 * (geometricError2 + photometricError2);
}

bool epipolar_matcher::push_new_data_in_buffer(dvo::core::RgbdImagePyramid && pyr,
                                               Eigen::Affine3d && transform_from_start)
{
  if(b_matrices_inited==false) init_matrices(pyr.level(0).intensity.size());
  last_images_buffer.push_back(
        std::make_pair<dvo::core::RgbdImagePyramid,Eigen::Affine3d>(
          std::forward<dvo::core::RgbdImagePyramid>(pyr),
          std::forward<Eigen::Affine3d>(transform_from_start)));
}

inline Eigen::Vector3d
epipolar_matcher::ProjectInZEqualOne(const Eigen::Vector4d &in)
{
  Eigen::Vector4d homogeneous_pt= in/ in[2];
  Eigen::Vector3d epipole_in_reference_z_equal_one(homogeneous_pt(0),homogeneous_pt(1),homogeneous_pt(2));
  return epipole_in_reference_z_equal_one;
}

inline Eigen::Vector4d
epipolar_matcher::UnProject(const Eigen::Vector3d &pt_3D)
{
  Eigen::Vector4d homogeneous_4D = Eigen::Vector4d(pt_3D(0),pt_3D(1),pt_3D(2),1);
  return homogeneous_4D;
}



inline cv::Point2d
epipolar_matcher::project_from_image_to_image(const cv::Point2d & p_in_1,const Eigen::Affine3d & se3_2_from_1, float distance)
{
  Eigen::Vector3d pt_in_z_equal_d
      = distance * intrinsics_matrix.inverse() * Eigen::Vector3d(p_in_1.x,p_in_1.y,1);

  Eigen::Vector4d pt_in_2_3D_homogeneous
      = se3_2_from_1 * UnProject(pt_in_z_equal_d);

  Eigen::Vector3d pt_in_2_2D_homogeneous
      = intrinsics_matrix * ProjectInZEqualOne(pt_in_2_3D_homogeneous);

  return cv::Point2d(pt_in_2_2D_homogeneous(0),pt_in_2_2D_homogeneous(1)) ;
}

void epipolar_matcher::triangulate_and_populate_observation(const cv::Point2d & p,const cv::Point2d & match,const Eigen::Affine3d & se3_ref_from_crt)
{
  cv::Mat intrinsics;
  cv::Mat reference_from_last_se3_cv;
  cv::eigen2cv(intrinsics_matrix,intrinsics);
  cv::eigen2cv(se3_ref_from_crt.matrix(),reference_from_last_se3_cv);

  // Les coordonnées des points seront exprimés dans crt
  cv::Mat p1 = cv::Mat(cv::Matx34d::eye());
  cv::Mat p2 = reference_from_last_se3_cv(cv::Rect(0,0,4,3));
  cv::Mat pt_last = intrinsics.inv() * cv::Mat(cv::Vec3d(p.x,p.y,1.));
  cv::Mat pt_ref =  intrinsics.inv() * cv::Mat(cv::Vec3d(match.x,match.y,1.));
  cv::Matx31d out = LinearLSTriangulation(pt_last.at<cv::Point3d>(0),p1,pt_ref.at<cv::Point3d>(0),p2);
  observed_depth_crt.at<float>(p) = float(out(2));
}

bool epipolar_matcher::compute_new_observation()
{
  dvo::core::RgbdImagePyramid & crt_pyramid = last_images_buffer.back().first;
  dvo::core::RgbdImagePyramid & ref_pyramid = last_images_buffer.front().first;

  Eigen::Affine3d & se3_world_from_crt = last_images_buffer.back().second;
  Eigen::Affine3d & se3_world_from_ref = last_images_buffer.front().second;

  Eigen::Affine3d se3_ref_from_crt = se3_world_from_ref.inverse() * se3_world_from_crt;

  Eigen::Vector4d epipole_in_reference = se3_ref_from_crt * Eigen::Vector4d(0,0,0,1);
  Eigen::Vector3d epipole_in_reference_z_equal_one = ProjectInZEqualOne(epipole_in_reference);
  Eigen::Vector3d epipole_in_reference_pixel = intrinsics_matrix * epipole_in_reference_z_equal_one;

  Eigen::Vector4d epipole_in_last = se3_ref_from_crt.inverse() * Eigen::Vector4d(0,0,0,1);
  Eigen::Vector3d epipole_in_last_z_equal_one = ProjectInZEqualOne(epipole_in_last);
  Eigen::Vector3d epipole_in_last_pixel = intrinsics_matrix * epipole_in_last_z_equal_one;

  cv::Point2d epipole_in_last_cv(epipole_in_last_pixel(0),epipole_in_last_pixel(1));
  cv::Point2d epipole_in_ref_cv(epipole_in_reference_pixel(0),epipole_in_reference_pixel(1));
#ifdef DRAW
  cv::Mat overlay_last= cv::Mat::zeros(crt_pyramid.level(0).intensity.size(),crt_pyramid.level(0).intensity.type());
  cv::Mat overlay_ref= cv::Mat::zeros(crt_pyramid.level(0).intensity.size(),crt_pyramid.level(0).intensity.type());
#endif
  observed_depth_crt.setTo(0);
  observed_inverse_depth_variance.setTo(0);


  for (int r = 0;  r < crt_pyramid.level(0).intensity.rows ; r++) {
    for (int c = 0;  c < crt_pyramid.level(0).intensity.cols ; c++){

      cv::Point2d p(c,r);
      observed_depth_crt(p) = 0;

      float d_prior_close;
      float d_prior_far;

      if(depth_prior.at<float>(p)==0 || isnan(depth_prior.at<float>(p)) ||
         depth_prior_variance.at<float>(p)==0 || isnan(depth_prior_variance.at<float>(p))){
            d_prior_close = 1./CLOSE_INVERSE_DISTANCE;
            d_prior_far = 1./FAR_INVERSE_DISTANCE;
      }
      else{
            d_prior_close = depth_prior.at<float>(p) - 2*depth_prior_variance(p);
            d_prior_far   = depth_prior.at<float>(p) + 2*depth_prior_variance(p);
      }
      cv::Point2d epipolar_close_ref = project_from_image_to_image(p,se3_ref_from_crt, d_prior_close);
      cv::Point2d epipolar_far_ref = project_from_image_to_image(p,se3_ref_from_crt, d_prior_far);

      cv::Vec2d epipole_direction_crt = cv::normalize(cv::Vec2d(p - epipole_in_last_cv));
      cv::Vec2d epipole_direction_ref = cv::normalize(cv::Vec2d(epipolar_far_ref-epipolar_close_ref));

      //--------------------------------------------------------------
      // Calcul du paramètre alpha de conversion en distance inverse
      //--------------------------------------------------------------
      double alpha;
      if(cv::norm(epipolar_close_ref-epipolar_far_ref) == 0)
      {
        observed_inverse_depth_variance(p) = 0;
        continue;
      }
      else if(cv::norm(epipolar_close_ref-epipolar_far_ref) > 400)
      {
        observed_inverse_depth_variance(p) = 0;
        continue;
      }
      else
      {
        //----------------------------------------------------
        // Calcul de la variance associée à la mesure courante
        //----------------------------------------------------
        alpha = (abs(1./d_prior_close-1./d_prior_far) / cv::norm(epipolar_close_ref-epipolar_far_ref));
        double variance = compute_error(p,epipole_direction_crt,1,16,alpha*alpha,crt_pyramid);

        observed_inverse_depth_variance(p) =  (variance < VARIANCE_MAX )? variance : 0;
      }
      if(observed_inverse_depth_variance.at<float>(p) == 0 || isnan(observed_inverse_depth_variance(p))){
        continue;
      }



      //---------------------------------------------------------------//
      // Parcours de l'épipolaire pour trouver un match dans référence //
      //---------------------------------------------------------------//
      bool bmatch=false;
      SSD_Subpixel_Matcher_Over_Line line_matcher(crt_pyramid.level(0).intensity,
                                                  ref_pyramid.level(0).intensity,
                                                  p,
                                                  epipolar_close_ref,
                                                  epipolar_far_ref,
                                                  epipole_direction_ref,
                                                  epipole_direction_crt,1);
      double error = line_matcher.match();
      if(error<SEUIL_ERROR_SSD) bmatch=true;
      cv::Point2d match = line_matcher.getMatch_point();
      //--------------------------------------------//
      // On a trouvé un match il reste à trianguler //
      //--------------------------------------------//

      if(bmatch==true){
#ifdef DRAW
        //-------------------------------------//
        // On trace des épipolaires pour debug //
        //-------------------------------------//
          if(r % 20 ==0 && c % 20==0){
              cv::line(overlay_last,p-10*cv::Point2d(epipole_direction_crt),p+10*cv::Point2d(epipole_direction_crt),1);
              cv::circle(overlay_last,p,8,Scalar(1));
              cv::line(overlay_ref,epipolar_close_ref,epipolar_far_ref,1);
              cv::circle(overlay_ref,match,8,Scalar(1));
              }
#endif

          //---------------//
          // Triangulation //
          //---------------//
          triangulate_and_populate_observation(p, match, se3_ref_from_crt);
      }
    }
  }
#ifdef DRAW
  cv::imshow("Variance",1./observed_inverse_depth_variance/10.);
  cv::imshow("Intensity",crt_pyramid.level(0).intensity/255 + overlay_last);
  cv::imshow("Intensity_Ref",ref_pyramid.level(0).intensity/255 + overlay_ref);
  //  cv::imshow("Intensity_dx",cv::abs(crt_pyramid.level(0).intensity_dx/64));
  //  cv::imshow("Intensity_dy",cv::abs(crt_pyramid.level(0).intensity_dy/64));
#endif
  return true;
}

cv::Mat epipolar_matcher::get_observed_depth() const
{
  return observed_depth_crt;
}
cv::Mat epipolar_matcher::get_observed_variance() const
{
  return observed_inverse_depth_variance;
}
}
