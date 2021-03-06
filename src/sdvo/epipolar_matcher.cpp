#include <sdvo/epipolar_matcher.h>
#include <utility>
#include <opencv2/core/eigen.hpp>
#include <sdvo/ssd_subpixel_matcher_over_line.h>
#include <queue>

#undef _OPENMP
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

using namespace std;
using namespace cv;

#define CLOSE_DISTANCE 0.5 // New hypothesis distance min
#define FAR_DISTANCE 128 // New hypothesis distance max

#define VARIANCE_MAX 0.08 // Typical 0.01 Variance estimation max for browsing epipolar segment
#define BUFFER_LENGTH 100 // Number of images to keep in buffer

#define SEUIL_ERROR2_SSD 500 // Typical SEUIL_DIFF_PIXEL_FOR_SSD*SEUIL_DIFF_PIXEL_FOR_SSD + epsilon  If the square error is larger than this number, stereo matching failed.
#define SEUIL_DIFF_PIXEL_FOR_SSD 10 // Typical 5 Max intensity level difference between two pixel to search for subpixel stereo matching accuracy
#define SSD_STEP 0.2 // Distance in pixel between two equidistant point in SSD

#define SIGMA_I2 49 // Typical 49 As defined in paper, involve in error which depends on gradient norm
#define SIGMA_L2 04 // Typical 4 As defined in paper, involve in error which depends on gradient direction only

#define LENGTH_EPIPOLAR_MAX 30 // Typical 30 When hypothesis exists, browse epipolar only if its size is smaller than this number
#define GRADIENT2_MIN 100 // Typical 50 Recjet all image area where gradient is smaller than this number

#define DRAW

namespace sdvo{

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
  last_images_buffer(BUFFER_LENGTH),
  b_matrices_inited(false),
  intrinsics_matrix(_intrinsics_matrix)
{
  ox = intrinsics_matrix(0,2);
  oy = intrinsics_matrix(1,2);
  fx = intrinsics_matrix(0,0);
  fy = intrinsics_matrix(1,1);
}

void epipolar_matcher::set_depth_prior(const Mat1f &depth)
{
  depth_prior = depth;
}

void epipolar_matcher::set_pixel_age(const Mat1b &age)
{
  pixel_age = age;
}

void
epipolar_matcher::set_depth_prior_variance(const cv::Mat1f & depth_variance){
  inverse_depth_prior_variance = depth_variance;
}

Mat1b & epipolar_matcher::get_pixel_age()
{
  return pixel_age;
}

Mat1f & epipolar_matcher::get_observed_depth()
{
  return observed_depth_crt;
}

cv::Mat1f & epipolar_matcher::get_depth_prior()
{
  return depth_prior;
}

cv::Mat1f & epipolar_matcher::get_depth_prior_variance()
{
  return inverse_depth_prior_variance;
}

cv::Mat1f & epipolar_matcher::get_observed_variance()
{
  return observed_inverse_depth_variance;
}

void epipolar_matcher::init_matrices(cv::Size size){
  observed_inverse_depth_variance.create(size);
  observed_depth_crt.create(size);
  b_matrices_inited=true;
}

double epipolar_matcher::compute_error(const cv::Point2d & point,
                                       const cv::Vec2d & epipole_direction,
                                       double sigma2_l,
                                       double sigma2_i,
                                       double alpha2,
                                       const cv::Mat1f & intensity_dx,
                                       const cv::Mat1f & intensity_dy
                                       ){
  double geometricError2 = INFINITY;
  double photometricError2 = INFINITY;

  for (int i = -2; i < 2; ++i) {
    cv::Point2d pt_i(point + i * SSD_STEP * cv::Point2d(epipole_direction));

    pt_i.x = std::max(0.,std::min(double(intensity_dx.cols-1),pt_i.x));
    pt_i.y = std::max(0.,std::min(double(intensity_dy.rows-1),pt_i.y));

    cv::Vec2d g = cv::Vec2d(intensity_dx(pt_i),
                            intensity_dy(pt_i));

    cv::Vec2d g_normalized = cv::normalize(g);

    double geometricError2_i = sigma2_l / std::pow(g_normalized.dot(epipole_direction),int(2));

    double photometricError2_i = 2 * sigma2_i/std::pow(g.dot(epipole_direction),int(2));

    {
      if(geometricError2_i < geometricError2)
        geometricError2=geometricError2_i;

      if(photometricError2_i < photometricError2)
        photometricError2 = photometricError2_i;
    }
  }
  return alpha2 * (geometricError2 + photometricError2);
}

bool epipolar_matcher::push_new_data_in_buffer(dvo::core::RgbdImagePyramid const& pyr,
                                               Eigen::Affine3d const& transform_from_start)
{
  if(!b_matrices_inited) init_matrices(pyr.level(0).intensity.size());
  if(last_images_buffer.full()) pixel_age = max(pixel_age - 1,0);
  last_images_buffer.push_back(
        std::make_pair<dvo::core::RgbdImagePyramid,Eigen::Affine3d>(
          dvo::core::RgbdImagePyramid(pyr), Eigen::Affine3d(transform_from_start)));
}

inline Eigen::Vector3d
epipolar_matcher::ProjectInZEqualOne(const Eigen::Vector4d &in)
{
  Eigen::Vector4d homogeneous_pt= in / in[2];
  return Eigen::Vector3d(homogeneous_pt(0),homogeneous_pt(1),homogeneous_pt(2));
  ;
}

inline Eigen::Vector4d
epipolar_matcher::UnProject(const Eigen::Vector3d &pt_3D)
{
  return Eigen::Vector4d(pt_3D(0),pt_3D(1),pt_3D(2),1);
}

inline cv::Point2d
epipolar_matcher::project_from_image_to_image(const cv::Point2d & p_in_1,const Eigen::Affine3d & se3_2_from_1, float distance)
{

  Eigen::Vector3d pt_in_z_equal_d(distance * (p_in_1.x-ox)/fx,distance * (p_in_1.y-oy)/fy,distance);

  Eigen::Vector3d pt_in_2_3D = se3_2_from_1 * pt_in_z_equal_d;

  return cv::Point2d(pt_in_2_3D(0) * fx / pt_in_2_3D(2) + ox, pt_in_2_3D(1) * fy / pt_in_2_3D(2) + oy) ;
}

bool
epipolar_matcher::triangulate_and_populate_observation(const cv::Point2d & p,
                                                       const cv::Point2d & m,
                                                       const Eigen::Affine3d & se3_ref_from_crt)
{
  cv::Mat reference_from_last_se3_cv;
  cv::eigen2cv(se3_ref_from_crt.matrix(),reference_from_last_se3_cv);

  // Les coordonnées des points seront exprimés dans crt
  cv::Mat p1(cv::Mat(cv::Matx34d::eye()));
  cv::Mat p2(reference_from_last_se3_cv(cv::Rect(0,0,4,3)));

  cv::Point3d pt_crt((p.x-ox)/fx,(p.y-oy)/fy,1.);

  cv::Point3d pt_ref((m.x-ox)/fx,(m.y-oy)/fy,1.);

  cv::Matx31d out = LinearLSTriangulation(pt_crt,p1,pt_ref,p2);

  if(out(2)<0){
    return false;
  }
  observed_depth_crt.at<float>(p) = float(out(2));
  return true;
}

float epipolar_matcher::find_min_var_over_neighbours(cv::Point2d p)
{
  float smooth_var = INFINITY;
  for(int x = -1; x < 2 ; x++){
    for(int y = -1; y < 2 ; y++){
      if(inverse_depth_prior_variance(p.y + y, p.x + x) < smooth_var &&
         inverse_depth_prior_variance(p.y + y, p.x + x) != 0){
        smooth_var = inverse_depth_prior_variance(p.y + y, p.x + x) ;
      }
    }
  }
  return smooth_var;
}

bool epipolar_matcher::compute_new_observation()
{
  int ref_age = 0;

  dvo::core::RgbdImagePyramid & crt_pyramid = last_images_buffer.back().first;
  const Eigen::Affine3d & se3_world_from_crt = last_images_buffer.back().second;

  observed_depth_crt.setTo(0.f);
  observed_inverse_depth_variance.setTo(0.f);

#ifdef DRAW
  std::vector<cv::Mat> ref;
  ref.reserve(BUFFER_LENGTH);
#endif
  std::queue<cv::Point2d> next_pixel_to_compute;
  std::queue<cv::Point2d> pixel_to_compute;
  int rows=crt_pyramid.level(0).intensity.rows;
  int cols=crt_pyramid.level(0).intensity.cols;
  cv::Mat1f* intensity_dx = (cv::Mat1f*)&crt_pyramid.level(0).intensity_dx;
  cv::Mat1f* intensity_dy = (cv::Mat1f*)&crt_pyramid.level(0).intensity_dy;
#ifdef DRAW
  cv::Mat1b  gradient_was_sufficent = cv::Mat1b::zeros(rows,cols);
#endif

  // On rejette l'endroit si ya pas de gradient
#pragma omp parallel_for schedule(static,1)
  for (int r = 5;  r < rows-5 ; r++) {
    for (int c = 5;  c < cols-5 ; c++){
      if((*intensity_dx)(r,c) * (*intensity_dx)(r,c) + (*intensity_dy)(r,c) * (*intensity_dy)(r,c) > GRADIENT2_MIN){
        next_pixel_to_compute.push(cv::Point2d(c,r));
#ifdef DRAW
        gradient_was_sufficent(r,c) = 255;
#endif
      }
    }
  }

#ifdef DRAW
  cv::Mat overlay_last(cv::Mat::zeros(crt_pyramid.level(0).intensity.size(),crt_pyramid.level(0).intensity.type()));
  std::vector<cv::Mat> overlay_ref_vector;
  overlay_ref_vector.reserve(BUFFER_LENGTH);
#endif

  for(boost::circular_buffer< std::pair<dvo::core::RgbdImagePyramid,Eigen::Affine3d> >::iterator ref_it = last_images_buffer.begin();
      !next_pixel_to_compute.empty() &&
      ref_it!=last_images_buffer.end()-1
      //&& ref_it!=last_images_buffer.begin() + 1  // On bloque
      ;
      ref_it++,ref_age++){

    std::swap (pixel_to_compute,next_pixel_to_compute);

    dvo::core::RgbdImagePyramid & ref_pyramid = ref_it->first;

#ifdef DRAW
    ref.push_back(ref_pyramid.level(0).intensity);
    overlay_ref_vector.push_back(cv::Mat::zeros(crt_pyramid.level(0).intensity.size(),
                                                crt_pyramid.level(0).intensity.type()));
    cv::Mat overlay_ref(overlay_ref_vector.back());
#endif

    const Eigen::Affine3d & se3_world_from_ref = ref_it->second;

    const Eigen::Affine3d se3_ref_from_crt = se3_world_from_ref.inverse() * se3_world_from_crt;

    //-----------------
    //Si l'angle est trop grand entre deux images, passer à l'image suivante.
    //-----------------
    //    Eigen::Vector3d eulers = se3_ref_from_crt.rotation().matrix().eulerAngles(2,1,3);
    //    if(eulers[0]>0.3 || eulers[1]>0.3){
    //      if(eulers[0]-3.14)
    //      continue;
    //    }
    const Eigen::Vector4d epipole_in_reference = se3_ref_from_crt * Eigen::Vector4d(0,0,0,1);
    const Eigen::Vector3d epipole_in_reference_z_equal_one = ProjectInZEqualOne(epipole_in_reference);
    const Eigen::Vector3d epipole_in_reference_pixel = intrinsics_matrix * epipole_in_reference_z_equal_one;

    const Eigen::Vector4d epipole_in_last = se3_ref_from_crt.inverse() * Eigen::Vector4d(0,0,0,1);
    const Eigen::Vector3d epipole_in_last_z_equal_one = ProjectInZEqualOne(epipole_in_last);
    const Eigen::Vector3d epipole_in_last_pixel = intrinsics_matrix * epipole_in_last_z_equal_one;

    const cv::Point2d epipole_in_last_cv(epipole_in_last_pixel(0),epipole_in_last_pixel(1));
    const cv::Point2d epipole_in_ref_cv(epipole_in_reference_pixel(0),epipole_in_reference_pixel(1));

    while(!pixel_to_compute.empty()) {
      cv::Point2d p = cv::Point2d(pixel_to_compute.front());
      pixel_to_compute.pop();
      //--------------------
      // On rejette les pixels qui sont trop vieux (Attention l'age est inversé en fait) //Marche pas bien en l'état
      //--------------------
      if(ref_age + 1 < pixel_age(p)){
        next_pixel_to_compute.push(p);
        continue;
      }
      float d_prior_close;
      float d_prior_far;
      bool newH;

      if(depth_prior(p)==0 || isnan(depth_prior(p)) ||
         inverse_depth_prior_variance.at<float>(p) == 0 ||
         isnan(inverse_depth_prior_variance.at<float>(p))){
        //Pas d'hypothèse
        d_prior_close = CLOSE_DISTANCE;
        d_prior_far = FAR_DISTANCE;
        newH=true;
      }
      else{
        //Présence d'hypothèse
        float smooth_var = find_min_var_over_neighbours(p);

        d_prior_close = 1./(1./depth_prior(p) + 2 * smooth_var);
        d_prior_far   = 1./(1./depth_prior(p) - 2 * smooth_var);
        if(d_prior_far < 0 || d_prior_far > FAR_DISTANCE) d_prior_far = FAR_DISTANCE;
        newH=false;
      }
      //      if(int(p.y) % 10 == 0 && int(p.x) % 10== 0 && inverse_depth_prior_variance(p)>0.005){
      //        cerr<<"Hypothesis:\tVariance= "<<inverse_depth_prior_variance(p)
      //            <<"\tPrior= "<<depth_prior(p)
      //            <<"\tmin= "<<d_prior_close
      //            <<"\tmax= "<<d_prior_far
      //            <<"\tinterval = "<<d_prior_far-d_prior_close<<endl;
      //      }
      cv::Point2d epipolar_close_ref = project_from_image_to_image(p,se3_ref_from_crt, d_prior_close);
      cv::Point2d epipolar_far_ref = project_from_image_to_image(p,se3_ref_from_crt, d_prior_far);

      cv::Vec2d epipole_direction_crt = cv::normalize(cv::Vec2d(p - epipole_in_last_cv));
      cv::Vec2d epipole_direction_ref = cv::normalize(cv::Vec2d(epipolar_far_ref-epipole_in_ref_cv));

      //--------------------------------------------------------------
      // Calcul du paramètre alpha de conversion en distance inverse
      //--------------------------------------------------------------

      if(cv::norm(epipolar_close_ref-epipolar_far_ref) < 1)
      {
        next_pixel_to_compute.push(p);
        continue;
      }
      else if( /*!newH &&*/ cv::norm(epipolar_close_ref-epipolar_far_ref) > LENGTH_EPIPOLAR_MAX)
      {
        next_pixel_to_compute.push(p);
        continue;
      }
      else
      {
        //----------------------------------------------------
        // Calcul de la variance associée à la mesure courante
        //----------------------------------------------------
        double alpha = (abs(1./d_prior_close-1./d_prior_far) / cv::norm(epipolar_close_ref - epipolar_far_ref));

        double variance = compute_error(p,
                                        epipole_direction_crt,
                                        SIGMA_L2,//* (BUFFER_LENGTH-ref_age),
                                        SIGMA_I2,
                                        alpha*alpha,
                                        *intensity_dx,
                                        *intensity_dy);

        observed_inverse_depth_variance(p) =  (variance < VARIANCE_MAX )? variance : observed_inverse_depth_variance(p);
      }
      if(observed_inverse_depth_variance(p) == 0 || isnan(observed_inverse_depth_variance(p))){
        next_pixel_to_compute.push(p);
        continue;
      }



      //---------------------------------------------------------------//
      // Parcours de l'épipolaire pour trouver un match dans référence //
      //---------------------------------------------------------------//
      bool bmatch = false;

      if(epipolar_close_ref.x<0 || epipolar_close_ref.x > 640 || epipolar_close_ref.y<0 || epipolar_close_ref.y > 480 ||
         epipolar_far_ref.x<0 || epipolar_far_ref.x > 640 || epipolar_far_ref.y<0 || epipolar_far_ref.y > 480)
      {
        next_pixel_to_compute.push(p);
        continue;
      }

      cv::LineIterator
          line_it(ref_pyramid.level(0).intensity,
                  epipolar_close_ref,
                  epipolar_far_ref,
                  4
                  );

      float intensity_crt = crt_pyramid.level(0).intensity.at<float>(p);

      float error = INFINITY;
      cv::Point2d match;

      for (int i = 0; i < line_it.count; ++i, line_it++)
      {
        cv::Point2d pos_line=(line_it.pos());

        float intensity_ref = ref_pyramid.level(0).intensity.at<float>(pos_line);
        float gradient_norm2_in_ref =
            ref_pyramid.level(0).intensity_dx.at<float>(pos_line) * ref_pyramid.level(0).intensity_dx.at<float>(pos_line) +
            ref_pyramid.level(0).intensity_dy.at<float>(pos_line) * ref_pyramid.level(0).intensity_dy.at<float>(pos_line);

        // Avec raffinement sous-pixelique
        if(std::abs(intensity_ref-intensity_crt) <= SEUIL_DIFF_PIXEL_FOR_SSD && gradient_norm2_in_ref > GRADIENT2_MIN/2){
          SSD_Subpixel_Matcher_Over_Line
              line_matcher(crt_pyramid.level(0).intensity,
                           ref_pyramid.level(0).intensity,
                           p,
                           pos_line - 5 * cv::Point2d(epipole_direction_ref),
                           pos_line + 5 * cv::Point2d(epipole_direction_ref),
                           epipole_direction_ref,
                           epipole_direction_crt,SSD_STEP,int(5./SSD_STEP) + int(5./SSD_STEP) % 2 - 1);

          if(line_matcher.get_error2() < error){
            error = line_matcher.get_error2();
            match = line_matcher.getMatch_point();
          }
          //          SSD_Subpixel_Matcher_Over_Line
          //              line_matcher2(crt_pyramid.level(0).intensity,
          //                           ref_pyramid.level(0).intensity,
          //                           p,
          //                           pos_line + 5 * cv::Point2d(epipole_direction_ref),
          //                           pos_line - 5 * cv::Point2d(epipole_direction_ref),
          //                           - epipole_direction_ref,
          //                           epipole_direction_crt,SSD_STEP,int(5./SSD_STEP) + int(5./SSD_STEP) % 2 - 1);

          //          if(line_matcher2.get_error2() < error){
          //            error = line_matcher.get_error2();
          //            match = line_matcher.getMatch_point();
          //          }
        }
        // Sans raffinement sous-pixellique
        /*
         * if(std::abs(intensity_ref-intensity_crt) <= error){
          error = std::abs(intensity_ref-intensity_crt);
          match = line_it.pos();
        }*/
      }


      if(error < SEUIL_ERROR2_SSD) bmatch=true;
      //--------------------------------------------//
      // On a trouvé un match il reste à trianguler //
      //--------------------------------------------//

      if(bmatch!=true){
        pixel_age(p) = ref_age + 1;
        next_pixel_to_compute.push(p);
        continue;
      }
      else
      {

        //---------------//
        // Triangulation //
        //---------------//
        bool success = triangulate_and_populate_observation(p, match,
                                                            se3_ref_from_crt);
        if(!success){
          //          std::cerr <<"ERROR triangulate failed!"<<endl;
          //          std::cerr <<  p     <<  " "
          //                     <<  match << " "<<std::endl;

          next_pixel_to_compute.push(p);
          continue;
        }
        else if(newH){
          pixel_age(p) = ref_age + 1;
        }

#ifdef DRAW
        //-------------------------------------//
        // On trace des épipolaires pour debug //
        //-------------------------------------//
        if(int(p.x) % 5 == 0 && int(p.y) % 5 == 0){
          cv::line(overlay_last,p - 5*cv::Point2d(epipole_direction_crt),
                   p + 5 * cv::Point2d(epipole_direction_crt),1);
          cv::circle(overlay_last,p,3,Scalar(1));
          cv::line(overlay_ref,epipolar_close_ref,epipolar_far_ref,1);
          cv::circle(overlay_ref,match,3,Scalar(1));
        }
#endif
      }
    }
  }
#ifdef DRAW
  cv::imshow("Intensity",crt_pyramid.level(0).intensity/255 + overlay_last);
  //  for (int i = 0; i < overlay_ref_vector.size(); ++i) {
  //    ostringstream oss;
  //    oss<<"Intensity_Ref"<<i;
  //    cv::imshow(oss.str(),ref[i]/255 + overlay_ref_vector[i]);
  //  }
  cv::imshow("EnoughGradient",gradient_was_sufficent);

#endif
  return true;
}
}
