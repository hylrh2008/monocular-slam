#include <sdvo/file_stream_input_image.h>
#include <sdvo/cvmat_to_rgbdpyramid.h>
#include <dvo/dense_tracking.h>
#include <sys/time.h>
class tracker_with_depth
{
public:
    tracker_with_depth():
        rgb_source("../dataset/rgb/",".png",CV_LOAD_IMAGE_COLOR),
        depth_source("../dataset/depth",".png",CV_LOAD_IMAGE_ANYDEPTH),
        create_rgbdpyramid()
    {
    }
    sdvo::file_stream_input_image rgb_source;
    sdvo::file_stream_input_image depth_source;
    sdvo::cvmat_to_rhbdpyramid create_rgbdpyramid;
    void run(){
        dvo::core::IntrinsicMatrix intrinsic = dvo::core::IntrinsicMatrix::create(525,525,319.5,139.5);
        dvo::DenseTracker tracker(intrinsic);
        cv::Mat rgb;
        cv::Mat depth;
        cv::Mat rgbnew = rgb_source.get_next_image();
        cv::Mat depthnew = depth_source.get_next_image();
        dvo::core::RgbdImagePyramid pyramid = create_rgbdpyramid(rgbnew,depthnew);
        dvo::core::RgbdImagePyramid pyramidnew = create_rgbdpyramid(rgbnew,depthnew);
        timeval start;
        timeval end;
        Eigen::Affine3d cumulated_transform(Eigen::Affine3d::Identity());
        while(true){
            rgb = rgbnew;
            depth = depthnew;
            pyramid  = pyramidnew;
            depthnew = depth_source.get_next_image();
            rgbnew = rgb_source.get_next_image();
            dvo::core::RgbdImagePyramid pyramidnew  = create_rgbdpyramid(rgbnew,depthnew);
            Eigen::Affine3d transform;
            gettimeofday(&start,NULL);
            tracker.match(pyramid,pyramidnew,transform);
            gettimeofday(&end,NULL);
            cumulated_transform = cumulated_transform*transform;
            std::cerr<<start.tv_sec<<" "<<start.tv_usec<<std::endl;
            std::cerr<<end.tv_sec<<" "<<end.tv_usec<<std::endl<<std::endl;
            std::cerr<<transform.matrix()<<std::endl;
            cv::imshow("Nouvelle",rgbnew);
            cv::imshow("Ancienne",rgb);
            cv::imshow("NouvelleD",depthnew*10.);
            cv::imshow("AncienneD",depth*10.);
            cv::waitKey();
        }
    }
};

int main(int argc, char *argv[])
{
    tracker_with_depth tracker;
    tracker.run();
    return 0;
}
