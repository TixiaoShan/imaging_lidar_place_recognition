#pragma once

#include "parameters.h"

using namespace Eigen;
using namespace std;
using namespace DVision;

class KeyFrame
{
public:

    double time_stamp; 
    int index;

    cv::Mat image;
    cv::Mat image_intensity;
    cv::Mat thumbnail;
    pcl::PointCloud<PointType>::Ptr cloud;

    // for 3d points
    vector<cv::Point3f> brief_point_3d;
    vector<cv::Point2f> brief_point_2d_uv;
    vector<cv::Point2f> brief_point_2d_norm;
    vector<cv::KeyPoint> brief_window_keypoints;
    vector<BRIEF::bitset> brief_window_descriptors;

    // for search 3d points' correspondences
    vector<cv::Point3f> search_brief_point_3d;
    vector<cv::Point2f> search_brief_point_2d_uv;
    vector<cv::Point2f> search_brief_point_2d_norm;
    vector<cv::KeyPoint> search_brief_keypoints;
    vector<BRIEF::bitset> search_brief_descriptors;

    // for ORB
    vector<cv::Point3f> orb_point_3d;
    vector<cv::Point2f> orb_point_2d_uv;
    vector<cv::Point2f> orb_point_2d_norm;
    vector<cv::KeyPoint> orb_window_keypoints;
    cv::Mat orb_window_descriptors;

    // for Search ORB
    vector<cv::Point3f> search_orb_point_3d;
    vector<cv::Point2f> search_orb_point_2d_uv;
    vector<cv::Point2f> search_orb_point_2d_norm;
    vector<cv::KeyPoint> search_orb_keypoints;
    cv::Mat search_orb_descriptors;

    // for BoW query
    vector<cv::Mat> bow_descriptors;

    KeyFrame(double _time_stamp, 
             int _index,
             const cv::Mat &_image_intensity, 
             const pcl::PointCloud<PointType>::Ptr _cloud);

    bool findConnection(KeyFrame* old_kf);
    void computeWindowOrbPoint();
    void computeWindowBriefPoint();
    void computeSearchOrbPoint();
    void computeSearchBriefPoint();
    void computeBoWPoint();

    int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b);

    bool searchInAera(const BRIEF::bitset window_descriptor,
                      const std::vector<BRIEF::bitset> &descriptors_old,
                      const std::vector<cv::Point2f> &keypoints_old,
                      const std::vector<cv::Point2f> &keypoints_old_norm,
                      cv::Point2f &best_match,
                      cv::Point2f &best_match_norm);

    void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                          std::vector<cv::Point2f> &matched_2d_old_norm,
                          std::vector<uchar> &status,
                          const std::vector<BRIEF::bitset> &descriptors_now,
                          const std::vector<BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::Point2f> &keypoints_old,
                          const std::vector<cv::Point2f> &keypoints_old_norm);


    void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                   const std::vector<cv::Point3f> &matched_3d,
                   std::vector<uchar> &status);

    void extractPoints(const vector<cv::Point2f>& in_point_2d_uv, 
                        vector<cv::Point3f>& out_point_3d,
                        vector<cv::Point2f>& out_point_2d_norm,
                        vector<uchar>& out_status);

    bool distributionValidation(const vector<cv::Point2f>& new_point_2d_uv, 
                          const vector<cv::Point2f>& old_point_2d_uv);

    void freeMemory();
};

