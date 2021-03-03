#pragma once

#include <ros/ros.h>
#include <ros/package.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int64MultiArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv/cv.h>
#include <cv_bridge/cv_bridge.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <cassert>


#include "DBoW3/DBoW3.h"
#include "ThirdParty/DVision/DVision.h"

using namespace std;

typedef pcl::PointXYZI PointType;

extern string PROJECT_NAME;
extern string CLOUD_TOPIC;
extern string PATH_TOPIC;
extern int IMAGE_WIDTH;
extern int IMAGE_HEIGHT;
extern int IMAGE_CROP;
extern int USE_BRIEF;
extern int USE_ORB;
extern int NUM_BRI_FEATURES;
extern int NUM_ORB_FEATURES;
extern int MIN_LOOP_FEATURE_NUM;
extern int MIN_LOOP_SEARCH_GAP;
extern double MIN_LOOP_SEARCH_TIME;
extern float MIN_LOOP_BOW_TH;
extern double SKIP_TIME;
extern int NUM_THREADS;
extern int DEBUG_IMAGE;
extern double MATCH_IMAGE_SCALE;
extern cv::Mat MASK;
extern map<int, int> index_match_container;
extern map<int, int> index_poseindex_container;
extern pcl::PointCloud<PointType>::Ptr cloud_traj;

extern ros::Publisher pub_match_img;
extern ros::Publisher pub_match_msg;
extern ros::Publisher pub_bow_img;
extern ros::Publisher pub_prepnp_img;
extern ros::Publisher pub_marker;
extern ros::Publisher pub_index;




struct PointOuster {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint8_t noise;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
}EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointOuster,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint8_t, noise, noise)
    (uint16_t, ring, ring)
)




class BriefExtractor
{
public:

    DVision::BRIEF m_brief;

    virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<DVision::BRIEF::bitset> &descriptors) const
    {
        m_brief.compute(im, keys, descriptors);
    }

    BriefExtractor(){};

    BriefExtractor(const std::string &pattern_file)
    {
        cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
        if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

        vector<int> x1, y1, x2, y2;
        fs["x1"] >> x1;
        fs["x2"] >> x2;
        fs["y1"] >> y1;
        fs["y2"] >> y2;

        m_brief.importPairs(x1, y1, x2, y2);
    }
};

extern BriefExtractor briefExtractor;