#include "parameters.h"
#include "keyframe.h"
#include "loop_detection.h"
#include "image_handler.h"

std::string PROJECT_NAME;
std::string CLOUD_TOPIC;
std::string PATH_TOPIC;
int IMAGE_WIDTH;
int IMAGE_HEIGHT;
int IMAGE_CROP;
int USE_BRIEF;
int USE_ORB;
int NUM_BRI_FEATURES;
int NUM_ORB_FEATURES;
int MIN_LOOP_FEATURE_NUM;
int MIN_LOOP_SEARCH_GAP;
double MIN_LOOP_SEARCH_TIME;
float MIN_LOOP_BOW_TH;
double SKIP_TIME = 0;
int NUM_THREADS;
int DEBUG_IMAGE;
double MATCH_IMAGE_SCALE;
cv::Mat MASK;
map<int, int> index_match_container;
map<int, int> index_poseindex_container;
pcl::PointCloud<PointType>::Ptr cloud_traj(new pcl::PointCloud<PointType>());

ros::Publisher pub_match_img;
ros::Publisher pub_match_msg;
ros::Publisher pub_bow_img;
ros::Publisher pub_prepnp_img;
ros::Publisher pub_marker;
ros::Publisher pub_index;

BriefExtractor briefExtractor;

ImageHandler *image_handler;
LoopDetector loopDetector;

void visualizeLoopClosure(ros::Publisher *pub_m,
                          ros::Publisher *pub_i, 
                          ros::Time timestamp, 
                          const map<int, int>& match_container, 
                          const map<int, int>& pose_container,
                          const pcl::PointCloud<PointType>::Ptr cloud_path)
{
    if (cloud_path->empty() || match_container.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    std_msgs::Int64MultiArray indexArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "odom";
    markerNode.header.stamp = timestamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
    markerNode.color.r = 0.0; markerNode.color.g = 1.0; markerNode.color.b = 0.0; markerNode.color.a = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "odom";
    markerEdge.header.stamp = timestamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0.0; markerEdge.color.a = 1;

    for (auto it = match_container.begin(); it != match_container.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        int cloud_pose_id = -1;
        geometry_msgs::Point p;

        cloud_pose_id = pose_container.find(key_cur)->second;
        p.x = cloud_traj->points[cloud_pose_id].x;
        p.y = cloud_traj->points[cloud_pose_id].y;
        p.z = cloud_traj->points[cloud_pose_id].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        indexArray.data.push_back(cloud_pose_id);

        cloud_pose_id = pose_container.find(key_pre)->second;
        p.x = cloud_traj->points[cloud_pose_id].x;
        p.y = cloud_traj->points[cloud_pose_id].y;
        p.z = cloud_traj->points[cloud_pose_id].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        indexArray.data.push_back(cloud_pose_id);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pub_m->publish(markerArray);
    pub_i->publish(indexArray);
};

void path_handler(const nav_msgs::PathConstPtr& path_msg)
{
    cloud_traj->clear();

    for (size_t i = 0; i < path_msg->poses.size(); ++i)
    {
        PointType p;
        p.x = path_msg->poses[i].pose.position.x;
        p.y = path_msg->poses[i].pose.position.y;
        p.z = path_msg->poses[i].pose.position.z;
        cloud_traj->push_back(p);
    }
}

void cloud_handler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
{
    image_handler->cloud_handler(cloud_msg);

    // control insertation frequency
    double cloud_time = cloud_msg->header.stamp.toSec();
    static double last_skip_time = -1;
    if (cloud_time - last_skip_time < SKIP_TIME)
        return;
    else
        last_skip_time = cloud_time;

    // new keyframe
    static int global_frame_index = 0;
    KeyFrame* keyframe = new KeyFrame(cloud_time,
                                      global_frame_index,
                                      image_handler->image_intensity,
                                      image_handler->cloud_track);   

    // detect loop
    loopDetector.addKeyFrame(keyframe, 1);

    // visualize loop
    index_poseindex_container[global_frame_index] = std::max((int)cloud_traj->size() - 1, 0);
    visualizeLoopClosure(&pub_marker, &pub_index, cloud_msg->header.stamp, index_match_container, index_poseindex_container, cloud_traj);

    global_frame_index++;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "loop_detection");
    ros::NodeHandle n;
    
    // Load params
    std::string config_file;
    n.getParam("lio_loop_config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    usleep(100);

    int LOOP_CLOSURE;
    fsSettings["loop_closure"] >> LOOP_CLOSURE;
    if (!LOOP_CLOSURE)
    {
        ros::spin();
        return 0;
    }

    // Initialize global params
    fsSettings["project_name"] >> PROJECT_NAME;
    fsSettings["cloud_topic"]  >> CLOUD_TOPIC;
    fsSettings["path_topic"]   >> PATH_TOPIC;
    fsSettings["image_width"]  >> IMAGE_WIDTH;
    fsSettings["image_height"] >> IMAGE_HEIGHT;
    fsSettings["image_crop"]   >> IMAGE_CROP;
    fsSettings["use_brief"]    >> USE_BRIEF;
    fsSettings["use_orb"]      >> USE_ORB;
    fsSettings["num_bri_features"] >> NUM_BRI_FEATURES;
    fsSettings["num_orb_features"] >> NUM_ORB_FEATURES;
    fsSettings["min_loop_feature_num"] >> MIN_LOOP_FEATURE_NUM;
    fsSettings["min_loop_search_gap"]  >> MIN_LOOP_SEARCH_GAP;
    fsSettings["min_loop_search_time"] >> MIN_LOOP_SEARCH_TIME;
    fsSettings["min_loop_bow_th"]      >> MIN_LOOP_BOW_TH;
    fsSettings["skip_time"]    >> SKIP_TIME;
    fsSettings["num_threads"]  >> NUM_THREADS;
    fsSettings["debug_image"]  >> DEBUG_IMAGE;
    fsSettings["match_image_scale"] >> MATCH_IMAGE_SCALE;
    string pkg_path = ros::package::getPath(PROJECT_NAME);

    // initialize vocabulary
    string vocabulary_file;
    fsSettings["vocabulary_file"] >> vocabulary_file;  
    vocabulary_file = pkg_path + vocabulary_file;
    loopDetector.loadVocabulary(vocabulary_file);

    // initialize brief extractor
    string brief_pattern_file;
    fsSettings["brief_pattern_file"] >> brief_pattern_file;  
    brief_pattern_file = pkg_path + brief_pattern_file;
    briefExtractor = BriefExtractor(brief_pattern_file);

    // create a mask for blocking feature extraction
    MASK = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, cv::Scalar(255));
    for (int i = 0; i < IMAGE_HEIGHT; ++i)
        for (int j = 0; j < IMAGE_WIDTH; ++j)
            if (j < IMAGE_CROP || j > IMAGE_WIDTH - IMAGE_CROP)
                MASK.at<uchar>(i,j) = 0;

    ros::Subscriber sub_cloud = n.subscribe(CLOUD_TOPIC, 1, cloud_handler);
    ros::Subscriber sub_path  = n.subscribe(PATH_TOPIC,  1, path_handler);
    pub_match_img  = n.advertise<sensor_msgs::Image>                ("loop_detector/image", 1);
    pub_match_msg  = n.advertise<std_msgs::Float64MultiArray>       ("loop_detector/time", 1);
    pub_bow_img    = n.advertise<sensor_msgs::Image>                ("loop_detector/bow", 1);
    pub_prepnp_img = n.advertise<sensor_msgs::Image>                ("loop_detector/prepnp", 1);
    pub_marker     = n.advertise<visualization_msgs::MarkerArray>   ("loop_detector/marker", 1);
    pub_index      = n.advertise<std_msgs::Int64MultiArray>         ("loop_detector/index", 1);

    image_handler = new ImageHandler();

    ROS_INFO("\033[1;32m----> Imaging Lidar Place Recognition Started.\033[0m");

    ros::spin();

    return 0;
}