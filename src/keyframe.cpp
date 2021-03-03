#include "keyframe.h"

template <typename Derived>
static void reduceVector(vector<Derived> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void keypointConverter(vector<cv::KeyPoint>& keypoints_in, vector<cv::Point2f>& points_in)
{
    points_in.resize(keypoints_in.size());
    for(size_t i = 0; i < keypoints_in.size(); i++)
    {
        points_in[i] = keypoints_in[i].pt;
    }
}

void keypointConverter(vector<cv::Point2f>& points_in, vector<cv::KeyPoint>& keypoints_in)
{
    keypoints_in.resize(points_in.size());
    for(size_t i = 0; i < points_in.size(); i++)
    {
        cv::KeyPoint key;
        key.pt = points_in[i];
        keypoints_in[i] = key;
    }
}

void pubMatchedImages(const ros::Publisher* this_pub,
                      const double time_stamp,
                      const bool draw_lines,
                      const int circle_size,
                      const cv::Scalar line_color,
                      const int index_new, const int index_old, 
                      const cv::Mat& image_new, const cv::Mat& image_old,
                      const vector<cv::Point2f>& features_new, const vector<cv::Point2f>& features_old)
{
    if (this_pub->getNumSubscribers() == 0)
        return;

    int gap = 1;
    cv::Mat gap_image(gap, image_new.size().width, CV_8UC1, cv::Scalar(255, 255, 255));
    cv::Mat gray_img, color_img;
    cv::Mat old_img = image_old;
    cv::vconcat(image_new, gap_image, gap_image);
    cv::vconcat(gap_image, old_img, gray_img);
    cv::cvtColor(gray_img, color_img, CV_GRAY2RGB);
    // plot features in current frame
    for(int i = 0; i< (int)features_new.size(); i++)
    {
        cv::Point2f cur_pt = features_new[i] * MATCH_IMAGE_SCALE;
        cv::circle(color_img, cur_pt, circle_size*MATCH_IMAGE_SCALE, line_color, MATCH_IMAGE_SCALE*2);
    }
    // plot features in previous frame
    for(int i = 0; i< (int)features_old.size(); i++)
    {
        cv::Point2f old_pt = features_old[i] * MATCH_IMAGE_SCALE;
        old_pt.y += image_new.size().height + gap;
        cv::circle(color_img, old_pt, circle_size*MATCH_IMAGE_SCALE, line_color, MATCH_IMAGE_SCALE*2);
    }
    // plot lines connecting features
    if (draw_lines)
    {
        for (int i = 0; i< (int)features_new.size(); i++)
        {
            cv::Point2f old_pt = features_old[i] * MATCH_IMAGE_SCALE;
            old_pt.y += image_new.size().height + gap;
            cv::line(color_img, features_new[i] * MATCH_IMAGE_SCALE, old_pt, line_color, MATCH_IMAGE_SCALE*2, 8, 0);
        }
    }
    // plot text
    // int text_height_pos = 20;
    // double text_scale = 0.8 * MATCH_IMAGE_SCALE;
    // int text_thickness = 2 * MATCH_IMAGE_SCALE;
    // cv::Scalar text_color = cv::Scalar(255,0,255);
    // cv::putText(color_img, "Frame: " + to_string(index_new), 
    //             cv::Point2f(5, text_height_pos),
    //             CV_FONT_HERSHEY_SIMPLEX,  text_scale, text_color, text_thickness);
    // cv::putText(color_img, "Frame: " + to_string(index_old),
    //             cv::Point2f(5, text_height_pos + image_new.size().height + gap),
    //             CV_FONT_HERSHEY_SIMPLEX,  text_scale, text_color, text_thickness);
    // publish matched image
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_img).toImageMsg();
    msg->header.stamp = ros::Time(time_stamp);
    this_pub->publish(msg);
}





















// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, int _index, const cv::Mat &_image_intensity, const pcl::PointCloud<PointType>::Ptr _cloud)
{
    time_stamp = _time_stamp;
    index = _index;
    cloud = _cloud;
    image = _image_intensity.clone();
    image_intensity = _image_intensity.clone();
    cv::resize(image, thumbnail, cv::Size(), MATCH_IMAGE_SCALE, MATCH_IMAGE_SCALE);

    #pragma omp parallel sections num_threads(NUM_THREADS)
    {
        #pragma omp section
        computeWindowOrbPoint();
        #pragma omp section
        computeWindowBriefPoint();
        #pragma omp section
        computeSearchOrbPoint();
        #pragma omp section
        computeSearchBriefPoint();
        #pragma omp section
        computeBoWPoint();
    }

    image_intensity.release();
    if(!DEBUG_IMAGE)
        image.release();
}

void KeyFrame::computeWindowOrbPoint()
{
    if (USE_ORB)
    {
        // ORB features
        vector<uchar> status;
        cv::Ptr<cv::ORB> detector = cv::ORB::create(NUM_ORB_FEATURES, 1.2f, 8, 1);
        detector->detect(image_intensity, orb_window_keypoints, MASK);
        keypointConverter(orb_window_keypoints, orb_point_2d_uv);
        extractPoints(orb_point_2d_uv, orb_point_3d, orb_point_2d_norm, status);
        reduceVector(orb_point_3d, status);
        reduceVector(orb_point_2d_uv, status);
        reduceVector(orb_point_2d_norm, status);
        reduceVector(orb_window_keypoints, status);
        detector->compute(image_intensity, orb_window_keypoints, orb_window_descriptors);
    }
}

void KeyFrame::computeWindowBriefPoint()
{
    if (USE_BRIEF)
    {
        // Corner feautres
        vector<uchar> status;
        cv::goodFeaturesToTrack(image_intensity, brief_point_2d_uv, NUM_BRI_FEATURES, 0.01, 10, MASK);
        extractPoints(brief_point_2d_uv, brief_point_3d, brief_point_2d_norm, status);
        reduceVector(brief_point_3d, status);
        reduceVector(brief_point_2d_uv, status);
        reduceVector(brief_point_2d_norm, status);
        keypointConverter(brief_point_2d_uv, brief_window_keypoints);
        briefExtractor(image_intensity, brief_window_keypoints, brief_window_descriptors);
    }
}

void KeyFrame::computeSearchOrbPoint()
{
    if (USE_ORB)
    {
        // ORB features
        vector<uchar> status;
        cv::Ptr<cv::ORB> detector = cv::ORB::create(NUM_ORB_FEATURES*5, 1.2f, 8, 5);
        detector->detect(image_intensity, search_orb_keypoints, MASK);
        keypointConverter(search_orb_keypoints, search_orb_point_2d_uv);
        extractPoints(search_orb_point_2d_uv, search_orb_point_3d, search_orb_point_2d_norm, status);
        reduceVector(search_orb_point_3d, status);
        reduceVector(search_orb_point_2d_uv, status);
        reduceVector(search_orb_point_2d_norm, status);
        reduceVector(search_orb_keypoints, status);
        detector->compute(image_intensity, search_orb_keypoints, search_orb_descriptors);
    }
}

void KeyFrame::computeSearchBriefPoint()
{
    if (USE_BRIEF)
    {
        // Corner feautres
        vector<uchar> status;
        cv::goodFeaturesToTrack(image_intensity, search_brief_point_2d_uv, NUM_BRI_FEATURES*5, 0.01, 2, MASK);
        extractPoints(search_brief_point_2d_uv, search_brief_point_3d, search_brief_point_2d_norm, status);
        reduceVector(search_brief_point_3d, status);
        reduceVector(search_brief_point_2d_uv, status);
        reduceVector(search_brief_point_2d_norm, status);
        keypointConverter(search_brief_point_2d_uv, search_brief_keypoints);
        briefExtractor(image_intensity, search_brief_keypoints, search_brief_descriptors);
    }
}

void KeyFrame::computeBoWPoint()
{
    cv::Mat descriptors;
    vector<cv::KeyPoint> keypoints;
    cv::Ptr<cv::ORB> detector = cv::ORB::create(2500, 1.2f, 8, 1);
    detector->detectAndCompute(image, MASK, keypoints, descriptors);

    bow_descriptors.resize(descriptors.rows);
    for (int i = 0; i < descriptors.rows; ++i)
        bow_descriptors[i] = descriptors.row(i);

    if (DEBUG_IMAGE)
    {
        cv::Mat imgShow;
        cv::drawKeypoints(image_intensity, keypoints, imgShow, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
        cv::resize(imgShow, imgShow, cv::Size(), 2, 2);
        cv::imshow("BOW Keypoints", imgShow );
        cv::waitKey(10);
    }
}

bool KeyFrame::findConnection(KeyFrame* old_kf)
{
    // visualize ORB features on two matched images from DBoW2
    pubMatchedImages(&pub_bow_img, time_stamp, false, 1, cv::Scalar(0, 255, 0), index, old_kf->index, thumbnail, old_kf->thumbnail, search_orb_point_2d_uv, old_kf->search_orb_point_2d_uv);

    if (!USE_ORB && !USE_BRIEF)
    {
        ROS_ERROR("No descriptor is used!");
        return false;
    }
    
    // ORB matching
    if (USE_ORB)
    {
        vector<cv::DMatch> matches, good_matches; 
        cv::BFMatcher matcher = cv::BFMatcher(cv::NORM_HAMMING); // https://docs.opencv.org/3.3.1/d3/da1/classcv_1_1BFMatcher.html
        matcher.match(orb_window_descriptors, old_kf->search_orb_descriptors, matches);

        std::sort(matches.begin(), matches.end());
        for (size_t i = 0; i < matches.size(); ++i)
        {
            good_matches.push_back(matches[i]);
            if (matches[i].distance > matches[0].distance * 2)
                break;
        }

        if ((int)good_matches.size() > MIN_LOOP_FEATURE_NUM)
        {
            vector<uchar> status;
            vector<cv::Point3f> matched_3d;
            vector<cv::Point2f> matched_2d_cur, matched_2d_old, matched_2d_old_norm;

            for (size_t i=0; i < good_matches.size(); i++)
            {
                int cur_index = good_matches[i].queryIdx;
                matched_3d.push_back(orb_point_3d[cur_index]);
                matched_2d_cur.push_back(orb_point_2d_uv[cur_index]);
                
                int old_index = good_matches[i].trainIdx;
                matched_2d_old.push_back(old_kf->search_orb_point_2d_uv[old_index]);
                matched_2d_old_norm.push_back(old_kf->search_orb_point_2d_norm[old_index]);
            }

            pubMatchedImages(&pub_prepnp_img, time_stamp, true, 5, cv::Scalar(0, 255, 0), index, old_kf->index, thumbnail, old_kf->thumbnail, matched_2d_cur, matched_2d_old);

            PnPRANSAC(matched_2d_old_norm, matched_3d, status);
            reduceVector(matched_3d, status);
            reduceVector(matched_2d_cur, status);
            reduceVector(matched_2d_old, status);
            reduceVector(matched_2d_old_norm, status);

            if ((int)matched_2d_cur.size() > MIN_LOOP_FEATURE_NUM && distributionValidation(matched_2d_cur, matched_2d_old))
            {
                pubMatchedImages(&pub_match_img, time_stamp, true, 5, cv::Scalar(0, 255, 0), index, old_kf->index, thumbnail, old_kf->thumbnail, matched_2d_cur, matched_2d_old);
                return true;
            }
        }
    }

    // BRIEF matching
    if (USE_BRIEF)
    {
        vector<uchar> status;
        vector<cv::Point3f> matched_3d;
        vector<cv::Point2f> matched_2d_cur, matched_2d_old, matched_2d_old_norm;

        matched_3d = brief_point_3d;
        matched_2d_cur = brief_point_2d_uv;

        searchByBRIEFDes(matched_2d_old, matched_2d_old_norm, status, brief_window_descriptors, old_kf->search_brief_descriptors, old_kf->search_brief_point_2d_uv, old_kf->search_brief_point_2d_norm);
        reduceVector(matched_3d, status);
        reduceVector(matched_2d_cur, status);
        reduceVector(matched_2d_old, status);
        reduceVector(matched_2d_old_norm, status);

        pubMatchedImages(&pub_prepnp_img, time_stamp, true, 5, cv::Scalar(0, 255, 0), index, old_kf->index, thumbnail, old_kf->thumbnail, matched_2d_cur, matched_2d_old);

        if ((int)matched_2d_cur.size() > MIN_LOOP_FEATURE_NUM)
        {
            status.clear();
            PnPRANSAC(matched_2d_old_norm, matched_3d, status);
            reduceVector(matched_3d, status);
            reduceVector(matched_2d_cur, status);
            reduceVector(matched_2d_old, status);
            reduceVector(matched_2d_old_norm, status);

            if ((int)matched_2d_cur.size() > MIN_LOOP_FEATURE_NUM && distributionValidation(matched_2d_cur, matched_2d_old))
            {
                pubMatchedImages(&pub_match_img, time_stamp, true, 5, cv::Scalar(0, 255, 0), index, old_kf->index, thumbnail, old_kf->thumbnail, matched_2d_cur, matched_2d_old);
                return true;
            }
        }
    }

    return false;
}


void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status)
{
    cv::Mat r, rvec, tvec, D, inliers;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);

    solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, tvec, false, 100, 0.025, 0.99, inliers);

    status.resize(matched_2d_old_norm.size(), 0);
    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }
}

void KeyFrame::extractPoints(const vector<cv::Point2f>& in_point_2d_uv, 
                        vector<cv::Point3f>& out_point_3d,
                        vector<cv::Point2f>& out_point_2d_norm,
                        vector<uchar>& out_status)
{
    assert(cloud->size() > 0);

    out_point_3d.resize(in_point_2d_uv.size());
    out_point_2d_norm.resize(in_point_2d_uv.size());
    out_status.resize(in_point_2d_uv.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (size_t i = 0; i < in_point_2d_uv.size(); ++i)
    {
        int col_id = cvRound(in_point_2d_uv[i].x);
        int row_id = cvRound(in_point_2d_uv[i].y);
        int index = row_id * IMAGE_WIDTH + col_id;
        PointType *pi = &cloud->points[index];

        cv::Point3f p_3d(0.f, 0.f, 0.f);
        cv::Point2f p_2d_n(0.f, 0.f);

        if (abs(pi->x) < 0.01)
        {
            out_status[i] = 0;
        } 
        else 
        {
            out_status[i] = 1;
            // lidar -> camera
            p_3d.x = -pi->y;
            p_3d.y = -pi->z;
            p_3d.z = pi->x;
            // normalize to projection plane
            p_2d_n.x = p_3d.x / p_3d.z;
            p_2d_n.y = p_3d.y / p_3d.z;
        }

        out_point_3d[i] = p_3d;
        out_point_2d_norm[i] = p_2d_n;
    }
}

bool KeyFrame::searchInAera(const BRIEF::bitset window_descriptor,
                            const std::vector<BRIEF::bitset> &descriptors_old,
                            const std::vector<cv::Point2f> &keypoints_old,
                            const std::vector<cv::Point2f> &keypoints_old_norm,
                            cv::Point2f &best_match,
                            cv::Point2f &best_match_norm)
{
    cv::Point2f best_pt;
    int bestDist = 128;
    int bestIndex = -1;
    for(int i = 0; i < (int)descriptors_old.size(); i++)
    {
        int dis = HammingDis(window_descriptor, descriptors_old[i]);
        if(dis < bestDist)
        {
            bestDist = dis;
            bestIndex = i;
        }
    }

    if (bestIndex != -1 && bestDist < 80)
    {
        best_match = keypoints_old[bestIndex];
        best_match_norm = keypoints_old_norm[bestIndex];
        return true;
    }
    else
        return false;
}

void KeyFrame::searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                                std::vector<cv::Point2f> &matched_2d_old_norm,
                                std::vector<uchar> &status,
                                const std::vector<BRIEF::bitset> &descriptors_now,
                                const std::vector<BRIEF::bitset> &descriptors_old,
                                const std::vector<cv::Point2f> &keypoints_old,
                                const std::vector<cv::Point2f> &keypoints_old_norm)
{
    status.resize(descriptors_now.size());
    matched_2d_old.resize(descriptors_now.size());
    matched_2d_old_norm.resize(descriptors_now.size());

    #pragma omp parallel for num_threads(NUM_THREADS)
    for(size_t i = 0; i < descriptors_now.size(); i++)
    {
        cv::Point2f pt(0.f, 0.f);
        cv::Point2f pt_norm(0.f, 0.f);
        if (searchInAera(descriptors_now[i], descriptors_old, keypoints_old, keypoints_old_norm, pt, pt_norm))
            status[i] = 1;
        else
            status[i] = 0;
        matched_2d_old[i] = pt;
        matched_2d_old_norm[i] = pt_norm;
    }
}

int KeyFrame::HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b)
{
    BRIEF::bitset xor_of_bitset = a ^ b;
    int dis = xor_of_bitset.count();
    return dis;
}

bool KeyFrame::distributionValidation(const vector<cv::Point2f>& new_point_2d_uv, const vector<cv::Point2f>& old_point_2d_uv)
{
    if (new_point_2d_uv.empty())
        return false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ> ());
    for (size_t i = 0; i < new_point_2d_uv.size(); ++i)
    {
        pcl::PointXYZ p;
        p.x = new_point_2d_uv[i].x;
        p.y = new_point_2d_uv[i].y;
        p.z = 0;
        new_cloud->push_back(p);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr old_cloud(new pcl::PointCloud<pcl::PointXYZ> ());
    for (size_t i = 0; i < old_point_2d_uv.size(); ++i)
    {
        pcl::PointXYZ p;
        p.x = old_point_2d_uv[i].x;
        p.y = old_point_2d_uv[i].y;
        p.z = 0;
        old_cloud->push_back(p);
    }

    Eigen::Vector4f new_xyz_centroid;
    Eigen::Matrix3f new_covariance_matrix;
    pcl::compute3DCentroid(*new_cloud, new_xyz_centroid);
    pcl::computeCovarianceMatrix(*new_cloud, new_xyz_centroid, new_covariance_matrix); 

    Eigen::Vector4f old_xyz_centroid;
    Eigen::Matrix3f old_covariance_matrix;
    pcl::compute3DCentroid(*old_cloud, old_xyz_centroid);
    pcl::computeCovarianceMatrix(*old_cloud, old_xyz_centroid, old_covariance_matrix);

    float new_cov_x = sqrt(new_covariance_matrix(0,0));
    float new_cov_y = sqrt(new_covariance_matrix(1,1));
    float old_cov_x = sqrt(old_covariance_matrix(0,0));
    float old_cov_y = sqrt(old_covariance_matrix(1,1));
    float cov_x_diff = abs(new_cov_x - old_cov_x);
    float cov_y_diff = abs(new_cov_y - old_cov_y);
    if (cov_x_diff > 3 * std::min(new_cov_x, old_cov_x) || cov_y_diff > 0.75 * std::min(new_cov_y, old_cov_y))
    {
        return false;
    }

    return true;
}

void KeyFrame::freeMemory()
{
    // these points are not used again
    brief_point_3d.clear(); brief_point_3d.shrink_to_fit();
    brief_point_2d_uv.clear(); brief_point_2d_uv.shrink_to_fit();
    brief_point_2d_norm.clear(); brief_point_2d_norm.shrink_to_fit();
    brief_window_keypoints.clear(); brief_window_keypoints.shrink_to_fit();
    brief_window_descriptors.clear(); brief_window_descriptors.shrink_to_fit();

    search_brief_point_3d.clear(); search_brief_point_3d.shrink_to_fit();
    search_brief_keypoints.clear(); search_brief_keypoints.shrink_to_fit();

    orb_point_3d.clear(); orb_point_3d.shrink_to_fit();
    orb_point_2d_uv.clear(); orb_point_2d_uv.shrink_to_fit();
    orb_point_2d_norm.clear(); orb_point_2d_norm.shrink_to_fit();
    orb_window_keypoints.clear(); orb_window_keypoints.shrink_to_fit();
    orb_window_descriptors.release();

    search_orb_point_3d.clear(); search_orb_point_3d.shrink_to_fit();
    search_orb_keypoints.clear(); search_orb_keypoints.shrink_to_fit();

    bow_descriptors.clear(); bow_descriptors.shrink_to_fit();
}