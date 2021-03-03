#include "loop_detection.h"

LoopDetector::LoopDetector(){}


void LoopDetector::loadVocabulary(std::string voc_path)
{
    voc = new DBoW3::Vocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

void LoopDetector::addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop)
{
    int loop_index = -1;
    if (flag_detect_loop)
    {
        loop_index = detectLoop(cur_kf, cur_kf->index);
    }
    else
    {
        addKeyFrameIntoVoc(cur_kf);
    }

    // check loop if valid using ransan and pnp
    if (loop_index != -1)
    {
        KeyFrame* old_kf = getKeyFrame(loop_index);

        if (abs(cur_kf->time_stamp - old_kf->time_stamp) > MIN_LOOP_SEARCH_TIME)
        {
            if (cur_kf->findConnection(old_kf))
            {
                std_msgs::Float64MultiArray match_msg;
                match_msg.data.push_back(cur_kf->time_stamp);
                match_msg.data.push_back(old_kf->time_stamp);
                pub_match_msg.publish(match_msg);

                index_match_container[cur_kf->index] = old_kf->index;
            }
        }
    }

    // add keyframe
    cur_kf->freeMemory();
    keyframelist.push_back(cur_kf);
}

KeyFrame* LoopDetector::getKeyFrame(int index)
{
    list<KeyFrame*>::iterator it = keyframelist.begin();
    for (; it != keyframelist.end(); it++)   
    {
        if((*it)->index == index)
            break;
    }
    if (it != keyframelist.end())
        return *it;
    else
        return NULL;
}

int LoopDetector::detectLoop(KeyFrame* keyframe, int frame_index)
{
    //first query; then add this frame into database!
    DBoW3::QueryResults ret;
    db.query(keyframe->bow_descriptors, ret, 4, frame_index - MIN_LOOP_SEARCH_GAP);
    db.add(keyframe->bow_descriptors); // ret[0] is the nearest neighbour's score. threshold change with neighour score

    if (DEBUG_IMAGE)
    {
        image_pool[frame_index] = keyframe->image.clone();

        cv::Mat bow_images = keyframe->image.clone();

        if (ret.size() > 0)
            putText(bow_images, "Index: " + to_string(frame_index), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);

        for (unsigned int i = 0; i < ret.size(); i++)
        {
            int tmp_index = ret[i].Id;
            auto it = image_pool.find(tmp_index);
            cv::Mat tmp_image = (it->second).clone();
            putText(tmp_image, "Index:  " + to_string(tmp_index) + ", BoW score:" + to_string(ret[i].Score), cv::Point2f(10, 50), CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);
            cv::vconcat(bow_images, tmp_image, bow_images);
        }

        cv::imshow("BoW images", bow_images);
        cv::waitKey(10);
    }

    if (frame_index - MIN_LOOP_SEARCH_GAP < 0)
        return -1;
    
    // a good match with its nerghbour
    bool find_loop = false;
    if (ret.size() >= 1 && ret[0].Score > MIN_LOOP_BOW_TH)
    {
        for (unsigned int i = 1; i < ret.size(); i++)
        {
            if (ret[i].Score > MIN_LOOP_BOW_TH)
            {          
                find_loop = true;
            }
        }
    }
    
    if (find_loop && frame_index > 5)
    {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++)
        {
            if (min_index == -1 || ((int)ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index;
    }
    else
        return -1;
}

void LoopDetector::addKeyFrameIntoVoc(KeyFrame* keyframe)
{
    if (DEBUG_IMAGE)
    {
        image_pool[keyframe->index] = keyframe->image.clone();
    }

    db.add(keyframe->bow_descriptors);
}