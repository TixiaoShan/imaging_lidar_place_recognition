#pragma once
#include "parameters.h"
#include "keyframe.h"

class LoopDetector
{
public:

    DBoW3::Database db;
    DBoW3::Vocabulary* voc;

    map<int, cv::Mat> image_pool;

    list<KeyFrame*> keyframelist;

    LoopDetector();
    void loadVocabulary(std::string voc_path);
    
    void addKeyFrame(KeyFrame* cur_kf, bool flag_detect_loop);
    void addKeyFrameIntoVoc(KeyFrame* keyframe);
    KeyFrame* getKeyFrame(int index);

    int detectLoop(KeyFrame* keyframe, int frame_index);
};
