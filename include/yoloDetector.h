#ifndef _YOLO_DETECTOR_H_
#define _YOLO_DETECTOR_H_

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

//cuda inlcudes
#define BLOCK 512
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#include "cudnn.h"
#endif

#ifdef __cplusplus
extern "C"{
#endif // __cplusplus

#include "darknet.h"
#ifdef __cplusplus
}
#endif // __cplusplus
//using namespace cv;

namespace yolo
{
    struct objectNode
    {
        cv::Rect    _rt;
        int         _class;
        float       _score;
    };

    class yoloDetector
    {
    public:
        yoloDetector(char* cfgfile, char* weightfile, float thresh, const int GPUNo);
        ~yoloDetector();
        void predict(cv::Mat & img_in,std::vector<objectNode> & det_out);
    private:
        network *_net;
        float _thresh;
    };


}

#endif //_YOLO_DETECTOR_H_
