#include "yoloDetector.h"

image mat_to_image(cv::Mat& src)
{
    unsigned char *data = (unsigned char *)src.data;
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();
    int step = src.step;
    image out = make_image(w, h, c);
    int i, j, k, count=0;;

    for(k= 0; k < c; ++k){
        for(i = 0; i < h; ++i){
            for(j = 0; j < w; ++j){
                out.data[count++] = data[i*step + j*c + 2 - k]/255.;
            }
        }
    }
    return out;
}

namespace yolo
{
    void yoloDetector::predict(cv::Mat & img_in,std::vector<objectNode> & det_out)
    {
        det_out.clear();
        float nms = 0.4;
        //cv::Mat resized;
        //cv::resize(img_in,resized,cv::Size(_net.w,_net.h));
        image im_yolo = mat_to_image(img_in);
	image im_in = letterbox_image(im_yolo, _net->w, _net->h);
        layer l = _net->layers[_net->n-1];
        //box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
        //for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
        network_predict(_net, im_in.data);
	int nboxes = 0;
	detection *dets = get_network_boxes(_net, img_in.cols, img_in.rows, _thresh, 0.5, 0,1,&nboxes); 
        //get_region_boxes(l, img_in.cols, img_in.rows,_net.w,_net.h, _thresh, probs, boxes, 0, 0,0,0.5,1);
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
        //if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

        /*for(int i = 0; i < l.w*l.h*l.n; ++i)
        {
            int cls = max_index(probs[i], l.classes);
            float prob = probs[i][cls];
            if (prob > _thresh)
            {
                box b = boxes[i];
                int left  = (b.x-b.w/2.)*img_in.cols;
                int right = (b.x+b.w/2.)*img_in.cols;
                int top   = (b.y-b.h/2.)*img_in.rows;
                int bot   = (b.y+b.h/2.)*img_in.rows;

                if(left < 0) left = 0;
                if(right > img_in.cols) right = img_in.cols;
                if(top < 0) top = 0;
                if(bot > img_in.rows) bot = img_in.rows;

                objectNode obj;
                obj._rt.x = left;
                obj._rt.y = top;
                obj._rt.width = right - left;
                obj._rt.height = bot - top;
                obj._score = prob;
                obj._class = cls;
   
                det_out.push_back(obj);
            }
        }*/

	for (int i = 0; i < nboxes; ++i) {
        	int cls = -1;
        	float prob = 0.0f;
        	for (int j = 0; j < l.classes; ++j) {
            	if (dets[i].prob[j] > _thresh) {
                	if (cls < 0) {
                    	cls = j;
               		}
                	prob = dets[i].prob[j];
            		}
        	}
        	if (cls >= 0) {
            		box b = dets[i].bbox;
            		objectNode obj;
            		obj._rt.x =  (b.x - b.w / 2.)*img_in.cols;
            		obj._rt.y = (b.y - b.h / 2.)*img_in.rows;
            		obj._rt.width = b.w * img_in.cols;
            		obj._rt.height = b.h * img_in.rows;
            		obj._score = prob;
            		obj._class = cls;
            		//obj.name = _configure.class_names.empty() ? "" : _configure.class_names[cls];
            		det_out.push_back(obj);

        }
    }

    free_detections(dets, nboxes);

	free_image(im_yolo);
        free_image(im_in);
        //free(boxes);
        //free_ptrs((void **)probs, l.w*l.h*l.n);
    }

    yoloDetector::yoloDetector(char* cfgfile, char* weightfile, float thresh, const int GPUNo)
    {
        cuda_set_device(GPUNo);
        printf("%s %s test\n",cfgfile,weightfile);
        //_net = parse_network_cfg(cfgfile);
	_net = load_network(cfgfile, weightfile, 0);
        //load_weights(&_net, weightfile);
        set_batch_network(_net, 1);
        _thresh = thresh;
    }

    yoloDetector::~yoloDetector()
    {
    }
}
