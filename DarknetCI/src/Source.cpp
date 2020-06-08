/*
 * Source.cpp
 *
 *  Created on: 08-Jun-2020
 *      Author: u1190432
 */

#include <opencv2/opencv.hpp>

#include "darknet.h"
#include <iostream>

int main(int argc, char **argv) {

	cuda_set_device(gpu_index);

	char* coco_data = "cfg/coco.data";
	char* cfg_file = "cfg/yolov3-face.cfg";
	char* weight_file = "yolov3-wider_16000.weights";
	char* input_filename = "img.jpg";
	char* output_filename = "out";

	float thresh = .5;
	int fullscreen = 0;

	list *options = read_data_cfg(coco_data);
	char *name_list = option_find_str(options, "names", "data/names.list");
	char **names = get_labels(name_list);

	image **alphabet = load_alphabet();
	network *net = load_network(cfg_file, weight_file, 0);
	set_batch_network(net, 1);
	srand(2222222);
	double time;
	char buff[256];
	char *input = buff;
	float nms = .45;

	while (1) {
		if (input_filename) {
			strncpy(input, input_filename, 256);
		} else {
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input)
				return 0;
			strtok(input, "\n");
		}
		image im = load_image_color(input, 0, 0);
		image sized = letterbox_image(im, net->w, net->h);
		//image sized = resize_image(im, net->w, net->h);
		//image sized2 = resize_max(im, net->w);
		//image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
		//resize_network(net, sized.w, sized.h);
		layer l = net->layers[net->n - 1];

		float *X = sized.data;
		time = what_time_is_it_now();
		network_predict(net, X);
		printf("%s: Predicted in %f seconds.\n", input,
				what_time_is_it_now() - time);
		int nboxes = 0;
		detection *dets = get_network_boxes(net, im.w, im.h, thresh, thresh, 0,
				1, &nboxes);
		//printf("%d\n", nboxes);
		//if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
		if (nms)
			do_nms_sort(dets, nboxes, l.classes, nms);
		draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
		free_detections(dets, nboxes);
		if (output_filename) {
			save_image(im, output_filename);
		} else {
			#ifdef OPENCV
			make_window("predictions", 512, 512, 0);
			show_image(im, "predictions", 0);
			#endif
		}

		free_image(im);
		free_image(sized);
		if (input_filename)
			break;
	}

	return 0;
}

