//
//  canny.h
//  Canny Edge Detector
//
//  Created by Hasan Akgün on 21/03/14.
//  Copyright (c) 2014 Hasan Akgün. All rights reserved.
//

#include "stdafx.h"

#ifndef _CANNY_
#define _CANNY_
#include "CImg.h"
/*#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"*/
#include <vector>
#include <iostream>
using namespace std;
using namespace cimg_library;


class canny {
private:
	CImg<float> img; //Original Image
	CImg<float> grayscaled; // Grayscale
	CImg<float> gFiltered; // Gradient
	CImg<float> sFiltered; //Sobel Filtered
	CImg<float> angles; //Angle Map
	CImg<float> non; // Non-maxima supp.
	CImg<float> thres; //Double threshold and final
public:

	canny(const char*); //Constructor
	CImg<float> toGrayScale();
	vector<vector<double>> createFilter(int, int, double); //Creates a gaussian filter
	CImg<float> useFilter(CImg<float>, vector<vector<double>>); //Use some filter
	CImg<float> sobel(); //Sobel filtering
	CImg<float> nonMaxSupp(); //Non-maxima supp.
	CImg<float> threshold(CImg<float>, int, int); //Double threshold and finalize picture
};

#endif
