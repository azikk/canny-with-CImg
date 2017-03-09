//
//  main.cpp
//  Canny Edge Detector
//
//  Created by Hasan Akgün on 21/03/14.
//  Copyright (c) 2014 Hasan Akgün. All rights reserved.
//

#include "stdafx.h"

#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
/*#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"*/
#include "canny.h"
#include "CImg.h"


using namespace std;
using namespace cimg_library;

using namespace std;

int main()
{
	const char* filePath = "lena.bmp"; //Filepath of input image
	canny mycanny(filePath); //直接在里面做处理然后展示
        
    return 0;
}

