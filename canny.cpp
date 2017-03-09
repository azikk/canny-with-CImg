//
//  canny.cpp
//  Canny Edge Detector
//
//  Created by Hasan Akgün on 21/03/14.
//  Copyright (c) 2014 Hasan Akgün. All rights reserved.
// Canny算子求边缘点具体算法步骤如下：
// 1. 用高斯滤波器平滑图像．
// 2. 用一阶偏导有限差分计算梯度幅值和方向.
// 3. 对梯度幅值进行非极大值抑制 ．
// 4. 用双阈值算法检测和连接边缘
//

#include "stdafx.h"

#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include "canny.h"
#include <math.h>
/*#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"*/

#include "CImg.h"
#include <cstring>
using namespace std;
using namespace cimg_library;



canny::canny(const char* filename)
{
	//const char *p = filename.c_str(); // or data()
	//img; //读入图片
	//CImg<float> img;
	img.load(filename);

	
	if (img.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;

	}
	else
	{ // if load successfully

	//	img.display();
		vector<vector<double>> filter = createFilter(3, 3, 1); //一个滤波器，做高斯滤波

    //Print filter
    for (int i = 0; i < filter.size(); i++) 
    {
        for (int j = 0; j < filter[i].size(); j++) 
        {
            cout << filter[i][j] << " ";
        }
    }
	cout << endl;
    grayscaled = toGrayScale(); //Grayscale the image
	
	
	gFiltered = useFilter(grayscaled, filter); //Gaussian Filter

	sFiltered = sobel(); //Sobel Filter
	//sFiltered.display();
	
	non = nonMaxSupp(); //Non-Maxima Suppression
//	non.display();

	thres = threshold(non, 20, 40); //Double Threshold and Finalize
	
	thres.display();
	/*grayscaled.display("grayscaled");
	gFiltered.display("gFiltered");
	sFiltered.display("sFiltered");

	non.display("non");
	thres.display("thres");*/

	
	/*namedWindow("Original");  
    namedWindow("GrayScaled");
    namedWindow("Gaussian Blur");
    namedWindow("Sobel Filtered");
    namedWindow("Non-Maxima Supp.");
    namedWindow("Final");

    imshow("Original", img);                  
    imshow("GrayScaled", grayscaled);
    imshow("Gaussian Blur", gFiltered);
    imshow("Sobel Filtered", sFiltered);
    imshow("Non-Maxima Supp.", non);
    imshow("Final", thres);*/
	
	}
}


CImg<float> canny::toGrayScale()
{
	int width = img.width();
	int height = img.height();

	CImg<float> grayscaled(width, height, 1, 3);
	cout << width << " " << height << endl;
	float b, g, r;
	for (int i = 0; i < width; i++) {
		int t = 0;
		for (t = 0; t < height; t++) {
			b = img(i, t, 0, 2);
			g = img(i, t, 0, 1);
			r = img(i, t, 0, 0);
			//cout << r << " " << g << " " << b << endl;

			float newValue = (r * 0.2126 + g * 0.7152 + b * 0.0722);
			grayscaled(i, t, 0, 0) = newValue;
			grayscaled(i, t, 0, 1) = newValue;
			grayscaled(i, t, 0, 2) = newValue;
		}
		
	}
	return grayscaled;
  /*  grayscaled = Mat(img.rows, img.cols, CV_8UC1); //To one channel
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			int b = img.at<Vec3b>(i, j)[0];
			int g = img.at<Vec3b>(i, j)[1];
			int r = img.at<Vec3b>(i, j)[2];

			double newValue = (r * 0.2126 + g * 0.7152 + b * 0.0722);
			grayscaled.at<uchar>(i, j) = newValue;

		}
    return grayscaled;*/
}

vector<vector<double>> canny::createFilter(int row, int column, double sigmaIn)
{
	vector<vector<double>> filter;

	for (int i = 0; i < row; i++)
	{
        vector<double> col;
        for (int j = 0; j < column; j++)
        {
            col.push_back(-1);
        }
		filter.push_back(col);
	}

	float coordSum = 0;
	float constant = 2.0 * sigmaIn * sigmaIn;

	// Sum is for normalization
	float sum = 0.0;

	for (int x = - row/2; x <= row/2; x++)
	{
		for (int y = -column/2; y <= column/2; y++)
		{
			coordSum = (x*x + y*y);
			//filter[x + row/2][y + column/2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
			filter[x + row / 2][y + column / 2] = (exp(-(coordSum) / constant)) / (M_PI * constant);
			sum += filter[x + row/2][y + column/2];
		}
	}

	// Normalize the Filter
	for (int i = 0; i < row; i++)
        for (int j = 0; j < column; j++)
            filter[i][j] /= sum;

	return filter;

}

CImg<float> canny::useFilter(CImg<float> img_in, vector<vector<double>> filterIn)
{
    int size = (int)filterIn.size()/2;
	//Mat filteredImg = Mat(img_in.rows - 2*size, img_in.cols - 2*size, CV_8UC1);
	CImg<float> filteredImg(img_in.width() - 2 * size, img_in.height() - 2 * size, 1, 3);
	for (int i = size; i < img_in.width() - size; i++)
	{
		for (int j = size; j < img_in.height() - size; j++)
		{
			double sum = 0;
            
			for (int x = 0; x < filterIn.size(); x++)
				for (int y = 0; y < filterIn.size(); y++)
				{
                    sum += filterIn[x][y] * (double)(img_in(i + x - size, j + y - size));
				}
            
            filteredImg(i-size, j-size, 0, 0) = sum;
			filteredImg(i - size, j - size, 0, 1) = sum;
			filteredImg(i - size, j - size, 0, 2) = sum;
		}

	}
	return filteredImg;
}

CImg<float> canny::sobel()
{

    //Sobel X Filter
    double x1[] = {-1.0, 0, 1.0};
    double x2[] = {-2.0, 0, 2.0};
    double x3[] = {-1.0, 0, 1.0};

    vector<vector<double>> xFilter(3);
    xFilter[0].assign(x1, x1+3);
    xFilter[1].assign(x2, x2+3);
    xFilter[2].assign(x3, x3+3);
    
    //Sobel Y Filter
    double y1[] = {1.0, 2.0, 1.0};
    double y2[] = {0, 0, 0};
    double y3[] = {-1.0, -2.0, -1.0};
    
    vector<vector<double>> yFilter(3);
    yFilter[0].assign(y1, y1+3);
    yFilter[1].assign(y2, y2+3);
    yFilter[2].assign(y3, y3+3);
    
    //Limit Size
    int size = (int)xFilter.size()/2;
    
	//Mat filteredImg = Mat(gFiltered.rows - 2*size, gFiltered.cols - 2*size, CV_8UC1);
	CImg<float> filteredImg(gFiltered.width() - 2 * size, gFiltered.height() - 2 * size, 1, 3);
    
    //angles = Mat(gFiltered.rows - 2*size, gFiltered.cols - 2*size, CV_32FC1); //AngleMap
	CImg<float> temp(gFiltered.width() - 2 * size, gFiltered.height() - 2 * size);
	int width = gFiltered.width();
	int height = gFiltered.height();

	for (int i = size; i < width - size; i++)
	{
		for (int j = size; j < height - size; j++)
		{
			double sumx = 0;
            double sumy = 0;
            
			for (int x = 0; x < xFilter.size(); x++)
				for (int y = 0; y < xFilter.size(); y++)
				{
                    sumx += xFilter[x][y] * (double)(gFiltered(i + x - size, j + y - size)); //Sobel_X Filter Value
                    sumy += yFilter[x][y] * (double)(gFiltered(i + x - size, j + y - size)); //Sobel_Y Filter Value
				}
            double sumxsq = sumx*sumx;
            double sumysq = sumy*sumy;
            
            double sq2 = sqrt(sumxsq + sumysq);
            
            if(sq2 > 255) //Unsigned Char Fix
                sq2 =255;
            filteredImg(i-size, j-size, 0 , 0) = sq2;
			filteredImg(i - size, j - size, 0, 1) = sq2;
			filteredImg(i - size, j - size, 0, 2) = sq2;
 
            if(sumx==0) //Arctan Fix
				temp(i - size, j - size) = 90;
		
                
            else
                temp(i-size, j-size) = atan(sumy/sumx);
		}
	}
    
	angles = temp;

    return filteredImg;
}


CImg<float> canny::nonMaxSupp()
{
  //  Mat nonMaxSupped = Mat(sFiltered.rows-2, sFiltered.cols-2, CV_8UC1);
	CImg<float> nonMaxSupped(sFiltered.width() - 2, sFiltered.height() - 2, 1, 3);

	int width = sFiltered.width();
	int height = sFiltered.height();


    for (int i=1; i< width - 1; i++) {
		for (int j = 1; j< height - 1; j++) {
            float Tangent = angles(i,j);

            nonMaxSupped(i-1, j-1, 0, 0) = sFiltered(i,j, 0, 0);
			nonMaxSupped(i - 1, j - 1, 0, 1) = sFiltered(i, j, 0, 1);
			nonMaxSupped(i - 1, j - 1, 0, 2) = sFiltered(i, j, 0, 2);
            //Horizontal Edge
            if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
            {
				if ((sFiltered(i, j) < sFiltered(i, j + 1)) || (sFiltered(i, j) < sFiltered(i, j - 1))) {
					nonMaxSupped(i - 1, j - 1, 0, 0) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 1) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 2) = 0;
				}
                    
            }
            //Vertical Edge
            if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
            {
				if ((sFiltered(i, j) < sFiltered(i + 1, j)) || (sFiltered(i, j) < sFiltered(i - 1, j))) {
					nonMaxSupped(i - 1, j - 1, 0, 0) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 1) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 2) = 0;
				}
                  //  nonMaxSupped(i-1, j-1) = 0;
            }
            
            //-45 Degree Edge
            if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
            {
				if ((sFiltered(i, j) < sFiltered(i - 1, j + 1)) || (sFiltered(i, j) < sFiltered(i + 1, j - 1))) {
					nonMaxSupped(i - 1, j - 1, 0, 0) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 1) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 2) = 0;
				}
                  //  nonMaxSupped(i-1, j-1) = 0;
            }
            
            //45 Degree Edge
            if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
            {
				if ((sFiltered(i, j) < sFiltered(i + 1, j + 1)) || (sFiltered(i, j) < sFiltered(i - 1, j - 1))) {
					nonMaxSupped(i - 1, j - 1, 0, 0) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 1) = 0;
					nonMaxSupped(i - 1, j - 1, 0, 2) = 0;
				}
                 //   nonMaxSupped(i-1, j-1) = 0;
            }
        }
    }
    return nonMaxSupped;
}

CImg<float> canny::threshold(CImg<float> imgin, int low, int high)
{
    if(low > 255)
        low = 255;
    if(high > 255)
        high = 255;
    
 //   Mat EdgeMat = Mat(imgin.rows, imgin.cols, imgin.type());
	CImg<float> EdgeMat(imgin.width(), imgin.height(), 1, 3);
    
    for (int i=0; i<imgin.width(); i++) 
    {
        for (int j = 0; j<imgin.height(); j++) 
        {
            EdgeMat(i,j, 0, 1) = imgin(i,j, 0, 1);
			EdgeMat(i, j, 0, 2) = imgin(i, j, 0, 2);
			EdgeMat(i, j, 0, 0) = imgin(i, j, 0, 0);
			if (EdgeMat(i, j) > high) {
				EdgeMat(i, j, 0, 0) = 255;
				EdgeMat(i, j, 0, 1) = 255;
				EdgeMat(i, j, 0, 2) = 255;
			}
                
			else if (EdgeMat(i, j) < low) {
				EdgeMat(i, j, 0, 0) = 0;
				EdgeMat(i, j, 0, 1) = 0;
				EdgeMat(i, j, 0, 2) = 0;
			}
                
            else
            {
                bool anyHigh = false;
                bool anyBetween = false;
                for (int x=i-1; x < i+2; x++) 
                {
                    for (int y = j-1; y<j+2; y++) 
                    {
                        if(x <= 0 || y <= 0 || EdgeMat.width() || y > EdgeMat.height()) //Out of bounds
                            continue;
                        else
                        {
                            if(EdgeMat(x,y) > high)
                            {
								EdgeMat(i, j, 0, 0) = 255;
								EdgeMat(i, j, 0, 1) = 255;
								EdgeMat(i, j, 0, 2) = 255;
                                anyHigh = true;
                                break;
                            }
                            else if(EdgeMat(x,y) <= high && EdgeMat(x,y) >= low)
                                anyBetween = true;
                        }
                    }
                    if(anyHigh)
                        break;
                }
                if(!anyHigh && anyBetween)
                    for (int x=i-2; x < i+3; x++) 
                    {
                        for (int y = j-1; y<j+3; y++) 
                        {
                            if(x < 0 || y < 0 || x > EdgeMat.width() || y > EdgeMat.height()) //Out of bounds
                                continue;
                            else
                            {
                                if(EdgeMat(x,y) > high)
                                {
									EdgeMat(i, j, 0, 0) = 255;
									EdgeMat(i, j, 0, 1) = 255;
									EdgeMat(i, j, 0, 2) = 255;
                                    anyHigh = true;
                                    break;
                                }
                            }
                        }
                        if(anyHigh)
                            break;
                    }
                if(!anyHigh)
                    EdgeMat(i,j) = 0;
            }
        }
    }
    return EdgeMat;
}
