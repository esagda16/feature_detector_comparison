#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include "opencv2/highgui.hpp" 
#include "opencv2/core/utility.hpp"
#include<opencv2/xfeatures2d/include/opencv2/xfeatures2d/nonfree.hpp>


#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
	time_t start, end;
	time(&start);
	Mat img1 = imread("C:/Users/egils/OneDrive/Pictures/test1.JPG", IMREAD_GRAYSCALE);
	Mat img2 = imread("C:/Users/egils/OneDrive/Pictures/test2.JPG", IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		cout << "Could not open or find the image!\n" << endl;

		return -1;
	}
	//-- Step 1: Detect the keypoints, compute the descriptors

//SIFT
	int minPoints = 1000;
	Ptr<Feature2D> sift = SIFT::create(minPoints);
	std::vector<KeyPoint> SIFTkeypoints1, SIFTkeypoints2;
	Mat SIFTdescriptors1, SIFTdescriptors2;
	sift->detectAndCompute(img1, Mat(), SIFTkeypoints1, SIFTdescriptors1);
	sift->detectAndCompute(img2, Mat(), SIFTkeypoints2, SIFTdescriptors2);

	//-- Step 2: Matching descriptor vectors with a brute force matcher

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

	std::vector< DMatch > SIFTmatches;

	matcher->match(SIFTdescriptors1, SIFTdescriptors2, SIFTmatches);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(SIFTdescriptors1, SIFTdescriptors2, knn_matches, 2);

	//-- Filter matches using the Lowe's ratio test
	const float ratio_thresh = 0.70f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}
	
	std::vector<Point2f> points1, points2;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		points1.push_back(SIFTkeypoints1[good_matches[i].queryIdx].pt);
		points2.push_back(SIFTkeypoints2[good_matches[i].trainIdx].pt);
	}

	// Calculating homography used for aligning the images:
	cv::Mat ransac_mat;
	std::cout << "Computing homography..." << std::endl;

	cv::Mat homography = cv::findHomography(points1, points2, ransac_mat, cv::RANSAC, 3.0);

	std::cout << "RANSAC information: " << ransac_mat << std::endl;

	float inlier = 0, outlier = 0;
	for (int i = 0; i < ransac_mat.rows; i++) {

		// We have an inlier:
		if ((int)ransac_mat.at<uchar>(i, 0) == 1) inlier = inlier + 1;

		// We have an outlier:
		else outlier = outlier + 1;
	}

	std::cout << "Total matches checked: " << ransac_mat.rows << std::endl;
	std::cout << "Inliers: " << inlier << std::endl;
	std::cout << "Outliers: " << outlier << std::endl;
	std::cout << "Procent inliers: " << (inlier / (ransac_mat.rows * 1.0)) * 100 << std::endl;
	std::cout << "Procent outliers: " << (outlier / (ransac_mat.rows * 1.0)) * 100 << std::endl;


	waitKey();




	return 0;
}
