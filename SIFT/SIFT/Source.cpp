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
	//-- Draw matches
	Mat img_matches;
	drawMatches(img1, SIFTkeypoints1, img2, SIFTkeypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	namedWindow("Good Matches", WINDOW_NORMAL);
	imshow("Good Matches", img_matches);

	//-- Draw matches
	Mat  img_SIFTmatches;
	drawMatches(img1, SIFTkeypoints1, img2, SIFTkeypoints2, SIFTmatches, img_SIFTmatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	namedWindow("SIFT", WINDOW_NORMAL);
	imshow("SIFT", img_SIFTmatches);
	double matches = SIFTmatches.size();
	double goodmatches = good_matches.size();
	double percentmatches = goodmatches / matches * 100;
	cout << "SIFT features found in image1:" << SIFTkeypoints1.size() << "\n";
	cout << "SIFT features found in image2:" << SIFTkeypoints2.size() << "\n";
	cout << "SIFT features matched:" << SIFTmatches.size() << "\n";
	cout << "SIFT good features matched:" << good_matches.size() << "\n";
	cout << "SIFT percent good features matched:" << percentmatches << "%" << "\n";
	time(&end);
	double time_taken = double(end - start);
	cout << "SIFT execution time:" << time_taken << "sec" << "\n";
	cout << "\n";

	waitKey();




	return 0;
}