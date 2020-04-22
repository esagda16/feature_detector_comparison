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

//ORB
	int minPoints = 1000;
Ptr<Feature2D> orb = ORB::create(minPoints);
std::vector<KeyPoint> ORBkeypoints1, ORBkeypoints2;
Mat ORBdescriptors1, ORBdescriptors2;
orb->detectAndCompute(img1, Mat(), ORBkeypoints1, ORBdescriptors1);
orb->detectAndCompute(img2, Mat(), ORBkeypoints2, ORBdescriptors2);



//-- Step 2: Matching descriptor vectors with a brute force matcher

Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);

std::vector< DMatch > ORBmatches;

matcher->match(ORBdescriptors1, ORBdescriptors2, ORBmatches);
std::vector< std::vector<DMatch> > knn_matches;
matcher->knnMatch(ORBdescriptors1, ORBdescriptors2, knn_matches, 2);

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
drawMatches(img1, ORBkeypoints1, img2, ORBkeypoints2, good_matches, img_matches, Scalar::all(-1),
	Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//-- Show detected matches
namedWindow("Good Matches", WINDOW_NORMAL);
imshow("Good Matches", img_matches);

//-- Draw matches
Mat  img_ORBmatches;

//ORB
drawMatches(img1, ORBkeypoints1, img2, ORBkeypoints2, ORBmatches, img_ORBmatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

double matches = ORBmatches.size();
double goodmatches = good_matches.size();
double percentmatches = goodmatches / matches * 100;

namedWindow("ORB", WINDOW_NORMAL);
imshow("ORB", img_ORBmatches);

cout << "ORB features found in image1:" << ORBkeypoints1.size() << "\n";
cout << "ORB features found in image2:" << ORBkeypoints2.size() << "\n";
cout << "ORB features matched:" << ORBmatches.size() << "\n";
cout << "ORB good features matched:" << good_matches.size() << "\n";
cout << "ORB percent good features matched:" << percentmatches << "%" << "\n";
time(&end);
double time_taken = (end - start);
cout << "ORB execution time:" << time_taken <<setprecision(5)<< "\n";
cout << "\n";

waitKey();




return 0;
}