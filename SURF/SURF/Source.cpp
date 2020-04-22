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
	int minPoints = 600;
	Ptr<Feature2D> surf = SURF::create(minPoints);
	std::vector<KeyPoint> SURFkeypoints1, SURFkeypoints2;
	Mat SURFdescriptors1, SURFdescriptors2;
	surf->detectAndCompute(img1, Mat(), SURFkeypoints1, SURFdescriptors1);
	surf->detectAndCompute(img2, Mat(), SURFkeypoints2, SURFdescriptors2);


	//-- Step 2: Matching descriptor vectors with a brute force matcher

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

	std::vector< DMatch > SURFmatches;

	matcher->match(SURFdescriptors1, SURFdescriptors2, SURFmatches);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(SURFdescriptors1, SURFdescriptors2, knn_matches, 2);


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
	drawMatches(img1, SURFkeypoints1, img2, SURFkeypoints2, good_matches, img_matches, Scalar::all(-1),
		Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	namedWindow("Good Matches", WINDOW_NORMAL);
	imshow("Good Matches", img_matches);

	//-- Draw matches
	Mat  img_SURFmatches;

	//SURF
	drawMatches(img1, SURFkeypoints1, img2, SURFkeypoints2, SURFmatches, img_SURFmatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	namedWindow("SURF", WINDOW_NORMAL);
	imshow("SURF", img_SURFmatches);
	double matches = SURFmatches.size();
	double goodmatches = good_matches.size();
	double percentmatches = goodmatches / matches * 100;

	cout << "SURF features found in image1:" << SURFkeypoints1.size() << "\n";
	cout << "SURF features found in image2:" << SURFkeypoints2.size() << "\n";
	//cout << "SURF features matched:" << SURFmatches.size() << "\n";
	cout << "SURF good features matched:" << good_matches.size() << "\n";
	cout << "SURF percent good features matched:" << percentmatches <<"%"<< "\n";
	time(&end);
	double time_taken = double(end - start);
	cout << "SURF execution time:" << time_taken << "sec" << "\n";
	cout << "\n";

	waitKey();

	return 0;
}