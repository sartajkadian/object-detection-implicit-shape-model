#include <opencv.hpp>
using namespace std;
using namespace cv;

int main()
{


	/**
	 *
	 */

	/**
	 * detector
	 */
//	mHarrisCornerDetector det(0, 17, 3, 0.006);
//	MserFeatureDetector det;
	SiftFeatureDetector det;
	vector<KeyPoint> kps;
	Mat im = imread("data/training/image_0001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	det.detect(im, kps);
	drawKeypoints(im, kps, im, Scalar(255,0,0));
	imshow("Image", im);
	waitKey(0);

}
