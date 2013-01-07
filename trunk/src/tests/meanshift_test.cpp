#include <opencv.hpp>
using namespace cv;

#include <iostream>
using namespace std;

#include <Util.h>
#include <mImplicitShapeModel.h>

/**
 * If the starting position at the start is important
 * If it starts at a place where probs are 0, there is no gradient change therefore the algorithm exits with no errors
 */
int main()
{
	mCodebook cb;
	cb.read("data/codebooks/codebook_sift_25_070_35_cut.xml");

	mISMimage_p im(new mISMimage(mAbstractAnnotation_p(new mCTAnnotation("", "data/testing/image_0115.jpg"))));

	SiftFeatureDetector det;
	vector<mMatch_p> matches = cb.match(im, &det);

	mPGHoughTransform t(im, matches);
	map<uint,vector<int> > patches_per_centre;
	Mat vhough = t.vote(patches_per_centre);
	Mat dhough = t.discretizeVotes(vhough, 100, 100, mPGHoughTransform::TYPE_PIXELS);
	dhough = mMaximaSearch::nonMaximaSuppression(dhough, 3, 0.5f);

	mDebug::showImage("Disc.", dhough, true, true, false);

	vector<Point2i> cs = mMaximaSearch::meanShiftMaximaSearch(dhough, 3, 0, 0.015);

	exit(0);

	int window_size = 3;
	int rad = 0.5*(float)(window_size-1);

	Size2i siz(13,13);
	Mat mat = Mat::zeros(siz.height,siz.width,CV_32F);
	mat(Util::rectFromOrigin(Point2i(6,6), siz, 5)) = 1;
	mat(Util::rectFromOrigin(Point2i(3,3), siz, 3)) = 1;
	mat(Util::rectFromOrigin(Point2i(9,5), siz, 3)) = 1;
	mat(Util::rectFromOrigin(Point2i(1,9), siz, 3)) = 1;
	mat(Util::rectFromOrigin(Point2i(1,9), siz, 3)) = 1;

	GaussianBlur(mat, mat, Size2i(7,7), -1);

	namedWindow("window", CV_WINDOW_KEEPRATIO);
	imshow("window", mat);

	vector<Point2i> clusters = mMaximaSearch::maxShiftMaximaSearch(mat, window_size, 0.5);

	for(int i = 0; i < clusters.size(); i++)
	{
		cout << "- " << i << " " << clusters[i] << " -> " << mat.at<float>(clusters[i]) << endl;
	}
//	cout << mmax << endl;
//
//	normalize(mmax, mmax, 0, 255*256, NORM_MINMAX);
//	namedWindow("MS P", CV_WINDOW_KEEPRATIO);
//	imshow("MS P", mmax);

	waitKey(0);
}
