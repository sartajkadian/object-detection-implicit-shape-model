
#include <mImplicitShapeModel.h>

int main()
{

	mISMimage_p image(new mISMimage(mAbstractAnnotation_p(new mCTAnnotation("", "data/training/image_0001.jpg"))));

	Mat m = image->getImage().clone();
	SiftFeatureDetector det;
	vector<KeyPoint> kps;
	det.detect(m, kps);
	drawKeypoints(m, kps, m, Scalar(255,255,255));
	imshow("Image SIFT", m);

	mDebug::showPatches("Image SIFT", image, &det);
	waitKey(0);
}
