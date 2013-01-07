#include <aggclustering.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat test_im = imread("/data/_work_repo_wd/datasets/VOCdevkit/VOC2010/JPEGImages/2007_000027.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	SiftFeatureDetector d;
	SiftDescriptorExtractor e;

	vector<KeyPoint> kp;
	Mat desc;

	d.detect(test_im, kp);
	e.compute(test_im, kp, desc);

	mAgglomerativeClusterer clus(new mNGCsimilarity(desc), 400, 0.1, 0);
	clus.cluster(desc);

	Mat cs = clus.getClusterCenters();
	Mat lbs = clus.getLabels();

	cout << "Num of clusters " << cs.rows << endl;

	RNG rnd(time(NULL));
	for(int i = 0; i < cs.rows; i++)
	{
		vector<KeyPoint> kpc;
		Scalar rnc(rnd(), rnd(), rnd());
		for(int j = 0; j < lbs.rows; j++)
		{
			int idx = lbs.at<int>(j, 0);
			if (i == idx)
			{
				kpc.push_back(kp[j]);
			}
		}
		drawKeypoints(test_im, kpc, test_im, rnc);
	}

	imshow("test_im", test_im);

	waitKey(0);
}
