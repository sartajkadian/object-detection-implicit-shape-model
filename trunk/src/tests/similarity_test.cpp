#include <mImplicitShapeModel.h>
#include <mDistance.h>
#include <Util.h>

float getNGCsim(const Mat& p1, const Mat& p2)
{
	if (p1.rows != p2.rows || p1.cols != p2.cols)
	{
		return 0;
	}

	float m1 = mean(p1)[0];
	float m2 = mean(p2)[0];

	Mat p1_n = p1-m1;
	Mat p2_n = p2-m2;

	return sum(p1_n.mul(p2_n))[0] / sqrt(sum(p1_n.mul(p1_n))[0] * sum(p2_n.mul(p2_n))[0]);
}

Mat mMatchTemplate(const Mat& im, const Mat& temp)
{
	mImageDescriptor::PATCH_SIZE = temp.rows;
	int rrad = 0.5*(temp.rows+1);
	int crad = 0.5*(temp.cols+1);

	Mat m = Mat::zeros(im.rows, im.cols, CV_32F);
	for(int r = rrad; r < im.rows-rrad; r++)
	{
		for(int c = crad; c < im.cols-crad; c++)
		{
			m.at<float>(r,c) = getNGCsim(mImageDescriptor::extractPatch(im, Point2i(c,r)), temp);
		}
	}
	return m;
}

int main()
{
//	vector<string> ds;
//	for(int i = 1; i < 15; i++)
//	{
//		ds.push_back("match_"+Util::numToString<int>(i)+"d.jpg");
//	}
//
//	mNGCsimilarity sim;
//
//	Mat cw = imread("codeword.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	for(int i = 0; i < ds.size(); i++)
//	{
//		Mat d = imread(ds[i], CV_LOAD_IMAGE_GRAYSCALE);
//		float s = sim.similarity(cw, d);
//		cout << ds[i] << " --> " << s << endl;
//	}

//	SiftFeatureDetector det;
//
//	mCodebook cb;
//	cb.read("data/codebook_20_seg_sift.xml");
//
//	mISMimage_p im(new mISMimage(mAbstractAnnotation_p(new mCTAnnotation("", "data/image_0092.jpg"))));
//
//	vector<mMatch_p> matches = cb.match(im, &det);

	Mat p1 = imread(/*"data/training/eye.jpg"*/"p1.png", CV_LOAD_IMAGE_GRAYSCALE);
	p1.convertTo(p1, CV_32F);
	p1 /= 255.0f;

	Mat p2 = imread(/*"data/training/image_0001.jpg"*/"p2.png", CV_LOAD_IMAGE_GRAYSCALE);
	p2.convertTo(p2, CV_32F);
	p2 /= 255.0f;

	cout << getNGCsim(p1, p2) << endl;

//	Mat res = mMatchTemplate(p2, p1);
//	Mat res;
//	matchTemplate(p2, eye, res, CV_TM_CCORR_NORMED);
//	res = mMaximaSearch::nonMaximaSuppression(res, 15);
//
//	double vmin, vmax;
//	Point2i pmin, pmax;
//	minMaxLoc(res, &vmin, &vmax, &pmin, &pmax);
//	cout << "min " << vmin << " " << pmin << endl;
//	cout << "max " << vmax << " " << pmax << endl;
//
//	namedWindow("match", CV_WINDOW_KEEPRATIO);
//	imshow("match", res);
//	waitKey(0);
}
