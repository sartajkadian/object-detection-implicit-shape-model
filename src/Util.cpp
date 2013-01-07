/*
 * Util.cpp
 *
 *  Created on: 14 Nov 2012
 *      Author: mmakhalaf
 */

#include <Util.h>

#include <sstream>
using namespace std;


/**
 * Util
 */
Point2i Util::getCenter(const Mat& m)
{
	return Point2i(0.5*m.cols, 0.5*m.rows);
}

//
float Util::getSqDistance(const Mat& p1, const Mat& p2)
{
	Mat tmp = Mat::zeros(p1.rows, p1.cols, p1.type());
	cv::absdiff(p1, p2, tmp);
	cv::pow(tmp, 2, tmp);
	return cv::sum(tmp)[0];
}
float Util::getDistance(const Mat& p1, const Mat& p2)
{
	return cv::sqrt(getSqDistance(p1, p2));
}

Mat Util::getMean(const vector<Mat>& m)
{
	if (m.empty())	return Mat();

	Mat meanM = Mat::zeros(Size(m[0].cols, 1), m[0].type());
	for (unsigned int i = 0; i < m.size(); i++)
	{
		meanM += m[i];
	}
	meanM = meanM / m.size();

#ifdef DEBUG_STATIC
	cout << "::getMean mean " << meanM << endl;
#endif

	return meanM;
}

bool Util::approximateEq(const Mat& m1, const Mat& m2, double threshold)
{
	Mat dst = Mat::zeros(Size(m1.cols,m1.rows),m1.type());
	cv::max(m1,m2,dst);
	double maxD = Util::getSqDistance(Mat::zeros(Size(m1.cols,m1.rows),m1.type()), dst);

#ifdef DEBUG_STATIC
	cout << "::approxEq maxDistance " << maxD << endl;
#endif

	for(int i = 0; i < m1.rows; i++)
	{
		double dist = Util::getSqDistance(m1.row(i), m2.row(i));
#ifdef DEBUG_STATIC
		cout << "::approxEq distance [" << i << "] " << m1.row(i) << " & " << m2.row(i) << dist << endl;
#endif

		if (dist > threshold*maxD)
		{
			return false;
		}
	}

	return true;
}

Mat Util::create2DGaussian(int siz) {
	int rad = 0.5 * (float) (siz - 1);
	float sigma = (float) siz / 3.0f;
	Mat kern(siz, siz, CV_32F);
	for (int x = 0; x < kern.cols; x++) {
		for (int y = 0; y < kern.rows; y++) {
			float xx = x - rad, yy = y - rad;
			kern.at<float>(y, x) = exp(
					-1.0f
							* ((pow(xx, 2.0f) + pow(yy, 2.0f))
									/ (2 * pow(sigma, 2.0f))))
					/ (2.0f * PI * pow(sigma, 2.0f));
		}
	}
	kern /= sum(kern)[0];
	return kern;
}

float Util::getMean(const Mat& m, Point2i& pmean, float& vmean) {
	// get the point with a value closest to the true mean

	float cmean = mean(m)[0];

	vmean = INFINITY;
	float tmp_d = 0;
	float min_d = 10*cmean;

	for (int r = 0; r < m.rows; r++)
	{
		for (int c = 0; c < m.cols; c++)
		{
			float v = m.at<float>(r, c);
			tmp_d = abs(v - cmean);

			if (tmp_d < min_d)
			{
				min_d = tmp_d;
				vmean = v;
				pmean = Point2i(c, r);
			}
		}
	}

	return cmean;
}

Point2i Util::originFromRect(const Rect& rec)
{
	return Point2i(rec.x+0.5*rec.width, rec.y+0.5*rec.height);
}

Rect Util::rectFromOrigin(const Point2i& p, const Size2i& siz, int wid_size, Point2i* new_origin)
{
	int rad = 0.5*(float)(wid_size-1.0f);

//	int x = cv::max<int>(p.x-rad, 0);
//	int y = cv::max<int>(p.y-rad, 0);
//	int w = cv::min<int>(wid_size, siz.width-x);
//	int h = cv::min<int>(wid_size, siz.height-y);

	Point2i tl(max<int>(p.x-rad, 0), max<int>(p.y-rad, 0));
	Point2i br(min<int>(p.x+rad+1, siz.width), min<int>(p.y+rad+1, siz.height));

	if (new_origin != NULL)
	{
//		*new_origin = Point2i(x+0.5*(float)w, y+0.5*(float)h);
		*new_origin = 0.5*(tl+br);
	}

//	return Rect(x, y, w, h);
	return Rect(tl, br);
}
