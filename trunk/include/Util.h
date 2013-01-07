/*
 * Util.h
 *
 *  Created on: 14 Nov 2012
 *      Author: Mohamed Massoud
 */

#ifndef UTIL_H_
#define UTIL_H_

#include <core/core.hpp>
using namespace cv;

#include <sstream>
using namespace std;

#define PI 3.142857143f

/**
 * Util
 *
 * Utility class that contains some useful functions
 */
class Util {
public:
	template<class T>
	static string numToString(T num) {
		string s;
		stringstream ss;
		ss << num;
		ss >> s;
		return s;
	}

	template<class T>
	static T stringToNum(const string& str) {
		T val;
		stringstream ss;
		ss << str;
		ss >> val;
		return val;
	}

	template<class T>
	static T clamp(T v, T min, T max) {
		if (v < min) {
			return min;
		} else if (v > max) {
			return max;
		} else {
			return v;
		}
	}

	static Point2i getCenter(const Mat& m);

	/**
	 * Get square the euclidean distance between 2 points
	 * p1 and p2 are 1xn matrix representing a point in nD space
	 */
	static float getSqDistance(const Mat& p1, const Mat& p2);
	static float getDistance(const Mat& p1, const Mat& p2);

	static Mat getMean(const vector<Mat>& m);
	static bool approximateEq(const Mat& m1, const Mat& m2, double threshold);


	static Mat create2DGaussian(int siz);

	static float getMean(const Mat& m, Point2i& pmean, float& vmean);

	static Point2i originFromRect(const Rect& rec);

	static Rect rectFromOrigin(const Point2i& p, const Size2i& siz, int wid_size, Point2i* new_origin = NULL);
};

#endif /* UTIL_H_ */
