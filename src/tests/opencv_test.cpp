#include <opencv2/opencv.hpp>
using namespace cv;

#include <iostream>
using namespace std;

#include <mImplicitShapeModel.h>

ostream& operator<<(ostream& out, const Rect& r)
{
	out << "[ " << r.x << ", " << r.y << ", " << r.width << ", " << r.height << " ]";
	return out;
}

int main()
{
	/**
	 * Matrix normalization and magnitude test
	 */
	Mat m = Mat_<float>(128, 1, 0.2f);

	Mat mm;
	magnitude(m, m, mm);

	Mat nm;
	normalize(m, nm);

	cout << "m " << m << endl << "mm " << mm << endl << "nm " << nm << endl;

	/**
	 * Matrix add delete test
	 */
	cout << "nm size " << nm.rows << endl;
	nm.push_back<float>(128);
	cout << "push_back m " << nm << endl;
	cout << "nm size " << nm.rows << endl;

	nm.release();
	cout << "release size: " << nm.rows << endl;

	nm.push_back<float>(0);
	cout << "add size " << nm.rows << endl;


	/**
	 * Rect intersection
	 */
	Rect r1(0,0,10,10);
	Rect r2(0,0,20,20);
	Rect r3(5,5,10,10);
	Rect r4(20,20,10,10);

	Rect r_contains = r1&r2;
	Rect r_intersect = r1&r3;
	Rect r_nointr = r1&r4;

	cout << "Rect contains  " << r_contains << endl;
	cout << "Rect intersect " << r_intersect << endl;
	cout << "Rect nointr    " << r_nointr << endl;

	/**
	 *
	 */
	Point2i p(1,1);
	Size2i siz(5,3);
	int lidx = 3;

	cout << "Linear " << p << "  is " << idxToLinear(p, siz.width) << endl;
	cout << "Idx of " << lidx << "  is " << linearToIdx(lidx, siz.width) << endl;
}
