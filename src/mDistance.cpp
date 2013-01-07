#include <mDistance.h>
#include <Util.h>


vector<float> centroid_mul(double n, vector<float> centroid) {
	int size = centroid.size();
	for (int i = 0; i < size; i++)
		centroid[i] = centroid[i] * n;
	return centroid;
}

vector<float> centroid_plus(vector<float> A, vector<float> B) {
	vector<float> centroid;
	int size = A.size();
	for (int i = 0; i < size; i++)
		centroid.push_back(A[i] + B[i]);
	return centroid;
}

vector<float> centroid_div(double n, vector<float> centroid) {
	int size = centroid.size();
	for (int i = 0; i < size; i++)
		centroid[i] = centroid[i] / n;
	return centroid;
}

vector<float> centroid_diff(vector<float> A, vector<float> B) {
	vector<float> centroid;
	int size = A.size();
	for (int i = 0; i < size; i++)
		centroid.push_back(A[i] - B[i]);
	return centroid;
}

double squared_magnitude(vector<float> vec) {
	int size = vec.size();
	double sum = 0;
	for (int i = 0; i < size; i++)
		sum += vec[i] * vec[i];
	return sum;
}

double timeval_diff(struct timeval *a, struct timeval *b) {
	return (double) (a->tv_sec + (double) a->tv_usec / 1000000)
			- (double) (b->tv_sec + (double) b->tv_usec / 1000000);
}


/**
 * mNGCsimilarity
 */
mNGCsimilarity::mNGCsimilarity() {}
mNGCsimilarity::~mNGCsimilarity() {}

double mNGCsimilarity::similarity(const Mat& p1, const Mat& p2)
{
	if (p1.rows != p2.rows || p1.cols != p2.cols)	return 0;

	float m1 = mean(p1)[0];
	float m2 = mean(p2)[0];

	Mat p1_n = p1-m1;
	Mat p2_n = p2-m2;

	return sum(p1_n.mul(p2_n))[0] / sqrt(sum(p1_n.mul(p1_n))[0] * sum(p2_n.mul(p2_n))[0]);
//	Mat res;
//	matchTemplate(p1, p2, res, CV_TM_CCORR_NORMED);
//	return res.at<float>(0);
}
double mNGCsimilarity::similarity(const Mat& data, const mCluster& c1, const mCluster& c2)
{
	// full link
	double sum = 0;
	for(uint i = 0; i < c1.data_index.size(); i++)
	{
		for(uint j = 0; j < c2.data_index.size(); j++)
		{
			sum += similarity(data.row(c1.data_index[i]), data.row(c2.data_index[j]));
		}
	}
	sum = sum / ((double)c1.data_index.size()*(double)c2.data_index.size());
	return sum;

	//average link
//	Mat p1(c1.centroid);
//	Mat p2(c2.centroid);
//	return similarity(p1, p2);
}


/**
 * mSquaredDiff
 */
mSquaredDiffsimilarity::mSquaredDiffsimilarity() {}
mSquaredDiffsimilarity::~mSquaredDiffsimilarity() {}

double mSquaredDiffsimilarity::similarity(const Mat& p1, const Mat& p2)
{
	return Util::getSqDistance(p1, p2);
}
double mSquaredDiffsimilarity::similarity(const Mat& data, const mCluster& c1, const mCluster& c2)
{
	vector<float> diff = centroid_diff(c1.centroid, c2.centroid);
	double sim = 0.5*(2-(c1.cvar + c2.cvar + squared_magnitude(diff)));
	return sim;
}
