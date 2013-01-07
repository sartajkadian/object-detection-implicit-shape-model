/*
 * mDistance.h
 *
 *  Created on: 30 Nov 2012
 *      Author: mmakhalaf
 */

#ifndef MDISTANCE_H_
#define MDISTANCE_H_

#include <opencv.hpp>
using namespace cv;

#include <list>
using namespace std;


//functions
vector<float> centroid_mul(double n, vector<float> centroid);
vector<float> centroid_plus(vector<float> A, vector<float> B);
vector<float> centroid_div(double n, vector<float> centroid);
vector<float> centroid_diff(vector<float> A, vector<float> B);
double squared_magnitude(vector<float>);
double timeval_diff(struct timeval *a, struct timeval *b);

// cluster data
class mCluster
{
public:
	vector<float> centroid; //centroid
	vector<unsigned int> data_index; //index of vectors of this cluster
	float cvar; //cluster varianceabout:startpage
};

//element structure
class mElement
{
public:
	list<mCluster>::iterator it;
	bool mask;
};
typedef vector<mElement> fmap;

//Struct of candidates to nn
class mCandidate
{
public:
	list<mCluster>::iterator it;
	unsigned int index;
};
typedef list<mCandidate> mSlice;


/**
 * mAbstractSimilarity
 */
class mAbstractSimilarity
{
public:
	mAbstractSimilarity() {}
	virtual ~mAbstractSimilarity() {}

	virtual double similarity(const Mat& p1, const Mat& p2) = 0;
	virtual double similarity(const Mat& data, const mCluster& c1, const mCluster& c2) = 0;
};

/**
 * mNGCsimilarity
 *
 * Similarity based on Normalised Grayscale Correlation between 2 clusters
 * Not optimal but better for comparing correlation
 */
class mNGCsimilarity : public mAbstractSimilarity
{
public:
	enum SimilarityType
	{
		AVERAGE_LINK,
		SINGLE_LINK,
		COMPLETE_LINK
	};

	mNGCsimilarity();
	~mNGCsimilarity();

	/**
	 * Measure normalized correlation between 2 vectors (linearized matrices)
	 */
	double similarity(const Mat& p1, const Mat& p2);

	/**
	 * Measure normalized correlation between 2 clusters
	 * Normalizes the sum of each vector in the cluster and each one in the other
	 */
	double similarity(const Mat& data, const mCluster& c1, const mCluster& c2);
private:
	SimilarityType _simType;
};

/**
 * mSquaredDiff
 *
 * Similarity is squared difference of the means + variance of the clusters
 * Efficient but the distance measure is more sensitive to changes
 */
class mSquaredDiffsimilarity : public mAbstractSimilarity
{
public:
	mSquaredDiffsimilarity();
	~mSquaredDiffsimilarity();

	double similarity(const Mat& p1, const Mat& p2);
	double similarity(const Mat& data, const mCluster& c1, const mCluster& c2);
};

template<class T>
ostream& operator<<(ostream& out, vector<T>& v)
{
	for(int i = 0; i < v.size(); i++)
	{
		out << v[i] << flush;
	}
	return out;
}

#endif /* MDISTANCE_H_ */
