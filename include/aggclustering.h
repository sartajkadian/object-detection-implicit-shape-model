//    Copyright (C) 2012  Roberto J. López-Sastre (robertoj.lopez@uah.es)
//                        Daniel Oñoro-Rubio
//			  Víctor Carrasco-Valdelvira
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.


#ifndef AGGCLUSTERING_H
#define AGGCLUSTERING_H

#include <opencv.hpp>
using namespace cv;

#include <mDistance.h>

/**
 * AgglomerativeClustering
 */
class mAgglomerativeClusterer
{
	mAbstractSimilarity* similarityMeasure;

	double thres_euclid;
	double thres;    //threshold for agglomerative clustering
	double ep; //threshold for slicing
	int free_top;
	unsigned int COMP;
	unsigned long ndist;

	cv::Mat _data;
	cv::Mat labels;
	cv::Mat cluster_centre;

	void init_map(fmap &f_map, list<mCluster> &X);
	void init_slices(fmap &f_map, list<mCluster> &X, mSlice &Si, mSlice &So, double V, double e);

	void agglomerate_clusters(mCluster &C1, const mCluster &C2);
	void get_nn(mCluster &C, list <mCluster> &R, double &sim, list <mCluster>::iterator &iNN);
	void get_nn_in_slices(mCluster &C, list <mCluster> &X, mSlice &Si, mSlice &So, double &sim, list <mCluster>::iterator &iNN, double limit);

	void insert_element(fmap &f_map, list<mCluster> &X, mCluster &V);
	void erase_element(fmap &f_map, list<mCluster> &X, list<mCluster>::iterator it);

	int unsigned binary_search_left(fmap &f_map, double d);
	unsigned int binary_search_right(fmap &f_map, double d);
	int bsearch(fmap &f_map, double d, unsigned int &b, unsigned int &t);
public:
	mAgglomerativeClusterer(const cv::Mat& data_points, mAbstractSimilarity* sim_measure, double thresh_euclid, double ep, unsigned int COMP);
	~mAgglomerativeClusterer();

	void cluster();

	cv::Mat getClusterCenters() { return cluster_centre; }
	cv::Mat getLabels() { return labels; }
};

/**
 * mPoint
 */
template<typename T>
class mPoint {
public:
	// The initializer constructor
	//
	mPoint(T index, double value) :
		_index(index), _value(value) {
	}

	// The initializer constructor
	//
	mPoint() :
		_index(0), _value(0) {
	}

	// The accessor functions
	//
	T get_index() const {
		return _index;
	}
	double get_value() const {
		return _value;
	}

	// The modifier functions
	//
	void set_index(T index) {
		_index = index;
	}
	void set_value(double v) {
		_value = v;
	}

	// A point is "less than" another point if the value is less
	//
	bool operator<(const mPoint& p) const {
		return get_value() < p.get_value();
	}

	// Whether the two points are equal
	//
	bool operator==(const mPoint& p) const {
		return (get_value() == p.get_value());
	}

	// Whether a point is greater than another point
	//
	bool operator>(const mPoint& p) const {
		// A point is greater than the other if the coordinate is greater
		//
		return get_value() > p.get_value();
	}

	// Whether a point is less than or equal
	//
	bool operator<=(const mPoint& p) const {
		return get_value() <= p.get_value();
	}

	// Whether a point is greater than or equal
	//
	bool operator>=(const mPoint& p) const {
		// A point is greater-than or equal if it is greater-than the other
		//
		return get_value() >= p.get_value();
	}

private:

	// The index
	//
	T _index;

	// The value
	//
	double _value;
};

template<class T>
ostream& operator<<(ostream& out, vector<T> v)
{
	for(uint i = 0; i < v.size(); i++)
	{
		out << v[i] << " ";
	}
	flush(out);
}

#endif
