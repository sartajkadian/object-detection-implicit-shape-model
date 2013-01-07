/*
 * mDataset.h
 *
 *  Created on: 24 Nov 2012
 *      Author: mmakhalaf
 */

#ifndef MDATASET_H_
#define MDATASET_H_

#include <opencv.hpp>
#include <libxml/parser.h>

#include <tr1/memory>
#include <vector>

using namespace std;
using namespace cv;

class mAbstractAnnotation;
typedef tr1::shared_ptr<mAbstractAnnotation> mAbstractAnnotation_p;

class mAbstractImageSet;
typedef tr1::shared_ptr<mAbstractImageSet> mAbstractImageSet_p;

/**
 * mAbstractAnnotation
 */
class mAbstractAnnotation
{
	string _nameId;
	string _path;
	map<string,vector<Rect> > _objectClasses;
//	vector<Rect> _objectsBoundingBox;
protected:
	void addObject(const string& classn, const Rect& bb)
	{
		if (_objectClasses.find(classn) == _objectClasses.end())
		{
			_objectClasses[classn] = vector<Rect>();
		}
		_objectClasses[classn].push_back(bb);
	}
public:
	mAbstractAnnotation(const string& name_id, const string& path) :
		_nameId(name_id),
		_path(path)
	{}
	virtual ~mAbstractAnnotation() {}

	virtual void parse(const string& file) = 0;

	map<string, vector<Rect> > getClassBB() { return _objectClasses; }

	string getNameId() { return _nameId; }
	string getImagePath() { return _path; }

	bool getBoundingBoxes(const string& classname, vector<Rect>& bb)
	{
		if (_objectClasses.find(classname) == _objectClasses.end())
		{
			return false;
		}

		bb = _objectClasses[classname];
		return true;
	}
};

/**
 * mPASCALAnnotation
 */
class mPASCALAnnotation : public mAbstractAnnotation
{
	void _parseObject(xmlNodePtr objnode, char** name, Rect& bb);
public:
	mPASCALAnnotation(const string& name_id, const string& path);
	~mPASCALAnnotation();

	void parse(const string& file);
};

/**
 * mCTAnnotation
 *
 * CalTech annotation after converting from Matlab format
 */
class mCTAnnotation : public mAbstractAnnotation
{
public:
	mCTAnnotation(const string& name_id, const string& path);
	~mCTAnnotation();

	void parse(const string& file);
};

/**
 * mAbstractImageSet
 */
class mAbstractImageSet
{
protected:
	string _imagePath;
	string _annotationPath;
public:
	mAbstractImageSet(const string& image_path, const string& annotation_path) :
		_imagePath(image_path),
		_annotationPath(annotation_path)
	{}
	virtual ~mAbstractImageSet() {}

	virtual vector<mAbstractAnnotation_p> parse(const string& image_set, int max_num_images, bool inclass_only) = 0;

	static void getStats(int n_truepos, int n_falsepos, int n_trueneg, int n_falseneg, float& accuracy, float& tpr, float& fpr, Mat& conf_mat)
	{
		conf_mat.release();
		conf_mat = Mat::zeros(2,2,CV_32S);
		conf_mat.at<int>(0,0) = n_truepos;
		conf_mat.at<int>(1,0) = n_falsepos;
		conf_mat.at<int>(0,1) = n_falseneg;
		conf_mat.at<int>(1,1) = n_trueneg;

		int totalpos = n_truepos+n_falseneg;
		int totalneg = n_falsepos+n_trueneg;

		accuracy = (float)(n_truepos+n_trueneg) / (float)(totalpos+totalneg);
		tpr = ((float)n_truepos/(float)totalpos);
		fpr = ((float)n_falsepos/(float)totalneg);
	}

	static void writeStats(FileStorage& fs, const Mat& conf_mat, float acc, float tpr, float fpr)
	{
		fs << "D" << "{";
		fs << "ConfMatrix" << conf_mat;
		fs << "Accuracy" << acc;
		fs << "TPR" << tpr;
		fs << "FPR" << fpr;
		fs << "}";
	}
};

/**
 * mPASCALImageSet
 */
class mPASCALImageSet : public mAbstractImageSet
{
public:
	mPASCALImageSet(const string& image_path, const string& annotation_path);
	~mPASCALImageSet();

	vector<mAbstractAnnotation_p> parse(const string& image_set, int max_num_images = -1, bool inclass_only = true);
};

/**
 * mCTImageSet
 */
class mCTImageSet : public mAbstractImageSet
{
public:
	mCTImageSet(const string& image_path);
	~mCTImageSet();

	vector<mAbstractAnnotation_p> parse(const string& image_set, int max_num_images = -1, bool inclass_only = true);
};


#endif /* MDATASET_H_ */
