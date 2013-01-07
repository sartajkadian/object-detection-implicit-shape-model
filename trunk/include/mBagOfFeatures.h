#ifndef M_BOF_H_
#define M_BOF_H_

#include <opencv.hpp>

using namespace std;
using namespace cv;

/**
 * mImage
 *
 * Image representation, hold its features and descriptors, can be matched to other images
 */
class mImage
{
private:
	string _name;
	Mat _image;
	vector<KeyPoint> _featureKeypoints;
	Mat _featureDescriptors;
	bool _isComputed;
	Mat _imageHistogram;
public:
	mImage(const string& im);
	~mImage();

	string getName() { return _name; }
	Mat getImage() { return _image; }
	vector<KeyPoint> getKeypoints() { return _featureKeypoints; }
	Mat getDescriptors() { return _featureDescriptors; }
	bool isComputed() { return _isComputed; }

	//TODO remove, store in mImageRep or something
	void setImageHistogram(const Mat& imD) { _imageHistogram = imD; }
	Mat getImageHistogram() { return _imageHistogram; }

	template<class Detector, class Extractor>
	void compute(Detector d, Extractor e)
	{
		cout << "* Detecting Features & computing descriptors for '" << _name << "'" << endl;
		d.detect(_image, _featureKeypoints);
		e.compute(_image, _featureKeypoints, _featureDescriptors);
		_isComputed = true;
	}

	/**
	 * TODO test this function, have a FLANN substitute
	 * Gets the matching features using K-NN
	 *
	 * for each feature in the first image
	 *  - get the first 2 matches in the second image
	 *  - if the distance is less than a threshold, consider it a match
	 */
	vector<pair<int,int> > matchFeatures(mImage* image);

	/**
	 * returns the index of closest match, and the distance to it
	 */
	unsigned int getClosestMatch(const vector<mImage*>& othImages, double& dist);

	Mat drawFeatures();
	Mat drawFeatures(const vector<int>& kpIndices, const Scalar& color);
};

/**
 * Abstract class for Cluster based classification
 */
class mAbstractClusterer
{
	int _numClusters;
	bool _doShiftCenters;

	Mat _descriptors;
	flann::Index _descriptorsIndex;

protected:
	int _getNumClusters() { return _numClusters; }
	virtual Mat _cluster(const Mat& descriptors) = 0;
public:
	/**
	 * @param numClusters Number of clusters
	 * @param shiftCenters Shift centers to their nearest neighbour in the set
	 */
	mAbstractClusterer(int numClusters, bool shiftCenters) :
		_numClusters(numClusters),
		_doShiftCenters(shiftCenters)
	{}
	virtual ~mAbstractClusterer() {}

	Mat cluster(const Mat& descriptors);
};

/**
 * mClassifierKMeans
 *
 * OpenCV's implementation of the algorithm
 */
class mClustererKMeans : public mAbstractClusterer
{
	Mat _cluster(const Mat& descriptors);
public:
	mClustererKMeans(int nClusters, bool shiftCenters);
	~mClustererKMeans();
};

/**
 * mClassifierCustomKMeans
 *
 * Implementation of K-Means algorithm
 */
class mClustererCustomKMeans : public mAbstractClusterer
{
	/**
	 * Generate words from one set of descriptors (ie. one image)
	 *  - Start with random cluster centers
	 *  - Repeat till the cluster center points stop changing
	 *  -  Assign each point in the space to its nearest cluster center
	 *  -  Recalculate the centers based on the new clusters
	 */
	Mat _cluster(const Mat& descriptors);
public:
	mClustererCustomKMeans(int nClusters, bool shiftCenters);
	~mClustererCustomKMeans();
};


/**
 * Extract codebook from a set of feature descriptors
 * Feature matching using K-NN, and clustering using K-Means
 */
class mClassCodebook
{
	mAbstractClusterer* _classifier;
	string _className;
	Mat _classDictionary;

	double maxDistance;
public:
	mClassCodebook(const string& name, mAbstractClusterer* classifier);
	~mClassCodebook();

	Mat getCodebook() { return _classDictionary; }

	/**
	 * Generate a codebook of visual words from the images
	 * Compute each image's descriptor
	 *  - Aggregate all features from all images
	 *  - Cluster descriptors using a classifier
	 *  - Result is some visual words for that category
	 */
	void createCodebook(const vector<mImage*>& trainingImages);

	/**
	 * Count the frequency of the word's occurence in the image
	 *  - for each feature, find the nearest center
	 *  - count the frequency of the center, histogram
	 * - When looking for nearest codeword, consider the distance compared to the maximum distance
	 */
	void calculateHistogram(mImage* image);

	/**
	 * For each image, compute its descriptor after the codebook is created
	 */
	void calculateHistograms(const vector<mImage*>& trainingImages);
};

/**
 * Dataset to store the images
 */
class Dataset
{
	string _className;
	string _rootPath;
	string _positiveImage;
	int _numPositiveImage;
	string _negativeImage;
	int _numNegativeImage;
	string _imageFormat;

	vector<string> _getListOfImages(const string& file, int num);
public:
	Dataset(	const string& className,
				const string& root_path,
				const string& pos, int nPositive,
				const string& neg, int nNegative,
				const string& imageFormat);
	~Dataset();

	vector<string> getPositiveImages();
	vector<string> getNegativeImages();
};

/**
 * Abstract class for classification
 *
 * Subclass can be like SVM, Naive Bayes
 *
 * TODO given a dataset
 *        convert to array of images;
 *        images must have their descriptors and histograms already computed
 */
class mAbstractImageClassifier
{
public:
	mAbstractImageClassifier() {}
	virtual ~mAbstractImageClassifier() {}

	virtual void train(const vector<mImage*>& positive, const vector<mImage*>& negative) = 0;
	virtual int classify(mImage* image) = 0;
};

/**
 * mNaiveBayesClassifier
 *
 * Naive Bayes Classifier
 */
class mNaiveBayesClassifier : public mAbstractImageClassifier
{
	mClassCodebook* _codebook;
	Mat _probCodewords;
public:
	mNaiveBayesClassifier(mClassCodebook* codebook);
	~mNaiveBayesClassifier();
	/**
	 * Take each codeword in the codebook as a separate feature, calculate its probability given a class (+/-)
	 * For each codeword, find the probability of it given a +/- class
	 * - For each codeword
	 *  - For each image
	 *   * find frequency of codeword appearance in the image
	 *   * add up the true positives and the false negatives
	 *  - compute the probability of that codeword given the hypothesis +/-
	 *      from the given class of the training images
	 */
	void train(const vector<mImage*>& positive, const vector<mImage*>& negative);

	/**
	 *
	 */
	int classify(mImage* image);
};

#endif
