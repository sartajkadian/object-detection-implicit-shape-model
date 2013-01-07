#include <mBagOfFeatures.h>
#include "Util.h"

#include <cstdio>
#include <iostream>

using namespace std;
using namespace cv;

//#define DEBUG
//#define DEBUG_STATIC

/**
 * mImage
 */
mImage::mImage(const string& im) :
	_name(im),
	_image(imread(im, CV_LOAD_IMAGE_GRAYSCALE)),
	_featureKeypoints(),
	_featureDescriptors(),
	_isComputed(false),
	_imageHistogram()
{}
mImage::~mImage() {}

vector<pair<int,int> > mImage::matchFeatures(mImage* image)
{
	vector<pair<int,int> > matches;

	Mat desc1 = _featureDescriptors;
	Mat desc2 = image->getDescriptors();

	for(int i = 0; i < desc1.rows; i++)
	{
		double nearest1 = 1000000, nearest2 = 1000000;
		int nearestInd = -1;

		for(int j = 0; j < desc2.rows; j++)
		{
			double dist = Util::getSqDistance(desc1.row(i), desc2.row(j));

			if (dist < nearest1)
			{
				nearest1 = dist;
				nearestInd = j;
			}
			else if (dist < nearest2)
			{
				nearest2 = dist;
			}
		}

		if (nearestInd != -1 && nearest1 < 0.6*nearest2)
		{
			matches.push_back(pair<int,int>(i,nearestInd));
		}
	}

	return matches;
}

unsigned int mImage::getClosestMatch(const vector<mImage*>& othImages, double& dist)
{
	unsigned int min_ind = -1;
	double min_dist = INFINITY;

	for(unsigned int i = 0; i < othImages.size(); i++)
	{
		double dist = Util::getSqDistance(_imageHistogram, othImages[i]->getImageHistogram());
		if (dist < min_dist)
		{
			min_dist = dist;
			min_ind = i;
		}
	}

	dist = sqrt(min_dist);
	return min_ind;
}

Mat mImage::drawFeatures()
{
	Mat tmpImg = Mat();
	drawKeypoints(_image, _featureKeypoints, tmpImg);
	return tmpImg;
}
Mat mImage::drawFeatures(const vector<int>& kpIndices, const Scalar& color)
{
	vector<KeyPoint> kp;
	for(unsigned int i = 0; i < kpIndices.size(); i++)
	{
		kp.push_back(_featureKeypoints[kpIndices[i]]);
	}

	Mat tmpImg = Mat();
	drawKeypoints(_image, kp, tmpImg, color);
	return tmpImg;
}

/**
 * mClassifier
 */
Mat mAbstractClusterer::cluster(const Mat& descriptors)
{
	_descriptors = descriptors;

	Mat clusterCenters = _cluster(_descriptors);

	// index the descriptors using KDtree algorithm for later search
	cv::flann::Index ind(_descriptors, flann::KDTreeIndexParams(4));
	// find the nearest descriptor to the centers and consider that the center
	for(int i = 0; i < clusterCenters.rows; i++)
	{
		Mat indices, dists;
		ind.knnSearch(clusterCenters.row(i), indices, dists, 1, flann::SearchParams());

		// update center position
		Mat dst = clusterCenters.row(i);
		_descriptors.row(indices.at<int>(0)).copyTo(dst);
	}

	return clusterCenters;
}

/**
 * mClassifierKMeans
 */
mClustererKMeans::mClustererKMeans(int nClusters, bool shiftCenters) :
	mAbstractClusterer(nClusters, shiftCenters)
{}
mClustererKMeans::~mClustererKMeans() {}

Mat mClustererKMeans::_cluster(const Mat& descriptors)
{
	Mat labels;
	Mat clusterCenters = Mat::zeros(Size(descriptors.cols, _getNumClusters()), descriptors.type());
	kmeans(descriptors, _getNumClusters(), labels, TermCriteria(TermCriteria::COUNT, 3, 10), 1, KMEANS_PP_CENTERS, clusterCenters);

#ifdef DEBUG
	cout << "KMeans: Best Labels " << labels << endl;
#endif

	return clusterCenters;
}

/**
 * mClassifierCustomKMeans
 */
mClustererCustomKMeans::mClustererCustomKMeans(int nClusters, bool shiftCenters) :
	mAbstractClusterer(nClusters, shiftCenters)
{

}
mClustererCustomKMeans::~mClustererCustomKMeans() {}

Mat mClustererCustomKMeans::_cluster(const Mat& descriptors)
{
	cout << "* Creating initial center points *" << endl;
	// start with random centers, choose from the set of points
	RNG rng;
	Mat center_points(Size(descriptors.cols,_getNumClusters()), descriptors.type());
	for(int i = 0; i < _getNumClusters(); i++)
	{
		int ridx = rng.uniform(0, descriptors.rows);
		//TODO alternative to use Mat::copyTo
		center_points.row(i) = descriptors.row(ridx) + Mat::zeros(Size(descriptors.row(ridx).cols, descriptors.row(ridx).rows), descriptors.type());
#ifdef DEBUG
		cout << " Center Point [" << i << "] " << center_points.row(i) << "; p " << ridx << endl;
#endif
	}

	Mat lastClusterCenters = center_points;

	do
	{
		lastClusterCenters = center_points;

		// get nearest points, store points
		cout << "* Grouping points in clusters *" << endl;
		vector<vector<Mat> > clusterd_points(_getNumClusters());
		for(int i = 0; i < descriptors.rows; i++)
		{
			// which cluster the point belongs to
			// get nearest cluster center to the point
			double minDist = 1000000;
			int nearestCluster = 0;
			for(int ic = 0; ic < _getNumClusters(); ic++)
			{
				double dist = Util::getSqDistance(descriptors.row(i), center_points.row(ic));
				if (dist < minDist)
				{
					minDist = dist;
					nearestCluster = ic;
				}
			}
			// store point in its corresponding cluster
			clusterd_points[nearestCluster].push_back(descriptors.row(i));
#ifdef DEBUG
			cout << " Added point " << descriptors.row(i) << " to cluster [" << nearestCluster << "]; Distance " << minDist << endl;
#endif
		}

		// recalculate cluster centers
		cout << "* Recalculate cluster centers *" << endl;
		for(unsigned int i = 0; i < clusterd_points.size(); i++)
		{
			Mat meanM = Util::getMean(clusterd_points[i]);
			center_points.row(i) = meanM + Mat::zeros(Size(meanM.cols, meanM.rows), meanM.type());
#ifdef DEBUG
			cout << " Update centre point [" << i << "] " << center_points.row(i) << endl;
#endif
		}

		cout << "* Checking end condition *" << endl;
#ifdef DEBUG
		cout << "Current center points " << center_points << endl << "Last center points " << lastClusterCenters << endl;
#endif
		// end if the centers stop changing
	} while(!Util::approximateEq(center_points, lastClusterCenters, 0.1));

	return center_points;
}


/**
 * mFeatureCodebookTrainer
 */
mClassCodebook::mClassCodebook(const string& name, mAbstractClusterer* classifier) :
	_classifier(classifier),
	_className(name),
	_classDictionary()
{}
mClassCodebook::~mClassCodebook() {}

void mClassCodebook::calculateHistogram(mImage* image)
{
	if (!image->isComputed())
	{
		cerr << "* Features have not been computed *" << endl;
 		return;
	}

#ifdef DEBUG
	cout << "* Calculating image histogram for '" << image->getName() << "' *" << endl;
#endif

	Mat image_histogram = Mat::zeros(_classDictionary.rows, 1, CV_32F);

	Mat im_descriptors = image->getDescriptors();

	//TODO consider normalizing, getting a matrix multiplication between the codebook MxN matrix and the Nx1 descriptor vector
	//TODO the result is a vector representing the dot product between each the codeword and the descriptor
	//TODO and then pick the minimum value, index of which corresponds to the closest center
	// foreach feature point, get closest center
	for(int imdi = 0; imdi < im_descriptors.rows; imdi++)
	{
		// closest center
		int min_center_ind = -1;
		double min_dist_sq = INFINITY;

		for(int ci = 0; ci < _classDictionary.rows; ci++)
		{
			double dist = Util::getSqDistance(im_descriptors.row(imdi), _classDictionary.row(ci));
			if (dist < min_dist_sq)
			{
				min_dist_sq = dist;
				min_center_ind = ci;
			}
		}

		// update center histogram
		if (min_center_ind != -1)
		{
#ifdef DEBUG
			cout << "- - Image feature " << imdi << endl << im_descriptors.row(imdi) << endl << "   matches visual word " << min_center_ind << endl << _classDictionary.row(min_center_ind) << endl;
#endif
			image_histogram.at<float>(0, min_center_ind) += 1.0f;
		}
	}

//	// normalize histogram
//	Scalar sum = cv::sum(image_histogram);
//	for(int i = 0; i < image_histogram.rows; i++)
//	{
//		image_histogram.at<float>(0, i) /= (float)sum[0];
//	}


#ifdef DEBUG
	cout << "---> Resulting Histogram " << endl << image_histogram << endl;
#endif
	image->setImageHistogram(image_histogram);
}
void mClassCodebook::calculateHistograms(const vector<mImage*>& trainingImages)
{
	cout << "* Calculating histogram of images for class '" << _className << "' *" << endl;
	for(unsigned int i = 0; i < trainingImages.size(); i++)
	{
		calculateHistogram(trainingImages[i]);
	}
}

void mClassCodebook::createCodebook(const vector<mImage*>& images)
{
	cout << "* Creating codebook for class '" << _className << "' *" << endl;

	int rows = 0;
	for(unsigned int i = 0; i < images.size(); i++)
	{
		rows += images[i]->getDescriptors().rows;
	}
	Mat total_descriptors = Mat::zeros(Size(images[0]->getDescriptors().cols, rows), images[0]->getDescriptors().type());

	rows = 0;
	for(unsigned int i = 0; i < images.size(); i++)
	{
		int tmpRend = rows + images[i]->getDescriptors().rows;
		Mat dst = total_descriptors.rowRange(rows, tmpRend);
		images[i]->getDescriptors().copyTo(dst);
	}

	// cluster aggregated descriptors from all images
	_classDictionary = _classifier->cluster(total_descriptors);
}

/**
 * Dataset
 */
Dataset::Dataset(	const string& className, const string& root_path,
		const string& pos, int nPositive,
		const string& neg, int nNegative,
		const string& imageFormat) :
	_className(className),
	_rootPath(root_path),
	_positiveImage(pos),
	_numPositiveImage(nPositive),
	_negativeImage(neg),
	_numNegativeImage(nNegative),
	_imageFormat(imageFormat)
{}
Dataset::~Dataset() {}

vector<string> Dataset::_getListOfImages(const string& file, int num)
{
	vector<string> list(num);
	for(int i = 0; i < num; i++)
	{
		char* full_path = new char[_rootPath.length()+file.length()+5+_imageFormat.length()];
		sprintf(full_path, "%04d", (i+1));
		strcpy(full_path, (_rootPath+file+full_path).c_str());
		strcat(full_path, _imageFormat.c_str());
		list[i] = string(full_path);
	}
	return list;
}

vector<string> Dataset::getPositiveImages()
{
	return _getListOfImages(_positiveImage, _numPositiveImage);
}
vector<string> Dataset::getNegativeImages()
{
	return _getListOfImages(_negativeImage, _numNegativeImage);
}

/**
 * mNaiveBayesClassifier
 */
mNaiveBayesClassifier::mNaiveBayesClassifier(mClassCodebook* codebook) :
	mAbstractImageClassifier(),
	_codebook(codebook)
{
	// for each codewords, and the hypothesis +/-
	// 0 for -, 1 for +
	_probCodewords = Mat::zeros(_codebook->getCodebook().rows, 2, CV_32F);

	//TODO load probabilities to file
}
mNaiveBayesClassifier::~mNaiveBayesClassifier()
{
	//TODO save probabilities to file, caching
}
void mNaiveBayesClassifier::train(const vector<mImage*>& positive, const vector<mImage*>& negative)
{
	Mat codeb = _codebook->getCodebook();
	for(int i = 0; i < codeb.rows; i++)
	{
		// number of codeword occurences in hypothesis +/-
		float num_cw = 0;
		// total number of codewords in hypothesis +/-
		float sum_cw = 0;
		for(unsigned int imi = 0; imi < positive.size(); imi++)
		{
			num_cw += positive[imi]->getImageHistogram().at<float>(0,i);
			sum_cw += sum(positive[imi]->getImageHistogram()).val[0];
		}
		_probCodewords.at<float>(1,i) = num_cw / sum_cw;

		num_cw = 0;
		sum_cw = 0;
		for(unsigned int imi = 0; imi < negative.size(); imi++)
		{
			num_cw += negative[imi]->getImageHistogram().at<float>(0,i);
			sum_cw += sum(positive[imi]->getImageHistogram()).val[0];
		}
		_probCodewords.at<float>(0,i) = num_cw / sum_cw;
	}
}
int mNaiveBayesClassifier::classify(mImage* image)
{
	cout << "Probability Matrix" << endl << _probCodewords << endl;
	return 0;
}
