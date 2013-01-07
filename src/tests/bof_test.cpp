#include <mBagOfFeatures.h>

#include <iostream>
using namespace std;


void genDataset(int num, vector<mImage*>& posset, vector<mImage*>& negset, vector<mImage*>& testingset)
{
	Dataset dataset("faces", "/home/mmakhalaf/Documents/bag_words_demo/images/", "faces2/image_", num, "background_caltech/image_", num, ".jpg");
	vector<string> positive_path = dataset.getPositiveImages();
	vector<string> negative_path = dataset.getNegativeImages();

	// create all images
	// detect keypoints, compute their descriptors

	vector<mImage*> all_pos(positive_path.size());
	vector<mImage*> all_neg(negative_path.size());

	SiftFeatureDetector siftFtD;
	SiftDescriptorExtractor siftFtEx;

	for(uint i = 0; i < all_pos.size(); i++)
	{
		all_pos[i] = new mImage(positive_path[i]);
		all_pos[i]->compute<SiftFeatureDetector,SiftDescriptorExtractor>(siftFtD, siftFtEx);
	}
	for(uint i = 0; i < all_neg.size(); i++)
	{
		all_neg[i] = new mImage(negative_path[i]);
		all_neg[i]->compute<SiftFeatureDetector,SiftDescriptorExtractor>(siftFtD, siftFtEx);
	}

	// generate random training set of image +/-
	Mat m = Mat::zeros(0.5*num,1,CV_32S);
	RNG r;
	r.fill(m, RNG::UNIFORM, Scalar(1), Scalar(num));

	// add randomly selected images to training set
	for(int i = 0; i < m.rows; i++)
	{
		uint ind = m.at<int>(0,i);
		if (ind < all_pos.size())
		{
			posset.push_back(all_pos[ind]);
		}
		if (ind < all_neg.size())
		{
			negset.push_back(all_neg[ind]);
		}
	}

	// add all images to testing set
	testingset.clear();
	for(uint i = 0; i < all_pos.size(); i++)
	{
		testingset.push_back(all_pos[i]);
	}
	for(uint i = 0; i < all_neg.size(); i++)
	{
		testingset.push_back(all_neg[i]);
	}
}

int main(int argc, char** argv)
{
	vector<mImage*> train_pos, train_neg, testset;
	genDataset(5, train_pos, train_neg, testset);

	mClassCodebook codebook("faces", new mClustererKMeans(100,false));
	codebook.createCodebook(testset);
	codebook.calculateHistograms(testset);

	mNaiveBayesClassifier classifier(&codebook);
	classifier.train(train_pos, train_neg);
	classifier.classify(NULL);
}
