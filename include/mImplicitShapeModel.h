
#ifndef M_ISM_H_
#define M_ISM_H_


#include <opencv.hpp>
using namespace cv;

#include <map>
#include <climits>
#include <tr1/memory>
using namespace std;

#include <mDataset.h>
#include <mDistance.h>

class mISMimage;
typedef tr1::shared_ptr<mISMimage> mISMimage_p;

class mImageDescriptor;
typedef tr1::shared_ptr<mImageDescriptor> mImageDescriptor_p;

class mCodebookEntry;
typedef tr1::shared_ptr<mCodebookEntry> mCodebookEntry_p;

class mMatch;
typedef tr1::shared_ptr<mMatch> mMatch_p;

class mCodebook;
typedef tr1::shared_ptr<mCodebook> mCodebook_p;

typedef vector<mCodebookEntry_p> vector_CodebookEntry;
typedef map<string, vector<mCodebookEntry_p> > map_CodebookEntry;


///**
// * mHarrisCornerDetector
// */
//class mHarrisCornerDetector : public FeatureDetector
//{
//	float _threshold;
//	// neighbourhood to consider for eigen vector maximas
//	int _blockSize;
//	// kernel size for edge detection
//	int _kSize;
//	// weighting trace in final value equation
//	double _sigma;
//	// what to do when pixels out bounds are required (how they're interpolated)
//	int _borderType;
//public:
//	mHarrisCornerDetector(float threshold, int bsize = 3, int ksize = 3, double sigma = 0.06, int borderType = BORDER_DEFAULT);
//	~mHarrisCornerDetector();
//protected:
//	void detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask=Mat()) const;
//};

/**
 * mAnnotatedImage
 */
class mISMimage
{
	mAbstractAnnotation_p _annotation;
	Mat _imageData;

	vector<mImageDescriptor_p> _compute(const Rect& bb, FeatureDetector* det, DescriptorExtractor* ext = NULL);
public:
	mISMimage(mAbstractAnnotation_p annotation);
	~mISMimage();

	void getImageDim(int& rows, int& cols) { rows = _imageData.rows; cols = _imageData.cols; }
	Mat getImage() { return _imageData; }

	vector<mImageDescriptor_p> compute(FeatureDetector* det, DescriptorExtractor* ext = NULL);
	vector<mImageDescriptor_p> compute(const string& classname, FeatureDetector* det, DescriptorExtractor* ext = NULL);

//	int getNumObjects(const string& classname);
//	vector<Rect> getObjectBoundingBox(const string& classname);
	Mat drawBoudingBox(const string& classname);

	void validateDetection(const string& classname, const vector<Rect>& det, bool& truepos, bool& falsepos, bool& trueneg, bool& falseneg, float det_wind_ratio = 0.5);

	static vector<mISMimage_p> createImages(const vector<mAbstractAnnotation_p>& ann)
	{
		vector<mISMimage_p> imageset(ann.size());
		for(uint i = 0; i < ann.size(); i++)
		{
			imageset[i] = mISMimage_p(new mISMimage(ann[i]));
		}
		return imageset;
	}
};

/**
 * mImageDescriptor
 */
class mImageDescriptor
{
	Mat _imagePatch;
	Mat _descriptor;
	Point2i _relativePosition;
	float _scale;
public:
	mImageDescriptor();
	mImageDescriptor(const Mat& patch, const Mat& descriptor, const Point2i& relative_position, float scale);
	~mImageDescriptor();

	Mat getDescriptor() { return _descriptor; }
	Mat getPatch() { return _imagePatch; }
	Point2i getRelativeLocation() { return _relativePosition; }
	float getScale() { return _scale; }

	// write codebook to file
	void write(FileStorage& fs);
	// read data from file
	void read(const FileNode& node);

	static int PATCH_SIZE;
	static Mat extractPatch(const Mat& m, const Point2i& p);
	static void addPatch(Mat& m, const Mat& patch, const Point2i& p);
};

/**
 * mCodebookEntry
 */
class mCodebookEntry
{
	string _className;
	Mat _clusterCenter;
	vector<mImageDescriptor_p> _contribDescriptors;
public:
	mCodebookEntry();
	mCodebookEntry(const string& classname, const Mat& center, vector<mImageDescriptor_p> contribDescriptors);
	~mCodebookEntry();

	string getClass() { return _className; }
	Mat getCodebookEntry() { return _clusterCenter; }
	vector<mImageDescriptor_p> getDescriptors() { return _contribDescriptors; }

	// write codebook to file
	void write(FileStorage& fs);
	// read data from file
	void read(const FileNode& node);
};


/**
 * mMatch
 *
 * Hold data on match between a given image patch and its nearest codeword
 *  Holds the centre points this match voted for
 */
class mMatch
{
	mISMimage_p _image;
	mImageDescriptor_p _imagePatch;
//	vector<Point2i> _votedCentres;
	vector<mCodebookEntry_p> _codewords;
public:
	mMatch(mISMimage_p image, mImageDescriptor_p im_desc) :
		_image(image),
		_imagePatch(im_desc),
		_codewords()
	{}

	mImageDescriptor_p getImageDescriptor() { return _imagePatch; }

	vector<mCodebookEntry_p> getMatchingCodewords() { return _codewords; }
	void addCodeword(mCodebookEntry_p cdw)
	{
		_codewords.push_back(cdw);
	}
};

/**
 * mCodebook
 */
class mCodebook
{
	vector<mCodebookEntry_p> _codebookEntries;
	double _threshold;
	double _epsilon;
	int _maxNumberOfCBE;
	Vec2i _minMax_NumDescriptors;

	vector<mImageDescriptor_p> _aggregateDescriptors(const vector<vector<mImageDescriptor_p> >& im_descs, Mat& descs_mat);

	vector<mImageDescriptor_p> _sortDescriptors(const vector<mImageDescriptor_p>& desc, const Mat& dists);
	vector<mCodebookEntry_p> _getHighestCBE(const vector<mCodebookEntry_p>& class_cbe);
public:
	mCodebook();
	mCodebook(double threshold, double ep, int max_num_codebookentries = INT_MAX, const Vec2i& num_descriptors_minmax = Vec2i(1,INT_MAX));
	~mCodebook();

	double getThreshold() { return _threshold; }

	/**
	 * Compute descriptors in regions that we know contains the objects
	 * Gather all image descriptors in one array
	 * Do agglomerative clustering on data points
	 * For each center
	 *  Store its descriptor
	 *  Store descriptors that contributed to it above threshold
	 *  Calculate their relative position to the center of the object
	 */
	void compute(const string& class_name, const vector<vector<mImageDescriptor_p> >& im_descs, mAbstractSimilarity* simm);// , const vector<mISMimage_p> images, FeatureDetector* det);

	/**
	 * TODO TEST
	 * Match a descriptor in the image to the codebook
	 * Matching codebook will have a similarity greater than the threshold
	 */
	vector<mMatch_p> match(mISMimage_p image, FeatureDetector* det) const;

	vector<mCodebookEntry_p> getCodebookEntries(const string& classname);
	vector<mCodebookEntry_p> getCodebookEntries() { return _codebookEntries; }

	//TODO write patch size
	// write codebook to file
	void write(const string& output_codebook);
	// read data from file
	bool read(const string& input_codebook);

	static vector<Rect> detect(mISMimage_p image, const mCodebook& cb, FeatureDetector* det, int xbins, int ybins, int kde_window, float kde_threshold);
};

/**
 * Output Stream Functions
 */
ostream& operator<<(ostream& out, mCodebook& cb)
{
	vector<mCodebookEntry_p> cbe = cb.getCodebookEntries();
	out << "Codebook size: " << cbe.size() << endl;
	out << "  Threshold: " << cb.getThreshold() << endl;
	return out;
}

ostream& operator<<(ostream& out, Rect& r)
{
	out << "[" << r.x << ", " << r.y << ", " << r.width << ", " << r.height << "]" << flush;
	return out;
}

uint idxToLinear(const Point2i& p, const int& height)
{
	return (p.x * height + p.y);
}

Point2i linearToIdx(uint idx, const int& height)
{
	Point2i p;
	p.x = (float)idx / (float)height;
	p.y = idx % height;
	return p;
}

/**
 * mPGHoughTransform
 *
 * TODO
 * Extract image patches in vector v
 * Get vector of matching codeword entries vw
 * Construct Hough space of x, y
 * For each codeoword matched,
 *   For each training patch with a stored relative position
 *     Increment the corresponding (x,y) in the hough space
 *     [For multiple classes] store the number of votes each class has cast for this point (x,y)
 * Group Hough space matrix into bins
 * Apply Non-maxima suppression on binned space
 * Mean-Shift Mode estimation on corresponding areas in continuous hough space to get the highest votes, store in a vector of (x,y)
 * Back projection, For each maxima, retrieve the codewords and matching image patches that voted for it
 */
class mPGHoughTransform
{
	mISMimage_p _image;
	vector<mMatch_p> _matches;
public:
	enum DiscreteType
	{
		TYPE_BINS,
		TYPE_PIXELS
	};

	mPGHoughTransform(mISMimage_p image, const vector<mMatch_p>& matches);
	~mPGHoughTransform();

	/**
	 * Each codeword with all its contributing descriptors vote for centre poisition
	 */
	Mat vote(map<uint,vector<int> >& patches_per_centre);

	/**
	 * Descritizes the hough space into a number of bins
	 * TODO option for defining size of the bin
	 */
	Mat discretizeVotes(const Mat& houghv, int xnbins, int ynbins, DiscreteType type = TYPE_BINS);


	/**
	 * Obtain global maxima
	 * Threshold the votes
	 */
	vector<vector<Point2i> > searchForMaxima(const Mat& houghv, const Mat& d_houghv, const vector<Point2i>& centers, int window_size, float thresh);

	/**
	 * each match holds the centres voted for by all contrib descriptors
	 * for each centre maxima
	 *     for each voted centre in match
	 *         if the match voted for the centre, add its index to corresponding centre
	 *              a match could have voted for multiple peaks
	 * parameters:
	 *  list of matches, contributing match for each centres, centre positions, new centres
	 */
	vector<vector<int> > backprojectPatches(const map<uint,vector<int> >& patches_per_centre, const vector<vector<Point2i> >& centre_p);

	/**
	 * Get the matching patches around a centre
	 *  draw bounding box around them
	 */
	Rect computeBoundingBox(const vector<int>& matches_idx);
	vector<Rect> computeBoundingBoxes(const vector<vector<int> >& matches_idx)
	{
		vector<Rect> vec;
		for(int i = 0; i < matches_idx.size(); i++)
		{
			vec.push_back(computeBoundingBox(matches_idx[i]));
		}
		return vec;
	}

	static Rect mergeRect(const Rect& r1, const Rect& r2)
	{
		Rect r = r1;
		if (r.x > r2.x)
		{
			r.x = r2.x;
			r.width += r1.x - r2.x;
		}
		if (r.y > r2.y)
		{
			r.y = r2.y;
			r.height += r1.y - r2.y;
		}

		if (r.width < r2.br().x-r.x)
		{
			r.width = r2.br().x-r.x;
		}
		if (r.height < r2.br().y-r.y)
		{
			r.height = r2.br().y-r.y;
		}
		return r;
	}
};

/**
 * Utility class for maxima search algorithms
 */
class mMaximaSearch
{
	static Point2i _getMean(const Mat& m, const Rect& reg);

public:

	/**
	 * Performas Non-maxiam suppression
	 */
	static vector<Point2i> nonMaximaSuppression(const Mat& mat, int window_size, float thresh);

	static vector<Point2i> meanShiftMaximaSearch(const Mat& mat, int wid_size, float kernel_bw, float thresh);

	/**
	 * Mean Shift Maxima search
	 *
	 * TODO keep points that voted for the maxima above a threshold
	 */
	static vector<Point2i> maxShiftMaximaSearch(const Mat& mat, int window_size = 3, float thresh = 0);
};


/**
 * Debug functions
 */
class mDebug
{
public:
	static void showMatch(const string& winname, mISMimage_p image, mMatch_p match, bool do_write = true);

	static void showMatches(const string& winname, mISMimage_p image, const vector<mMatch_p>& matches, bool disp_image = false, bool disp_centres = false, bool do_write = true);

	static void showImage(const string& winname, const Mat& mat, bool norm = true, bool disp_minmax = true, bool do_write = true);

	static void showPoints(const string& winname, const vector<vector<Point2i> >& points, Mat& im, bool writeToImage = false, bool do_write = true);

	static void showCenters(const string& winname, const Mat& image, const vector<mMatch_p>& matches, const vector<vector<Point2i> >& centres_max, const vector<vector<int> >& contrib_patches, bool do_write = true);

	static void showPatches(const string& winname, mISMimage_p image, vector<mImageDescriptor_p> desc, bool do_write = true);

	static void showPatches(const string& winname, mISMimage_p image, FeatureDetector* det, bool do_write = true);

	static void backprojection(const string& winname, mISMimage_p image, const vector<mMatch_p>& matches, const vector<vector<int> > contrib_patches, bool do_write = true);
};

#endif
