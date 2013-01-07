#include <map>
#include <exception>
using namespace std;

#include <mImplicitShapeModel.h>
#include <aggclustering.h>
#include <Util.h>

using namespace cv;

///**
// * mHarrisCornerDetector
// */
//mHarrisCornerDetector::mHarrisCornerDetector(float threshold, int bsize, int ksize, double sigma, int borderType) :
//	_threshold(threshold),
//	_blockSize(bsize),
//	_kSize(ksize),
//	_sigma(sigma),
//	_borderType(borderType)
// {}
//mHarrisCornerDetector::~mHarrisCornerDetector() {}
//
//void mHarrisCornerDetector::detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask) const
//{
//	Mat dst;
//	cornerHarris(image, dst, _blockSize, _kSize, _sigma, _borderType);
//
//	dst = mMaximaSearch::nonMaximaSuppression(dst, 3);
//
//	double vmin, vmax;
//	Point2i pmin, pmax;
//	minMaxLoc(dst, &vmin, &vmax, &pmin, &pmax);
//
//	float thresh = vmin + (vmax-vmin)*_threshold;
//	cout << "** Threshold " << thresh << endl;
//	cout << "min " << vmin << " " << pmin << endl;
//	cout << "max " << vmax << " " << pmax << endl;
//
//	for(int r = 0; r < dst.rows; r++)
//	{
//		for(int c = 0; c < dst.cols; c++)
//		{
//			if (dst.at<float>(r,c) > thresh)
//			{
//				keypoints.push_back(KeyPoint(Point2f(c,r), 1, 0, dst.at<float>(r,c), 1));
//			}
//		}
//	}
//
//	imshow("Harr", dst);
//	waitKey(0);
//}

/**
 * mAnnotatedImage
 */
mISMimage::mISMimage(mAbstractAnnotation_p annotation) :
	_annotation(annotation),
	_imageData(imread(annotation->getImagePath(), CV_LOAD_IMAGE_GRAYSCALE))
{
	if (_imageData.empty())
	{
		cout << "** Image " << annotation->getImagePath() << " not found" << endl;
	}
}
mISMimage::~mISMimage()
{
	_imageData.release();
}

vector<mImageDescriptor_p> mISMimage::_compute(const Rect& bb, FeatureDetector* det, DescriptorExtractor* ext)
{
	Mat im;
	if (bb.width == 0)
	{
		im = _imageData;
	}
	else
	{
		im = Mat(_imageData, bb);
	}

	vector<KeyPoint> nkps;
	det->detect(im, nkps);

	// sample keypoints so the spaces between them are 0.25 of their size
	vector<KeyPoint> kps;
	for (uint ni = 0; ni < nkps.size(); ni++)
	{
		bool doadd = true;
		for(uint i = 0; i < kps.size(); i++)
		{
			KeyPoint a = nkps[ni], b = kps[i];
			float dist = sqrt(pow(a.pt.x - b.pt.x, 2) + pow(a.pt.y -b.pt.y, 2));
			if (dist != 0 && dist < 0.25*(float)mImageDescriptor::PATCH_SIZE)
			{
				doadd = false;
				break;
			}
		}

		if (doadd)
		{
			kps.push_back(nkps[ni]);
		}
	}

	Mat descs;
	if (ext != NULL)
	{
		ext->compute(im, kps, descs);
	}

	// center of object as defined from bounding box
	// or 0,0 if no bounding box defined (in the case of processing whole image)
	Point2i c;
	if (bb.width == 0)
	{
		c = Point2i(0,0);
	}
	else
	{
		c = Util::getCenter(im);
	}

	vector<mImageDescriptor_p> descriptors(kps.size());

	for(uint i = 0; i < kps.size(); i++)
	{
		// relative position to center
		Point2i rp(kps[i].pt.x-c.x, kps[i].pt.y-c.y);

		// keypoint position in bigger image
		Point2f kp_global(kps[i].pt.x+bb.x, kps[i].pt.y+bb.y);

		// image patch
		Mat patch = mImageDescriptor::extractPatch(_imageData, kp_global);
		patch.convertTo(patch, CV_32F);
		patch /= 255.0f;
		Mat desc;
		if (ext == NULL)
		{
			// use patch as descriptor
			desc = patch.clone();
			desc = desc.reshape(0,1);
		}
		else
		{
			desc = descs.row(i).clone();
			normalize(desc, desc);
		}

		// store representational part
		descriptors[i] = mImageDescriptor_p(new mImageDescriptor(patch, desc, rp, kps[i].size));
	}

	return descriptors;
}

vector<mImageDescriptor_p> mISMimage::compute(FeatureDetector* det, DescriptorExtractor* ext)
{
	return _compute(Rect(0,0,0,0), det, ext);
}

vector<mImageDescriptor_p> mISMimage::compute(const string& classname, FeatureDetector* det, DescriptorExtractor* ext)
{
	vector<mImageDescriptor_p> descs;

	if (!_annotation)
	{
		cout << " *-* Image is not annotated" << endl;
	}

	vector<Rect> bbs;
	if (!_annotation->getBoundingBoxes(classname, bbs))
	{
		cerr << "* No objects of class " << classname << " in image [" << _annotation->getImagePath() << "]" << endl;
		return descs;
	}

	cout << "* Computing Descriptors for class " << classname << " in [" << _annotation->getImagePath() << "]" << endl;

	for(uint i = 0; i < bbs.size(); i++)
	{
		vector<mImageDescriptor_p> bb_descs = _compute(bbs[i], det, ext);
		descs.insert(descs.end(), bb_descs.begin(), bb_descs.end());
	}
	return descs;
}

//int mISMimage::getNumObjects(const string& classname)
//{
//	vector<Rect> bbs;
//	if (!_annotation->getBoundingBoxes(classname, bbs))
//	{
//		return 0;
//	}
//	return bbs.size();
//}
//
//vector<Rect> mISMimage::getObjectBoundingBox(const string& classname)
//{
//	vector<Rect> bbs;
//	_annotation->getBoundingBoxes(classname, bbs);
//	return bbs;
//}

Mat mISMimage::drawBoudingBox(const string& classname)
{
	Mat m = _imageData.clone();
	vector<Rect> bbs;
	if (!_annotation->getBoundingBoxes(classname, bbs))
	{
		cerr << "* No objects of class " << classname << " in " << _annotation->getImagePath() << endl;
		return m;
	}

	for(uint i = 0; i < bbs.size(); i++)
	{
		rectangle(m, bbs[i], Scalar(255,255,255), 2);
	}
	return m;
}

void mISMimage::validateDetection(const string& classname, const vector<Rect>& det, bool& truepos, bool& falsepos, bool& trueneg, bool& falseneg, float det_wind_ratio)
{
	// if no detections, and no classes, true negative
	truepos = false;
	falsepos = false;
	trueneg = false;
	falseneg = false;

	vector<Rect> truedet;
	_annotation->getBoundingBoxes(classname, truedet);

	// no object, no detections, true negative
	if (truedet.empty() && det.empty())
	{
		trueneg = true;
		return;
	}

	// no object, yes detections, false positive
	if (truedet.empty() && !det.empty())
	{
		falsepos = true;
		return;
	}

	// yes object, no detections, false negative
	if (!truedet.empty() && det.empty())
	{
		falseneg = true;
		return;
	}


	// yes object, yes detections, true positive
	for(vector<Rect>::const_iterator itr = det.begin(); itr != det.end(); itr++)
	{
		for(vector<Rect>::iterator titr = truedet.begin(); titr != truedet.end(); titr++)
		{
			float ar = titr->area();
			if (((*titr)&(*itr)).area() > det_wind_ratio*ar)
			{
				truepos = true;
			}
		}
	}

	// yes object, no detections
	if (!truepos)
	{
		falseneg = true;
	}
}

/**
 * mImageDescriptor
 */

int mImageDescriptor::PATCH_SIZE = 30;

mImageDescriptor::mImageDescriptor() :
	_descriptor(),
	_imagePatch(),
	_relativePosition(),
	_scale()
{}
mImageDescriptor::mImageDescriptor(const Mat& patch, const Mat& descriptor, const Point2i& relative_position, float scale) :
	_relativePosition(relative_position),
	_scale(scale)
{
	patch.copyTo(_imagePatch);
//	_imagePatch.convertTo(_imagePatch, CV_32F);
//	_imagePatch = _imagePatch / 255.0f;

	descriptor.copyTo(_descriptor);
//	_imagePatch.copyTo(_descriptor);
//	_descriptor = _descriptor.reshape(0,1);
}
mImageDescriptor::~mImageDescriptor()
{
	_imagePatch.release();
	_descriptor.release();
}

void mImageDescriptor::write(FileStorage& fs)
{
	/**
	 * Descriptor: **
	 * RelativePosition: **
	 * Scale: **
	 */

	fs << "Descriptor" << _descriptor;
	fs << "ImagePatch" << _imagePatch;
	fs << "RelativePosition" << _relativePosition;
	fs << "Scale" << _scale;
}

void mImageDescriptor::read(const FileNode& node)
{
	node["Descriptor"] >> _descriptor;
	node["ImagePatch"] >> _imagePatch;
//	_imagePatch = _descriptor.reshape(0, sqrt<int>(_descriptor.cols));

	vector<int> relp;
	node["RelativePosition"] >> relp;
	if (relp.size() >= 2)
	{
		_relativePosition = Point2i(relp[0], relp[1]);
	}

	node["Scale"] >> _scale;
}

Mat mImageDescriptor::extractPatch(const Mat& m, const Point2i& p)
{
	int x = cv::max<int>(p.x - (int)(0.5f*(float)PATCH_SIZE), 0);
	int y = cv::max<int>(p.y - (int)(0.5f*(float)PATCH_SIZE), 0);
	int width = cv::min<int>(PATCH_SIZE, m.cols - p.x);
	int height = cv::min<int>(PATCH_SIZE, m.rows - p.y);

	return Mat(m, Rect(x,y,width,height));
}

void mImageDescriptor::addPatch(Mat& m, const Mat& patch, const Point2i& p)
{
	int x = cv::max<int>(p.x - (int)(0.5f*(float)patch.cols), 0);
	int y = cv::max<int>(p.y - (int)(0.5f*(float)patch.rows), 0);
	int width = cv::min<int>(patch.cols, m.cols - x);
	int height = cv::min<int>(patch.rows, m.rows - y);

	Rect roi = Rect(x,y,width,height);

	Mat im_roi = Mat(m, roi);
	patch.copyTo(im_roi);
}

/**
 * mCodebookEntry
 */
mCodebookEntry::mCodebookEntry() :
	_clusterCenter(),
	_contribDescriptors()
{}
mCodebookEntry::mCodebookEntry(const string& classname, const Mat& center, vector<mImageDescriptor_p> contribDescriptors) :
	_className(classname),
	_clusterCenter(center),
	_contribDescriptors(contribDescriptors)
{}
mCodebookEntry::~mCodebookEntry() {}

void mCodebookEntry::write(FileStorage& fs)
{
	/**
	 * A Cluster object
	 * {
	 * 	ClusterCenter: **,
	 * 	Descriptors
	 * 	[
	 * 		- {**},
	 * 		- {**}
	 * 	]
	 * }
	 */

	fs << "ClassName" << _className;
	fs << "ClusterCenter" << _clusterCenter;
	fs << "Descriptors" << "[";
	for(uint i = 0; i < _contribDescriptors.size(); i++)
	{
		fs << "{";
		_contribDescriptors[i]->write(fs);
		fs << "}";
	}
	fs << "]";
}

void mCodebookEntry::read(const FileNode& node)
{
	node["ClassName"] >> _className;
	node["ClusterCenter"] >> _clusterCenter;
	FileNode desc_list = node["Descriptors"];
	for(FileNodeIterator itr = desc_list.begin(); itr != desc_list.end(); itr++)
	{
		mImageDescriptor_p desc(new mImageDescriptor());
		desc->read(*itr);
		_contribDescriptors.push_back(desc);
	}
}

/**
 * mCodebook
 */
mCodebook::mCodebook() :
	_codebookEntries(),
	_threshold(-1),
	_epsilon(-1),
	_maxNumberOfCBE(INFINITY),
	_minMax_NumDescriptors(2,INT_MAX)
{}
mCodebook::mCodebook(double threshold, double ep, int max_num_codebookentries, const Vec2i& num_descriptors_minmax) :
	_codebookEntries(),
	_threshold(threshold),
	_epsilon(ep),
	_maxNumberOfCBE(max_num_codebookentries),
	_minMax_NumDescriptors(num_descriptors_minmax)
{}
mCodebook::~mCodebook() {}

vector<mImageDescriptor_p> mCodebook::_aggregateDescriptors(const vector<vector<mImageDescriptor_p> >& im_descs, Mat& descs_mat)
{
	vector<mImageDescriptor_p> descs;
	for(uint i = 0; i < im_descs.size(); i++)
	{
		vector<mImageDescriptor_p> im_d = im_descs[i];//->getDescriptors();
		for(uint j = 0; j < im_d.size(); j++)
		{
			descs.push_back(im_d[j]);
		}
	}

	descs_mat = Mat::zeros(descs.size(), descs[0]->getDescriptor().cols, descs[0]->getDescriptor().type());
	for(int i = 0; i < descs_mat.rows; i++)
	{
		Mat m = descs_mat.row(i);
		descs[i]->getDescriptor().copyTo(m);
	}

	return descs;
}

vector<mImageDescriptor_p> mCodebook::_sortDescriptors(const vector<mImageDescriptor_p>& desc, const Mat& dists)
{
	vector<mImageDescriptor_p> desc_sorted;
	Mat dist_sortidx;
	// sort descriptor based on distance to the center
	sortIdx(dists, dist_sortidx, CV_SORT_ASCENDING + CV_SORT_EVERY_COLUMN);

	for(int i = 0; i < dist_sortidx.rows; i++)
	{
		int idx = dist_sortidx.at<int>(i,0);
		desc_sorted.push_back(desc[idx]);
	}
	return desc_sorted;
}

vector<mCodebookEntry_p> mCodebook::_getHighestCBE(const vector<mCodebookEntry_p>& class_cbe)
{
	if (_maxNumberOfCBE >= class_cbe.size())
	{
		return class_cbe;
	}

	// store number of descriptors for sorting
	Mat n_desc;
	for(uint i = 0; i < class_cbe.size(); i++)
	{
		n_desc.push_back<int>(class_cbe[i]->getDescriptors().size());
	}

	// sort
	Mat n_desc_sortidx;
	sortIdx(n_desc, n_desc_sortidx, CV_SORT_DESCENDING);

	// store first n in new array
	vector<mCodebookEntry_p> class_cbe_sorted;
	for(int i = 0; i < _maxNumberOfCBE; i++)
	{
		int idx = n_desc_sortidx.at<int>(i);
		class_cbe_sorted.push_back(class_cbe[idx]);
	}

	return class_cbe_sorted;
}

void mCodebook::compute(const string& class_name, const vector<vector<mImageDescriptor_p> >& im_descs, mAbstractSimilarity* simm)
{
//	cout << "* Feature Detection and extraction *" << endl;
//	vector<vector<mImageDescriptor_p> > im_descs;
//	for(uint i = 0; i < images.size(); i++)
//	{
//		vector<mImageDescriptor_p> im_d = images[i]->compute(class_name, det);
//		im_descs.push_back(im_d);
//	}

	// put all descriptors in one struct
	Mat agg_descs;
	vector<mImageDescriptor_p> agg_descs_rep = _aggregateDescriptors(im_descs, agg_descs);

	cout << "* Starting Agglomerative Clustering on " << agg_descs.rows << " descriptors *" << endl;

	// agglomerative clustering
	mAgglomerativeClusterer clustering(agg_descs, simm, _threshold, _epsilon, 0);
	clustering.cluster();
	// retrieve centers and all contrib descriptors
	Mat centers = clustering.getClusterCenters();
	Mat labels = clustering.getLabels();

	cout << "* Found " << centers.rows << " Clusters *" << endl;
	// store in codebook entry
	// codebook entry is the cluster center
	vector<mCodebookEntry_p> class_cbe;
	vector<mImageDescriptor_p> contrib;
	Mat contrib_dist;
	for(int c_i = 0; c_i < centers.rows; c_i++)
	{
		// get all descriptors in this cluster
		for(int d_i = 0; d_i < labels.rows; d_i++)
		{
			int idx = labels.at<int>(d_i,0);
			if (idx == c_i)
			{
				// compare if distance less than threshold, store with codebook entry, otherwise, discard
				float sim = simm->similarity(centers.row(c_i), agg_descs_rep[d_i]->getDescriptor());
				if (sim >= _threshold)
				{
					contrib.push_back(agg_descs_rep[d_i]);
					contrib_dist.push_back<float>(sim);
				}
			}
		}

		// discard codebook entries with only minimum number of descriptors
		//  or greater number descriptors than max
		if (contrib.size() > _minMax_NumDescriptors[0] && contrib.size() < _minMax_NumDescriptors[1])
		{
			// sort descriptor based on distance to the center
			contrib = _sortDescriptors(contrib, contrib_dist);

			class_cbe.push_back(mCodebookEntry_p(new mCodebookEntry(class_name, centers.row(c_i), contrib)));

			cout << "* Added center with " << contrib.size() << " descriptors" << endl;
		}

		contrib.clear();
		contrib_dist.release();
	}

	// pick highest n number of codebook entries based on the number of contributing descriptors
	class_cbe = _getHighestCBE(class_cbe);

	// add to codebook
	if (!class_cbe.empty())
	{
		_codebookEntries.insert(_codebookEntries.end(), class_cbe.begin(), class_cbe.end());
	}
}

vector<mMatch_p> mCodebook::match(mISMimage_p image, FeatureDetector* det) const
{
	vector<mMatch_p> matches;
	mNGCsimilarity simm;

	vector<mImageDescriptor_p> desc = image->compute(det);
	cout << "** Matching Codebook entries **" << endl;
	cout << " ** Found " << desc.size() << " interest points" << endl;

	// for each descriptor (patch)
	//   find closest codebook patch over the threshold value
	for(uint di = 0; di < desc.size(); di++)
	{
		mImageDescriptor_p d = desc[di];
		mMatch_p match(new mMatch(image, d));
		for(uint ci = 0; ci < _codebookEntries.size(); ci++)
		{
			float s = simm.similarity(_codebookEntries[ci]->getCodebookEntry().reshape(0,mImageDescriptor::PATCH_SIZE), d->getPatch());
			if (s > _threshold)
			{
				match->addCodeword(_codebookEntries[ci]);
			}
		}

		if (!match->getMatchingCodewords().empty())
		{
			matches.push_back(match);
		}
	}
	cout << " ** Found " << matches.size() << " matches" << endl;

	return matches;
}

vector<mCodebookEntry_p> mCodebook::getCodebookEntries(const string& classname)
{
	vector<mCodebookEntry_p> v_cbe;
	for(uint i = 0; i < _codebookEntries.size(); i++)
	{
		if (_codebookEntries[i]->getClass() == classname)
		{
			v_cbe.push_back(_codebookEntries[i]);
		}
	}
	return v_cbe;
}

void mCodebook::write(const string& output_codebook)
{
	/**
	 * Codebook array
	 * [
	 * 	- {*mCodebookEntry*},
     *	- {**}
	 * ]
	 */
	cout << "* Writing " << _codebookEntries.size() << " Codebook entries" << endl;

	FileStorage fs(output_codebook, FileStorage::WRITE);

	fs << "Threshold" << _threshold;
	fs << "Epsilon" << _epsilon;
	fs << "PatchSize" << mImageDescriptor::PATCH_SIZE;

	fs << "Codebook" << "[";
	for(uint i = 0; i < _codebookEntries.size(); i++)
	{
		fs << "{";
		_codebookEntries[i]->write(fs);
		fs << "}";
	}
	fs << "]";

	fs.release();
}
bool mCodebook::read(const string& input_codebook)
{
	_codebookEntries.clear();

	FileStorage fs(input_codebook, FileStorage::READ);

	if (!fs.isOpened())	return false;

	FileNode node = fs.root();

	node["Threshold"] >> _threshold;
	node["Epsilon"] >> _epsilon;
	node["PatchSize"] >> mImageDescriptor::PATCH_SIZE;

	FileNode c_list = node["Codebook"];
	for(FileNodeIterator c_itr = c_list.begin(); c_itr != c_list.end(); c_itr++)
	{
		mCodebookEntry_p cbe(new mCodebookEntry());
		cbe->read(*c_itr);
		_codebookEntries.push_back(cbe);
	}

	cout << "* Reading " << _codebookEntries.size() << " Codebook Entries" << endl;

	fs.release();

	return true;
}

vector<Rect> mCodebook::detect(mISMimage_p image, const mCodebook& cb, FeatureDetector* det, int xbins, int ybins, int kde_window, float kde_threshold)
{
	vector<mMatch_p> matches = cb.match(image, det);
	mPGHoughTransform pght(image, matches);
	map<uint, vector<int> > patches_per_center;
	Mat hough = pght.vote(patches_per_center);
	Mat d_hough = pght.discretizeVotes(hough, xbins, ybins, mPGHoughTransform::TYPE_PIXELS);


	vector<Point2i> max_points = mMaximaSearch::nonMaximaSuppression(d_hough, 3, kde_threshold);
	vector<vector<Point2i> > centres_max = pght.searchForMaxima(hough, d_hough, max_points, kde_window, 0.5*kde_threshold);
	vector<vector<int> > contrib_patches = pght.backprojectPatches(patches_per_center, centres_max);
	vector<Rect> detbbs = pght.computeBoundingBoxes(contrib_patches);

	return detbbs;
}

/**
 * mPGHoughTransform
 */

mPGHoughTransform::mPGHoughTransform(mISMimage_p image, const vector<mMatch_p>& matches) :
	_image(image),
	_matches(matches)
{}
mPGHoughTransform::~mPGHoughTransform()
{
	_matches.clear();
}

Mat mPGHoughTransform::vote(map<uint,vector<int> >& patches_per_centre)
{
	cout << "** Generalized Hough Transform **" << endl;

	int rows, cols;
	_image->getImageDim(rows, cols);

	Mat hough_space = Mat::zeros(rows, cols, CV_32F);

	patches_per_centre.clear();

	for(uint i = 0; i < _matches.size(); i++)
	{
		// for each match, for each codebook that contributed to the match
		//  update the hough space with votes cast from all the contributing descriptors
		mMatch_p m = _matches[i];

		vector<mCodebookEntry_p> m_cdw = m->getMatchingCodewords();

		for(uint mi = 0; mi < m_cdw.size(); mi++)
		{
			vector<mImageDescriptor_p> contrib_d = m_cdw[mi]->getDescriptors();

			for(uint ci = 0; ci < contrib_d.size(); ci++)
			{
				Point2i centre = m->getImageDescriptor()->getRelativeLocation() - contrib_d[ci]->getRelativeLocation();
				Point2i t_centre;
				t_centre.x = Util::clamp<int>(centre.x, 0, hough_space.cols-1);
				t_centre.y = Util::clamp<int>(centre.y, 0, hough_space.rows-1);

				// only update if the centre  voted was inside the image
				if (centre.x == t_centre.x && centre.y == t_centre.y)
				{
					float p_o = 1.0f / (float)contrib_d.size();
					float p_c = 1.0f / (float)m_cdw.size();

					hough_space.at<float>(centre) += p_c*p_o;
					patches_per_centre[idxToLinear(centre, rows)].push_back(i);
				}
			}
		}
	}

	return hough_space;
}

Mat mPGHoughTransform::discretizeVotes(const Mat& houghv, int xval, int yval, DiscreteType type)
{
	double xrem = remainder(houghv.cols, xval);
	while (xrem != 0)
	{
		xval++;
		xrem = remainder(houghv.cols, xval);
	}

	double yrem = remainder(houghv.rows, yval);
	while (yrem != 0)
	{
		yval++;
		yrem = remainder(houghv.rows, yval);
	}

	int nxbins, nybins, w, h;

	if (type == TYPE_BINS)
	{
		nxbins = xval;
		nybins = yval;
		w = (float)houghv.cols / (float)nxbins;
		h = (float)houghv.rows / (float)nybins;
	}
	else
	{
		w = xval;
		h = yval;
		nxbins = (float)houghv.cols / (float)w;
		nybins = (float)houghv.rows / (float)h;
	}

	Mat d_hv(nybins, nxbins, CV_32F);

	cout << "* Discrete Hough space from " << houghv.cols << "x" << houghv.rows << " to " << nxbins << "x" << nybins << " bins *" << endl;
	cout << "  * Discrete Width: " << w << "; Height: " << h << " *" << endl;

	// for each bin x, y
	//  get the average of the equivalent window in continuous space
	for(int r = 0; r < d_hv.rows; r++)
	{
		for(int c = 0; c < d_hv.cols; c++)
		{
			Mat m(houghv, Rect(c*w, r*h, w, h));
			d_hv.at<float>(r,c) = sum(m)[0]/((float)(w*h));
		}
	}

	double vmin, vmax;
	Point2i pmin, pmax;
	minMaxLoc(houghv, &vmin, &vmax, &pmin, &pmax);
	normalize(d_hv, d_hv, vmin, vmax, NORM_MINMAX);

	return d_hv;
}

vector<vector<Point2i> > mPGHoughTransform::searchForMaxima(const Mat& houghv, const Mat& d_houghv, const vector<Point2i>& centers, int window_size, float thresh)
{
	cout << "* Search for Local Maxima *" << endl;

	int w = (float)houghv.cols / (float)d_houghv.cols;
	int h = (float)houghv.rows / (float)d_houghv.rows;
	int rad = 0.5*(float)(window_size-1);

	thresh *= 0.01;

	vector<vector<Point2i> > max_centers;
	vector<Point2i> all_centres;

	for(uint i = 0; i < centers.size(); i++)
	{
		vector<Point2i> m_centers;

		// for each maxima in the discrete space
		//  consider it and its neighbourhood for continuous space maxima search
		Point2i d_tl(max<int>(centers[i].x-1, 0), max<int>(centers[i].y-1, 0));
		Point2i d_br(min<int>(centers[i].x+1, d_houghv.cols), min<int>(centers[i].y+1, d_houghv.rows));
		Point2i c_tl(d_tl.x*w, d_tl.y*h);
		Point2i c_br(d_br.x*w, d_br.y*h);
		Rect roi(c_tl, c_br);
		Mat m(houghv, roi);

		// for each maxima in the continuous space in this region
		//   add the surrounding region as an object centre
		//   only add if the mean of the window is greater than a threshold
		//       0.02 of the threshold used for finding high value points
		vector<Point2i> m_clusters = mMaximaSearch::maxShiftMaximaSearch(m, window_size, thresh);
		for(int ci = 0; ci < m_clusters.size(); ci++)
		{
			Point2i c_center = m_clusters[ci]+c_tl;
			float m = mean(Mat(houghv, Util::rectFromOrigin(c_center, Size2i(houghv.cols, houghv.rows), window_size)))[0];
			if (m >= thresh)
			{
				all_centres.push_back(c_center);
			}
		}
	}

	vector<Rect> recs;
	Mat idc = Mat::zeros(1, all_centres.size(), CV_32S);
	cout << "-*- thresh " << thresh << endl;
	cout << "-- tot. num rects: " << all_centres.size() << endl;
	Mat tmp = houghv.clone();

	// merge the resulting maxima if there is overlap
	for(uint i = 0; i < all_centres.size(); i++)
	{
		Point2i p1 = all_centres[i];
		Rect r1 = Util::rectFromOrigin(p1, Size2i(houghv.cols, houghv.rows), window_size);
		rectangle(tmp, r1, Scalar(255,255,255), 1);

		if (idc.at<int>(0,i) != 0)	continue;
		Rect rnn_chain = r1;

		for(uint j = 0; j < all_centres.size(); j++)
		{
			if (i == j)	continue;

			Point2i p2 = all_centres[j];
			Rect r2 = Util::rectFromOrigin(p2, Size2i(houghv.cols, houghv.rows), window_size);

			if ((rnn_chain&r2).area() > 0)
			{
				if (idc.at<int>(0,j) != 0)
				{
					rnn_chain = mergeRect(rnn_chain, recs[idc.at<int>(0,j)]);
				}
				else
				{
					rnn_chain = mergeRect(rnn_chain, r2);
					idc.at<int>(0,j) = recs.size();
				}
			}
		}

		idc.at<int>(0,i) = recs.size();
		recs.push_back(rnn_chain);
	}


	cout << "-- fin. num: " << recs.size() << endl;
	// threshold the resulting merges
	for(vector<Rect>::iterator itr = recs.begin(); itr != recs.end(); itr++)
	{
		Rect rr = *itr;
		float m = mean(Mat(houghv,rr))[0];
		cout << "    " << m << endl;
		if (m >= thresh)
		{
			vector<Point2i> obj_c;
			for(int c = rr.x; c < rr.x+rr.width; c++)
			{
				for(int r = rr.y; r < rr.y+rr.height; r++)
				{
					if (houghv.at<float>(r,c) != 0)
					{
						obj_c.push_back(Point2i(c,r));
					}
				}
			}
			max_centers.push_back(obj_c);

			rectangle(tmp, rr, Scalar(255,255,255), 2);
		}
	}

	imshow("Maxima's", tmp);

	cout << "  ** Found " << max_centers.size() << " Maximas" << endl;

	return max_centers;
}

vector<vector<int> > mPGHoughTransform::backprojectPatches(const map<uint,vector<int> >& patches_per_centre, const vector<vector<Point2i> >& centre_p)
{
	cout << "* Find Contributing Patches *" << endl;

	int rows, cols;
	_image->getImageDim(rows, cols);

	vector<vector<int> > contrib_patches;

	// go through each big maxima
	//  calc new centre from small maximas
	//  get associated patches for each centre
	for(uint cci = 0; cci < centre_p.size(); cci++)
	{
		vector<Point2i> centres = centre_p[cci];
		vector<int> ccontrib;
		for(uint ci = 0; ci < centres.size(); ci++)
		{
			Point2i ccentre = centres[ci];

			// find contrib patches
			//  patches who voted for that centre
			try {
				vector<int> cc_p = patches_per_centre.at(idxToLinear(ccentre, rows));
				for(uint i = 0; i < cc_p.size(); i++)
				{
					ccontrib.push_back(cc_p[i]);
				}
			} catch(...) {}
		}
		contrib_patches.push_back(ccontrib);
	}

	return contrib_patches;
}

Rect mPGHoughTransform::computeBoundingBox(const vector<int>& matches_idx)
{
	cout << "* Calculate Bounding Box *" << endl;

	if (matches_idx.size() < 2)
	{
		cout << "*  Not enough matches" << endl;
		return Rect();
	}

	int rad = 0.5*(float)(mImageDescriptor::PATCH_SIZE-1);

	Rect_<int> bb(_matches[matches_idx[0]]->getImageDescriptor()->getRelativeLocation(), _matches[matches_idx[1]]->getImageDescriptor()->getRelativeLocation());
	for(uint i = 2; i < matches_idx.size(); i++)
	{
		Point2i matchp = _matches[matches_idx[i]]->getImageDescriptor()->getRelativeLocation();

		// update x
		if (matchp.x < bb.x)
		{
			float diff = bb.x - matchp.x;
			bb.x = matchp.x - rad;
			bb.width += diff + rad;
		}

		// update y
		if (matchp.y < bb.y)
		{
			float diff = bb.y - matchp.y;
			bb.y = matchp.y - rad;
			bb.height += diff + rad;
		}

		// update width
		if (matchp.x > bb.x+bb.width)
		{
			bb.width = (matchp.x - bb.x) + rad;
		}

		// update height
		if (matchp.y > bb.y+bb.height)
		{
			bb.height = (matchp.y - bb.y) + rad;
		}
	}
	return bb;
}

/**
 * mMaximaSearch
 */
Point2i mMaximaSearch::_getMean(const Mat& m, const Rect& reg)
{
	float sub_mean_num = 0;
	Point2i sub_mean(0,0);

	for(int r = reg.y; r < reg.y+reg.height; r++)
	{
		for(int c = reg.x; c < reg.x+reg.width; c++)
		{
			if (m.at<float>(r,c) != 0)
			{
				sub_mean += Point2i(c, r);
				sub_mean_num++;
			}
		}
	}

	if (sub_mean_num == 0)	sub_mean_num++;
	return Point2i((float)sub_mean.x/sub_mean_num, (float)sub_mean.y/sub_mean_num);
}

vector<Point2i> mMaximaSearch::nonMaximaSuppression(const Mat& mat, int window_size, float thresh)
{
	vector<Point2i> maxs;
	double vmin, vmax;
	Point2i pmin, pmax;

	for(int r = 0; r < mat.rows; r++)
	{
		for(int c = 0; c < mat.cols; c++)
		{
			Rect rr = Util::rectFromOrigin(Point2i(c,r), Size2i(mat.cols, mat.rows), window_size);
			Mat m(mat, rr);
			minMaxLoc(m, &vmin, &vmax, &pmin, &pmax);
			if (pmax == Point2i(0.5*((float)window_size-1), 0.5*((float)window_size-1)) && vmax > thresh)
			{
//				cout << Point2i(c,r) << "   " << rr << endl;
//				cout << m << endl;
				maxs.push_back(Point2i(c,r));
			}
		}
	}
	return maxs;
}


vector<Point2i> mMaximaSearch::maxShiftMaximaSearch(const Mat& mat, int window_size, float thresh)
{
	if (remainder(window_size, 2) == 0)
	{
		window_size++;
	}

	int rad = 0.5*(float)(window_size-1);

	Mat votedPoints = Mat::zeros(mat.rows, mat.cols, CV_32S);
	map<int, vector<Point2i> > clusterIdx;
	for(int r = rad; r < mat.rows-rad; r++)
	{
		for(int c = rad; c < mat.cols-rad; c++)
		{
			// initial position of window is around the point
			Rect sub_r = Util::rectFromOrigin(Point2i(c,r), Size2i(mat.cols, mat.rows), window_size);
			Mat sub(mat, sub_r);

			Point2i pmin, pmax;
			double vmin, vmax;

			Point2i last_p(c,r);

			Point2i vmove(window_size,window_size);

			// stop when local mean is found
			// stop when the mean location stops moving
			while (vmove != Point2i(0,0))
			{
				minMaxLoc(sub, &vmin, &vmax, &pmin, &pmax);

				pmax += sub_r.tl();
				vmove = pmax - last_p;
				last_p = pmax;

				// move window
				sub_r = Util::rectFromOrigin(pmax, Size2i(mat.cols, mat.rows), window_size);
				sub = Mat(mat, sub_r);
			}

			clusterIdx[idxToLinear(pmax,mat.rows)].push_back(Point2i(c,r));
		}
	}

	vector<Point2i> clusters;

	// threshold clusters based on the mean of their neighbourhood
	for(map<int,vector<Point2i> >::iterator itr = clusterIdx.begin(); itr != clusterIdx.end(); itr++)
	{
		// if the maxima is above a threshold
		Point2i idx = linearToIdx(itr->first, mat.rows);
//		float m = mat.at<float>(idx);
		Rect r = Util::rectFromOrigin(idx, Size2i(mat.cols, mat.rows), window_size);
		float m = mean(Mat(mat, r))[0];
		if (m >= thresh/* && itr->second.size() > 2*/)
		{
//			cout << "   " << m << endl;
			clusters.push_back(idx);

//			cout << " - " << m << endl;
//
//			float tmp = 0;
//
//			vector<Point2i> points = itr->second;
//			for(int j = 0; j < points.size(); j++)
//			{
//				float v = mat.at<float>(points[j]);
//				if (v >= thresh)
//				{
//					mmax.at<int>(points[j]) = clusters.size()-1;
//					tmp += mat.at<float>(points[j]);
//				}
//			}
//			cout << "   " << tmp/(float)points.size() << endl;
		}
	}

	clusterIdx.clear();
	return clusters;
}

vector<Point2i> mMaximaSearch::meanShiftMaximaSearch(const Mat& mat, int wid_size, float kernel_bw, float thresh)
{
//		namedWindow("IM", CV_WINDOW_KEEPRATIO);
//		Mat tmp = Mat::zeros(mat.rows, mat.cols, mat.type());
	vector<Point2i> clusters;
	for(int r = 0; r < mat.rows; r++)
	{
		for(int c = 0; c < mat.cols; c++)
		{
			Rect sub_r = Util::rectFromOrigin(Point2i(c,r), Size2i(mat.cols, mat.rows), wid_size);

			Point2i m = _getMean(mat, sub_r);
			Point2i last_m = m;

			do
			{
//					rectangle(tmp, sub_r, Scalar(255,255,255));
				last_m = m;

				sub_r = Util::rectFromOrigin(last_m, Size2i(mat.cols, mat.rows), wid_size);

				m = _getMean(mat, sub_r);
			} while(last_m != m);

			if (m != Point2i(c,r))
			{
				float mmean = mean(Mat(mat, sub_r))[0];
				if (mmean > thresh)
				{
//						cout << "p " << Point2i(c,r) << endl;
//						cout << "-- mean " << m << endl;
//						cout << "   " << mmean << endl;
//
//						tmp.at<float>(m) = 1;
//						imshow("IM", tmp);
//						waitKey(5);

				}
			}
		}
	}
//		cout << "*FINISHED*" << endl;
//		waitKey();
	return clusters;
}


/**
 * mDebug
 */
void mDebug::showMatch(const string& winname, mISMimage_p image, mMatch_p match, bool do_write)
{
	mImageDescriptor_p md = match->getImageDescriptor();
	vector<mCodebookEntry_p> mcbe = match->getMatchingCodewords();

	Mat m = image->getImage().clone();
	m.convertTo(m, CV_32F);
	m /= 255.0f;

	for(uint i = 0; i < mcbe.size(); i++)
	{
		mCodebookEntry_p cbe = mcbe[i];
		vector<mImageDescriptor_p> mcbed = cbe->getDescriptors();
		for(uint di = 0; di < mcbed.size(); di++)
		{
			Point2i p = md->getRelativeLocation() - mcbed[di]->getRelativeLocation();
			line(m, p, md->getRelativeLocation(), Scalar(255,0,0));
		}
	}

	mImageDescriptor::addPatch(m, md->getPatch(), md->getRelativeLocation());

	namedWindow(winname, CV_WINDOW_KEEPRATIO);
	imshow(winname, m);
	if (do_write)
	{
		imwrite(winname+".png", m*255.0f);
	}
}

void mDebug::showMatches(const string& winname, mISMimage_p image, const vector<mMatch_p>& matches, bool disp_image, bool disp_centres, bool do_write)
{
	int rows, cols;
	image->getImageDim(rows, cols);
	Mat tmp;
	if (disp_image)
	{
		tmp = image->getImage().clone();
		tmp.convertTo(tmp, CV_32F);
		tmp /= 255.0f;
	}
	else
	{
		tmp = Mat::zeros(rows, cols, CV_32F);
	}

	for(uint i = 0; i < matches.size(); i++)
	{
		Point2i patchp = matches[i]->getImageDescriptor()->getRelativeLocation();
		mImageDescriptor::addPatch(tmp, matches[i]->getMatchingCodewords()[0]->getCodebookEntry().reshape(0,mImageDescriptor::PATCH_SIZE), patchp);
		if (disp_centres)
		{
			for(int ci = 0; ci < matches[i]->getMatchingCodewords().size(); ci++)
			{
				mCodebookEntry_p cb = matches[i]->getMatchingCodewords()[ci];

				for(int di = 0; di < cb->getDescriptors().size(); di++)
				{
					Point2i centre = patchp-cb->getDescriptors()[di]->getRelativeLocation();
					circle(tmp, centre, 1, Scalar(255,255,255));
					line(tmp, centre, patchp, Scalar(255,255,255));
				}
			}
		}
	}

	namedWindow(winname, CV_WINDOW_KEEPRATIO);
	imshow(winname, tmp);
	if (do_write)
	{
		imwrite(winname+".png", tmp*255.0f);
	}
}

void mDebug::showImage(const string& winname, const Mat& mat, bool norm, bool disp_minmax, bool do_write)
{
	if (disp_minmax)
	{
		double vmin, vmax;
		Point2i pmin, pmax;
		minMaxLoc(mat, &vmin, &vmax, &pmin, &pmax);
		cout << "--- " << winname << endl;
		cout << "  min: " << vmin << " at " << pmin << endl;
		cout << "  max: " << vmax << " at " << pmax << endl;
	}

	Mat m = mat.clone();
	if (norm)
	{
		if (m.type() == CV_32F)
		{
			normalize(m, m, 0, 1, NORM_MINMAX);
		}
		else
		{
			normalize(m, m, 0, 255*256, NORM_MINMAX);
		}
	}

	namedWindow(winname, CV_WINDOW_KEEPRATIO);
	imshow(winname, m);
	if(do_write)
	{
		imwrite(winname+".png", m*255.0f);
	}
}

void mDebug::showPoints(const string& winname, const vector<vector<Point2i> >& points, Mat& im, bool writeToImage, bool do_write)
{
	Mat t_im;
	if (writeToImage)
	{
		t_im = Mat::zeros(im.rows, im.cols, CV_32F);
	}
	else
	{
		t_im = im.clone();
	}

	for(uint i = 0; i < points.size(); i++)
	{
		for(uint j = 0; j < points[i].size(); j++)
		{
//			circle(t_im, points[i][j], 2, Scalar(255,255, 255));
			t_im.at<float>(points[i][j]) = 1;
		}
	}
	namedWindow(winname, CV_WINDOW_KEEPRATIO);
	imshow(winname, t_im);
	if(do_write)
	{
		imwrite(winname+".png", t_im*255.0f);
	}
}

void mDebug::showCenters(const string& winname, const Mat& image, const vector<mMatch_p>& matches, const vector<vector<Point2i> >& centres_max, const vector<vector<int> >& contrib_patches, bool do_write)
{
	Mat im_data = image.clone();
	for(uint ci = 0; ci < centres_max.size(); ci++)
	{
		Point2i new_centre(0,0);
		for(int mci = 0; mci < centres_max[ci].size(); mci++)
		{
			new_centre += centres_max[ci][mci];
			circle(im_data, centres_max[ci][mci], 2, Scalar(255,255,255), 2);
		}
		new_centre = Point2i(new_centre.x/centres_max[ci].size(), new_centre.y/centres_max[ci].size());

		for(uint pi = 0; pi < contrib_patches[ci].size(); pi++)
		{
			Point2i patch_loc = matches[contrib_patches[ci][pi]]->getImageDescriptor()->getRelativeLocation();

			line(im_data, patch_loc, new_centre, Scalar(255,255,255));

			Mat patch = matches[contrib_patches[ci][pi]]->getMatchingCodewords()[0]->getCodebookEntry().clone();
			patch = patch.reshape(0, sqrt(patch.cols));
			mImageDescriptor::addPatch(im_data, patch, patch_loc);
		}
	}
	namedWindow(winname, CV_WINDOW_KEEPRATIO);
	imshow(winname, im_data);
	if (do_write)
	{
		imwrite(winname+".png", im_data*255.0f);
	}
}

void mDebug::showPatches(const string& winname, mISMimage_p image, vector<mImageDescriptor_p> desc, bool do_write)
{
	Mat im = image->getImage().clone();

	vector<KeyPoint> kps(desc.size());
	for(uint i = 0; i < desc.size(); i++)
	{
		kps[i] = KeyPoint(desc[i]->getRelativeLocation(), 1, 1, 1, 1, 1);
	}

	drawKeypoints(im, kps, im, Scalar(255,0,0));

	imshow(winname+" KeyPoints", im);
	if (do_write)
	{
		imwrite(winname+" KeyPoints", im);
	}

	int dim_w = sqrt(desc.size()) + 1;
	int dim_h = ((float)desc.size() / (float)dim_w);

	int width = dim_w*(mImageDescriptor::PATCH_SIZE+1);
	int height = dim_h*(mImageDescriptor::PATCH_SIZE+1);

	Mat patches = Mat::zeros(height, width, CV_32F);
	Point2i offset(0.5*mImageDescriptor::PATCH_SIZE+1, 0.5*mImageDescriptor::PATCH_SIZE+1);
	Point2i currentp(offset);

	for(int i = 0; i < desc.size(); i++)
	{
		try {
			mImageDescriptor::addPatch(patches, desc[i]->getPatch(), currentp);
		} catch(cv::Exception& e) {
			cout << e.what() << endl;
		}

		currentp.x += 2*offset.x;
		if (currentp.x >= width)
		{
			currentp.x = offset.x;
			currentp.y += 2*offset.y;
		}
	}

	namedWindow(winname, CV_WINDOW_KEEPRATIO);
	imshow(winname+" Patches", patches);
	if (do_write)
	{
		imwrite(winname+" Patches.png", patches*255.0f);
	}
}

void mDebug::showPatches(const string& winname, mISMimage_p image, FeatureDetector* det, bool do_write)
{
	vector<mImageDescriptor_p> desc = image->compute(det);

	showPatches(winname, image, desc, do_write);
}

void mDebug::backprojection(const string& winname, mISMimage_p image, const vector<mMatch_p>& matches, const vector<vector<int> > contrib_patches, bool do_write)
{
	int rows, cols;
	image->getImageDim(rows, cols);

	Mat m = Mat::zeros(rows, cols, CV_32F);

	for(uint i = 0; i < contrib_patches.size(); i++)
	{
		for(uint j = 0; j < contrib_patches[i].size(); j++)
		{
			mImageDescriptor_p d = matches[contrib_patches[i][j]]->getImageDescriptor();
			mImageDescriptor::addPatch(m, d->getPatch(), d->getRelativeLocation());
		}
	}

	namedWindow(winname, CV_WINDOW_KEEPRATIO);
	imshow(winname, m);
	if(do_write)
	{
		imwrite(winname+".png", m*255.0f);
	}
}
