#include <mImplicitShapeModel.h>
#include <Util.h>
using namespace std;

string detector_type = "sift";
string pathToCB = "data/codebooks/codebook_sift_13_070_35.xml";
string imageToMatch = "data/testing/image_0136.jpg";
int nxbins = 13;
int nybins = 8;
mPGHoughTransform::DiscreteType binType = mPGHoughTransform::TYPE_BINS;
int min_num_patches = 4;
float kde_window = 35;
float kde_threshold = 0.5;
bool debug = false;
bool dowrite = false;

void parse_cmd(int argc, char** argv)
{
	for(int i = 1; i < argc; i += 2)
	{
		char* k = argv[i];

		if (!strcmp(k, "--cb"))
		{
			pathToCB = argv[i+1];
		}
		else if (!strcmp(k, "-i") || !strcmp(k, "--input"))
		{
			imageToMatch = argv[i+1];
		}
		else if (!strcmp(k, "--detector"))
		{
			detector_type = argv[i+1];
		}
		else if (!strcmp(k, "--nxbins"))
		{
			nxbins = Util::stringToNum<int>(argv[i+1]);
		}
		else if (!strcmp(k, "--nybins"))
		{
			nybins = Util::stringToNum<int>(argv[i+1]);
		}
		else if (!strcmp(k, "--pixels"))
		{
			binType = mPGHoughTransform::TYPE_PIXELS;
			i--;
		}
		else if (!strcmp(k, "--kde_window"))
		{
			kde_window = Util::stringToNum<double>(argv[i+1]);
		}
		else if(!strcmp(k, "--kde_threshold"))
		{
			kde_threshold = Util::stringToNum<double>(argv[i+1]);
		}
		else if (!strcmp(k, "-d") || !strcmp(k, "--debug"))
		{
			debug = true;
			i--;
		}
		else if (!strcmp(k, "-w") || !strcmp(k, "--write"))
		{
			dowrite = true;
			i--;
		}
		else if (!strcmp(k, "-h") || !strcmp(k, "--help"))
		{
			cout << "Usage:" << endl;
			cout << "    --cb             Codebook to use" << endl;
			cout << "    -i | --input     Input image to match" << endl;
			cout << "    --detector       Detector Type (mser, surf, sift)" << endl;
			cout << "    --nxbins         Number of bins for X for Hough space descretization [Default=" << nxbins << "]" << endl;
			cout << "    --nybins         Number of bins for Y for Hough space descretization [Default=" << nybins << "]" << endl;
			cout << "    --pixels         Values provided by nxbins, nybins are size of a bin in pixels" << endl;
			cout << "    --kde_window     Window size for Kernel in Mean Shift [Default=" << kde_window << "]" << endl;
			cout << "    --kde_threshold  Probability threshold for choosing maximas in Mean Shift [Default=" << kde_threshold << "]" << endl;
			cout << "    -d | --debug     Display debug information [Default=" << debug << "]" << endl;
			cout << "    -w | --write     Write debugging images information [Default=" << dowrite << "]" << endl;
			exit(EXIT_SUCCESS);
		}
		else
		{
			cout << "Invalid argument " << k << endl;
			exit (EXIT_FAILURE);
		}
	}
}

void printMinMax(const Mat& m, const string& id)
{
	double vmax, vmin;
	Point2i pmax, pmin;

	minMaxLoc(m, &vmin, &vmax, &pmin, &pmax);
	cout << id << endl;
	cout << "min " << pmin << "  -> " << vmin << endl;
	cout << "max " << pmax << "  -> " << vmax << endl;
	cout << "--" << endl;
}

int main(int argc, char** argv)
{
	parse_cmd(argc, argv);

	cout << "** parameters " << endl;
	cout << "      codebook    " << pathToCB << endl;
	cout << "      image       " << imageToMatch << endl;
	cout << "      threshold   " << kde_threshold << endl;
	cout << "      window size " << kde_window << endl;
	cout << "      detector    " << detector_type << endl;
	cout << "------------------" << endl;

	FeatureDetector* det;
	if (detector_type == "mser")
	{
		det = new MserFeatureDetector();
	}
	else if (detector_type == "sift")
	{
		det = new SiftFeatureDetector();
	}
	else if (detector_type == "surf")
	{
		det = new SurfFeatureDetector();
	}
	else
	{
		cout << "** Detector type not supported" << endl;
		exit(EXIT_FAILURE);
	}

	mCodebook cb;
	cb.read(pathToCB);
	cout << cb << endl;

	mISMimage_p im(new mISMimage(mAbstractAnnotation_p(new mCTAnnotation("", imageToMatch))));

	Mat im_data = im->getImage();
	im_data.convertTo(im_data, CV_32F);
	im_data /= 255.0f;

	Mat tmp;

	/**
	 * Matching to codebook
	 */
	vector<mMatch_p> matches = cb.match(im, det);
	if (debug)
	{
		mDebug::showMatches("Matches", im, matches, false, false, dowrite);
	}

	/**
	 * Hough transform
	 */
	mPGHoughTransform pght(im, matches);
	map<uint, vector<int> > patches_per_centre;
	Mat hough_votes = pght.vote(patches_per_centre);
	if (debug)
	{
		mDebug::showImage("Cont. Hough", hough_votes, true, true, dowrite);
	}

	/**
	 * discretize hough voting matrix
	 * scale to be same as the continuous' range
	 */
	Mat d_houghv = pght.discretizeVotes(hough_votes, nxbins, nybins, binType);

//	double vmin, vmax;
//	Point2i pmin, pmax;
//	minMaxLoc(hough_votes, &vmin, &vmax, &pmin, &pmax);
//	normalize(d_houghv, d_houghv, vmin, vmax, NORM_MINMAX);
	if (debug)
	{
		mDebug::showImage("Disc. Hough", d_houghv, true, true, dowrite);
	}

	/**
	 * find local maximas in discrete hough matrix
	 */
	vector<Point2i> max_means = mMaximaSearch::nonMaximaSuppression(d_houghv, 3, kde_threshold);
//	vector<Point2i> max_means = mMaximaSearch::maxShiftMaximaSearch(d_houghv, 3, kde_threshold);
	if (debug)
	{
		vector<vector<Point2i> > tmp;
		tmp.push_back(max_means);
		mDebug::showPoints("Mean Shift Disc. Hough Max", tmp, d_houghv, true, dowrite);
		cout << "  Found " << max_means.size() << " clusters" << endl;
	}

	/**
	 * find local maximas in continuous hough matrix for maximas in discrete matrix
	 */
	vector<vector<Point2i> > centres_max = pght.searchForMaxima(hough_votes, d_houghv, max_means, kde_window, kde_threshold);
	if (debug)
	{
		mDebug::showPoints("Maxima Centres", centres_max, hough_votes, false, dowrite);
	}

	/**
	 * Get patches that contributed to the maximas
	 */
	vector<vector<int> > contrib_patches = pght.backprojectPatches(patches_per_centre, centres_max);
	if (debug)
	{
		mDebug::showCenters("Centers & Patches", im_data, matches, centres_max, contrib_patches, dowrite);
		mDebug::backprojection("BackProjection", im, matches, contrib_patches, dowrite);
	}

	/**
	 * Create a bounding box around each centre depending on the contributing patches
	 */
	for(uint ci = 0; ci < centres_max.size(); ci++)
	{
		rectangle(im_data, pght.computeBoundingBox(contrib_patches[ci]), Scalar(255,255,255), 2);
	}
	imshow("BoundingBox", im_data);
	imwrite("BoundingBox.png", im_data*255.0f);

	waitKey(0);
}
