#include <mImplicitShapeModel.h>
#include <mDataset.h>
#include <Util.h>

// training
//  num of images to train on
//  threshold value for clustering
//  patch size
//  input dataset (images/annotations)
//  input training file
//  name of class
//  output number of codebook entries
//  output codebook file
// testing
//  save image patches
//  output patches directory
int num_images = 25;
int max_num_codebookentries = INT_MAX;
int min_num_descriptors = 2;
double clustering_threshold = 0.7;
string detector_type = "sift";
string class_name = "face";
string dataset = "./data/training";
//string dataset_images = "/data/_work_repo_wd/datasets/caltech_faces";
//string dataset_annotations = "/data/_work_repo_wd/datasets/caltech_faces";
string training_file = "./data/caltech_face_training.txt";

string output_codebook = "";
bool output_codebook_isinput = false;

bool usedescriptor = false;

void print_usage()
{
	cout << "Usage:" << endl;
	cout << "    --num_im       Number of images for training" << endl;
	cout << "    --threshold    Threshold used for clustering" << endl;
	cout << "    --maxn_cbe     Maximum number of codebook entries" << endl;
	cout << "    --min_d        Minimum number of descriptors to consider in a cluster ( > 1)" << endl;
	cout << "    --psize        Size of the patch" << endl;
	cout << "    --detector     Feature detector to use (mser, surf, sift)" << endl;
	cout << "    --descriptor   Use SIFT descriptor" << endl;
	cout << "    --class        Name of class; based on image annotation" << endl;
	cout << "    --dataset      Path to images folder" << endl;
	cout << "    --annotation   Path to annotation folder" << endl;
	cout << "    --trainingset  Input training file" << endl;
	cout << "    --output | -o  Output codebook file" << endl;
}

void parse_cmd(int argc, char** argv)
{
	mImageDescriptor::PATCH_SIZE = 25;
	for(int i = 1; i < argc; i += 2)
	{
		char* k = argv[i];
		if (!strcmp(k, "--maxn_cbe"))
		{
			// max number of codebook entries to pick
			max_num_codebookentries = Util::stringToNum<int>(argv[i+1]);
		}
		else if (!strcmp(k, "--num_im"))
		{
			// number of images to train on
			num_images = Util::stringToNum<int>(argv[i+1]);
		}
		else if (!strcmp(k, "--min_d"))
		{
			min_num_descriptors = Util::stringToNum<int>(argv[i+1]);
		}
		else if (!strcmp(k, "--threshold"))
		{
			// threshold for clustering
			clustering_threshold = Util::stringToNum<double>(argv[i+1]);
		}
		else if (!strcmp(k, "--psize"))
		{
			// patch size
			mImageDescriptor::PATCH_SIZE = Util::stringToNum<int>(argv[i+1]);
		}
		else if (!strcmp(k, "--detector"))
		{
			detector_type = argv[i+1];
		}
		else if (!strcmp(k, "--descriptor"))
		{
			usedescriptor = true;
			i--;
		}
		else if (!strcmp(k, "--class"))
		{
			// class name
			class_name = argv[i+1];
		}
//		else if (!strcmp(k, "--dataset"))
//		{
//			// image dataset
//			dataset_images = argv[i+1];
//		}
//		else if (!strcmp(k, "--annotation"))
//		{
//			// annotation dataset
//			dataset_annotations = argv[i+1];
//		}
		else if (!strcmp(k, "--trainingset"))
		{
			// training file
			training_file = argv[i+1];
		}
		else if (!strcmp(k, "-o"))
		{
			// output codebook
			output_codebook = argv[i+1];
			output_codebook_isinput = true;
		}
		else if (!strcmp(k, "-h") || !strcmp(k, "--help"))
		{
			print_usage();
			exit(EXIT_SUCCESS);
		}
		else
		{
			cout << "Invalid argument " << k << endl;
			print_usage();
			exit(EXIT_FAILURE);
		}
	}

	if (!output_codebook_isinput)
	{
		cout << "Must enter output codebook file name" << endl;
		print_usage();
		exit(EXIT_FAILURE);
	}
}

int main(int argc, char** argv)
{
	parse_cmd(argc, argv);

	cout << "** parameters" << endl;
	cout << "      Minimum descriptors " << min_num_descriptors << endl;
	cout << "      Threshold           " << clustering_threshold << endl;
	cout << "      Patch size          " << mImageDescriptor::PATCH_SIZE << endl;
	cout << "      Detector            " << detector_type << endl;
	cout << "--------------------------" << endl;

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

//	mPASCALImageSet im_set(dataset_images, dataset_annotations);
	mCTImageSet im_set(dataset);
	vector<mAbstractAnnotation_p> im_set_data = im_set.parse(training_file, num_images);

	vector<mISMimage_p> images(im_set_data.size());
	for(uint i = 0; i < im_set_data.size(); i++)
	{
		images[i] = mISMimage_p(new mISMimage(im_set_data[i]));
	}

	SiftDescriptorExtractor* ext = NULL;
	if (usedescriptor)
	{
		ext = new SiftDescriptorExtractor();
	}

	// compute
//	Mat im = Mat::zeros(1000, 1000, CV_32F);
	vector<vector<mImageDescriptor_p> > im_descs;
	for(uint i = 0; i < images.size(); i++)
	{
		vector<mImageDescriptor_p> im_d = images[i]->compute(class_name, det, ext);
		im_descs.push_back(im_d);

//		for(int d = 0; d < im_d.size(); d++)
//		{
//			mImageDescriptor::addPatch(im, im_d[d]->getPatch(), Point2i(500,500)+im_d[d]->getRelativeLocation());
//		}
	}

//	imshow("Patches ", im);
//	waitKey(0);

	// codebook
	mAbstractSimilarity* simm;
	if (usedescriptor)
	{
		simm = new mSquaredDiffsimilarity();
	}
	else
	{
		simm = new mNGCsimilarity();
	}
	mCodebook_p codebook(new mCodebook(clustering_threshold, 0.1, max_num_codebookentries, Vec2i(min_num_descriptors, INT_MAX)));
	codebook->compute(class_name, im_descs, simm);
	codebook->write(output_codebook);
	cout << *codebook << endl;

//	delete det;
//	if (ext != NULL)	delete ext;
//	delete simm;

	return EXIT_SUCCESS;
}
