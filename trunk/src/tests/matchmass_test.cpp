
#include <mImplicitShapeModel.h>
#include <Util.h>

#include <thread>
#include <vector>
using namespace std;

//TODO error when importing training set

string dataset_path_true = "/data/_work_repo_wd/datasets/caltech_faces";
string dataset_path_false = "/data/_work_repo_wd/datasets/caltech_cars";
string testingfile_true = "./data/caltech_face_testing.txt";
string testingfile_false = "./data/caltech_car_testing.txt";
string output_path = "./data/testing/";
string codebook_path = "./data/codebooks/codebook_sift_13_070_35.xml";
string classname = "face";

int nxbins = 50;
int nybins = 50;
int kde_window = 35;
float kde_threshold = 0.5;

int num_parallel = 30;
int max_num = INT_MAX;

float detection_window_ratio = 0.75;

//vector<mISMimage_p> getImages(const vector<mAbstractAnnotation_p>& ann)
//{
//	vector<mISMimage_p> imageset(ann.size());
//	for(uint i = 0; i < ann.size(); i++)
//	{
//		imageset[i] = mISMimage_p(new mISMimage(ann[i]));
//	}
//	return imageset;
//}

int num_true_positive = 0;
int num_false_positive = 0;
int num_true_negative = 0;
int num_false_negative = 0;
mCodebook codebook;

void detect_class(mISMimage_p image, const string& name)
{
	SiftFeatureDetector det;
	vector<Rect> detbbs = mCodebook::detect(image, codebook, &det, nxbins, nybins, kde_window, kde_threshold);

	bool truepos, falsepos, trueneg, falseneg;
	image->validateDetection(classname, detbbs, truepos, falsepos, trueneg, falseneg, detection_window_ratio);
	if (truepos)
	{
		num_true_positive++;
	}
	if (falsepos)
	{
		num_false_positive++;
	}
	if (trueneg)
	{
		num_true_negative++;
	}
	if(falseneg)
	{
		num_false_negative++;
	}
	cout << "** Done **" << endl;
}

int main(int argc, char** argv)
{
	if (!codebook.read(codebook_path))
	{
		cout << "** Error reading codebook " << codebook_path << endl;
		exit(EXIT_FAILURE);
	}

	FileStorage fs;
	fs.open("data/testing/threshold_vs_confmat.xml", FileStorage::WRITE);
	fs << "Data" << "[";

	// images containing
	mCTImageSet imageset_true(dataset_path_true);
	vector<mAbstractAnnotation_p> v_imageset_true = imageset_true.parse(testingfile_true);
	vector<mISMimage_p> images_true = mISMimage::createImages(v_imageset_true);

	// images not containing
	mCTImageSet imageset_false(dataset_path_false);
	vector<mAbstractAnnotation_p> v_imageset_false = imageset_false.parse(testingfile_false);
	vector<mISMimage_p> images_false = mISMimage::createImages(v_imageset_false);

	for(kde_threshold = 0.1; kde_threshold < 1; kde_threshold+=0.1)
	{
		num_true_positive = 0;
		num_false_positive = 0;
		num_true_negative = 0;
		num_false_negative = 0;

		for(uint i = 0; i < images_true.size() && i < max_num; i+=num_parallel)
		{
			vector<thread> ths;

			for(int p = i; p < i+num_parallel; p++)
			{
				if (p >= images_true.size())	break;
				ths.push_back(thread(detect_class, images_true[p], v_imageset_true[p]->getNameId()));
			}

			for(uint i = 0; i < ths.size(); i++)
			{
				ths[i].join();
			}
		}

		for(uint i = 0; i < images_false.size() && i < max_num; i+=num_parallel)
		{
			vector<thread> ths;

			for(int p = i; p < i+num_parallel; p++)
			{
				if (p >= images_false.size())	break;
				ths.push_back(thread(detect_class, images_false[p], v_imageset_false[p]->getNameId()));
			}

			for(uint i = 0; i < ths.size(); i++)
			{
				ths[i].join();
			}
		}

		fs << "{";
		fs << "Threshold" << kde_threshold;
		Mat conf_mat;
		float acc, tpr, fpr;
		mAbstractImageSet::getStats(num_true_positive, num_false_positive, num_true_negative, num_false_negative, acc, tpr, fpr, conf_mat);
		mAbstractImageSet::writeStats(fs, conf_mat, acc, tpr, fpr);
		fs << "}";

		cout << "****" << endl << conf_mat << endl;
	}

	fs << "]";
	fs.release();
}
