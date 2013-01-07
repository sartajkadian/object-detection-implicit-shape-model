#include <thread>
#include <mImplicitShapeModel.h>

string dataset_path_true = "/data/_work_repo_wd/datasets/caltech_faces";
string dataset_path_false = "/data/_work_repo_wd/datasets/caltech_cars";
string testingfile_true = "./data/caltech_face_testing.txt";
string testingfile_false = "./data/caltech_car_testing.txt";
string output_path = "./data/testing/";
string codebook_path = "./data/codebooks/codebook_sift_25_070_35_cut.xml";
string classname = "face";

int nxbins = 50;
int nybins = 50;
int kde_window = 35;
float kde_threshold = 0.5;

int num_parallel = 30;
int max_num = INT_MAX;

float detection_window_ratio = 0.75;

int n_truepos;
int n_falsepos;
int n_trueneg;
int n_falseneg;

mCodebook cb;

void detect_class(mISMimage_p image, const string& name)
{
	SiftFeatureDetector det;
	vector<Rect> detbbs = mCodebook::detect(image, cb, &det, nxbins, nybins, kde_window, kde_threshold);

	bool truepos, falsepos, trueneg, falseneg;
	image->validateDetection(classname, detbbs, truepos, falsepos, trueneg, falseneg, detection_window_ratio);
	if (truepos)
	{
		n_truepos++;
	}
	if (falsepos)
	{
		n_falsepos++;
	}
	if (trueneg)
	{
		n_trueneg++;
	}
	if(falseneg)
	{
		n_falseneg++;
	}
	cout << "** Done **" << endl;
}

int main()
{
	int images_num[] = {1, 3, 5, 7, 10, 13, 16, 20, 25};
	int len = sizeof(images_num) / sizeof(int);

	FileStorage fs;
	fs.open("data/testing/trainingnum_vs_accuracy.xml", FileStorage::WRITE);
	fs << "Data" << "[";

	// images containing
	mCTImageSet imageset_true(dataset_path_true);
	vector<mAbstractAnnotation_p> v_imageset_true = imageset_true.parse(testingfile_true);
	vector<mISMimage_p> images_true = mISMimage::createImages(v_imageset_true);

	// images not containing
	mCTImageSet imageset_false(dataset_path_false);
	vector<mAbstractAnnotation_p> v_imageset_false = imageset_false.parse(testingfile_false);
	vector<mISMimage_p> images_false = mISMimage::createImages(v_imageset_false);

	for(int i = 0; i < len; i++)
	{
		char cb_path[512];
		sprintf(cb_path, "data/codebooks/codebook_sift_%02d_070_35.xml", images_num[i]);

		cout << cb_path << endl;
		if(!cb.read(cb_path))
		{
			cout << "** wrong path " << cb_path << endl;
			exit(0);
		}

		n_truepos = 0;
		n_trueneg = 0;
		n_falsepos = 0;
		n_falseneg = 0;

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


		Mat conf_mat;
		float acc, tpr, fpr;
		fs << "{";
		fs << "NumTraining" << images_num[i];
		mAbstractImageSet::getStats(n_truepos, n_falsepos, n_trueneg, n_falseneg, acc, tpr, fpr, conf_mat);
		mAbstractImageSet::writeStats(fs, conf_mat, acc, tpr, fpr);
		fs << "}";
	}

	fs << "]";
	fs.release();
}
