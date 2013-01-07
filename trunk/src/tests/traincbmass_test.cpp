#include <thread>
#include <iostream>
#include <vector>
using namespace std;

#include <mImplicitShapeModel.h>
#include <Util.h>

float thresh_min = 0.6;
float thresh_max = 0.95;
float thresh_inc = 0.05;
int nimages_start = 25;
int nimages_max = 25;

int patch_size = 35;

int min_num_descriptors = 3;

string classname = "face";
string cb_base_name = "./data/codebooks/codebook";
string dataset = "./data/training/";
string training_file = "./data/caltech_face_training.txt";

void create_codebook(vector<vector<mImageDescriptor_p> > im_descs, float thresh)
{
	string o = cb_base_name + "_sift_" + Util::numToString<uint>(im_descs.size()) + "_" + Util::numToString<float>(thresh)+"_"+Util::numToString<float>(patch_size)+".xml";
	cout << "-- out " << o << endl;
	// codebook
	mNGCsimilarity simm;
	mCodebook_p codebook(new mCodebook(thresh, 0.1, INFINITY, Vec2i(min_num_descriptors,INT_MAX)));
	codebook->compute(classname, im_descs, &simm);
	codebook->write(o);
}

int main(int argc, char** argv)
{
	int num_images = nimages_max+2;
	mCTImageSet im_set(dataset);
	vector<mAbstractAnnotation_p> im_set_data = im_set.parse(training_file, num_images);
	vector<mISMimage_p> images(im_set_data.size());
	for(uint i = 0; i < im_set_data.size(); i++)
	{
		images[i] = mISMimage_p(new mISMimage(im_set_data[i]));
	}

	mImageDescriptor::PATCH_SIZE = patch_size;
	cout << "* Feature Detection and extraction *" << endl;
	SiftFeatureDetector det;
	vector<thread> ths;
	for (int n = nimages_start; n <= nimages_max; n++)
	{
		for (float t = thresh_min; t <= thresh_max; t += thresh_inc)
		{
			// compute
			vector<vector<mImageDescriptor_p> > im_descs;
			for(uint i = 0; i < n; i++)
			{
				vector<mImageDescriptor_p> im_d = images[i]->compute(classname, &det);
				im_descs.push_back(im_d);
			}
			ths.push_back(thread(create_codebook, im_descs, t));
		}
	}

	for(uint i = 0; i < ths.size(); i++)
	{
		ths[i].join();
	}
	cout << "*** ENDING **" << endl;
}
