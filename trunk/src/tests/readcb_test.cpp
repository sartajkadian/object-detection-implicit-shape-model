#include <mImplicitShapeModel.h>
#include <Util.h>

using namespace cv;
using namespace std;

// input/output directory
// display centers
// num close descriptors

bool write_centers = false;
int num_descriptors = 0;
string input_codebook = "./data/";
string output_dir = "./data/";

void parse_cmd(int argc, char** argv)
{
	for(int i = 1; i < argc; i += 2)
	{
		char* k = argv[i];

		if (!strcmp(k, "-i") || !strcmp(k, "--input"))
		{
			input_codebook = argv[i+1];
		}
		else if (!strcmp(k, "-o") || !strcmp(k, "--output"))
		{
			output_dir = argv[i+1];
		}
		else if (!strcmp(k, "--wcenters"))
		{
			write_centers = true;
			i--;
		}
		else if (!strcmp(k, "--ndesc"))
		{
			num_descriptors = Util::stringToNum<int>(argv[i+1]);
		}
		else if (!strcmp(k, "-h"))
		{
			cout << "Usage: " << endl;
			exit(EXIT_SUCCESS);
		}
		else
		{
			cout << "Invalid argument " << k << endl;
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char** argv)
{
	parse_cmd(argc, argv);

	mCodebook_p codeb(new mCodebook());
	if (!codeb->read(input_codebook))
	{
		cout << "** ERROR READING CODEBOOK" << endl;
		exit(EXIT_FAILURE);
	}

	cout << *codeb << endl;

	vector<mCodebookEntry_p> codee = codeb->getCodebookEntries();

	int h = num_descriptors+2;
	int w = codee.size();

	int wid = w*(mImageDescriptor::PATCH_SIZE+1);
	int height = h*(mImageDescriptor::PATCH_SIZE+1);

	Point2i offset(0.5*mImageDescriptor::PATCH_SIZE+1, 0.5*mImageDescriptor::PATCH_SIZE+1);
	Point2i currentp(offset);
	cout << "init " << currentp << endl;
	cout << "wid " << wid << ", " << height << endl;

	Mat im = Mat::zeros(height, wid, CV_32F);

	for(uint i = 0; i < codee.size(); i++)
	{
		try {
			mImageDescriptor::addPatch(im, codee[i]->getCodebookEntry().reshape(0,mImageDescriptor::PATCH_SIZE), currentp);
		} catch(cv::Exception& e) {
			cout << e.what() << endl;
		}

		currentp.y += 2*offset.y;
		for(uint di = 0; di < num_descriptors && di < codee[i]->getDescriptors().size(); di++)
		{
			currentp.y += 2*offset.y;
			mImageDescriptor::addPatch(im, codee[i]->getDescriptors()[di]->getPatch(), currentp);
		}
		currentp.x += 2*offset.x;
		currentp.y = offset.y;
		cout << "Centre " << i << " -> " << codee[i]->getDescriptors().size() << " descriptors" << endl;
	}

	if (write_centers)
	{
		imwrite(input_codebook+".jpg", im*255.0f);
	}

	namedWindow("Codebook", CV_WINDOW_KEEPRATIO);
	imshow("Codebook", im);
	waitKey(0);
}
