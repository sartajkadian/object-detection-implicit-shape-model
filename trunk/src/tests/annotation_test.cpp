#include <mDataset.h>


using namespace std;

ostream& operator<<(ostream& out, Rect r)
{
	return out << r.x << ", " << r.y << ", " << r.width << ", " << r.height;
}

void printAnnotation(mAbstractAnnotation* pAnnot)
{
	map<string, vector<Rect> > classes = pAnnot->getClassBB();

	cout << "num classes in  " << pAnnot->getImagePath() << "  -> " << classes.size() << endl;
	for(map<string, vector<Rect> >::iterator itr = classes.begin(); itr != classes.end(); itr++)
	{
		vector<Rect> bb = itr->second;
		cout << "ClassName: " << itr->first << " ;  BB: " << bb.size() << endl;
		for(uint i = 0; i < bb.size(); i++)
		{
			cout << "  - " << bb[i] << endl;
		}
	}
}

int main(int argc, char** argv)
{
//	mPASCALAnnotation pAnnot("IMAGE_PATH");
//	pAnnot.parse("file:///data/_work_repo_wd/datasets/VOCdevkit/VOC2010/Annotations/2010_002168.xml");
//	printAnnotation(&pAnnot);
//	mPASCALImageSet im_set("/data/_work_repo_wd/datasets/VOCdevkit/VOC2010/JPEGImages", "/data/_work_repo_wd/datasets/VOCdevkit/VOC2010/Annotations");
//	vector<mAbstractAnnotation_p> annots = im_set.parse("/data/_work_repo_wd/datasets/VOCdevkit/VOC2010/ImageSets/Main/car_train.txt");
//	for(uint i = 0; i < annots.size(); i++)
//	{
//		printAnnotation(annots[i].get());
//	}

	mCTAnnotation annot("/data/_work_repo_wd/datasets/caltech_faces/image_0001.jpg");
	annot.parse("data/_work_repo_wd/datasets/caltech_faces/image_0001.txt");
	printAnnotation(&annot);

	mCTImageSet im_set("/data/_work_repo_wd/datasets/caltech_faces");
	vector<mAbstractAnnotation_p> annots = im_set.parse("data/face_caltech.txt");
	for(uint i = 0; i < annots.size(); i++)
	{
		printAnnotation(annots[i].get());
	}
}
