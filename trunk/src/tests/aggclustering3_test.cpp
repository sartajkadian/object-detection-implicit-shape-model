#include <Util.h>
#include <mImplicitShapeModel.h>
#include <aggclustering.h>
#include <mDistance.h>

#include <vector>
using namespace std;

int main(int argc, char** argv)
{
//	float c1m[] = {1,1,1,1,1,1};
//	mCluster c1;
//	c1.centroid = vector<float>(c1m, c1m + sizeof(c1m)/sizeof(float));
//
//	float c2m[] = {1,1,1,1,1,1};
//	mCluster c2;
//	c2.centroid = vector<float>(c2m, c2m + sizeof(c2m)/sizeof(float));
//
//	mNGCsimilarity sim;
//	cout << sim.similarity(c1, c2);

	string class_name = "face";
	mAbstractAnnotation_p ann1(new mCTAnnotation("data/image_0001.jpg"));
	ann1->parse("data/image_0001.txt");
	mAbstractAnnotation_p ann2(new mCTAnnotation("data/image_0002.jpg"));
	ann2->parse("data/image_0002.txt");

	SurfFeatureDetector det;

	mISMimage_p im1(new mISMimage(ann1));
	mISMimage_p im2(new mISMimage(ann2));
	vector<mImageDescriptor_p> descs1 = im1->compute(class_name, &det);
	vector<mImageDescriptor_p> descs2 = im2->compute(class_name, &det);

	vector<mImageDescriptor_p> descs;
	descs.insert(descs.end(), descs1.begin(), descs1.end());
	descs.insert(descs.end(), descs2.begin(), descs2.end());

	cout << "num features " << descs.size() << endl;
	for(uint i = 0; i < descs.size(); i++)
	{
//		cout << descs[i]->getDescriptor() << endl;
		imshow("Patch " + Util::intToString(i), descs[i]->getPatch());
	}

//	// TEST NGS similarity measure
//	double maxs = 0;
//	int maxi = 0;
	mNGCsimilarity sim;
//	for(uint i = 1; i < descs.size(); i++)
//	{
//		double s = sim.similarity(descs[0]->getPatch(), descs[i]->getPatch());
//		if (s > maxs)
//		{
//			maxs = s;
//			maxi = i;
//		}
//	}
//	cout << maxi << endl;
//	cout << sim.similarity(descs[49]->getDescriptor(), descs[43]->getDescriptor()) << endl;
//	imshow("1", descs[49]->getPatch());
//	imshow("2", descs[43]->getPatch());

	// TEST clustering
	Mat data;//(descs.size(), descs[0]->getDescriptor().cols, descs[0]->getDescriptor().type());
	for(uint i = 0; i < descs.size(); i++)
	{
		data.push_back(descs[i]->getDescriptor());
	}
//	cout << data << endl;
//	cout << "d " << descs[0]->getDescriptor().rows << "x" << descs[0]->getDescriptor().cols << endl;
//	cout << data.rows << "x" << data.cols << endl;
	mAgglomerativeClusterer clusterer(new mNGCsimilarity(data), 0.8, 0.1, 0);
	clusterer.cluster(data);
	Mat cs = clusterer.getClusterCenters();
	Mat lbl = clusterer.getLabels();
	cout << "Clusters " << cs.rows << endl;
	for(int i = 0; i < cs.rows; i++)
	{
		Mat m;
		cs.row(i).copyTo(m);
		m = m.reshape(0, mImageDescriptor::PATCH_SIZE);
		imshow("c " + Util::intToString(i), m);
		imwrite("data/C_"+Util::intToString(i)+".jpg", m*255.0f);
	}

	return cvWaitKey(0);
}
