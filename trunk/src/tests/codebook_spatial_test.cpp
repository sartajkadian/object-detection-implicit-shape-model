
#include <mImplicitShapeModel.h>

int main()
{
	mCodebook cb;
	if (!cb.read("data/codebooks/codebook_sift_25_070_35_cut.xml"))
	{
		cout << "*ERROR reading codebook" << endl;
		exit(0);
	}

	mISMimage_p image(new mISMimage(mAbstractAnnotation_p(new mCTAnnotation("", "data/training/image_0001.jpg"))));

	SiftFeatureDetector det;
	vector<mMatch_p> matches = cb.match(image, &det);

	mDebug::showMatch("Image Spatial", image, matches[15]);

//	mMatch_p match = matches[15];
//	mImageDescriptor_p md = match->getImageDescriptor();
//	vector<mCodebookEntry_p> mcbe = match->getMatchingCodewords();
//
//	Mat m = image->getImage().clone();
//	m.convertTo(m, CV_32F);
//	m /= 255.0f;
//
//	for(uint i = 0; i < mcbe.size(); i++)
//	{
//		mCodebookEntry_p cbe = mcbe[i];
//		vector<mImageDescriptor_p> mcbed = cbe->getDescriptors();
//		for(uint di = 0; di < mcbed.size(); di++)
//		{
//			Point2i p = md->getRelativeLocation() - mcbed[di]->getRelativeLocation();
//			line(m, p, md->getRelativeLocation(), Scalar(255,0,0));
//		}
//	}
//
//	mImageDescriptor::addPatch(m, md->getPatch(), md->getRelativeLocation());
//
//	imshow("Image Spatial", m);
	waitKey(0);
}
