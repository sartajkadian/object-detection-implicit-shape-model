#include <aggclustering.h>

#include <sys/time.h>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;


Mat gen_data(int n_points, int range)
{
	Mat rnd_d = Mat::zeros(3*n_points, 2, CV_32F);
	cout << "TYPE " << rnd_d.type() << endl;

	int start = 0, end = n_points;
	int start_r = 0, end_r = range;

	for(int i = 0; i < 3; i++)
	{
		Mat sub_rnd_d = rnd_d.rowRange(start, end);
		RNG rnd(time(NULL));
		rnd.fill(sub_rnd_d, RNG::UNIFORM, Scalar(start_r, start_r), Scalar(end_r, end_r));

		start = end; end = end + n_points;
		start_r = end_r; end_r = end_r + range;
	}

	return rnd_d;
}

int main(int argc, char *argv[])
{
	//list with all the vectors
	Mat point_data = gen_data(100, 1);

	cout << "-- Gen Data " << point_data.rows << " points" << endl;

	// component for slicing
	unsigned int COMP = 0;
	// threshold for the agglomertive clustering (dependent on the dim of the space)
	double thres_euclid = 0.8;
	// epsilon (parameter for slicing)
	double ep = 0.1;

	cout << endl;
	cout << "================" << endl;
	cout << "    FAST-RNN    " << endl;
	cout << "================" << endl << endl;

	//fast-rnn clustering
	mAgglomerativeClusterer* clustering = new mAgglomerativeClusterer(new mSquaredDiffsimilarity(), thres_euclid, ep, COMP);
	clustering->cluster(point_data);

	cout << "-- Number of clusters " << clustering->getClusterCenters().rows << endl;
	cout << clustering->getClusterCenters() << endl;

	cout << "Labels" << endl;
	cout << clustering->getLabels() << endl;

	//Free memory, we will not need it anymore!
	delete clustering;

	return 0;
}
