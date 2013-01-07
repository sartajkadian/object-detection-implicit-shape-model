//    Copyright (C) 2012  Roberto J. López-Sastre (robertoj.lopez@uah.es)
//                        Daniel Oñoro-Rubio
//			  Víctor Carrasco-Valdelvira
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <aggclustering.h>

#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

/**
 * AgglomerativeClustering
 */
mAgglomerativeClusterer::mAgglomerativeClusterer(const cv::Mat& data_points, mAbstractSimilarity* sim_measure, double thresh_euclid, double ep, unsigned int COMP) :
		_data(data_points),
		similarityMeasure(sim_measure),
		thres_euclid(thresh_euclid),
		thres(thresh_euclid * thresh_euclid),
		ep(ep),
		COMP(COMP) {
}
mAgglomerativeClusterer::~mAgglomerativeClusterer()
{
	delete similarityMeasure;
}

/**
 * Agglomerate 2 clusters C1 and C2 and put result in C1
 */
void mAgglomerativeClusterer::agglomerate_clusters(mCluster &C1, const mCluster &C2) {
	//Agglomerate two clusters
	//C1=C1+C2

	unsigned int m = C2.data_index.size();
	unsigned int n = C1.data_index.size();
	double d_sum = double(n + m);

	//Copy index values
	for (unsigned int i = 0; i < m; i++) {
		C1.data_index.push_back(C2.data_index[i]);
	}

	//update centroid
	C1.centroid = centroid_div(d_sum, centroid_plus(centroid_mul(double(n), C1.centroid), centroid_mul(double(m), C2.centroid)));

	//update variance
	vector<float> diff = centroid_diff(C1.centroid, C2.centroid);
	C1.cvar = ((double(n) * C1.cvar) + (double(m) * C2.cvar)
			+ (((double(n * m)) / (d_sum)) * (squared_magnitude(diff))))
			/ (d_sum);
}

//Get Nearest Neighbor
/**
 * Return the nearest neighbouring cluster to a cluster along with the similarity between them
 * @param C point to get nn to
 * @param R set to pick nn from
 * @return similarity between the nn clusters based on their means and variation
 * @return iterator to the nearest neighbour in the given cluster
 */
void mAgglomerativeClusterer::get_nn(mCluster &C, list<mCluster> &R, double &sim, list<mCluster>::iterator &iNN) {
	//Return the NN of cluster C in the list R.
	//iNN is the iterator of the NN in R, and sim is the similarity.

	unsigned int n = R.size();
	list<mCluster>::iterator it, itBegin = R.begin(), itEnd = R.end();
//	vector<float> diff;
	double tmp_sim;

	if (n > 0) {
		//First iteration
		it = itBegin;
//		diff = centroid_diff(C.centroid, (*it).centroid);
//		sim = -(C.cvar + (*it).cvar + squared_magnitude(diff));

		sim = similarityMeasure->similarity(_data, C, *it);

//		cout << "**** NN ****" << endl;
//		cout << "   -- sim  [" << C.data_index << "]x[" << it->data_index << "]  ->" << sim << endl;

		iNN = it;
		it++;
		while(it != itEnd)
		{
//			diff = centroid_diff(C.centroid, (*it).centroid);
//			tmp_sim = -(C.cvar + (*it).cvar + squared_magnitude(diff));
			tmp_sim = similarityMeasure->similarity(_data, C, *it);
//			cout << "   -- sim  [" << C.data_index << "]x[" << it->data_index << "]  ->" << tmp_sim << endl;

			if (tmp_sim > sim) {
				sim = tmp_sim;
				iNN = it;
			}
			it++;
		}
	} else {
		cout << "Warning: R is empty (function: get_nn)" << endl;
		sim = 0;
	}
}

/**
 * Search for NN of cluster C  within Si and So (interior and exterior slices) with similarity less than 'limit'
 * Return the similarity of the 2 clusters, and which cluster in the list was the NN
 */
void mAgglomerativeClusterer::get_nn_in_slices(mCluster &C, list<mCluster> &X,
		mSlice &Si, mSlice &So, double &sim, list<mCluster>::iterator &iNN,
		double limit) {
	//Search within the interior and exterior slices, Si and So respectively
	mSlice::iterator itS, endS;
	list<mCluster>::iterator it;
	bool isfirst = true;
//	vector<float> diff;
	double tmp_sim;

	//Search first in the interior slice
	if (Si.size() > 0)
	{
		endS = Si.end();
		//First iteration
		isfirst = false;
		itS = Si.begin();
		it = (*itS).it;
//		diff = centroid_diff(C.centroid, (*it).centroid);
//		sim = -(C.cvar + (*it).cvar + squared_magnitude(diff));
		sim = similarityMeasure->similarity(_data, C, *it);

		iNN = it;
		itS++;
		for (; itS != endS; itS++)
		{
			it = (*itS).it;
//			diff = centroid_diff(C.centroid, (*it).centroid);
//			tmp_sim = -(C.cvar + (*it).cvar + squared_magnitude(diff));
			tmp_sim = similarityMeasure->similarity(_data, C, *it);
			if (tmp_sim > sim)
			{
				sim = tmp_sim;
				iNN = it;
			}
		}

		//DEBUG
		ndist += Si.size();

		//Do we need to search in the exterior slice?
		if (sim >= limit)
		{
			return; //NO
		}
	}

	//Search in the exterior slice (if any)
	if (So.size() > 0) {
		endS = So.end();
		if (isfirst) {
			//First iteration
			isfirst = false;
			itS = So.begin();
			it = (*itS).it;
//			diff = centroid_diff(C.centroid, (*it).centroid);
//			sim = -(C.cvar + (*it).cvar + squared_magnitude(diff));
			sim = similarityMeasure->similarity(_data, C, *it);
			iNN = it;
		}

		for (itS = So.begin(); itS != endS; itS++) {
			it = (*itS).it;
//			diff = centroid_diff(C.centroid, (*it).centroid);
//			tmp_sim = -(C.cvar + (*it).cvar + squared_magnitude(diff));
			tmp_sim = similarityMeasure->similarity(_data, C, *it);
			if (tmp_sim > sim) {
				sim = tmp_sim;
				iNN = it;
			}
		}

		//DEBUG
		ndist += So.size();
	}
}

/**
 * Do a binary search on the map 'fmap' to the left
 */
int unsigned mAgglomerativeClusterer::binary_search_left(fmap &f_map,
		double d) {
	//Search in f_map to the left
	int b = 0, c, t = f_map.size() - 1, aux_c;
	double q;
	bool end = false;

	// Move until we find a top value not erased
	while (!f_map[t].mask) {
		t--;
	}

	// Move until we find a base value not erased
	while (!f_map[b].mask) {
		b++;
	}

	while (((t - b) > 1) && !end) {
		//the middle
		c = (b + t) >> 1;

		//erased
		if (!f_map[c].mask) {
			// Search a not erased element

			// Fitst iteration
			aux_c = c + 1;

			//Searching upward
			while (!f_map[aux_c].mask && (aux_c < t)) {
				aux_c++;
			}

			//Do we need to search downward?
			if (!f_map[aux_c].mask || (aux_c >= t)) {
				aux_c = c - 1;

				// Searching downward
				while (!f_map[aux_c].mask && (aux_c > b))
					aux_c--;
			}

			if (aux_c == b) {
				end = true;
			} else {
				c = aux_c;
			}

		} //if erased

		if (!end && !f_map[c].mask) {
			cout << "bsearchL failed" << endl;
			exit(-1);
		}

		if (!end) {
			q = (*f_map[c].it).centroid[COMP];

			if (d < q) {
				t = c;
			} else if (d > q) {
				b = c;
			} else {
				return c;
			}
		}
	}

	if (!f_map[b].mask || !f_map[t].mask) {
		cout << "Error: bsearchL failed" << endl;
		exit(-1);
	}

	return ((d <= (*f_map[b].it).centroid[COMP]) ? b : t);
}

/**
 * perform binary search to the right of an fmap
 */
unsigned int mAgglomerativeClusterer::binary_search_right(fmap &f_map,
		double d) {
	int b = 0, c, t = f_map.size() - 1, aux_c;
	double q;
	bool end = false;

	// Move until find a top not erased
	while (!f_map[t].mask) {
		t--;
	}

	// Move until find a base not erased
	while (!f_map[b].mask) {
		b++;
	}

	while ((t - b > 1) && !end) {

		c = (b + t) >> 1;

		//erased
		if (!f_map[c].mask) {
			aux_c = c + 1;

			// Search upward
			while (!f_map[aux_c].mask && (aux_c < t)) {
				aux_c++;
			}

			//Do we need to search downward
			if (!f_map[aux_c].mask || (aux_c >= t)) {
				aux_c = c - 1;

				// Search downward
				while (!f_map[aux_c].mask && (aux_c > b)) {
					aux_c--;
				}
			}

			if (aux_c == b) {
				end = true;
			} else {
				c = aux_c;
			}
		}

		if (!end && !f_map[c].mask) {
			cout << "error: bsearchR failed" << endl;
			exit(-1);
		}

		if (!end) {
			q = (*f_map[c].it).centroid[COMP];
			if (d < q) {
				t = c;
			} else if (d > q) {
				b = c;
			} else {
				return c;
			}
		}
	}

	if (!f_map[b].mask || !f_map[t].mask) {
		cout << "error: bsearchR failed" << endl;
		exit(-1);
	}

	return ((d >= (*f_map[t].it).centroid[COMP]) ? t : b);
}

int mAgglomerativeClusterer::bsearch(fmap &f_map, double d, unsigned int &b,
		unsigned int &t) {
	unsigned int leng = f_map.size() - 1;
	b = 0;
	t = leng;
	unsigned int c, aux_c;
	double q;
	bool end = false;

	//highest no deleted position
	while ((t > 0) && !f_map[t].mask) {
		t--;
	}

	//lowest no deleted position
	while ((b < leng) && !f_map[b].mask) {
		b++;
	}

	//Check conditions
	if (b > t) {
		return -1;
	}

	//Binary search
	while (((t - b) > 1) && !end) {
		c = (b + t) >> 1;

		//is it erased?
		if (!f_map[c].mask) {
			aux_c = c + 1;

			// Search upward
			while (!f_map[aux_c].mask && (aux_c < t))
				aux_c++;

			// Do we have to search downward?
			if (!f_map[aux_c].mask || (aux_c >= t)) {
				aux_c = c - 1;

				// Search downward
				while (!f_map[aux_c].mask && (aux_c > b))
					aux_c--;
			}

			if (aux_c == b)
				end = true;
			else
				c = aux_c;
		}

		if (!end && !f_map[c].mask) {
			cout << "error: bsearch error" << endl;
			exit(-1);
		}

		if (!end) {
			q = (*f_map[c].it).centroid[COMP];
			if (d < q) {
				t = c;
			} else if (d > q) {
				b = c;
			} else {
				return c;
			}
		}
	}

	if (!f_map[b].mask || !f_map[t].mask) {
		cout << "error: bsearch error" << endl;
		exit(-1);
	}

	if (b == t) {
		return ((*f_map[b].it).centroid[COMP] >= d) ? t : t + 1;
	} else {
		return ((*f_map[b].it).centroid[COMP] >= d) ? b : t;
	}
}

void mAgglomerativeClusterer::init_slices(fmap &f_map, list<mCluster> &X,
		mSlice &Si, mSlice &So, double V, double e) {
	//Generate the slice in the space where the NN candidates are. The slice has 2e width.
	unsigned int min, max, bmax, bmin, tmax, tmin, i;
	mCandidate c;

	//Three slices? (recall: ep is the parameter for slicing)
	if (e > ep) {
		//Build interior slice
		min = binary_search_left(f_map, V - ep);
		max = binary_search_right(f_map, V + ep);

		for (i = min; i <= max; i++) {
			if (f_map[i].mask) {
				c.it = f_map[i].it;
				c.index = i;
				Si.push_back(c);
			}
		}

		if (min != 0) //generate bottom candidate list
		{
			bmax = min - 1;
			//Build bottom slice
			bmin = binary_search_left(f_map, V - e);

			for (i = bmin; i <= bmax; i++) {
				if (f_map[i].mask) {
					c.it = f_map[i].it;
					c.index = i;
					So.push_back(c);
				}
			}
		}

		if (max != (f_map.size() - 1)) {
			tmin = max + 1;

			//Build top slice
			tmax = binary_search_right(f_map, V + e);

			for (i = tmin; i <= tmax; i++) {
				if (f_map[i].mask) {
					c.it = f_map[i].it;
					c.index = i;
					So.push_back(c);
				}
			}
		}
	} else {
		//only one slice
		min = binary_search_left(f_map, V - e);
		max = binary_search_right(f_map, V + e);

		for (i = min; i <= max; i++) {
			if (f_map[i].mask) {
				c.it = f_map[i].it;
				c.index = i;
				Si.push_back(c);
			}
		}
	}
}

void mAgglomerativeClusterer::insert_element(fmap &f_map, list<mCluster> &X,
		mCluster &V) {
	//Insert element V in X and update f_map
	mElement elem2insert, aux_elem;
	int pos = 0;
	unsigned int b, t;
	bool update_free_top;

	//Push back the element in X
	X.push_back(V);

	// Initialize the element to insert
	elem2insert.mask = true;

	elem2insert.it = X.end();
	elem2insert.it--;

	//Search for a position
	pos = bsearch(f_map, V.centroid[COMP], b, t);

	// f_map is empty
	if (-1 == pos) {
		pos = t;
		f_map[pos] = elem2insert;
		return;
	}

	//Is pos the last position?
	if (f_map.size() == (unsigned int) pos) {
		pos--;
		if (!f_map[pos].mask) {
			f_map[pos] = elem2insert;
			return;
		}

		//Insert downwards

		while (elem2insert.mask) {

			//Save current pos element
			aux_elem = f_map[pos];

			//Insert element in pos
			f_map[pos] = elem2insert;

			//Update elem2insert
			elem2insert = aux_elem;

			pos--;

		}
		free_top = pos;
		return;

	}

	if (pos >= free_top) {
		//Insert downwards
		while (elem2insert.mask) {
			pos--;
			//Save current pos element
			aux_elem = f_map[pos];

			//Insert element in pos
			f_map[pos] = elem2insert;

			//Update elem2insert
			elem2insert = aux_elem;
		}
		//update free_top?
		update_free_top = true;

	} else //upwards
	{
		if (f_map[pos].mask)
			if (V.centroid[COMP] >= (*f_map[pos].it).centroid[COMP])
				pos++;

		while (elem2insert.mask) {
			//Save current pos element
			aux_elem = f_map[pos];

			//Insert element in pos
			f_map[pos] = elem2insert;

			//Update elem2insert
			elem2insert = aux_elem;

			//Update index
			pos++;
		}

		//update free_top?
		update_free_top = (pos < free_top) ? false : true;

	}

	//Update free_top just in case it has been occupied
	if (update_free_top) {
		free_top = pos - 1;
		while ((free_top > 0) && (f_map[free_top].mask))
			free_top--;
	}

}

void mAgglomerativeClusterer::erase_element(fmap &f_map, list<mCluster> &X,
		list<mCluster>::iterator it) {
	int l = f_map.size(), i;

	for (i = 0; i < l; i++) {
		if (f_map[i].mask && f_map[i].it == it) {
			f_map[i].mask = false;
			X.erase(it);

			//Update free_top
			free_top = (i > free_top) ? i : free_top;
			return;
		}
	}

	cout << "error: erasing element" << endl;
	exit(-1);
}

void mAgglomerativeClusterer::init_map(fmap &f_map, list<mCluster> &X) {
	//Create forward map
	list<mCluster>::iterator itX, endX;
	list<mPoint<list<mCluster>::iterator> > aux_list;
	list<mPoint<list<mCluster>::iterator> >::iterator it, itend;
	mElement eaux;

	//Get the list of iterators
	endX = X.end();
	for (itX = X.begin(); itX != endX; itX++) {
		aux_list.push_back(
				mPoint<list<mCluster>::iterator>(itX, itX->centroid[COMP]));
	}

	//Sorting
	aux_list.sort();

	eaux.mask = true; //Mask = true for all elements

	//Convert the sorted list to f_map
	itend = aux_list.end();
	for (it = aux_list.begin(); it != itend; it++) {
		eaux.it = (*it).get_index(); //save the iterators
		f_map.push_back(eaux);
	}

	//Update free top
	free_top = -1;
}

void mAgglomerativeClusterer::cluster()
{
	//This function computes the Fast RNN (reciprocal nearest neighbors) clustering.
	unsigned int dim = _data.cols;
	unsigned int n = _data.rows;

	//Chain for the clusters R
	list<mCluster> R(n);
	list<mCluster>::iterator it, itEnd = R.end(), iNN, penult;
	//	matrix_data::const_iterator data_it = data_points.begin();
	unsigned int Xindex = 0;
	double sim = 0;
	//TODO positive for sq distance ??
	double l_agg_thres = (1 * thres_euclid);
	double epsilon;
	mSlice Si, So; //slices with candidates
	bool RNNfound = false;

	//DEBUG
	ndist = 0;

	//Initialize list R - each vector in a separate cluster (start point of the algorithm)
	for (it = R.begin(); it != itEnd; it++, Xindex++)
	{
		//update index of the vector
		it->data_index.push_back(Xindex);
		//update centroid
		vector<float> cen_v(_data.cols);
		for (int i = 0; i < _data.cols; i++)
		{
			cen_v[i] = _data.at<float>(Xindex, i);
		}
		it->centroid = cen_v;
		//update variance
		it->cvar = 0;
	}

	//NN-Chain (the pair at the end of this chain is always a RNN)
	list<mCluster> L;
	//Chain for similarities
	list<double> Lsim;
	//chain for the (final) clusters C
	list<mCluster> C;

	//Create forward map
	fmap f_map;
	init_map(f_map, R);

	//The algorithm starts with a random cluster
	srand(time(NULL));
	unsigned int rp = (unsigned int) rand() % n; //random integer [0,n-1]

	//Add to L
	it = R.begin();
	advance(it, rp);
	L.push_back(*it);

	//R\rp -> delete the cluster in R and mark as erased in f_map
	erase_element(f_map, R, it);

	//First iteration
	if (R.size() > 0) {
		//Get nearest neighbor
		get_nn(L.back(), R, sim, iNN);

		//DEBUG
		ndist += R.size();

		//Add to the NN chain
		L.push_back(*iNN); //add to L
		erase_element(f_map, R, iNN); //delete from R
		Lsim.push_back(sim); //add to Lsim

		//Only two clusters?
		if (R.size() == 0) {
			penult = L.end();
			penult--;
			penult--;
			//check the similarity (last element)
			if (sim > l_agg_thres) {
				//Agglomerate clusters
				agglomerate_clusters(L.back(), *penult);
				C.push_back(L.back());
			} else {
				//Save in C separately
				C.push_back(*penult);
				C.push_back(L.back());
			}
			L.clear(); //free memory
		}
	} else {
		//R is empty
		if (L.size() == 1) {
			C.push_back(L.back()); // Only one vector, only one cluster
		}

		L.clear(); //free memory
	}

	//Main loop
	while (R.size() > 0) {
		RNNfound = false;

		//Clear slices
		Si.clear();
		So.clear();

		//Update epsilon with the last sim
		epsilon = sqrt(-1 * Lsim.back());

		epsilon = (epsilon < ep) ? epsilon : ep;

		//Identify slices
		init_slices(f_map, R, Si, So, L.back().centroid[COMP], epsilon);

		if ((Si.size() > 0) || (So.size() > 0)) //Search for a NN within the candidate list
		{
			get_nn_in_slices(L.back(), R, Si, So, sim, iNN, l_agg_thres);

			if (sim > Lsim.back()) //no RNN
			{

				//No RNNs, add s to the NN chain
				L.push_back(*iNN); //add to L
				erase_element(f_map, R, iNN); //delete from R
				Lsim.push_back(sim); //add to Lsim

				if (R.size() == 0) //R has been emptied
				{
					//check the last similarity
					if (Lsim.back() > l_agg_thres) {
						//Agglomerate clusters
						penult = L.end();
						penult--;
						penult--;
						agglomerate_clusters(L.back(), *penult);
						insert_element(f_map, R, L.back());

						//delete the last two elements in L
						L.pop_back();
						L.pop_back();

						//delete similarities
						Lsim.pop_back();
						if (Lsim.size() >= 1)
							Lsim.pop_back();

						//Initialize the chain with the following nearest neighbour
						if (L.size() == 1) {
							//Get nearest neighbor
							get_nn(L.back(), R, sim, iNN);

							ndist += R.size();

							//Add to the NN chain
							L.push_back(*iNN); //add to L
							erase_element(f_map, R, iNN); //delete from R
							Lsim.push_back(sim); //add to Lsim

							if (R.size() == 0) //R has been emptied?
							{
								penult = L.end();
								penult--;
								penult--;

								//check the similarity
								if (Lsim.back() > l_agg_thres) {
									//Agglomerate clusters
									agglomerate_clusters(L.back(), *penult);
									C.push_back(L.back()); //add the cluster to C
								} else {
									//Save in C
									C.push_back(*penult);
									C.push_back(L.back());
								}
								break; //end main while
							}
						}
					} else {
						//Add the clusters to C (separate clusters)
						itEnd = L.end();
						for (it = L.begin(); it != itEnd; it++)
							C.push_back(*it);
						break; //end main while
					}
				}
			} else
				//A RNN
				RNNfound = true;
		}

		//RNN found
		if (RNNfound || ((Si.size() == 0) && (So.size() == 0))) {

			if (Lsim.back() > l_agg_thres) //can they be agglomerated?
			{
				//Agglomerate clusters
				penult = L.end();
				penult--;
				penult--;
				agglomerate_clusters(L.back(), *penult);

				insert_element(f_map, R, L.back());

				L.pop_back();
				L.pop_back();

				//delete similarities
				Lsim.pop_back();
				if (Lsim.size() >= 1)
					Lsim.pop_back();

				if (L.size() == 1) {
					//Get nearest neighbor
					get_nn(L.back(), R, sim, iNN);

					//DEBUG
					ndist += R.size();

					//Add the NN chain
					//Add to the NN chain
					L.push_back(*iNN); //add to L
					erase_element(f_map, R, iNN); //delete from R
					Lsim.push_back(sim); //add to Lsim

					if (R.size() == 0) //R has been emptied?
					{
						penult = L.end();
						penult--;
						penult--;
						//check the similarity
						if (Lsim.back() > l_agg_thres) {
							//Agglomerate clusters
							agglomerate_clusters(L.back(), *penult);
							C.push_back(L.back()); //add the cluster to C

						} else {
							//Save in C
							C.push_back(*penult);
							C.push_back(L.back());
						}

						break;
					}
				}
			} else //discard this chain
			{
				//Add the clusters to C (separate clusters)
				itEnd = L.end();
				for (it = L.begin(); it != itEnd; it++)
					C.push_back(*it);

				L.clear();
			}
		}

		//Do we need to start a new chain?
		if (L.size() == 0) {
			//Initialize a new chain
			Lsim.clear();

			//random point
			srand(time(NULL));
			rp = rand() % R.size(); //random point

			//Add to L
			it = R.begin();
			advance(it, rp);
			L.push_back(*it);

			//R\rp -> delete the cluster in R and mark as erased in f_map
			erase_element(f_map, R, it);

			//First iteration
			if (R.size() > 0) {
				//Get nearest neighbor
				get_nn(L.front(), R, sim, iNN);

				//DEBUG
				ndist += R.size();

				//Add to the NN chain
				L.push_back(*iNN); //add to L
				erase_element(f_map, R, iNN); //delete from R
				Lsim.push_back(sim); //add to Lsim

				//Only two clusters?
				if (R.size() == 0) {
					penult = L.end();
					penult--;
					penult--;

					//check the similarity (last element)
					if (Lsim.back() > l_agg_thres) {
						//Agglomerate clusters
						agglomerate_clusters(L.back(), *penult);
						C.push_back(L.back());
					} else {
						//Save in C separately
						C.push_back(*penult);
						C.push_back(L.back());
					}
					break; //end main while
				}
			} else //R is empty
			{
				if (L.size() == 1)
					C.push_back(L.front());
			}
		}
	}

	//Chain C contains all the clusters
	int nc = C.size(); //number of clusters

	// take the dimension from the first element in C
	if (nc > 0) {
		dim = (C.front()).centroid.size();
	}

	// initialize cluster centers
	labels = Mat::zeros(n, 1, CV_32S);
	cluster_centre = Mat::zeros(nc, dim, CV_32F);

	//	labels = vector<unsigned int>(n);
	//	vector<double> aux_centroid(dim);

	itEnd = C.end();
	unsigned int c_label = 0, num_labels = 0, s;

	int center_index = 0;
	for (it = C.begin(); it != itEnd; it++, center_index++) {
		//convert from vcl to vnl
		//add centroid
		for (s = 0; s < dim; s++) {
			cluster_centre.at<float>(center_index, s) = it->centroid[s];
		}

		num_labels += (*it).data_index.size();

		// label data points
		for (s = 0; s < (*it).data_index.size(); s++) {
			//			labels[(*it).data_index[s]] = c_label;
			labels.at<int>(it->data_index[s], 0) = c_label;
		}

		c_label++;
	}

	//Were all the points asigned?.
	if (num_labels != n) {
		cout << "Warning: all the points were not assigned to a cluster!" << endl;
		cout << "Num. Labels = " << num_labels << " Num. Points = " << n << endl;
	}
}
