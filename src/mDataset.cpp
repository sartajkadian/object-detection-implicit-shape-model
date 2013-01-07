#include <mDataset.h>
#include <Util.h>

#include <fstream>

using namespace std;
using namespace cv;


/**
 * mPASCALAnnotation
 */
mPASCALAnnotation::mPASCALAnnotation(const string& name_id, const string& name) :
	mAbstractAnnotation(name_id, name)
{}
mPASCALAnnotation::~mPASCALAnnotation() {}

void mPASCALAnnotation::_parseObject(xmlNodePtr objnode, char** name, Rect& bb)
{
	for(xmlNodePtr objc = objnode->children; objc != NULL; objc = objc->next)
	{
		if (xmlStrcmp(objc->name, (const xmlChar*)"name") == 0)
		{
			*name = new char[sizeof(xmlNodeGetContent(objc))];
			strcpy(*name, (const char*)xmlNodeGetContent(objc));
		}
		if (xmlStrcmp(objc->name, (const xmlChar*)"bndbox") == 0)
		{
			int xmax, xmin, ymax, ymin;
			for(xmlNodePtr objcc = objc->children; objcc != NULL; objcc = objcc->next)
			{
				if (xmlStrcmp(objcc->name, (const xmlChar*)"xmax") == 0)
				{
					xmax = Util::stringToNum<int>((const char*)xmlNodeGetContent(objcc));
				}
				if (xmlStrcmp(objcc->name, (const xmlChar*)"xmin") == 0)
				{
					xmin = Util::stringToNum<int>((const char*)xmlNodeGetContent(objcc));
				}
				if (xmlStrcmp(objcc->name, (const xmlChar*)"ymax") == 0)
				{
					ymax = Util::stringToNum<int>((const char*)xmlNodeGetContent(objcc));
				}
				if (xmlStrcmp(objcc->name, (const xmlChar*)"ymin") == 0)
				{
					ymin = Util::stringToNum<int>((const char*)xmlNodeGetContent(objcc));
				}
			}
			bb.x = xmin;
			bb.y = ymin;
			bb.width = xmax-xmin;
			bb.height = ymax-ymin;
		}
	}
}

void mPASCALAnnotation::parse(const string& file)
{
	cout << "* Parsing Annotation File [" << file << "]" << endl;

	xmlDocPtr doc = xmlParseFile(file.c_str());
	if (doc == NULL)
	{
		cerr << "No document [" << file << "]" << endl;
		return;
	}

	xmlNodePtr curNode = xmlDocGetRootElement(doc);
	if (curNode == NULL)
	{
		cerr << "Document [" << file << "] is empty" << endl;
		return;
	}

	if (xmlStrcmp(curNode->name, (const xmlChar*)"annotation") != 0)
	{
		cerr << "Wrong Format; Root node is not called 'annotation' --> ;" << curNode->name << ";" << endl;
		return;
	}

	// loop through children
	curNode = curNode->children;
	while (curNode != NULL)
	{
		if (xmlStrcmp(curNode->name, (const xmlChar*)"object") == 0)
		{
			char* name = NULL;
			Rect bb;
			_parseObject(curNode, &name, bb);
			addObject(name, bb);
		}
		curNode = curNode->next;
	}

	xmlFreeDoc(doc);
}

/**
 *
 */
mCTAnnotation::mCTAnnotation(const string& name_id, const string& name) :
	mAbstractAnnotation(name_id, name)
{}
mCTAnnotation::~mCTAnnotation() {}

void mCTAnnotation::parse(const string& file)
{
	cout << "* Parsing Annotation File [" << file << "]" << endl;

	ifstream sinput(file.c_str(), ifstream::in);
	string sline;

	char cname[20];
	int xbl, ybl, xtl, ytl, xtr, ytr, xbr, ybr;
	Rect bb;
	bool hasname = false, hasbb = false;

	while(std::getline(sinput, sline))
	{
		if (sscanf(sline.c_str(), "%d %d %d %d %d %d %d %d", &xbl, &ybl, &xtl, &ytl, &xtr, &ytr, &xbr, &ybr) == 8)
		{
			bb.x = xtl;
			bb.y = ytl;
			bb.width = xtr - xtl;
			bb.height = ybl - ytl;
			hasbb = true;
		}
		else if (sscanf(sline.c_str(), "%s", cname) == 1)
		{
			hasname = true;
		}
	}

	sinput.close();

	if (!(hasname && hasbb))
	{
		cerr << "Incorrect annotation format " << file << endl;
		return;
	}

	addObject(cname, bb);
}


/**
 * mPASCALImageSet
 */
mPASCALImageSet::mPASCALImageSet(const string& image_path, const string& annotation_path) :
	mAbstractImageSet(image_path, annotation_path)
{}
mPASCALImageSet::~mPASCALImageSet() {}

vector<mAbstractAnnotation_p> mPASCALImageSet::parse(const string& image_set, int max_num_images, bool inclass_only)
{
	cout << "* Parsing Image Set [" << image_set << "]" << endl;
	vector<mAbstractAnnotation_p> ann_v;
	ifstream sinput(image_set.c_str(), ifstream::in);
	string sline;
	while(std::getline(sinput, sline))
	{
		int isclass;
		char im_name[20];
		//TODO remove != EOF and test
		if (sscanf(sline.c_str(), "%s %d", im_name, &isclass) != EOF)
		{
			mAbstractAnnotation_p ann(new mPASCALAnnotation(im_name, _imagePath+"/"+im_name+".jpg"));
			ann->parse(_annotationPath+"/"+im_name+".xml");

			// if not in class
			if (inclass_only && isclass == -1)	continue;

			ann_v.push_back(ann);

			if (max_num_images != -1 && ann_v.size() >= max_num_images)
			{
				break;
			}
		}
	}
	return ann_v;
}

/**
 * mCTImageSet
 */
mCTImageSet::mCTImageSet(const string& ds_path) :
		mAbstractImageSet(ds_path, ds_path)
{}
mCTImageSet::~mCTImageSet() {}

vector<mAbstractAnnotation_p> mCTImageSet::parse(const string& image_set, int max_num_images, bool inclass_only)
{
	cout << "* Parsing Image Set [" << image_set << "]" << endl;
	vector<mAbstractAnnotation_p> ann_v;
	ifstream sinput(image_set.c_str(), ifstream::in);
	string sline;
	while(std::getline(sinput, sline))
	{
		char im_name[20];
		if (sscanf(sline.c_str(), "%s", im_name) != EOF)
		{
			mAbstractAnnotation_p ann(new mCTAnnotation(im_name, _imagePath+"/"+im_name+".jpg"));
			ann->parse(_annotationPath+"/"+im_name+".txt");

			ann_v.push_back(ann);

			if (max_num_images != -1 && ann_v.size() >= max_num_images)
			{
				break;
			}
		}
	}
	return ann_v;
}
