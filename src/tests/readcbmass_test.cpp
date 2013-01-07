#include <mImplicitShapeModel.h>
#include <Util.h>

using namespace std;

float thresh_min = 0.9;
float thresh_max = 0.98;
int nimages_max = 20;

string classname = "face";
string cb_base_name = "./data/codebooks/codebook";
int main(int argc, char** argv)
{
	for(int i = 1; i <= nimages_max; i++)
	{
		for (float t = thresh_min; t <= thresh_max; t += 0.01)
		{
			cout << "------" << endl;
			string name = cb_base_name + "_" + Util::numToString<uint>(i) + "_" + Util::numToString<float>(t)+".xml";
			mCodebook_p codeb(new mCodebook());
			if (!codeb->read(name)) continue;
			cout << name << endl;

			vector<mCodebookEntry_p> vcodebe = codeb->getCodebookEntries();
			for(uint icb = 0; icb < vcodebe.size(); icb++)
			{
				mCodebookEntry_p codebe = vcodebe[icb];
				cout << "  Num Contrib. Patches: " << codebe->getDescriptors().size() << endl;
			}
		}
	}
}
