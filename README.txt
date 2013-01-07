The algorithm constructs an non-parametric implicit representation of an object at training time by creating clusters of the object's parts (image patches extracted around SIFT interest points), keeping a spatial distribution relative to the centre point of the object, resulting in a codebook.

Generalized Hough Transform is performed to vote for centre of the object in continuous space, and the maxima is found. Mean-Shift should be used to find the maxima, but for this implementation, something else was made up (which works most of the time, but definitely needs to be improved). The result is a bounding box around the object(s) computed from the image patches that contributed to the centres at the maxima of the distribution.

75% accuracy rate has been achieved, working best on images of the same scale as the used for training, so I suspect there was some over-fitting on the CALTech dataset, as their frontal face images are all of similar scale.

The codebooks provided are over different detectors, numbers of images used for training, thresholds, and patch sizes, respectively to the naming schema used for the codebooks.

Dependencies:
	- g++ (supports c++0x experimental features)
	- opencv 2.3
	- libxml2

Install:
	- All tests are built to ./bin
	- The library is built to ./lib
	- command line, navigate to root directory, make
	- To build codebook training, make traincb_test
	- To build codebook reading, make readcb_test
	- To build codebook matching, make match_test

Data (Data.7z):
	- codebooks in ./data/codebooks
	- multi-scale, multi-rotation, partial occlusion in ./data/testing
	- training set in ./data/training
	- training file ./data/caltech_face_training.txt
	- testing file positive ./data/caltech_face_testing.txt --> dataset ./data/caltech_faces
	- testing file negative ./data/caltech_car_testing.txt --> dataset ./data/caltech_cars
	- testing file negative ./data/caltech_noface_testing.txt --> dataset ./data/caltech_background

Run:
	- 'bin/traincb_test -h' --> needs similarity threshold, patch size, class name (faces)
	- 'bin/readcb_test -h' --> number of contributing patches to display, whether to store
	- 'bin/match_test -h' --> codebook to use, maxima window size, threshold
