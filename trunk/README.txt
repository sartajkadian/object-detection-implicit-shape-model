To Build/Run copy the DVD onto the local disk

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

Data:
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
