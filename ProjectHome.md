The algorithm constructs an non-parametric implicit representation of an object at training time by creating clusters of the object's parts (image patches extracted around SIFT interest points), keeping a spatial distribution relative to the centre point of the object, resulting in a codebook.

Generalized Hough Transform is performed to vote for centre of the object in continuous space, and the maxima is found. Mean-Shift should be used to find the maxima, but for this implementation, something else was made up (which works most of the time, but definitely needs to be improved). The result is a bounding box around the object(s) computed from the image patches that contributed to the centres at the maxima of the distribution.

75% accuracy rate has been achieved, working best on images of the same scale as the used for training, so I suspect there was some over-fitting on the CALTech dataset, as their frontal face images are all of similar scale.

The codebooks provided are over different detectors, numbers of images used for training, thresholds, and patch sizes, respectively to the naming schema used for the codebooks.