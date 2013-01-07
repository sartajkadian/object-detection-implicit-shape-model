ROOT = $(PWD)

G_INC = /usr/include
G_LIB = /usr/lib

SRC_PATH = $(ROOT)/src
OBJ_PATH = $(ROOT)/obj
INC_PATH = $(ROOT)/include
LIB_PATH = $(ROOT)/lib
TEST_PATH = $(SRC_PATH)/tests

BIN_PATH = $(ROOT)/bin

SOURCE = aggclustering mBagOfFeatures mImplicitShapeModel Util mDataset mDistance

LIB_HEADERS = $(addprefix $(INC_PATH)/, $(addsuffix .h, $(SOURCE)))
LIB_SRC = $(addprefix $(SRC_PATH)/, $(addsuffix .cpp, $(SOURCE)))
LIB_OBJ = $(addprefix $(OBJ_PATH)/, $(addsuffix .o, $(SOURCE)))

INC = -I$(G_INC)/opencv2 -I$(G_INC)/libxml2 -I$(INC_PATH)
LIB = -L$(G_LIB) -L$(LIB_PATH)
LIBS = -lopencv_core -lopencv_highgui -lopencv_flann -lopencv_features2d -lopencv_video -lopencv_imgproc -lxml2

COMPILE_CMD = g++ -g -c -fPIC -std=c++0x
LINK_APP_CMD = g++ -g -std=c++0x
LINK_LIB_CMD = g++ -g -shared -fPIC -std=c++0x

LIB_NAME = miniproj_1
TARGET_LIB = $(LIB_PATH)/lib$(LIB_NAME).so

.SUFFIXES:
#implicit rule
# compile cpp library files
$(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp $(INC_PATH)/%.h
	@echo ---------
	@echo Compiling $<
	@echo ---------
	$(COMPILE_CMD) $< $(INC) -o $@

# compile cpp test files
%_test:	$(TEST_PATH)/%_test.cpp $(TARGET_LIB)
	@echo ---------
	@echo Compiling $<
	@echo ---------
	$(LINK_APP_CMD) $< $(INC) $(LIB) $(LIBS) -l$(LIB_NAME) -o $(BIN_PATH)/$@

# link obj files into library
$(TARGET_LIB):	$(LIB_OBJ)
	@echo -------
	@echo Linking $(TARGET_LIB)
	@echo -------
	$(LINK_LIB_CMD) -Wl,-soname,$(TARGET_LIB) $(LIB_OBJ) $(LIBS) -o $(TARGET_LIB)
	ln -sf $(TARGET_LIB) $(TARGET_LIB).0
	ln -sf $(TARGET_LIB) $(TARGET_LIB).0.0
	ln -sf $(TARGET_LIB) $(TARGET_LIB).0.0.1

all:	$(TARGET_LIB)

clean:
	@echo ----------------
	@echo Cleaning Library..
	@echo ----------------
	rm -rf $(LIB_PATH)/*
	rm -rf $(OBJ_PATH)/*
	@echo -------------
	@echo Removing Apps
	@echo -------------
	rm -rf $(BIN_PATH)/*