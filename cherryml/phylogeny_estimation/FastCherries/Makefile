CXX = g++
CFLAGS = -g -std=c++11 -Wall -O3 -c -finline-functions -funroll-loops 
LFLAGS = -g 

TEST_DIR = ./tests
BUILD_DIR = ./build

# A phony target is one that is not really the name of a file;
# rather it is just a name for a recipe to be executed when you make an explicit request.
# You can read more about them here: https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html
.PHONY : all clean

all : test

# compiles all source files

matrix_exponential.o: build_dir
	$(CXX) $(CFLAGS) matrix_exponential/matrix_exponential.cpp -o $(BUILD_DIR)/matrix_exponential.o
	$(CXX) $(CFLAGS) matrix_exponential/r8lib.cpp -o $(BUILD_DIR)/r8lib.o
	$(CXX) $(CFLAGS) matrix_exponential/c8lib.cpp -o $(BUILD_DIR)/c8lib.o

io_helpers.o: build_dir
	$(CXX) $(CFLAGS) io_helpers.cpp -o $(BUILD_DIR)/io_helpers.o

pairing_algorithms.o: build_dir
	$(CXX) $(CFLAGS) pairing_algorithms.cpp -o $(BUILD_DIR)/pairing_algorithms.o

branch_length_estimation.o: build_dir
	$(CXX) $(CFLAGS) branch_length_estimation.cpp -o $(BUILD_DIR)/branch_length_estimation.o

fast_cherries.o: pairing_algorithms.o branch_length_estimation.o io_helpers.o matrix_exponential.o build_dir
	$(CXX) $(CFLAGS)  fast_cherries.cpp -o $(BUILD_DIR)/fast_cherries.o

# compiles all test files
test_io_helpers.o : build_dir
	$(CXX) $(CFLAGS) $(TEST_DIR)/test_io_helpers.cpp -o $(BUILD_DIR)/test_io_helpers.o

test_pairing_algorithms.o: pairing_algorithms.o build_dir
	$(CXX) $(CFLAGS) $(TEST_DIR)/test_pairing_algorithms.cpp -o $(BUILD_DIR)/test_pairing_algorithms.o

test_branch_length_estimation.o: branch_length_estimation.o build_dir
	$(CXX) $(CFLAGS) $(TEST_DIR)/test_branch_length_estimation.cpp -o $(BUILD_DIR)/test_branch_length_estimation.o

# runs test files
test_io_helpers: test_io_helpers.o io_helpers.o matrix_exponential.o build_dir
	$(CXX) $(LFLAGS) $(BUILD_DIR)/r8lib.o $(BUILD_DIR)/c8lib.o $(BUILD_DIR)/matrix_exponential.o $(BUILD_DIR)/test_io_helpers.o $(BUILD_DIR)/io_helpers.o -o $(BUILD_DIR)/run_test
	$(BUILD_DIR)/run_test

test_pairing_algorithms: test_pairing_algorithms.o 
	$(CXX) $(LFLAGS) $(BUILD_DIR)/test_pairing_algorithms.o $(BUILD_DIR)/pairing_algorithms.o -o $(BUILD_DIR)/run_test
	$(BUILD_DIR)/run_test

test_branch_length_estimation: test_branch_length_estimation.o io_helpers.o matrix_exponential.o
	$(CXX) $(LFLAGS) $(BUILD_DIR)/r8lib.o $(BUILD_DIR)/c8lib.o $(BUILD_DIR)/matrix_exponential.o $(BUILD_DIR)/test_branch_length_estimation.o $(BUILD_DIR)/io_helpers.o $(BUILD_DIR)/branch_length_estimation.o -o $(BUILD_DIR)/run_test
	$(BUILD_DIR)/run_test

build_dir :
	mkdir -p $(BUILD_DIR)

main : fast_cherries.o
	$(CXX) $(LFLAGS) $(BUILD_DIR)/r8lib.o $(BUILD_DIR)/c8lib.o $(BUILD_DIR)/matrix_exponential.o $(BUILD_DIR)/pairing_algorithms.o $(BUILD_DIR)/io_helpers.o $(BUILD_DIR)/fast_cherries.o $(BUILD_DIR)/branch_length_estimation.o -o $(BUILD_DIR)/fast_cherries

clean :
	rm -f $(BUILD_DIR)/*