# Defines the compiler
CXX = g++

# Defines basic flags to build the binary.
CXXFLAGS := -std=c++11 -g -Wall -Wextra -Wshadow
CXXFLAGS := $(CXXFLAGS) -I ../code/

# Define the set of flags for either release or debugging.
# In such small application like the feature extractor, this has 
# a very small impact in the program efficiency.
CXXFLAGS := $(CXXFLAGS) -O0 
#CXXFLAGS = $(CXXFLAGS) -O3

# All dependencies there. As the binary application is so small, we 
# do not check for changes in its dependencies. Instead, we always
# recompile the binary.
all: featExt

featExt: extractFeatures.cpp
	$(CXX) $(CXXFLAGS) -o featExt \
		extractFeatures.cpp \
		../code/common/data.cpp \
		../code/constructor/constructor.cpp

clean:
	rm -f featExt

.PHONY: featExt clean

