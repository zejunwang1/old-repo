#
# Copyright (c) 2018-present
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

CXX = c++
CXXFLAGS = -pthread -std=c++0x -march=native
OBJS = args.o dictionary.o matrix.o vector.o model.o utils.o algebra.o cooccurrence.o lsa.o shuffle.o hypertext.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops
opt: hypertext

debug: CXXFLAGS += -g -O0 -fno-inline
debug: hypertext

args.o: src/args.cpp src/args.h
	$(CXX) $(CXXFLAGS) -c src/args.cpp

dictionary.o: src/dictionary.cpp src/dictionary.h src/args.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cpp

matrix.o: src/matrix.cpp src/matrix.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cpp

vector.o: src/vector.cpp src/vector.h src/utils.h
	$(CXX) $(CXXFLAGS) -c src/vector.cpp

model.o: src/model.cpp src/model.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/model.cpp

utils.o: src/utils.cpp src/utils.h
	$(CXX) $(CXXFLAGS) -c src/utils.cpp

algebra.o: src/algebra.cpp src/algebra.h 
	$(CXX) $(CXXFLAGS) -c src/algebra.cpp

cooccurrence.o: src/cooccurrence.cpp src/cooccurrence.h src/args.h src/dictionary.h
	$(CXX) $(CXXFLAGS) -c src/cooccurrence.cpp

lsa.o: src/lsa.cpp src/args.h src/algebra.h src/fastsvd.h
	$(CXX) $(CXXFLAGS) -c src/lsa.cpp

shuffle.o: src/shuffle.cpp src/shuffle.h src/args.h
	$(CXX) $(CXXFLAGS) -c src/shuffle.cpp

hypertext.o: src/hypertext.cpp src/*.h
	$(CXX) $(CXXFLAGS) -c src/hypertext.cpp

hypertext: $(OBJS) src/hypertext.cpp
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cpp -o hypertext

clean:
	rm -rf *.o hypertext
