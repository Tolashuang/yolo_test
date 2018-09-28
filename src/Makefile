CXX := g++

#CFLAGS :=  -g -Wall -fPIC
CFLAGS :=  -Ofast -Wall -fPIC -fopenmp
CFLAGS+= -DGPU -DCUDNN
INCLUDE_PATH := -I. -I ../include -I /usr/local/cuda-9.0/include -I/usr/local/include
LIBS = -L ../lib -L /usr/local/cuda-9.0/lib64 -ldarknet -lcuda -lcudart -lcublas -lcurand -lpthread -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lcudnn

LIB_OUT = ../lib/libyoloPed.so
APP_OUT = ../bin/uDetect
CPPSRCS = $(wildcard *.cpp)
CPPOBJS := $(patsubst %.cpp, ./%.o, $(CPPSRCS))
./%.o: %.cpp
	        $(CXX) $(CFLAGS) $(INCLUDE_PATH) -c $< -o $@

all:$(CPPOBJS)
	        $(CXX) $(CFLAGS) -o $(APP_OUT) $(CPPOBJS) $(LIBS)
	        #$(CXX) -shared $(CFLAGS) -o $(LIB_OUT) $(CPPOBJS) $(LIBS)

clean:
	        rm -f *.o $(LIB_OUT)
