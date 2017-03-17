#VPATH = liblinear:liblinear/blas

CXXFLAGS = $(shell pkg-config --cflags opencv)
LDLIBS = $(shell pkg-config --libs opencv)

VLROOT = /home/wangs/vlfeat-0.9.20

CC = g++

objects = main.cpp utils.cpp dsift.cpp dictionary.cpp llc.cpp train.cpp predict.cpp \
          liblinear/linear.cpp liblinear/tron.cpp liblinear/blas/daxpy.c liblinear/blas/ddot.c \
          liblinear/blas/dnrm2.c liblinear/blas/dscal.c

classify : $(objects)
	$(CC) -o classify $(objects) $(CXXFLAGS) $(LDLIBS) -I$(VLROOT) -L$(VLROOT)/bin/glnxa64/ -lvl

clean:
	/bin/rm -f classify *.o

clean-all: clean
	/bin/rm -f *~ 

