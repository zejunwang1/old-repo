
#if !defined(predictHRG_INCLUDE)
#define predictHRG_INCLUDE

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include "stdlib.h"
#include "time.h"
#include "math.h"

#include "MersenneTwister.h"
#include "dendro_pr.h"
#include "graph_pr.h"
#include "graph_simp.h"
#include "rbtree.h"

using namespace std;

// ****************************************************************************************************

#if !defined(pblock_INCLUDED)
#define pblock_INCLUDED
struct pblock { double L; int i; int j; };
#endif

// ******** Function Prototypes ***************************************************************************


string	num2str(const unsigned int);
void		QsortMain(pblock*, int, int);
int		QsortPartition(pblock*, int, int, int);

// ******** Structures and Constants **********************************************************************

struct iopredict {
	int			n;				// number vertices in input graph
	int			m;				// number of edges in input graph

	int			timer;			// timer for reading input
	bool			flag_timer;		// (flag) for timer
	bool			flag_compact;		// (flag) T: compress number of trials

	string		start_time;		// time simulation was started
};

// ******** Global Variables ******************************************************************************

iopredict	iop;				// program parameters
rbtree		namesLUT;				// look-up table; translates .graph indices to .pairs indices
rbtree		namesLUTr;			// look-up table; translates .pairs indices to .graph indices
dendro*		d;					// inferred dendrograph data structure
simpleGraph*	g;					// base graph read from file
int			num_samples;			// number of samples to take for predictions
int			num_bins;				// number of bins in edge statistics histogram
MTRand		mtr;					// Mersenne Twister random number generator instance
pblock*		br_list;				// store adjacencies and their averaged likelihoods
int			br_length;			// length of br_list
int			mk;					// number of missing edges (n \choose 2) - m
double			bestL;


// ******** Function Definitions **************************************************************************

string num2str(const unsigned int input) {
	// input must be a positive integer
	unsigned int temp = input;
	string str  = "";
	if (input == 0) { str = "0"; } else {
		while (temp != 0) {
			str  = char(int(temp % 10)+48) + str;
			temp = (unsigned int)temp/10;
		}
	}
	return str;
}

// ********************************************************************************************************

void QsortMain (pblock* array, int left, int right) {
	if (right > left) {
//		cout << "left right = " << left << " " << right << "\n";
		int pivot = left;
//		cout << "QsortPartition\n";
		int part  = QsortPartition(array, left, right, pivot);
//		cout << "QsortMain_left\n";
		QsortMain(array, left,   part-1);
//		cout << "QsortMain_right\n";
		QsortMain(array, part+1, right  );
	}
	return;
}

int QsortPartition (pblock* array, int left, int right, int index) {
	if (left < 0) { cout << "ERROR: left = " << left << " < 0\n"; }
	if (right >= br_length) { cout << "ERROR: right = " << right << " >= 0\n"; }
	if (index >= br_length or index < 0) { cout << "ERROR: index = " << index << "\n"; }
	// 27 May 2012: There is a bug in my implementation of partition() that produces a
	//              seg fault under some circumstances. The work around I have put in
	//              place is to write out the *unsorted* candidate list first, before
	//              sorting. If quicksort runs properly, this file will be overwritten
	//              with the sorted version; if it fails, at least you have the unsorted
	//              version.

	pblock p_value, temp;
	p_value.L = array[index].L;
	p_value.i = array[index].i;
	p_value.j = array[index].j;

	// swap(array[p_value], array[right])
	temp.L   	   = array[right].L;			temp.i         = array[right].i;			temp.j         = array[right].j;
	array[right].L = array[index].L;			array[right].i = array[index].i;			array[right].j = array[index].j;
	array[index].L = temp.L;					array[index].i = temp.i;					array[index].j = temp.j;

	int stored = left;
	for (int i=left; i<right; i++) {
		if (array[i].L <= p_value.L) {
			// swap(array[stored], array[i])
			temp.L     = array[i].L;			temp.i	 = array[i].i;				temp.j	 = array[i].j;
			array[i].L = array[stored].L;		array[i].i = array[stored].i;		array[i].j = array[stored].j;
			array[stored].L = temp.L;			array[stored].i = temp.i;			array[stored].j = temp.j;
			stored++;
			if (stored >= br_length or stored < 0) { cout << "ERROR: stored = " << stored << "\n"; }
		}
	}
	// swap(array[right], array[stored])
	temp.L		 = array[stored].L;				temp.i		 = array[stored].i;			temp.j		 = array[stored].j;
	array[stored].L = array[right].L;			array[stored].i = array[right].i;		array[stored].j = array[right].j;
	array[right].L  = temp.L;					array[right].i  = temp.i;				array[right].j  = temp.j;

	return stored;
}

// ********************************************************************************************************
// ********************************************************************************************************

#endif
