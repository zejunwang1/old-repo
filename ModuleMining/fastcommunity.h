
#if !defined(fastcommunity_INCLUDED)
#define fastcommunity_INCLUDED

#include <iostream>
#include <fstream>
#include <string>
#include "stdlib.h"
#include "time.h"
#include "math.h"

#include "maxheap.h"
#include "vector.h"

using namespace std;

// ------------------------------------------------------------------------------------
// Edge object - defined by a pair of vertex indices and *edge pointer to next in linked-list
#if !defined(edgem_INCLUDED)
#define edgem_INCLUDED
class edgem {
public:
	int     so;					// originating node
	int     si;					// terminating node
	edgem    *next;					// pointer for linked list of edges

	edgem();						// default constructor
	~edgem();						// default destructor
};
edgem::edgem()  { so = 0; si = 0; next = NULL; }
edgem::~edgem() {}
#endif

// ------------------------------------------------------------------------------------
// Nodenub object - defined by a *node pointer and *node pointer
struct nodenub {
	tuple	*heap_ptr;			// pointer to node(max,i,j) in max-heap of row maxes
	vektor    *v;					// pointer stored vector for (i,j)
};

// ------------------------------------------------------------------------------------
// tuple object - defined by an real value and (row,col) indices
#if !defined(TUPLE_INCLUDED)
#define TUPLE_INCLUDED
struct tuple {
	double    m;					// stored value
	int		i;					// row index
	int		j;					// column index
	int		k;					// heap index
};
#endif

// ordered pair structures (handy in the program)
struct apair { int x; int y; };
#if !defined(DPAIR_INCLUDED)
#define DPAIR_INCLUDED
class dpair {
public:
	int x; double y; dpair *next;
	dpair(); ~dpair();
};
dpair::dpair()  { x = 0; y = 0.0; next = NULL; }
dpair::~dpair() {}
#endif

// ------------------------------------------------------------------------------------
// List object - simple linked list of integers
class listm {
public:
	int		index;				// node index
	listm		*next;				// pointer to next element in linked list
	listm();   ~listm();
};
listm::listm()  { index= 0; next = NULL; }
listm::~listm() {}

// ------------------------------------------------------------------------------------
// Community stub object - stub for a community list
class stub {
public:
	bool		valid;				// is this community valid?
	int		size;				// size of community
	listm		*members;				// pointer to list of community members
	listm		*last;				// pointer to end of list
	stub();   ~stub();
};
stub::stub()  { valid = false; size = 0; members = NULL; last = NULL; }
stub::~stub() {
	listm *current;
	if (members != NULL) {
		current = members;
		while (current != NULL) { members = current->next; delete current; current = members; }
	}
}


// PROGRAM PARAMETERS -----------------------------------------------------------------

struct netparameters {
	int			n;				// number of nodes in network
	int			m;				// number of edges in network
	int			maxid;			// maximum node id
	int			minid;			// minimum node id
}; netparameters    gparm;

struct groupstats {
	int			numgroups;		// number of groups
	double		meansize;			// mean size of groups
	int			maxsize;			// size of largest group
	int			minsize;			// size of smallest group
	double		*sizehist;		// distribution of sizes
}; groupstats		gstats;

struct iocommunity {
	short int		textFlag;			// 0: no console output
								// 1: writes file outputs
	bool			suppFlag;			// T: no support(t) file
								// F: yes support(t) file
	short int		fileFlag;			//
	string		f_joins_temp;			// (file) community hierarchy
	string		f_support_temp;		// (file) dQ support as a function of time
	string		f_net_temp;			// (file) .wpairs file for .cutstep network
	string		f_group_temp;			// (file) .list of indices in communities at .cutstep
	string		f_gstats_temp;			// (file) distribution of community sizes at .cutstep

	int			timer;			// timer for displaying progress reports
	bool			timerFlag;		// flag for setting timer
}; iocommunity	ioc;

// ------------------------------------------------------------------------------------
// ----------------------------------- GLOBAL VARIABLES -------------------------------

edgem		*e;				// initial adjacency matrix (sparse)
edgem		*elist;			// list of edges for building adjacency matrix
nodenub   *dq;				// dQ matrix
maxheap   *h;				// heap of values from max_i{dQ_ij}
double    *Q;				// Q(t)
dpair     Qmax;			// maximum Q value and the corresponding time t
double    *a;				// A_i
apair	*joins;			// list of joins
stub		*c;				// link-lists for communities

enum {NONE};

int    supportTot;
double supportAve;



// ------------------------------------------------------------------------------------

#endif
