

#include <stdio.h>
#include <string>
#include <map>
#include <stdlib.h>
#include "strtokenizer.h"
#include "utils.h"
#include "model.h"

using namespace std;

int utils::parse_args(int argc, char ** argv, model * pmodel) {
    string dir = "";
    string model_name = "";
    string dfile = "";
	string rfile = "";
    double alpha = -1.0;
    double beta = -1.0;
    int K = 0;
    int niters = 0;
    int savestep = 0;
    int twords = 0;
    int withrawdata = 0;

    int i = 0; 
	dfile=argv[1];
	rfile=argv[2];

    
	pmodel->model_status = 1;
	
	    // default value for alpha
	
	pmodel->twords = 20;
	//dfile is the source file 
	pmodel->dfile = dfile;
	pmodel->rfile = rfile;
	string::size_type idx = dfile.find_last_of("/");

	//the "/" doesn't exist 
	//this part get the dir and dfile name
	if (idx == string::npos) {
	    pmodel->dir = "./";
	} else {
	    pmodel->dir = dfile.substr(0, idx + 1);
	    pmodel->dfile = dfile.substr(idx + 1, dfile.size() - pmodel->dir.size());
		pmodel->rfile = rfile.substr(idx + 1, rfile.size() - pmodel->dir.size());
	    printf("dir = %s\n", pmodel->dir.c_str());
	    printf("dfile = %s\n", pmodel->dfile.c_str());
		printf("rfile = %s\n", pmodel->rfile.c_str());
	}
    
  
    
    return 0;
}

int utils::read_and_parse(string filename, model * pmodel) {
    // open file <model>.others to read:
    // alpha=?
    // beta=?
    // ntopics=?
    // ndocs=?
    // nwords=?
    // citer=? // current iteration (when the model was saved)
    
    FILE * fin = fopen(filename.c_str(), "r");
    if (!fin) {
	printf("Cannot open file: %s\n", filename.c_str());
	return 1;
    }
    
    char buff[BUFF_SIZE_SHORT];
    string line;
    
    while (fgets(buff, BUFF_SIZE_SHORT - 1, fin)) {
	line = buff;
	strtokenizer strtok(line, "= \t\r\n");
	int count = strtok.count_tokens();
	
	if (count != 2) {
	    // invalid, ignore this line
	    continue;
	}

	string optstr = strtok.token(0);
	string optval = strtok.token(1);
	
	 if (optstr == "ntopics") {
	    pmodel->K = atoi(optval.c_str());
	
	} else if (optstr == "ndocs") {	   
	    pmodel->M = atoi(optval.c_str());
	 
	} else if (optstr == "nwords") {
	    pmodel->V = atoi(optval.c_str());
	
	} else if (optstr == "liter") {
	    pmodel->liter = atoi(optval.c_str());
	
	} else {
	    // any more?
	}
    }
    
    fclose(fin);
    
    return 0;
}

string utils::generate_model_name(int iter) {
    string model_name = "model-";

    //char buff[BUFF_SIZE_SHORT];
    
 //   if (0 <= iter && iter < 10) {
	//sprintf(buff, "0000%d", iter);
 //   } else if (10 <= iter && iter < 100) {
	//sprintf(buff, "000%d", iter);
 //   } else if (100 <= iter && iter < 1000) {
	//sprintf(buff, "00%d", iter);
 //   } else if (1000 <= iter && iter < 10000) {
	//sprintf(buff, "0%d", iter);
 //   } else {
	//sprintf(buff, "%d", iter);
 //   }
    
 //   if (iter >= 0) {
	//model_name += buff;
 //   } else {
	model_name += "final";
    //}
    
    return model_name;
}

void utils::sort(vector<double> & probs, vector<int> & words) {
    for (int i = 0; i < probs.size() - 1; i++) {
	for (int j = i + 1; j < probs.size(); j++) {
	    if (probs[i] < probs[j]) {
		double tempprob = probs[i];
		int tempword = words[i];
		probs[i] = probs[j];
		words[i] = words[j];
		probs[j] = tempprob;
		words[j] = tempword;
	    }
	}
    }
}

void utils::quicksort(vector<pair<int, double> > & vect, int left, int right) {
    int l_hold, r_hold;
    pair<int, double> pivot;
    
    l_hold = left;
    r_hold = right;    
    int pivotidx = left;
    pivot = vect[pivotidx];

    while (left < right) {
	while (vect[right].second <= pivot.second && left < right) {
	    right--;
	}
	if (left != right) {
	    vect[left] = vect[right];
	    left++;
	}
	while (vect[left].second >= pivot.second && left < right) {
	    left++;
	}
	if (left != right) {
	    vect[right] = vect[left];
	    right--;
	}
    }

    vect[left] = pivot;
    pivotidx = left;
    left = l_hold;
    right = r_hold;
    
    if (left < pivotidx) {
	quicksort(vect, left, pivotidx - 1);
    }
    if (right > pivotidx) {
	quicksort(vect, pivotidx + 1, right);
    }    
}

