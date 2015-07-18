

#include "model.h"
#include <stdio.h>
using namespace std;



int main(int argc, char ** argv) {
    model lda;

    lda.init(argc, argv);

	lda.estimate();

    
    return 0;
}


