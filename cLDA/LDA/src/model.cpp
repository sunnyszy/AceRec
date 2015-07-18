#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include "constants.h"
#include "strtokenizer.h"
#include "utils.h"
#include "dataset.h"
#include "model.h"

using namespace std;

int rand_multi(int m)
{
	int tmp=(int)(((double)rand() / RAND_MAX) * m);
	while(tmp==m)
	{
		tmp=(int)(((double)rand() / RAND_MAX) * m);
	}
	return tmp;
}
double rand_uni(double m)
{
	double tmp=((double)rand() / RAND_MAX) * m;
	while(tmp==m)
	{
		tmp=((double)rand() / RAND_MAX) * m;
	}
	return tmp;
}

model::~model() {
    if (p) {
	delete p;
    }

    if (ptrndata) {
	delete ptrndata;
    }
    
    if (pnewdata) {
	delete pnewdata;
    }

 
}

void model::set_default_values() {
    wordmapfile = "wordmap.txt";
	referencemapfile = "referencemap.txt";
    trainlogfile = "trainlog.txt";
    tassign_suffix = ".tassign";
    theta_suffix = ".theta";
    phi_suffix = ".phi";
    others_suffix = ".others";
    twords_suffix = ".twords";
	delta_suffix = ".delta";
	lambda_suffix= ".lambda";
    
    dir = "./";
    dfile = "trndocs.dat";
    model_name = "model-final";    
    model_status = MODEL_STATUS_UNKNOWN;
    
    ptrndata = NULL;
    pnewdata = NULL;
    
    M = 0;
    V = 0;
    K = 30;

    niters = 1000;
    liter = 0;
    savestep = 200;    
    twords = 20;
    withrawstrs = 0;
    
	alpha_vector=0.5;
	beta_vector=0.01;
	eta_vector=0.1;
	alpha_lambda_n=0.1;
	alpha_lambda_c=0.1;

    p = NULL;

}

int model::parse_args(int argc, char ** argv) {
    return utils::parse_args(argc, argv, this);
}

int model::init(int argc, char ** argv) {
    // call parse_args
	// this function get the alpha ¡¢beta and other character of the model, as well as the source file the path 
    if (parse_args(argc, argv)) {
	//parse fail
	printf("Error: invalid parameter!\n");
	return 1;
    }
    
	// estimating the model from scratch
	//get all the word from the source file. map it with a number. and initial all the containers. 
	if (init_est()) {
	    return 1;
	}
	
    
    return 0;
}


int model::save_model(string model_name) {
    if (save_model_tassign(dir + model_name + tassign_suffix)) {
	return 1;
    }
    
    if (save_model_others(dir + model_name + others_suffix)) {
	return 1;
    }
    
    if (save_model_theta(dir + model_name + theta_suffix)) {
	return 1;
    }
    
    if (save_model_phi(dir + model_name + phi_suffix)) {
	return 1;
    }

	if (save_model_delta(dir + model_name + delta_suffix)) {
	return 1;
    }

	if (save_model_lambda(dir + model_name + lambda_suffix)) {
	return 1;
    }
    
    if (twords > 0) {
	if (save_model_twords(dir + model_name + twords_suffix)) {
	    return 1;
	}
    }
    
    return 0;
}

int model::save_model_tassign(string filename) {
    int i, j;
    
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to save!\n", filename.c_str());
	return 1;
    }

    // wirte docs with topic assignments for words
    for (i = 0; i < ptrndata->M; i++) {    
	for (j = 0; j < ptrndata->docs[i]->length; j++) {
	    fprintf(fout, "%d:%d ", ptrndata->docs[i]->words[j], z_mn[i][j]);
	}
	fprintf(fout, "\n");
    }

    fclose(fout);
    
    return 0;
}

int model::save_model_theta(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to save!\n", filename.c_str());
	return 1;
    }
    
    for (int i = 0; i < M; i++) {
	for (int j = 0; j < K; j++) {
	    fprintf(fout, "%f ", theta_mk[i][j]);
	}
	fprintf(fout, "\n");
    }
    
    fclose(fout);
    
    return 0;
}

int model::save_model_phi(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to save!\n", filename.c_str());
	return 1;
    }
    
    for (int i = 0; i < K; i++) {
	for (int j = 0; j < V; j++) {
	    fprintf(fout, "%f ", phi_kv[i][j]);
	}
	fprintf(fout, "\n");
    }
    
    fclose(fout);    
    
    return 0;
}


int model::save_model_delta(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to save!\n", filename.c_str());
	return 1;
    }
    
    for (int i = 0; i < M; i++) {
	for (int j = 0; j < rn_m[i]; j++) {
	    fprintf(fout, "%f ", delta_mm[i][j]);
	}
	fprintf(fout, "\n");
    }
    
    fclose(fout);
    
    return 0;
}

int model::save_model_lambda(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to save!\n", filename.c_str());
	return 1;
    }
    
    for (int i = 0; i < M; i++) {
	    fprintf(fout, "%f ", lambda_m[i]);
		fprintf(fout, "\n");
    }
    
    fclose(fout);
    
    return 0;
}

 bool Comp(const pair<int, double> &a,const pair<int, double> &b)
{
    return a.second>b.second;
}


int model::save_model_others(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to save!\n", filename.c_str());
	return 1;
    }

    fprintf(fout, "alpha=%f\n", alpha_vector);
    fprintf(fout, "beta=%f\n", beta_vector);
    fprintf(fout, "ntopics=%d\n", K);
    fprintf(fout, "ndocs=%d\n", M);
    fprintf(fout, "nwords=%d\n", V);
    fprintf(fout, "liter=%d\n", liter);
    
    fclose(fout);    
    
    return 0;
}

int model::save_model_twords(string filename) {
    FILE * fout = fopen(filename.c_str(), "w");
    if (!fout) {
	printf("Cannot open file %s to save!\n", filename.c_str());
	return 1;
    }
    
    if (twords > V) {
	twords = V;
    }
    mapid2word::iterator it;
    
    for (int k = 0; k < K; k++) {
	vector<pair<int, double> > words_probs;
	pair<int, double> word_prob;
	for (int w = 0; w < V; w++) {
	    word_prob.first = w;
	    word_prob.second = phi_kv[k][w];
	    words_probs.push_back(word_prob);
	}
    
        // quick sort to sort word-topic probability
	//utils::quicksort(words_probs, 0, words_probs.size() - 1);
	
	sort(words_probs.begin(),words_probs.end(),Comp);

	fprintf(fout, "Topic %dth:\n", k);
	for (int i = 0; i < twords; i++) {
	    it = id2word.find(words_probs[i].first);
	    if (it != id2word.end()) {
		fprintf(fout, "\t%s   %f\n", (it->second).c_str(), words_probs[i].second);
	    }
	}
    }
    
    fclose(fout);    
    
    return 0;    
}


int model::init_est() {
    int m, n, w, k;

    p = new double[K];
	p_binomial = new double [2];
    // + read training data
    ptrndata = new dataset;
    if (ptrndata->read_trndata(dir + dfile,dir+rfile, dir + wordmapfile,dir + referencemapfile)) {
        printf("Fail to read training data!\n");
        return 1;
    }
		
    // + allocate memory and assign values for variables
    M = ptrndata->M;
    V = ptrndata->V;
    // K: from command line or default value
    // alpha, beta: from command line or default values
    // niters, savestep: from command line or default values



	//int **n0_mk;
	n0_mk=new int*[M];
	for (w = 0; w < M; w++) 
	{
		n0_mk[w] = new int[K];
		for (k = 0; k < K; k++) 
		{
  	    n0_mk[w][k] = 0;
        }
    }
	//int **n_mk;
	n_mk=new int*[M];
	for (w = 0; w < M; w++) 
	{
		n_mk[w] = new int[K];
		for (k = 0; k < K; k++) 
		{
  	    n_mk[w][k] = 0;
        }
    }

	//int **n_kv;
	n_kv=new int*[K];
	for (w = 0; w < K; w++) 
	{
		n_kv[w] = new int[V];
		for (k = 0; k < V; k++) 
		{
  	    n_kv[w][k] = 0;
        }
    }
	//int *nc0_m;
	nc0_m=new int[M];
	for (w = 0; w < M; w++) 
	{
		nc0_m[w] = 0;
    }

	//int *nc_m;
	nc_m=new int[M];
	for (w = 0; w < M; w++) 
	{
		nc_m[w] = 0;
    }
	//int *nc_k;
	nc_k=new int[K];
	for (w = 0; w < K; w++) 
	{
		nc_k[w] = 0;
    }

	
	//int **g_mm;
	g_mm=new int*[M];
	for (w=0;w<M;w++)
	{
		if(ptrndata->reference_matrix[w][0]>0)
		g_mm[w]=new int[ptrndata->reference_matrix[w][0]];
		for (k = 0; k <ptrndata->reference_matrix[w][0] ; k++) {
    	    g_mm[w][k] = ptrndata->reference_matrix[w][k+1];
        }
	}

	//int **r_mm;
	r_mm=new int*[M];
	for (w=0;w<M;w++)
	{
		if(ptrndata->reference_matrix[w][0]>0)
		r_mm[w]=new int[ptrndata->reference_matrix[w][0]];
		for (k = 0; k <ptrndata->reference_matrix[w][0] ; k++) {
    	    r_mm[w][k] = 0;
        }
	}
	//int *rc_m;
	rc_m=new int[M];
	for(w=0;w<M;w++){
	rc_m[w]=0;
	}
	//int *rn_m;
	rn_m=new int[M];
	for (w=0;w<M;w++)
	{
		rn_m[w]=ptrndata->reference_matrix[w][0];
	}


    srand(time(0)); // initialize for rand number generation

	w_mn=new int*[M];


	z_mn=new int*[M];


	c_mn=new int*[M];


	r_mn=new int*[M];

	s_mn=new int*[M];

	


    for (m = 0; m < ptrndata->M; m++) {
	//N is the total word number in document m
	int N = ptrndata->docs[m]->length;
	w_mn[m]=new int[N];
	z_mn[m]=new int[N];
	c_mn[m]=new int[N];
	s_mn[m]=new int[N];
	r_mn[m]=new int[N];
        // initialize for z
        for (n = 0; n < N; n++) {
			w_mn[m][n]=ptrndata->docs[m]->words[n];
			//randomly pick a topic
    	    int topic = rand_multi(K);
    	    z_mn[m][n] = topic;
			s_mn[m][n] =rand_multi(2);
			if(rn_m[m]==0)
				s_mn[m][n]=0;
			if(s_mn[m][n]==0)
			{
				r_mn[m][n]=-1;
				c_mn[m][n]=m;
				n0_mk[m][topic]++;
				nc0_m[m]++;
				n_kv[topic][w_mn[m][n]]++;
				nc_k[topic]++;
			}
			else
			{
				int local_id=rand_multi(rn_m[m]);
				r_mn[m][n]=local_id;
				c_mn[m][n]=g_mm[m][local_id];
				n_mk[c_mn[m][n]][topic]++;
				nc_m[c_mn[m][n]]++;
				r_mm[m][local_id]++;
				rc_m[m]++;
				n_kv[topic][w_mn[m][n]]++;
				nc_k[topic]++;
			}

    	   
        } 
       
    }
    
    theta_mk = new double*[M];
    for (m = 0; m < M; m++) {
        theta_mk[m] = new double[K];
    }
	
    phi_kv = new double*[K];
    for (k = 0; k < K; k++) {
        phi_kv[k] = new double[V];
    }   

	delta_mm =new double *[M];
	for (m=0;m<M;m++)
	{
		if(rn_m[m])
		{
			delta_mm[m]=new double[rn_m[m]];
		}
	}

	lambda_m=new double [M];

    
    return 0;
}



void model::estimate() {
	int s_wave,k_wave,c_wave,r_wave;
    if (twords > 0) {
	// print out top words per topic
	dataset::read_wordmap(dir + wordmapfile, &id2word);
    }

    printf("Sampling %d iterations!\n", niters);

    int last_iter = liter;
	//liter is the current iteration number
    for (liter = last_iter + 1; liter <= niters + last_iter; liter++) {
	printf("Iteration %d ...\n", liter);
	
	for (int m = 0; m < M; m++) {
	    for (int n = 0; n < ptrndata->docs[m]->length; n++) {
		if(s_mn[m][n]==0)
		{
			n0_mk[m][z_mn[m][n]]--;
			nc0_m[m]--;
		}
		else
		{
			n_mk[c_mn[m][n]][z_mn[m][n]]--;
			nc_m[c_mn[m][n]]--;
			r_mm[m][r_mn[m][n]]--;
			rc_m[m]--;
		}
		s_wave=sampling_s_wave(m,n);//sample s;
		s_mn[m][n]=s_wave;
		if(s_wave==0)
		{
			n_kv[z_mn[m][n]][w_mn[m][n]]--;
			nc_k[z_mn[m][n]]--;
			k_wave=sampling_k_wave_first(m,n);//sample k;
			z_mn[m][n]=k_wave;
			n0_mk[m][k_wave]++;
			nc0_m[m]++;
			n_kv[k_wave][w_mn[m][n]]++;
			nc_k[k_wave]++;
		}
		else
		{
			r_wave=sampling_r_wave(m,n);//sample c
			c_wave=g_mm[m][r_wave];
			r_mn[m][n]=r_wave;
			c_mn[m][n]=c_wave;

			r_mm[m][r_wave]++;
			rc_m[m]++;
			n_kv[z_mn[m][n]][w_mn[m][n]]--;
			nc_k[z_mn[m][n]]--;
			k_wave=sampling_k_wave_second(m,n,c_wave);//sample k
			//update
			z_mn[m][n]=k_wave;


			n_mk[c_wave][k_wave]++;
			nc_m[c_wave]++;
			n_kv[k_wave][w_mn[m][n]]++;
			nc_k[k_wave]++;
		}


	    }
	}
	}
  
    printf("Gibbs sampling completed!\n");
    printf("Saving the final model!\n");
    compute_theta();
    compute_phi();
	compute_delta();
	compute_lambda();
    liter--;
    save_model(utils::generate_model_name(-1));
}



int model::sampling_s_wave(int m, int n) {
	
	int topic = z_mn[m][n];
	int tmp_s_wave;
	if(rn_m[m]==0)
		return 0;
	p_binomial[0]=0;
	p_binomial[1]=0;
	//p_binomial 0  1 

	for(int k=0;k<rn_m[m];k++)
	{
		//double tmp_denominator=0;
		//for(int h=0;h<K;h++)
		//{
		//	tmp_denominator+=n_mk[c_mn[m][k]][h]+alpha_vector;
		//}
		p_binomial[1]+=(n_mk[c_mn[m][k]][topic]+alpha_vector)/
						(nc_m[c_mn[m][k]]+K*alpha_vector);
	}
	p_binomial[1]*=(nc_m[m]+alpha_lambda_c);

	double tmp_demominator=0;
	//for(int k=0;k<K;k++)
	//{
	//	tmp_demominator+=n0_mk[m][k]+alpha_vector;
	//}
    p_binomial[0] =(n0_mk[m][topic]+alpha_vector)/
						(nc0_m[m]+K*alpha_vector)*
						(nc0_m[m]+alpha_lambda_n);

	p_binomial[1] += p_binomial[0];
    
    // scaled sample because of unnormalized p[]
    double u = rand_uni (p_binomial[1]);
    
    for (tmp_s_wave = 0; tmp_s_wave < 2; tmp_s_wave++) {
	if (p_binomial[tmp_s_wave] > u) {
	    break;
	}
    }
    
   
    
    return tmp_s_wave;
}

//int model::sampling_k_wave_first(int m, int n) {
//    // remove z_i from the count variables
//    int topic = z_mn[m][n];
//
//  
//    // do multinomial sampling via cumulative method
//    for (int k = 0; k < K; k++) 
//	{
//		p[k]=n_kv[k][w_mn[m][n]]+beta_vector;
//		double tmp_denominator_1=0;
//		for(int t=0;t<V;t++)
//		{
//			tmp_denominator_1+=n_kv[k][t]+beta_vector;
//		}
//		double tmp_denominator_2=0;
//		for(int h=0;h<K;h++)
//		{
//			tmp_denominator_2+=n_mk[m][h]+n0_mk[m][h]+alpha_vector;
//		}
//		p[k]*=(n0_mk[m][k]+n_mk[m][k]+alpha_vector)/tmp_denominator_1/tmp_denominator_2;
//    }
//    // cumulate multinomial parameters
//    for (int k = 1; k < K; k++) {
//	p[k] += p[k - 1];
//    }
//    // scaled sample because of unnormalized p[]
//    double u = rand_uni( p[K - 1]);
//    
//    for (topic = 0; topic < K; topic++) {
//	if (p[topic] > u) {
//	    break;
//	}
//    }
//    
//    
//    return topic;
//}

int model::sampling_k_wave_first(int m, int n) {
    // remove z_i from the count variables
    int topic = z_mn[m][n];

  
    // do multinomial sampling via cumulative method
    for (int k = 0; k < K; k++) 
	{
		//p[k]=n_kv[k][w_mn[m][n]]+beta_vector;
		//double tmp_denominator_1=0;
		//for(int t=0;t<V;t++)
		//{
		//	tmp_denominator_1+=n_kv[k][t]+beta_vector;
		//}
		//double tmp_denominator_2=0;
		//for(int h=0;h<K;h++)
		//{
		//	tmp_denominator_2+=n_mk[m][h]+n0_mk[m][h]+alpha_vector;
		//}
		//p[k]*=(n0_mk[m][k]+n_mk[m][k]+alpha_vector)/tmp_denominator_1/tmp_denominator_2;

		p[k]=(n_kv[k][w_mn[m][n]]+beta_vector)/
			(nc_k[k]+V*beta_vector)*
			(n0_mk[m][k]+n_mk[m][k]+alpha_vector)/
			(nc0_m[m]+nc_m[m]+K*alpha_vector);



    }
    // cumulate multinomial parameters
    for (int k = 1; k < K; k++) {
	p[k] += p[k - 1];
    }
    // scaled sample because of unnormalized p[]
    double u = rand_uni( p[K - 1]);
    
    for (topic = 0; topic < K; topic++) {
	if (p[topic] > u) {
	    break;
	}
    }
    
    
    return topic;
}


int model::sampling_k_wave_second(int m, int n,int c_mn) {
    // remove z_i from the count variables
    int topic = z_mn[m][n];

  
    // do multinomial sampling via cumulative method
    for (int k = 0; k < K; k++) 
	{
		//p[k]=n_kv[k][w_mn[m][n]]+beta_vector;
		//double tmp_denominator_1=0;
		//for(int t=0;t<V;t++)
		//{
		//	tmp_denominator_1+=n_kv[k][t]+beta_vector;
		//}
		//double tmp_denominator_2=0;
		//for(int h=0;h<K;h++)
		//{
		//	tmp_denominator_2+=n_mk[c_mn][h]+n0_mk[c_mn][h]+alpha_vector;
		//}
		//p[k]*=(n0_mk[c_mn][k]+n_mk[c_mn][k]+alpha_vector)/tmp_denominator_1/tmp_denominator_2;
		p[k]=(n_kv[k][w_mn[m][n]]+beta_vector)/
			(nc_k[k]+V*beta_vector)*
			(n0_mk[c_mn][k]+n_mk[c_mn][k]+alpha_vector);
    }
    // cumulate multinomial parameters
    for (int k = 1; k < K; k++) {
	p[k] += p[k - 1];
    }
    // scaled sample because of unnormalized p[]
    double u = rand_uni(p[K - 1]);
    
    for (topic = 0; topic < K; topic++) {
	if (p[topic] > u) {
	    break;
	}
    }
    
    
    return topic;
}

int model::sampling_r_wave(int m, int n) {
    // remove z_i from the count variables
	double *p_reference=new double[rn_m[m]];
	int tmp_r_wave;
    // do multinomial sampling via cumulative method
    for (int r = 0; r < rn_m[m]; r++) 
	{
		//int c_mn=g_mm[m][r];
		//double denominator_1=0;
		//double denominator_2=0;
		//for (int h=0;h<rn_m[m];h++)
		//{
		//	denominator_1+=r_mm[m][r]+eta_vector;
		//}
		//for(int k=0;k<K;k++)
		//{
		//	denominator_2+=n0_mk[c_mn][k]+n_mk[c_mn][k]+alpha_vector;
		//}
		//p_reference[r]=(r_mm[m][c_mn]+eta_vector)*
		//	(n0_mk[c_mn][z_mn[m][n]]+n_mk[c_mn][z_mn[m][n]]+alpha_vector)
		//	/denominator_1/denominator_2;
		p_reference[r]=(r_mm[m][r]+eta_vector)/
						(rc_m[m]+eta_vector*rn_m[m])*
						(n0_mk[r][z_mn[m][n]]+n_mk[r][z_mn[m][n]]+alpha_vector)/
						(nc0_m[r]+nc_m[r]+K*alpha_vector);


    }
    // cumulate multinomial parameters
    for (int r = 1; r < rn_m[m]; r++) {
	p_reference[r] += p_reference[r - 1];
    }
    // scaled sample because of unnormalized p[]
    double u = rand_uni( p_reference[rn_m[m] - 1]);
    

    for (tmp_r_wave = 0; tmp_r_wave < p_reference[rn_m[m]]; tmp_r_wave++) {
	if (p_reference[tmp_r_wave] > u) {
	    break;
	}
    }
    delete p_reference;
    return tmp_r_wave;
}



void model::compute_theta() {
    for (int m = 0; m < M; m++) {
	for (int k = 0; k < K; k++) {
	    /*double denominator=0;
		for(int h=0;h<K;h++)
		{	
			denominator+=n0_mk[m][h]+alpha_vector;
		}*/
		/*theta_mk[m][k]=(n0_mk[m][k]+alpha_vector)/denominator;*/
		theta_mk[m][k]=(n0_mk[m][k]+alpha_vector)/
						(nc_m[m]+alpha_vector);
	}
    }
}

void model::compute_phi() {
    for (int k = 0; k < K; k++) {
	for (int w = 0; w < V; w++) {
	    /*double denominator=0;
		for(int h=0;h<V;h++)
		{
			denominator+=n_kv[k][h]+beta_vector;
		}
		phi_kv[k][w]=(n_kv[k][w]+beta_vector)/denominator;*/
		phi_kv[k][w]=(n_kv[k][w]+beta_vector)/
						(nc_k[k]+beta_vector);
	}
    }
}

void model::compute_delta() {
    for (int m = 0; m < M; m++) 
	{
		if(!rn_m[m])continue;
	for (int w = 0; w < rn_m[m]; w++) {
	    /*double denominator=0;
		for(int h=0;h<rn_m[m];h++)
		{
			denominator+=r_mm[m][h]+eta_vector;
		}
		delta_mm[m][w]=(r_mm[m][w]+eta_vector)/denominator;*/
		delta_mm[m][w]=(r_mm[m][w]+eta_vector)/
			(rc_m[m]+eta_vector);
	}
    }
}

void model::compute_lambda() {
    for (int m = 0; m < M; m++) 
	{
		lambda_m[m]=(nc_m[m]+alpha_lambda_c)/(nc0_m[m]+nc_m[m]+alpha_lambda_c+alpha_lambda_n);
    }
}
//
//
//
//
//void model::compute_newtheta() {
//    for (int m = 0; m < newM; m++) {
//	for (int k = 0; k < K; k++) {
//	    newtheta[m][k] = (newnd[m][k] + alpha) / (newndsum[m] + K * alpha);
//	}
//    }
//}
//
//void model::compute_newphi() {
//    map<int, int>::iterator it;
//    for (int k = 0; k < K; k++) {
//	for (int w = 0; w < newV; w++) {
//	    it = pnewdata->_id2id.find(w);
//	    if (it != pnewdata->_id2id.end()) {
//		newphi[k][w] = (nw[it->second][k] + newnw[w][k] + beta) / (nwsum[k] + newnwsum[k] + V * beta);
//	    }
//	}
//    }
//}

