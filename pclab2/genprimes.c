#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<string.h>
int main(int argc, char* argv[]){
    double tstart=0.0, ttaken;
    int thread_count=strtol(argv[2], NULL, 10);
    int N=strtol(argv[1], NULL, 10);
    FILE * fp;
    char output[100]="";
    int Max_Factor=(N+1)/2;
    int Composite[N-1];
    memset(Composite, 0, (N-1)*sizeof(int));
    //stores at position i the indicator whether (i+2) is a compositenumber or not
    tstart=omp_get_wtime();

    //parallel part
    int pos, i, j;
#   pragma omp parallel for num_threads(thread_count) default(none) private(pos, i, j) shared(Max_Factor, N, Composite) schedule(static,1)
    for(i=2;i<=Max_Factor;i++){
        if(Composite[i-2]==1){
            //do nothing
        }else{
	    for(j=2*i;j<=N;j=j+i){
	        pos=j-2;
                Composite[pos]=1;
	    }
        }
    }
    ttaken=omp_get_wtime()-tstart;
    printf("Time taken for the main part: %f\n", ttaken);

    //output part
    sprintf(output, "%d.txt", N);
    fp=fopen(output, "w");
    if(!fp){
	printf("Cannot create the file %s\n", output);
	exit(1);
    }
    int count=0;
    int pre_prime_pos=0;
    for(i=0;i<N-1;i++){
	if(Composite[i]==0){
	    count++;
	    fprintf(fp, "%d, %d, %d\n", count, i+2, i-pre_prime_pos);
	    pre_prime_pos=i;
	}
    }
    
    fclose(fp);
    exit(0);
}
