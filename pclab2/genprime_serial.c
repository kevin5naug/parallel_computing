#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int main(int argc, char* argv[]){
    int thread_count=strtol(argv[1], NULL, 10);
    int N=strtol(argv[2], NULL, 10);
    FILE * fp;
    char output[100]="";
    int Max_Factor=(N+1)/2;
    int Composite[N-1]={0};
    //stores at position i the indicator whether (i+2) is a compositenumber or not
    
    //parallel part
    int end=-1;
    int pos;
    for(int i=2;i<=Max_Factor;i++){
	for(int j=i+1;j<=N;j++){
	    pos=j-2;
	    if(j%i==0){
		Composite[pos]=1;
	    }
	}
    }

    //output part
    sprintf(output, "%d.txt", N);
    fp=fopen(output, "w");
    if(!fp){
	printf("Cannot create the file %s\n", output);
	exit(1);
    }
    int count=0;
    int pre_prime_pos=0;
    for(int i=0;i<N-1;i++){
	if(Composite[i]==0){
	    count++;
	    printf(fp, "%d, %d, %d\n", count, i+2, i-pre_prime_pos);
	    pre_prime_pos=i
	}
    }
    
    fclose(fp);
    exit(0);
}
