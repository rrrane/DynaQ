
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "utils.h"
#include "rlglue/RL_glue.h"
#include "common.h"

#define	N_EPISODES_PER_RUN	50
#define	N_RUNS			30
#define	N_STEPS_PER_EPISODE	0
#define N_ALPHAS		7 /* Number of different values of alpha to test */
#define N_THETAS		6 /* Number of different values of theta to test */

void saveResults(double* data, int dataSize, const char* filename);


int main(int argc, char *argv[]) {

	srand(time(NULL));

#ifdef ONE

	double n_steps_to_goal[N_EPISODES_PER_RUN];

		
	const char *msg[N_THETAS] = {"1", "2", "3", "4", "5", "6"};

	const char *name[N_THETAS] = {"0", "0_1", "0_2", "0_3", "0_4", "0_5"};

	for(int i = 0; i < N_THETAS; i++){

		for(int j = 0; j < N_EPISODES_PER_RUN; j++)
			n_steps_to_goal[j] = 0.0;


		for(int j = 0; j < N_RUNS; j++){
			RL_init();
			
			RL_agent_message(msg[i]);
					
			for(int k = 0; k < N_EPISODES_PER_RUN; k++){
				RL_episode(N_STEPS_PER_EPISODE);
				
				n_steps_to_goal[k] += RL_num_steps();
			}

			printf(".");
		}
		printf("\nDone\n");

		for(int j = 0; j < N_EPISODES_PER_RUN; j++){
			n_steps_to_goal[j] /= (double)N_RUNS;
		}

  		/* Save data to a file */
		char filename[100];
		strcpy(filename, "OUT_");
		strcat(filename, name[i]);
		strcat(filename, ".dat");
		printf("Saving file %s ...\n", filename);
		saveResults(n_steps_to_goal, N_EPISODES_PER_RUN, (const char *)filename);
	}
#endif

#ifdef TWO
	
	double n_steps_to_goal[N_ALPHAS];

	for(int i = 0; i < N_ALPHAS; i++)
		n_steps_to_goal[i] = 0.0;
	
	const char *msg[N_ALPHAS] = {"1", "2", "3", "4", "5", "6", "7"};

	for(int i = 0; i < N_ALPHAS; i++){
		for(int j = 0; j < N_RUNS; j++){
			RL_init();
			
			RL_agent_message(msg[i]);

			for(int k = 0; k < N_EPISODES_PER_RUN; k++){
				RL_episode(N_STEPS_PER_EPISODE);
				
				n_steps_to_goal[i] += RL_num_steps();
			}

			printf(".");
		}
		printf("\nDone\n");

	}

	for(int i = 0; i < N_ALPHAS; i++){
		n_steps_to_goal[i] /= (double)N_RUNS;
	}

  
  	/* Save data to a file */
	saveResults(n_steps_to_goal, N_ALPHAS, "OUT_ALPHA.dat");

#endif
  	return 0;
}


void saveResults(double* data, int dataSize, const char* filename) {
  FILE *dataFile;
  int i;
  dataFile = fopen(filename, "w");
  for(i = 0; i < dataSize; i++){
    fprintf(dataFile, "%lf\n", data[i]);
  }
  fclose(dataFile);
}
