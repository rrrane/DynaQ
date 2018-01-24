
#include <stdio.h>
#include <string.h>


#include "rlglue/Agent_common.h" /* Required for RL-Glue */
#include "PriorityQueue.h"
#include "utils.h"
#include "common.h"


static gsl_vector* local_action;
static action_t* this_action;
static gsl_vector* last_observation;

static double Q[GRIDWIDTH][GRIDHEIGHT][N_ACTIONS];
static double model_reward[GRIDWIDTH][GRIDHEIGHT][N_ACTIONS];
static int model_states[GRIDWIDTH][GRIDHEIGHT][N_ACTIONS][2];

static const double epsilon = 0.1;
static const double disc_factor = 0.95;
static const int n = 5;

static double alpha = 0.5;
static double theta = 0.1;

void agent_init()
{

	//Allocate Memory

	local_action = gsl_vector_calloc(1);
	this_action = local_action;
	last_observation = gsl_vector_calloc(2);
	
	for(int i = 0; i < GRIDWIDTH; i++){
		for(int j = 0; j < GRIDHEIGHT; j++){
			for(int k = 0; k < N_ACTIONS; k++){
				Q[i][j][k] = 0;
				model_reward[i][j][k] = 0.0;
				model_states[i][j][k][0] = 0;
				model_states[i][j][k][1] = 0;
			}

		}
	}
	
}

const action_t *agent_start(const observation_t *this_observation) {

  	//Read State

	int x = (int)gsl_vector_get(this_observation, 0);
	int y = (int)gsl_vector_get(this_observation, 1);
	

	//Get optimal and suboptimal actions
	int opt_actions[N_ACTIONS] = {0, 0, 0, 0};
	int sub_opt_actions[N_ACTIONS] = {0, 0, 0, 0};

	int maxcount = 0;
	int subcount = 0;
	
	double maxval = Q[x][y][0];
	for(int i = 1; i < N_ACTIONS; i++){
		if(maxval < Q[x][y][i])
			maxval = Q[x][y][i];
	}

	for(int i = 0; i < N_ACTIONS; i++){
		if(Q[x][y][i] == maxval)
			opt_actions[maxcount++] = i;
		else
			sub_opt_actions[subcount++] = i;
	}
	

	//Randomly select optimal action
	int maxindex = randInRange(maxcount);
	int opt_act = opt_actions[(maxindex < N_ACTIONS) ? maxindex : N_ACTIONS - 1];

	//Randomly select suboptimal action
	int subindex = randInRange(subcount);
	int sub_opt_act = sub_opt_actions[(subindex < N_ACTIONS) ? subindex : N_ACTIONS - 1];
	
	//Select action based on epsilon-greedy policy
	double p = rand_un();

	int act = randInRange(N_ACTIONS);

	act = (act < N_ACTIONS) ? act : N_ACTIONS - 1;
	
	if(p >= epsilon || subcount == 0)
		act = opt_act;
	else
		act = sub_opt_act;
	
	//Save action in local_action
	gsl_vector_set(local_action, 0, act);

	//Save last observation
	gsl_vector_memcpy(last_observation, this_observation);

  	return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {


  	//Read State S'
	int x_prime = (int)gsl_vector_get(this_observation, 0);
	int y_prime = (int)gsl_vector_get(this_observation, 1);
	
	//Read Previous State S
	int x = (int)gsl_vector_get(last_observation, 0);
	int y = (int)gsl_vector_get(last_observation, 1);

	//Get last action A
	int act = (int)gsl_vector_get(local_action, 0);
	
	//Update model M(S,A) <- R, S'
	model_reward[x][y][act] = reward;
	model_states[x][y][act][0] = x_prime;
	model_states[x][y][act][1] = y_prime;

	//Update priority P
	
	double maxval = Q[x_prime][y_prime][0];
	
	for(int i = 1; i < N_ACTIONS; i++){
		if(maxval < Q[x_prime][y_prime][i])
			maxval = Q[x_prime][y_prime][i];

	}

	double P = reward + disc_factor * maxval - Q[x][y][act];
	P = (P >= 0) ? P : -P;

	if(P > theta)
	{
		struct state_action_pair sa_pair;
		sa_pair.x = x;
		sa_pair.y = y;
		sa_pair.A = act;

		insert(sa_pair, P);
	}

	int count = 0;

	while(count < n && !is_empty_queue()){
		struct state_action_pair sa_pair;
		sa_pair = remove();
		
		int x_update = sa_pair.x;
		int y_update = sa_pair.y;
		int act_update = sa_pair.A;
		
		double R = model_reward[x_update][y_update][act_update];
		int x_update_prime = model_states[x_update][y_update][act_update][0];
		int y_update_prime = model_states[x_update][y_update][act_update][1];
		
		double maxval_update_prime = Q[x_update_prime][y_update_prime][0];
		double maxval_update = Q[x_update][y_update][0];

		for(int i = 1; i < N_ACTIONS; i++){
			if(maxval_update_prime < Q[x_update_prime][y_update_prime][i])
				maxval_update_prime = Q[x_update_prime][y_update_prime][i];
			if(maxval_update < Q[x_update][y_update][i])
				maxval_update = Q[x_update][y_update][i];

		}

		Q[x_update][y_update][act_update] = Q[x_update][y_update][act_update] + alpha * (R + disc_factor * maxval_update_prime - Q[x_update][y_update][act_update]);

		//Write code to update Q values.
		struct state_action *predicted_state_actions;
		int pred_st_act_count = 0;
		
		for(int i = 0; i < GRIDWIDTH; i++){
			for(int j = 0; j < GRIDHEIGHT; j++){
				for(int k = 0; k < N_ACTIONS; k++){
					if(model_states[i][j][k][0] == x_update && model_states[i][j][k][1] == y_update){
						int x_bar = i;
						int y_bar = j;
						int A_bar = k;

						double R_bar = model_reward[x_bar][y_bar][A_bar];
						double P_bar = R_bar + disc_factor * maxval_update - Q[x_bar][y_bar][A_bar];
						P_bar = (P_bar < 0) ? -P_bar : P_bar;

						if(P_bar > theta)
						{
							struct state_action_pair sa_pair_bar;
							sa_pair_bar.x = x_bar;
							sa_pair_bar.y = y_bar;
							sa_pair_bar.A = A_bar;

							insert(sa_pair_bar, P_bar);
						}
					}
				}
			}
		}
		

		count++;
	}



	//Get set of optimal and suboptimal actions

	int opt_actions[N_ACTIONS] = {0, 0, 0, 0};
	int sub_opt_actions[N_ACTIONS] = {0, 0, 0, 0};


	int maxcount = 0;
	int subcount = 0;

	for(int i = 0; i < N_ACTIONS; i++){
		if(Q[x_prime][y_prime][i] >= maxval)
			opt_actions[maxcount++] = i;
		else
			sub_opt_actions[subcount++] = i;
	}
	
	//Randomly select optimal action
	int maxindex = randInRange(maxcount);
	int opt_act = opt_actions[(maxindex < N_ACTIONS) ? maxindex : N_ACTIONS - 1];

	//Randomly select suboptimal action
	int subindex = randInRange(subcount);
	int sub_opt_act = sub_opt_actions[(subindex < N_ACTIONS) ? subindex : N_ACTIONS - 1];
	

	//Select action based on epsilon-greedy policy
	double p = rand_un();

	int act_prime = randInRange(N_ACTIONS);

	act_prime = (act_prime < N_ACTIONS) ? act_prime : N_ACTIONS - 1;
	
	if(p >= epsilon || subcount == 0)
		act_prime = opt_act;
	else
		act_prime = sub_opt_act;
	
	
	//Save action in local_action
	gsl_vector_set(local_action, 0, act_prime);

	//Save last observation
	gsl_vector_memcpy(last_observation, this_observation);

  	return this_action;
}


void agent_end(double reward) {

	//Read Previous State S
	int x = (int)gsl_vector_get(last_observation, 0);
	int y = (int)gsl_vector_get(last_observation, 1);

	//Get last action A
	int act = (int)gsl_vector_get(local_action, 0);
	int x_prime = x;
	int y_prime = y;
	
	if(reward == 1.0){
		x_prime = GOAL_X;
		y_prime = GOAL_Y;
	}
		
	//Update model M(S,A) <- R, S'
	model_reward[x][y][act] = reward;
	model_states[x][y][act][0] = x_prime;
	model_states[x][y][act][1] = y_prime;

	//Update priority P
	
	double maxval = Q[x_prime][y_prime][0];
	
	for(int i = 1; i < N_ACTIONS; i++){
		if(maxval < Q[x_prime][y_prime][i])
			maxval = Q[x_prime][y_prime][i];

	}

	double P = reward + disc_factor * maxval - Q[x][y][act];
	P = (P >= 0) ? P : -P;

	if(P > theta)
	{
		struct state_action_pair sa_pair;
		sa_pair.x = x;
		sa_pair.y = y;
		sa_pair.A = act;

		insert(sa_pair, P);
	}

	int count = 0;

	while(count < n && !is_empty_queue()){
		struct state_action_pair sa_pair;
		sa_pair = remove();
		
		int x_update = sa_pair.x;
		int y_update = sa_pair.y;
		int act_update = sa_pair.A;
		
		double R = model_reward[x_update][y_update][act_update];
		int x_update_prime = model_states[x_update][y_update][act_update][0];
		int y_update_prime = model_states[x_update][y_update][act_update][1];
		
		double maxval_update_prime = Q[x_update_prime][y_update_prime][0];
		double maxval_update = Q[x_update][y_update][0];

		for(int i = 1; i < N_ACTIONS; i++){
			if(maxval_update_prime < Q[x_update_prime][y_update_prime][i])
				maxval_update_prime = Q[x_update_prime][y_update_prime][i];
			if(maxval_update < Q[x_update][y_update][i])
				maxval_update = Q[x_update][y_update][i];

		}

		Q[x_update][y_update][act_update] = Q[x_update][y_update][act_update] + alpha * (R + disc_factor * maxval_update_prime - Q[x_update][y_update][act_update]);

		//Write code to update Q values.
		struct state_action *predicted_state_actions;
		int pred_st_act_count = 0;
		
		
		for(int i = 0; i < GRIDWIDTH; i++){
			for(int j = 0; j < GRIDHEIGHT; j++){
				for(int k = 0; k < N_ACTIONS; k++){
					if(model_states[i][j][k][0] == x_update && model_states[i][j][k][1] == y_update){
						int x_bar = i;
						int y_bar = j;
						int A_bar = k;

						double R_bar = model_reward[x_bar][y_bar][A_bar];
						double P_bar = R_bar + disc_factor * maxval_update - Q[x_bar][y_bar][A_bar];
						P_bar = (P_bar < 0) ? -P_bar : P_bar;

						if(P_bar > theta)
						{
							struct state_action_pair sa_pair_bar;
							sa_pair_bar.x = x_bar;
							sa_pair_bar.y = y_bar;
							sa_pair_bar.A = A_bar;

							insert(sa_pair_bar, P_bar);
						}
					}
				}
			}
		}
		
		count++;
	}

}

void agent_cleanup() {
  /* clean up mememory */
  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);
}

const char* agent_message(const char* inMessage) {
  	/* might be useful to get information from the agent */

#ifdef ONE
	if(strcmp(inMessage,"1")==0){
	  	theta = 0.0;
  		return "Nothing";
  	}else if(strcmp(inMessage, "2")==0){
		theta = 0.1;
		return "Nothing";
	}else if(strcmp(inMessage, "3")==0){
		theta = 0.2;
		return "Nothing";
	}else if(strcmp(inMessage, "4")==0){
		theta = 0.3;
		return "Nothing";
	}else if(strcmp(inMessage, "5")==0){
		theta = 0.4;
		return "Nothing";
	}else if(strcmp(inMessage, "6")==0){
		theta = 0.5;
		return "Nothing";
	}

#endif
#ifdef TWO
  	if(strcmp(inMessage,"1")==0){
	  	alpha = 0.025;
  		return "Nothing";
  	}else if(strcmp(inMessage, "2")==0){
		alpha = 0.05;
		return "Nothing";
	}else if(strcmp(inMessage, "3")==0){
		alpha = 0.1;
		return "Nothing";
	}else if(strcmp(inMessage, "4")==0){
		alpha = 0.2;
		return "Nothing";
	}else if(strcmp(inMessage, "5")==0){
		alpha = 0.4;
		return "Nothing";
	}else if(strcmp(inMessage, "6")==0){
		alpha = 0.5;
		return "Nothing";
	}else if(strcmp(inMessage, "7")==0){
		alpha = 0.8;
		return "Nothing";
	}
#endif
  	/* else */
  	return "I don't know how to respond to your message";
}
