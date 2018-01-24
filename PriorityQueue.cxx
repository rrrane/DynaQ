#include "PriorityQueue.h"

static std::map<double, struct state_action_pair> pq_map;

void insert(struct state_action_pair p, double priority){
	pq_map.insert(std::pair<double, struct state_action_pair>(priority, p));
}

struct state_action_pair remove(){
	double largest_key = pq_map.rbegin()->first;
	struct state_action_pair p = pq_map.at(largest_key);
	pq_map.erase(largest_key);
	return p;

}

bool is_empty_queue(){
	return pq_map.empty();
}
