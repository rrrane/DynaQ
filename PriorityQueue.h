#include <map>

struct state_action_pair{
	int x;
	int y;
	int A;
};

void insert(struct state_action_pair, double);

struct state_action_pair remove();

bool is_empty_queue();
