from enum import Enum
import heapq as minheap
import itertools

# Global values
N = 3
GOAL_STATE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0],
]
TEST_STATE = [
    [1, 2, 3],
    [0, 4, 5],
    [7, 8, 6],
]
counter = itertools.count()         # For tie-breaking when comparing nodes with the same f(n) value
repeated_states = set()             # To avoid re-exploring repeated states
chosen_heuristic_function = None

# Classes
class Node:
    def __init__(self):
        self.state = []         # The positions of tiles as a 2D array
        self.g = 0              # G(n) value = cost from initial state based on depth
        self.h = 0              # H(n) value = cost from goal state based on heuristic function
        self.f = 0              # F(n) value = G(n) + H(n) used for sorting the search nodes by A* algorithm
        self.blank_rc = (2, 2)  # storing blank tile index makes it easier
    
    def clone(self):
        """
            Util to create a deep copy of the node. 
            Useful when applying operators to create new nodes without modifying the original node.
        """
        new_node = Node()
        new_node.state = [row[:] for row in self.state]
        new_node.g = self.g
        new_node.h = self.h
        new_node.f = self.f
        new_node.blank_rc = self.blank_rc
        return new_node

    def hash(self):
        """
            Simple hash key generator for the node state.
            Example: For state [[1, 2, 3], [0, 4, 5], [7, 8, 6]], the hash will be "1 2 3,0 4 5,7 8 6"
        """
        return ",".join([" ".join(map(str, row)) for row in self.state])

class Problem:
    def __init__(self, initial_state, operators):
        self.init_state = initial_state
        self.operators = operators
    
    def goal_test(self, node_state: list[list[int]]):
        """Checks if the given node state matches the goal state."""
        # A single mismatch concludes that node is not yet the goal state.
        for i in range(N):
            for j in range(N):
                if node_state[i][j] != GOAL_STATE[i][j]:
                    return False
        return True

class Operator(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    def apply(self, node: Node):
        """Applies the operator to the given node and returns a new node with the updated state."""
        if node is None:
            return None
        
        r, c = node.blank_rc
        new_blank_rc = (r, c)
        if self.name == 'UP':
            # If top row, can't move blank tile up
            if r == 0:
                return None
            # Swap upper cell with this blank cell
            node.state[r - 1][c], node.state[r][c] = node.state[r][c], node.state[r - 1][c]
            new_blank_rc = (r - 1, c)
        elif self.name == 'DOWN':
            # If bottom row, can't move blank tile down
            if r == N - 1:
                return None
            # Swap lower cell with this blank cell
            node.state[r + 1][c], node.state[r][c] = node.state[r][c], node.state[r + 1][c]
            new_blank_rc = (r + 1, c)
        elif self.name == 'LEFT':
            # If leftmost column, can't move blank tile left
            if c == 0:
                return None
            # Swap left cell with this blank cell
            node.state[r][c - 1], node.state[r][c] = node.state[r][c], node.state[r][c - 1]
            new_blank_rc = (r, c - 1)
        elif self.name == 'RIGHT':
            # If rightmost column, can't move blank tile right
            if c == N - 1:
                return None
            # Swap right cell with this blank cell
            node.state[r][c + 1], node.state[r][c] = node.state[r][c], node.state[r][c + 1]
            new_blank_rc = (r, c + 1)
        else:
            return None
        node.blank_rc = new_blank_rc
        return node


def compute_h_uniform_cost(node: Node):
    """In Uniform Cost Search, heuristic cost is 0."""
    return 0


def compute_h_misplaced_tiles(node: Node):
    """We just check the number of misplaced tiles compared to the goal state, except the blank tile."""
    count = 0
    for i in range(N):
        for j in range(N):
            if node.state[i][j] != 0 and node.state[i][j] != GOAL_STATE[i][j]:
                count += 1
    return count

def compute_h_manhattan_distance(node: Node):
    """We compute and accumulate the Manhattan distance of each tile from its goal position, except the blank tile."""
    distance = 0
    for i in range(N):
        for j in range(N):
            tile = node.state[i][j]
            if tile != 0:
                goal_r = (tile - 1) // N            # We just compute this instead of searching/hashing the goal state position
                goal_c = (tile - 1) % N
                distance += abs(i - goal_r) + abs(j - goal_c)
    return distance

def queue_function(nodes: list[tuple[int, int, Node]], new_nodes: list[Node]):
    """
        For each new node explored, we compute the heuristic cost h(n),
          and insert it into the priority queue (min-heap) based on the least f(n) value.
        
        f(n) is sum of g(n) and h(n), where,
          g(n) is the cost from the initial state to the current node (which is just the depth of the node in our case), and
          h(n) is the heuristic cost from the current node to the goal state.
    """
    for node in new_nodes:
        node.h = chosen_heuristic_function(node)    # Compute heuristic cost as per chosen heuristic function
        node.f = node.g + node.h
        minheap.heappush(nodes, (node.f, next(counter), node))  # The ordering is based on the least f(n) value among the nodes
    return nodes

def expand(node: Node, operators: list[Operator]):
    """Expands the given node by applying the given operators and returns a list of new nodes."""
    new_nodes = []
    for operator in operators:
        node_copy = node.clone()                # Create a copy of the current node
        new_node = operator.apply(node_copy)    # Apply operator

        if new_node is None:
            # Invalid operation in this state, so skip
            continue

        # Repetition check
        new_node_hash = new_node.hash()
        if new_node_hash in repeated_states:
            # We repeat a state, so skip
            continue
        
        repeated_states.add(new_node_hash)  # Add the new node's hash to the set
        new_node.g = node.g + 1             # g(n) is just the depth of node, so increase by 1 wrt parent node
        new_nodes.append(new_node)          # This just holds expanded nodes. Actual insertion happens in the queue function
    return new_nodes


def make_queue(node: Node):
    """Initializes the queue with the initial node."""
    return [(node.f, next(counter), node)]

def make_node(state: list[list[int]]):
    """Creates a new node with the given state. Also stores blank tile position."""
    node = Node()
    node.state = state
    for i in range(N):
        for j in range(N):
            if state[i][j] == 0:
                node.blank_rc = (i, j)
                break
    return node


def is_empty(queue: list[tuple[int, int, Node]]):
    """Simple util to check if the queue is empty."""
    return len(queue) == 0


def remove_front(queue: list[tuple[int, int, Node]]):
    """Pops and returns node with the least f(n) value from the queue."""
    element = minheap.heappop(queue)
    return element[2]                   # our element is a tuple of (f(n), tie-breaker, node), so, return the node only


# ############################# THE GENERIC SEARCH ALGORITHM #############################

def generic_search(problem: Problem, queue_function):
    """Implements the generic search algorithm for the 8-puzzle problem."""
    initial_node = make_node(problem.init_state)
    initial_node.h = chosen_heuristic_function(initial_node)
    initial_node.f = initial_node.g + initial_node.h
    
    queue = make_queue(initial_node)

    while True:
        if is_empty(queue):
            return None                 # Indicates failure to find a solution
        
        node = remove_front(queue)
        
        if problem.goal_test(node.state):
            return node                 # Goal state found, return the solution node
        
        new_nodes = expand(node, problem.operators)
        queue = queue_function(queue, new_nodes)

# ########################################################################################

def main():
    ALL_OPERATORS = [Operator.UP, Operator.DOWN, Operator.LEFT, Operator.RIGHT]
    global chosen_heuristic_function

    test_mode = int(input("Please enter 1 for manual testing and 2 for running all test cases: "))
    if test_mode == 1:
         # Custom test case
        print("Enter the initial state of the 8-puzzle (use 0 for blank tile and whitespace for separators):")
        custom_state = []
        for i in range(N):
            row = list(map(int, input(f"Row {i + 1}: ").split()))
            custom_state.append(row)
        problem = Problem(custom_state, ALL_OPERATORS)
    elif test_mode == 2:
        print("Running all test cases...")
        problem = Problem(TEST_STATE, ALL_OPERATORS)
    else:
        print("Invalid input. Exiting.")
        return

    search_type = int(input("Please enter 1 for Uniform Cost Search, 2 for A* with Misplaced Tiles heuristic, and 3 for A* with Manhattan Distance heuristic: "))
    if search_type == 1:
        print("Running Uniform Cost Search...")
        chosen_heuristic_function = compute_h_uniform_cost
    elif search_type == 2:
        print("Running A* Search with Misplaced Tiles heuristic...")
        chosen_heuristic_function = compute_h_misplaced_tiles
    elif search_type == 3:
        print("Running A* Search with Manhattan Distance heuristic...")
        chosen_heuristic_function = compute_h_manhattan_distance
    else:
        print("Invalid input. Exiting.")
        return

    # Run the search algorithm and print the solution
    solution_node = generic_search(problem, queue_function)
    if solution_node is not None:
        print("Solution found with g(n) =", solution_node.g)
    else:
        print("No solution found")


main()
