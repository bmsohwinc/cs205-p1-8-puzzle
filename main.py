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
counter = itertools.count()

# Classes
class Node:
    def __init__(self):
        self.state = []         # The positions of tiles as a 2D array
        self.g = 0              # G(n) value = cost from initial state based on depth
        self.h = 0              # H(n) value = cost from goal state based on heuristic function
        self.f = 0              # F(n) value = G(n) + H(n) used for sorting the search nodes by A* algorithm
        self.blank_rc = (2, 2)  # storing blank tile index makes it easier
    
    def clone(self):
        new_node = Node()
        new_node.state = [row[:] for row in self.state]
        new_node.g = self.g
        new_node.h = self.h
        new_node.f = self.f
        new_node.blank_rc = self.blank_rc
        return new_node

class Problem:
    def __init__(self, initial_state, operators):
        self.init_state = initial_state
        self.operators = operators
    
    def goal_test(self, node_state: list[list[int]]):
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
    return 0


def compute_h_misplaced_tiles(node: Node):
    count = 0
    for i in range(N):
        for j in range(N):
            if node.state[i][j] != 0 and node.state[i][j] != GOAL_STATE[i][j]:
                count += 1
    return count

def compute_h_manhattan_distance(node: Node):
    distance = 0
    for i in range(N):
        for j in range(N):
            tile = node.state[i][j]
            if tile != 0:
                goal_r = (tile - 1) // N            # We can just compute this instead of searching/hashing the goal state position
                goal_c = (tile - 1) % N
                distance += abs(i - goal_r) + abs(j - goal_c)
    return distance

def queue_function_uniform_cost(nodes: list[tuple[int, int, Node]], new_nodes: list[Node]):
    for node in new_nodes:
        node.h = compute_h_uniform_cost(node)
        node.f = node.g + node.h
        minheap.heappush(nodes, (node.f, next(counter), node))
    return nodes


def queue_function_misplaced_tiles(nodes: list[tuple[int, int, Node]], new_nodes: list[Node]):
    for node in new_nodes:
        node.h = compute_h_misplaced_tiles(node)
        node.f = node.g + node.h
        minheap.heappush(nodes, (node.f, next(counter), node))
    return nodes

def queue_function_manhattan_distance(nodes: list[tuple[int, int, Node]], new_nodes: list[Node]):
    for node in new_nodes:
        node.h = compute_h_manhattan_distance(node)
        node.f = node.g + node.h
        minheap.heappush(nodes, (node.f, next(counter), node))
    return nodes

def expand(node: Node, operators: list[Operator]):
    new_nodes = []
    for operator in operators:
        node_copy = node.clone()            # Create a copy of the current node
        new_node = operator.apply(node_copy)
        if new_node is not None:
            new_node.g = node.g + 1         # g(n) is just the depth of node, so increase by 1 wrt parent node
            new_nodes.append(new_node)      # This just holds expanded nodes. Actual insertion happens in the queue function
    return new_nodes


def make_queue(node: Node):
    return [(node.f, next(counter), node)]

def make_node(state: list[list[int]]):
    node = Node()
    node.state = state
    for i in range(N):
        for j in range(N):
            if state[i][j] == 0:
                node.blank_rc = (i, j)
                break
    return node


def is_empty(queue: list[tuple[int, int, Node]]):
    return len(queue) == 0


def remove_front(queue: list[tuple[int, int, Node]]):
    element = minheap.heappop(queue)
    return element[2]                   # our element is a tuple of (f(n), tie-breaker, node), so, return the node only


def generic_search(problem: Problem, queue_function):
    initial_node = make_node(problem.init_state)
    initial_node.h = compute_h_uniform_cost(initial_node)
    initial_node.f = initial_node.g + initial_node.h
    queue = make_queue(initial_node)

    while True:
        if is_empty(queue):
            return None                # Indicates failure to find a solution
        
        node = remove_front(queue)
        # print("---\n", node, "\n---\n", node.state, "\n---\n")
        
        if problem.goal_test(node.state):
            return node
        
        new_nodes = expand(node, problem.operators)
        queue = queue_function(queue, new_nodes)


def main():
    print("Hello")
    ALL_OPERATORS = [Operator.UP, Operator.DOWN, Operator.LEFT, Operator.RIGHT]
    problem = Problem(TEST_STATE, ALL_OPERATORS)
    
    # solution_node = generic_search(problem, queue_function_uniform_cost)
    # solution_node = generic_search(problem, queue_function_misplaced_tiles)
    solution_node = generic_search(problem, queue_function_manhattan_distance)
    if solution_node is not None:
        print("Solution found with g(n) =", solution_node.g)
    else:
        print("No solution found")


main()
