import sys
import math
import random
import bisect
from sys import maxsize as infinity


class Problem:
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        raise NotImplementedError

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self):
        raise NotImplementedError




class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0  # search depth
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):

        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))

    def solution(self):
        return [node.action for node in self.path()[1:]]

    def solve(self):
        return [node.state for node in self.path()[0:]]

    def path(self):
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        result.reverse()
        return result


    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)




class Queue:

    def __init__(self):
        raise NotImplementedError

    def append(self, item):
        raise NotImplementedError

    def extend(self, items):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __contains__(self, item):
        raise NotImplementedError


class Stack(Queue):


    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop()

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class FIFOQueue(Queue):


    def __init__(self):
        self.data = []

    def append(self, item):
        self.data.append(item)

    def extend(self, items):
        self.data.extend(items)

    def pop(self):
        return self.data.pop(0)

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return item in self.data


class PriorityQueue(Queue):


    def __init__(self, order=min, f=lambda x: x):
        assert order in [min, max]
        self.data = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort_right(self.data, (self.f(item), item))

    def extend(self, items):
        for item in items:
            bisect.insort_right(self.data, (self.f(item), item))

    def pop(self):
        if self.order == min:
            return self.data.pop(0)[1]
        return self.data.pop()[1]

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.data)

    def __getitem__(self, key):
        for _, item in self.data:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.data):
            if item == key:
                self.data.pop(i)




def tree_search(problem, fringe):
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        print(node.state)
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None


def breadth_first_tree_search(problem):
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    return tree_search(problem, Stack())




def graph_search(problem, fringe):
    closed = set()
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        if problem.goal_test(node.state):
            return node
        if node.state not in closed:
            closed.add(node.state)
            fringe.extend(node.expand(problem))
    return None


def breadth_first_graph_search(problem):
    return graph_search(problem, FIFOQueue())


def depth_first_graph_search(problem):
    return graph_search(problem, Stack())


def depth_limited_search(problem, limit=50):
    def recursive_dls(node, problem, limit):
        cutoff_occurred = False
        if problem.goal_test(node.state):
            return node
        elif node.depth == limit:
            return 'cutoff'
        else:
            for successor in node.expand(problem):
                result = recursive_dls(successor, problem, limit)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
        if cutoff_occurred:
            return 'cutoff'
        return None

    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result is not 'cutoff':
            return result


def uniform_cost_search(problem):

    return graph_search(problem, PriorityQueue(min, lambda a: a.path_cost))




def memoize(fn, slot=None):
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}
    return memoized_fn


def best_first_graph_search(problem, f):
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None


def greedy_best_first_graph_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def recursive_best_first_search(problem, h=None):
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result




def distance(a, b):

    return math.hypot((a[0] - b[0]), (a[1] - b[1]))


class Graph:
    def __init__(self, dictionary=None, directed=True):
        self.dict = dictionary or {}
        self.directed = directed
        if not directed:
            self.make_undirected()
        else:
            nodes_no_edges = list({y for x in self.dict.values()
                                   for y in x if y not in self.dict})
            for node in nodes_no_edges:
                self.dict[node] = {}

    def make_undirected(self):
        for a in list(self.dict.keys()):
            for (b, dist) in self.dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, node_a, node_b, distance_val=1):
        self.connect1(node_a, node_b, distance_val)
        if not self.directed:
            self.connect1(node_b, node_a, distance_val)

    def connect1(self, node_a, node_b, distance_val):
        self.dict.setdefault(node_a, {})[node_b] = distance_val

    def get(self, a, b=None):
        links = self.dict.get(a)
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):

        return list(self.dict.keys())


def UndirectedGraph(dictionary=None):
    return Graph(dictionary=dictionary, directed=False)


def RandomGraph(nodes=list(range(10)), min_links=2, width=400, height=300,
                curvature=lambda: random.uniform(1.1, 1.5)):
    g = UndirectedGraph()
    g.locations = {}
    # Build the cities
    for node in nodes:
        g.locations[node] = (random.randrange(width), random.randrange(height))
    # Build roads from each city to at least min_links nearest neighbors.
    for i in range(min_links):
        for node in nodes:
            if len(g.get(node)) < min_links:
                here = g.locations[node]

                def distance_to_node(n):
                    if n is node or g.get(node, n):
                        return math.inf
                    return distance(g.locations[n], here)

                neighbor = nodes.index(min(nodes, key=distance_to_node))
                d = distance(g.locations[neighbor], here) * curvature()
                g.connect(node, neighbor, int(d))
    return g


class GraphProblem(Problem):


    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, state):
        return list(self.graph.get(state).keys())

    def result(self, state, action):
        return action

    def path_cost(self, c, state1, action, state2):
        return c + (self.graph.get(state1, state2) or math.inf)

    def h(self, node):
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return math.inf


obstacles = [(6,1),(6,2),(4,2),(2,3),(1,4),(1,6),(1,8),
             (2,9),(6,4),(5,5),(6,7),(5,7),(4,7),(4,8)]


def move_atom_up(atom1,atom2,atom3):
    while (8 > atom1[0] > 0 and 10 > atom1[1] > 0 and
           (atom1 not in obstacles) and (atom1 not in [atom2,atom3])):
        x = atom1[0]
        x = x - 1
        atom1 = x, atom1[1]
    pos_new = atom1[0] + 1,atom1[1]
    return pos_new

def move_atom_down(atom1,atom2,atom3):
    while (8 > atom1[0] > 0 and 10 > atom1[1] > 0 and
           (atom1 not in obstacles) and (atom1 not in [atom2,atom3])):
        x = atom1[0]
        x = x + 1
        atom1 = x,atom1[1]
    pos_new = atom1[0] - 1,atom1[1]
    return pos_new

def move_atom_left(atom1,atom2,atom3):
    while (8 > atom1[0] > 0 and 10 > atom1[1] > 0 and
           (atom1 not in obstacles) and (atom1 not in [atom2,atom3])):
        y = atom1[1]
        y = y - 1
        atom1 = atom1[0],y
    pos_new = atom1[0],atom1[1] + 1
    return pos_new

def move_atom_right(atom1,atom2,atom3):
    while (8 > atom1[0] > 0 and 10 > atom1[1] > 0 and
           (atom1 not in obstacles) and (atom1 not in [atom2,atom3])):
        y = atom1[1]
        y = y + 1
        atom1 = atom1[0],y
    pos_new = atom1[0],atom1[1] - 1
    return pos_new

class Molecule(Problem):

    def __init__(self,initial):
        self.initial = initial

    def goal_test(self, state):
        h1_x = state[0]
        h1_y = state[1]
        o_x = state[2]
        o_y = state[3]
        h2_x = state[4]
        h2_y = state[5]
        return h1_x == o_x and o_x == h2_x and \
                h1_y == o_y - 1 and o_y == h2_y - 1

    def successor(self, state):
        successors = dict()
        h1 = state[0], state[1]
        o = state[2], state[3]
        h2 = state[4], state[5]

        #Up H1
        h1_new = move_atom_up(h1,o,h2)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['GoreH1'] = state_new

        #Left H1
        h1_new = move_atom_left(h1, o, h2)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['LevoH1'] = state_new

        #Right H1
        h1_new = move_atom_right(h1, o, h2)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['DesnoH1'] = state_new

        #Down H1
        h1_new = move_atom_down(h1, o, h2)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['DoluH1'] = state_new


        #Right O
        o_new = move_atom_right(o,h1,h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            successors['DesnoO'] = state_new

        #Left O
        o_new = move_atom_left(o,h1,h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            successors['LevoO'] = state_new

        #Up O
        o_new = move_atom_up(o, h1, h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            successors['GoreO'] = state_new

        #Down O
        o_new = move_atom_down(o, h1, h2)
        state_new = h1 + o_new + h2
        if state_new != state:
            successors['DoluO'] = state_new


        # Up H2
        h1_new = move_atom_up(h2, o, h1)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['GoreH2'] = state_new

        # Left H2
        h1_new = move_atom_left(h2, o, h1)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['LevoH2'] = state_new

        # Right H2
        h1_new = move_atom_right(h2, o, h1)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['DesnoH2'] = state_new

        # Down H1
        h1_new = move_atom_down(h2, o, h1)
        state_new = h1_new + o + h2
        if state_new != state:
            successors['DoluH2'] = state_new

        return successors

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]


if __name__ == '__main__':

    h1_atom_row = int(input())
    h1_atom_column = int(input())
    o_atom_row = int(input())
    o_atom_column = int(input())
    h2_atom_row = int(input())
    h2_atom_column = int(input())

    molecule = Molecule((h1_atom_row,h1_atom_column , o_atom_row,o_atom_column,
                         h2_atom_row,h2_atom_column))

    answer = breadth_first_graph_search(molecule)
    print(answer.solution())
