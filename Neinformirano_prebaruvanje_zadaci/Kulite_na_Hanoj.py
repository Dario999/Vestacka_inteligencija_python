import sys
import math
import random
import bisect
from sys import maxsize as infinity

"""
Дефинирање на класа за структурата на проблемот кој ќе го решаваме со пребарување.
Класата Problem е апстрактна класа од која правиме наследување за дефинирање на основните
карактеристики на секој проблем што сакаме да го решиме
"""


class Problem:
    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def successor(self, state):
        """За дадена состојба, врати речник од парови {акција : состојба}
       достапни од оваа состојба. Ако има многу следбеници, употребете
       итератор кој би ги генерирал следбениците еден по еден, наместо да
       ги генерирате сите одеднаш.

       :param state: дадена состојба
       :return:  речник од парови {акција : состојба} достапни од оваа
                 состојба
       :rtype: dict
       """
        raise NotImplementedError

    def actions(self, state):
        """За дадена состојба state, врати листа од сите акции што може да
       се применат над таа состојба

       :param state: дадена состојба
       :return: листа на акции
       :rtype: list
       """
        raise NotImplementedError

    def result(self, state, action):
        """За дадена состојба state и акција action, врати ја состојбата
       што се добива со примена на акцијата над состојбата

       :param state: дадена состојба
       :param action: дадена акција
       :return: резултантна состојба
       """
        raise NotImplementedError

    def goal_test(self, state):
        """Врати True ако state е целна состојба. Даденава имплементација
       на методот директно ја споредува state со self.goal, како што е
       специфицирана во конструкторот. Имплементирајте го овој метод ако
       проверката со една целна состојба self.goal не е доволна.

       :param state: дадена состојба
       :return: дали дадената состојба е целна состојба
       :rtype: bool
       """
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Врати ја цената на решавачкиот пат кој пристигнува во состојбата
       state2 од состојбата state1 преку акцијата action, претпоставувајќи
       дека цената на патот до состојбата state1 е c. Ако проблемот е таков
       што патот не е важен, оваа функција ќе ја разгледува само состојбата
       state2. Ако патот е важен, ќе ја разгледува цената c и можеби и
       state1 и action. Даденава имплементација му доделува цена 1 на секој
       чекор од патот.

       :param c: цена на патот до состојбата state1
       :param state1: дадена моментална состојба
       :param action: акција која треба да се изврши
       :param state2: состојба во која треба да се стигне
       :return: цена на патот по извршување на акцијата
       :rtype: float
       """
        return c + 1

    def value(self):
        """За проблеми на оптимизација, секоја состојба си има вредност.
       Hill-climbing и сличните алгоритми се обидуваат да ја максимизираат
       оваа вредност.

       :return: вредност на состојба
       :rtype: float
       """
        raise NotImplementedError


"""
Дефинирање на класата за структурата на јазел од пребарување.
Класата Node не се наследува
"""


class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Креирај јазол од пребарувачкото дрво, добиен од parent со примена
       на акцијата action

       :param state: моментална состојба (current state)
       :param parent: родителска состојба (parent state)
       :param action: акција (action)
       :param path_cost: цена на патот (path cost)
       """
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
        """Излистај ги јазлите достапни во еден чекор од овој јазол.

       :param problem: даден проблем
       :return: листа на достапни јазли во еден чекор
       :rtype: list(Node)
       """

        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """Дете јазел

       :param problem: даден проблем
       :param action: дадена акција
       :return: достапен јазел според дадената акција
       :rtype: Node
       """
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next_state))

    def solution(self):
        """Врати ја секвенцата од акции за да се стигне од коренот до овој јазол.

       :return: секвенцата од акции
       :rtype: list
       """
        return [node.action for node in self.path()[1:]]

    def solve(self):
        """Врати ја секвенцата од состојби за да се стигне од коренот до овој јазол.

       :return: листа од состојби
       :rtype: list
       """
        return [node.state for node in self.path()[0:]]

    def path(self):
        """Врати ја листата од јазли што го формираат патот од коренот до овој јазол.

       :return: листа од јазли од патот
       :rtype: list(Node)
       """
        x, result = self, []
        while x:
            result.append(x)
            x = x.parent
        result.reverse()
        return result

    """Сакаме редицата од јазли кај breadth_first_search или
   astar_search да не содржи состојби - дупликати, па јазлите што
   содржат иста состојба ги третираме како исти. [Проблем: ова може
   да не биде пожелно во други ситуации.]"""

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


"""
Дефинирање на помошни структури за чување на листата на генерирани, но непроверени јазли
"""


class Queue:
    """Queue е апстрактна класа / интерфејс. Постојат 3 типа:
        Stack(): Last In First Out Queue (стек).
        FIFOQueue(): First In First Out Queue (редица).
        PriorityQueue(order, f): Queue во сортиран редослед (подразбирливо,од најмалиот кон
                                најголемиот јазол).
    """

    def __init__(self):
        raise NotImplementedError

    def append(self, item):
        """Додади го елементот item во редицата

       :param item: даден елемент
       :return: None
       """
        raise NotImplementedError

    def extend(self, items):
        """Додади ги елементите items во редицата

       :param items: дадени елементи
       :return: None
       """
        raise NotImplementedError

    def pop(self):
        """Врати го првиот елемент од редицата

       :return: прв елемент
       """
        raise NotImplementedError

    def __len__(self):
        """Врати го бројот на елементи во редицата

       :return: број на елементи во редицата
       :rtype: int
       """
        raise NotImplementedError

    def __contains__(self, item):
        """Проверка дали редицата го содржи елементот item

       :param item: даден елемент
       :return: дали queue го содржи item
       :rtype: bool
       """
        raise NotImplementedError


class Stack(Queue):
    """Last-In-First-Out Queue."""

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
    """First-In-First-Out Queue."""

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
    """Редица во која прво се враќа минималниот (или максималниот) елемент
   (како што е определено со f и order). Оваа структура се користи кај
   информирано пребарување"""
    """"""

    def __init__(self, order=min, f=lambda x: x):
        """
       :param order: функција за подредување, ако order е min, се враќа елементот
                     со минимална f(x); ако order е max, тогаш се враќа елементот
                     со максимална f(x).
       :param f: функција f(x)
       """
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


"""
Неинформирано пребарување во рамки на дрво.
Во рамки на дрвото не разрешуваме јамки.
"""


def tree_search(problem, fringe):
    """ Пребарувај низ следбениците на даден проблем за да најдеш цел.

   :param problem: даден проблем
   :param fringe:  празна редица (queue)
   :return: Node
   """
    fringe.append(Node(problem.initial))
    while fringe:
        node = fringe.pop()
        print(node.state)
        if problem.goal_test(node.state):
            return node
        fringe.extend(node.expand(problem))
    return None


def breadth_first_tree_search(problem):
    """Експандирај го прво најплиткиот јазол во пребарувачкото дрво.

   :param problem: даден проблем
   :return: Node
   """
    return tree_search(problem, FIFOQueue())


def depth_first_tree_search(problem):
    """Експандирај го прво најдлабокиот јазол во пребарувачкото дрво.

   :param problem:даден проблем
   :return: Node
   """
    return tree_search(problem, Stack())


"""
Неинформирано пребарување во рамки на граф
Основната разлика е во тоа што овде не дозволуваме јамки,
т.е. повторување на состојби
"""


def graph_search(problem, fringe):
    """Пребарувај низ следбениците на даден проблем за да најдеш цел.
    Ако до дадена состојба стигнат два пата, употреби го најдобриот пат.

   :param problem: даден проблем
   :param fringe: празна редица (queue)
   :return: Node
   """
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
    """Експандирај го прво најплиткиот јазол во пребарувачкиот граф.

   :param problem: даден проблем
   :return: Node
   """
    return graph_search(problem, FIFOQueue())


def depth_first_graph_search(problem):
    """Експандирај го прво најдлабокиот јазол во пребарувачкиот граф.

   :param problem: даден проблем
   :return: Node
   """
    return graph_search(problem, Stack())


def depth_limited_search(problem, limit=50):
    def recursive_dls(node, problem, limit):
        """Помошна функција за depth limited"""
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
    """Експандирај го прво јазолот со најниска цена во пребарувачкиот граф."""
    return graph_search(problem, PriorityQueue(min, lambda a: a.path_cost))


"""
Информирано пребарување во рамки на граф
"""


def memoize(fn, slot=None):
    """ Запамети ја пресметаната вредност за која била листа од
   аргументи. Ако е специфициран slot, зачувај го резултатот во
   тој slot на првиот аргумент. Ако slot е None, зачувај ги
   резултатите во речник.

   :param fn: зададена функција
   :param slot: име на атрибут во кој се чуваат резултатите од функцијата
   :return: функција со модификација за зачувување на резултатите
   """
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
    """Пребарувај низ следбениците на даден проблем за да најдеш цел. Користи
    функција за евалуација за да се одлучи кој е сосед најмногу ветува и
    потоа да се истражи. Ако до дадена состојба стигнат два пата, употреби
    го најдобриот пат.

   :param problem: даден проблем
   :param f: дадена функција за евристика
   :return: Node or None
   """
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
    """ Greedy best-first пребарување се остварува ако се специфицира дека f(n) = h(n).

   :param problem: даден проблем
   :param h: дадена функција за евристика
   :return: Node or None
   """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, h)


def astar_search(problem, h=None):
    """ A* пребарување е best-first graph пребарување каде f(n) = g(n) + h(n).

   :param problem: даден проблем
   :param h: дадена функција за евристика
   :return: Node or None
   """
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


def recursive_best_first_search(problem, h=None):
    """Recursive best first search - ја ограничува рекурзијата
   преку следење на f-вредноста на најдобриот алтернативен пат
   од било кој јазел предок (еден чекор гледање нанапред).

   :param problem: даден проблем
   :param h: дадена функција за евристика
   :return: Node or None
   """
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0  # (втората вредност е неважна)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Подреди ги според најниската f вредност
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


"""
Пребарување низ проблем дефиниран како конечен граф
"""


def distance(a, b):
    """Растојание помеѓу две (x, y) точки."""
    return math.hypot((a[0] - b[0]), (a[1] - b[1]))


class Graph:
    def __init__(self, dictionary=None, directed=True):
        self.dict = dictionary or {}
        self.directed = directed
        if not directed:
            self.make_undirected()
        else:
            # додади празен речник за линковите на оние јазли кои немаат
            # насочени врски и не се дадени како клучеви во речникот
            nodes_no_edges = list({y for x in self.dict.values()
                                   for y in x if y not in self.dict})
            for node in nodes_no_edges:
                self.dict[node] = {}

    def make_undirected(self):
        """Ориентираниот граф претвори го во неориентиран со додавање
       на симетричните ребра."""
        for a in list(self.dict.keys()):
            for (b, dist) in self.dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, node_a, node_b, distance_val=1):
        """Додади ребро од A до B со дадено растојание, а додади го и
       обратното ребро (од B до A) ако графот е неориентиран."""
        self.connect1(node_a, node_b, distance_val)
        if not self.directed:
            self.connect1(node_b, node_a, distance_val)

    def connect1(self, node_a, node_b, distance_val):
        """Додади ребро од A до B со дадено растојание, но само во
       едната насока."""
        self.dict.setdefault(node_a, {})[node_b] = distance_val

    def get(self, a, b=None):
        """Врати растојание придружено на ребро или пак врати речник
       чии елементи се од обликот {јазол : растојание}.
       .get(a,b) го враќа растојанието или пак враќа None
       .get(a) враќа речник со елементи од обликот {јазол : растојание},
           кој може да биде и празен – {}."""
        links = self.dict.get(a)
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Врати листа од јазлите во графот."""
        return list(self.dict.keys())


def UndirectedGraph(dictionary=None):
    """Изгради Graph во кој што секое ребро (вклучувајќи ги и идните
   ребра) е двонасочно."""
    return Graph(dictionary=dictionary, directed=False)


def RandomGraph(nodes=list(range(10)), min_links=2, width=400, height=300,
                curvature=lambda: random.uniform(1.1, 1.5)):
    """Construct a random graph, with the specified nodes, and random links.
   The nodes are laid out randomly on a (width x height) rectangle.
   Then each node is connected to the min_links nearest neighbors.
   Because inverse links are added, some nodes will have more connections.
   The distance between nodes is the hypotenuse times curvature(),
   where curvature() defaults to a random number between 1.1 and 1.5."""
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
    """Проблем на пребарување на граф од еден до друг јазол."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, state):
        """Акциите кај јазол во граф се едноставно - неговите соседи."""
        return list(self.graph.get(state).keys())

    def result(self, state, action):
        """Резултат на одењето кон сосед е самиот тој сосед."""
        return action

    def path_cost(self, c, state1, action, state2):
        return c + (self.graph.get(state1, state2) or math.inf)

    def h(self, node):
        """Функцијата h е праволиниско растојание од состојбата зачувана
       во тековниот јазол до целната состојба."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return math.inf


class Kuli(Problem):
    def __init__(self, initial, goal):
        super().__init__(initial, goal)

    def actions(self, state):
        return self.successor(state).keys()

    def result(self, state, action):
        possible = self.successor(state)
        return possible[action]

    def goal_test(self, state):
        return state == self.goal

    def successor(self, state):
        kast = list(state)

        successors = dict()

        # print("STATE: " + str(state) + " VS GOAL: " + str(self.goal))

        for i in range(len(kast)):
            for j in range(len(kast)):
                if i == j:
                    continue
                newL = kast

                stolbFrom = list(newL[i])
                stolbTo = list(newL[j])

                # we are looking for a plate on an empty stick
                if (len(stolbFrom) <= 0):
                    continue

                # here is our val if it isnt empty, this should always be the lowest
                from_top = stolbFrom[len(stolbFrom) - 1]

                # black magic to find the other, if it's empty we go with 0 otherwise we find it!
                to_top = -1
                if (len(stolbTo) > 0):
                    to_top = stolbTo[len(stolbTo) - 1]

                if to_top != -1:
                    if from_top > to_top:
                        continue
                stolbTo.append(stolbFrom.pop(len(stolbFrom) - 1))

                bnl = list()
                for x in range(len(kast)):
                    if x == i:
                        bnl.append(tuple(stolbFrom))
                    elif x == j:
                        bnl.append(tuple(stolbTo))
                    else:
                        bnl.append(newL[x])

                successors['MOVE TOP BLOCK FROM PILLAR ' + str(i + 1) + ' TO PILLAR ' + str(j + 1)] = tuple(bnl)

        # print(str(state) + "  VS  " + str(self.goal))
        return successors


if __name__ == "__main__":
    s = input()
    initial_towers = tuple([tuple(map(int, x.split(','))) if x != '' else () for x in s.split(';')])
    s = input()
    goal_towers = tuple([tuple(map(int, x.split(','))) if x != '' else () for x in s.split(';')])
    print(goal_towers)

    kuli = Kuli(initial_towers, goal_towers)
    answ = breadth_first_graph_search(kuli)
    print("Number of steps required: " + str(len(answ.solution())))
    print("Solution:")
    for x in answ.solution():
        print(x)