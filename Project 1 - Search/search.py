# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):

    stack = util.Stack()

    #create tuple of (state, action) so no need to create double stack
    stack.push((problem.getStartState(), []))

    #Set to track visited states
    visited = set()

    while not stack.isEmpty():
        # Pop state to get new state and actions
        currentState, actions = stack.pop()

        # Return actions, if the currentstate is the goal
        if problem.isGoalState(currentState):
            return actions

        # In case a state is yet to be visited
        if currentState not in visited:
            visited.add(currentState)

            # in each successor state, push successors to the stack
            for successor, action, _ in problem.getSuccessors(currentState):
                newActions = actions + [action]
                stack.push((successor, newActions))

    # return empty list if no actions were found to the goal
    return []
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
  


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    queue.push((problem.getStartState(), []))

    visited = set()

    while not queue.isEmpty():
        currentState, actions = queue.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited:
            visited.add(currentState)

            # in each successor state, push successors to the queue
            for successor, action, _ in problem.getSuccessors(currentState):
                newActions = actions + [action]
                queue.push((successor, newActions))
    return []


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    heap = util.PriorityQueue()
    heap.push((problem.getStartState(), []), 0)
    
    visited = set()

    while not heap.isEmpty():
        # Pop a state and the actions that led to it.
        currentState, actions = heap.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited:
            visited.add(currentState)

            # in each successor state, push successors to the priority queue
            for successor, action, cost in problem.getSuccessors(currentState):
                newActions = actions + [action]
                new_cost = problem.getCostOfActions(newActions)
                if successor not in visited:
                    heap.push((successor, newActions), new_cost)
    return []
    


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    heap = util.PriorityQueue()
    startState = problem.getStartState()
    heap.push((startState, []), heuristic(startState, problem)) #inital cost will be heuristic of the initial node.

    visited = set()

    while not heap.isEmpty():
        currentState, actions = heap.pop()

        if problem.isGoalState(currentState):
            return actions

        if currentState not in visited:
            visited.add(currentState)

            # in each successor state, push successors to the priority queue
            for successor, action, cost in problem.getSuccessors(currentState):
                newActions = actions + [action]
                if successor not in visited:
                    new_cost = problem.getCostOfActions(newActions) #g(n)
                    h_cost = heuristic(successor, problem)  # h(n)
                    total_cost = new_cost + h_cost #total cost of a* is the sum of cost and heuristic
                    heap.push((successor, newActions), total_cost)
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
