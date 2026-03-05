from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createUSTCMap,
    createHefeiMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch

# BEGIN_YOUR_CODE (You may add some codes here to assist your coding below if you want, but don't worry if you deviate from this.)

# END_YOUR_CODE

# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Check out the docstring for `State` in `util.py` for more details and code.

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(self.startLocation, None)  # 初始状态为起点
    
        raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.endTag in self.cityMap.tags[state.location]  # 检查当前节点是否为目标标签
    
        raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        successors = []
        current = state.location
        for neighbor, cost in self.cityMap.distances[current].items():
            # 将动作名称设为邻居的位置标签
            successors.append((neighbor, State(neighbor, None), cost))
        return successors
    
        raise NotImplementedError("Override me")
        # END_YOUR_CODE


########################################################################################
# Problem 1b: Custom -- Plan a Route through USTC


def getUSTCShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of USTC, specifying your own
    `startLocation`/`endTag`.

    Run `python mapUtil.py > readableUSTCMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/USTC-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "coffee", "food")
    """
    cityMap = createUSTCMap()

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    startLocation = locationFromTag(makeTag("landmark", "8348"), cityMap)  # 西区1958
    endTag = makeTag("landmark", "3rd_teaching_building")  # 三教标签
    # END_YOUR_CODE
    return ShortestPathProblem(startLocation, endTag, cityMap)
    raise NotImplementedError("Override me")



########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(self.startLocation, frozenset())        
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        # 需满足终点标签且所有途径点被覆盖
        return (self.endTag in self.cityMap.tags[state.location]) and \
               (state.memory.issuperset(self.waypointTags))
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
        successors = []
        current = state.location
        collected = set(state.memory)
        for tag in self.cityMap.tags[current]:
            if tag in self.waypointTags:
                collected.add(tag)
        for neighbor, cost in self.cityMap.distances[current].items():
            # 这里只将邻居位置作为动作，而不是 'move'
            successors.append((neighbor, State(neighbor, frozenset(collected)), cost))
        return successors
        # END_YOUR_CODE


########################################################################################
# Problem 2b: Custom -- Plan a Route with Unordered Waypoints through USTC


def getUSTCWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem using the map of USTC, specifying your own
    `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableUSTCMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createUSTCMap()
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    startLocation = locationFromTag(makeTag("landmark", "8348"), cityMap)
    waypointTags = [makeTag("landmark", "1958-WEST"), makeTag("landmark", "west_campus_library")]
    endTag = makeTag("landmark", "3rd_teaching_building")

    # END_YOUR_CODE
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)

########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def __init__(self,):
            # BEGIN_YOUR_CODE (our solution is 3 line of code, but don't worry if you deviate from this)
            self.problem = problem
            self.heuristic = heuristic
            self.startLocation = problem.startLocation  # 初始化 startLocation 属性
            self.endTag = problem.endTag  # 初始化 endTag 属性
            self.cityMap = problem.cityMap  # 初始化 cityMap 属性
            # END_YOUR_CODE

        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            # 新问题的起始状态与原问题相同
            return self.problem.startState()         
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            # 新问题的结束条件与原问题相同
            return self.problem.isEnd(state)
            # END_YOUR_CODE

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
            # 获取原问题的后继状态和代价
            successors = self.problem.successorsAndCosts(state)
            new_successors = []
            for action, next_state, cost in successors:
                # 计算新的代价，结合启发式函数
                new_cost = cost + self.heuristic.evaluate(next_state) - self.heuristic.evaluate(state)
                new_successors.append((action, next_state, new_cost))
            return new_successors
            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 3c: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        # 预计算所有具有目标标签的位置
        self.end_locations = [loc for loc, tags in self.cityMap.tags.items() if endTag in tags]
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        current_location = state.location
        min_distance = float('inf')
        # 计算当前位置到所有目标位置的直线距离，并取最小值
        for end_location in self.end_locations:
            if current_location in self.cityMap.geoLocations and end_location in self.cityMap.geoLocations:
                distance = computeDistance(self.cityMap.geoLocations[current_location], self.cityMap.geoLocations[end_location])
                min_distance = min(min_distance, distance)
        return min_distance
        # END_YOUR_CODE


########################################################################################
# Problem 3e: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        # Precompute
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
        self.endTag = endTag
        self.cityMap = cityMap
        # 预计算所有具有目标标签的位置
        self.end_locations = [loc for loc, tags in self.cityMap.tags.items() if endTag in tags]
        self.distances = {}
        # 预计算所有位置到目标位置的最小距离
        for start_loc in self.cityMap.distances:
            self.distances[start_loc] = float('inf')
            for end_loc in self.end_locations:
                if start_loc in self.cityMap.distances and end_loc in self.cityMap.distances[start_loc]:
                    self.distances[start_loc] = min(self.distances[start_loc], self.cityMap.distances[start_loc][end_loc])
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.distances[state.location]
        # END_YOUR_CODE


########################################################################################
# Problem 3f: Plan a Route through Hefei with or without a Heuristic

def getHefeiShortestPathProblem() -> ShortestPathProblem:
    """
    Create a search problem using the map of Hefei
    """
    cityMap = createHefeiMap()
    startLocation=locationFromTag(makeTag("landmark", "USTC"), cityMap)
    endTag=makeTag("landmark", "Chaohu")
    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    # 创建一个不使用启发式函数的最短路径问题
    return ShortestPathProblem(startLocation, endTag, cityMap)
    # END_YOUR_CODE

def getHefeiShortestPathProblem_withHeuristic() -> ShortestPathProblem:
    """
    Create a search problem with Heuristic using the map of Hefei
    """
    cityMap = createHefeiMap()
    startLocation=locationFromTag(makeTag("landmark", "USTC"), cityMap)
    endTag=makeTag("landmark", "Chaohu")
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    # 创建直线距离启发式函数
    heuristic = StraightLineHeuristic(endTag, cityMap)
    # 创建最短路径问题
    problem = ShortestPathProblem(startLocation, endTag, cityMap)
    # 使用 A* 到 UCS 的归约
    new_problem = aStarReduction(problem, heuristic)
    return new_problem
    # END_YOUR_CODE
