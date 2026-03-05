from calendar import c
import re
from tracemalloc import start
from typing import List, Tuple

from networkx import neighbors

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
        return State(self.startLocation, memory=None)   # 最开始的时候memory为空
        raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.endTag in self.cityMap.tags[state.location]
        raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        return [
            (neighbor, State(neighbor, memory=None), distance)
            for neighbor, distance in self.cityMap.distances[state.location].items()
        ]
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
    """使用中国科学技术大学（USTC）的地图创建你自己的搜索问题，
    指定你自己的 “起始位置（startLocation）” 和 “目标标签（endTag）”。
    运行 “python mapUtil.py> readableUSTCMap.txt” 命令，以生成一个包含位置列表及其相关标签的文件；
    你可能会发现搜索以下这些（以及其他的）标签关键字会很有用：
    “landmark=”—— 人工定义的地标（来自 “data/USTC-landmarks.json” 文件）
    “amenity=”—— 各种设施类型（例如，“咖啡”“食物”）
    """
    cityMap = createUSTCMap()

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    startLocation = '3643576549'

    # endTag = "entrance=yes"
    endTag = 'name=中科大西区东门'
    return ShortestPathProblem(startLocation, endTag, cityMap)

    raise NotImplementedError("Override me")
    # END_YOUR_CODE
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Think carefully about what `memory` representation your States should have!
    """
    """定义了一个搜索问题，
    该问题对应于找到从 “起始位置（startLocation）” 到具有指定 “目标标签（endTag）” 的任意位置的最短路径，
    并且此路径还需经过包含 “路径点标签（waypointTags）” 集合中标签的各个位置。
    仔细思考你的 “状态（State）” 对象的 “记忆（memory）” 部分应该采用怎样的表示形式！
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
        return State(self.startLocation,memory=tuple(1 if tag in self.cityMap.tags[self.startLocation] 
                                                     else 0 for tag in self.waypointTags))
        raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
        return self.endTag in self.cityMap.tags[state.location] and all(state.memory)
        raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
        neighbors = []
        for neighbor, distance in self.cityMap.distances[state.location].items():
            state_memory = tuple(max(state.memory[i], 
                                     1 if tag in self.cityMap.tags[neighbor] else 0) 
                                 for i, tag in enumerate(self.waypointTags))
            neighbors.append((neighbor, State(neighbor, memory=tuple(state_memory),), distance))
        return neighbors
        raise NotImplementedError("Override me")
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
    """使用中国科学技术大学（USTC）的地图创建你自己的搜索问题，指定你自己的 “起始位置（startLocation）”、“途经点标签（waypointTags）” 和 “终点标签（endTag）”。
    与问题 1b 类似，使用 readableUSTCMap.txt 文件来确定可能的位置和标签。
    """

    cityMap = createUSTCMap()
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    startLocation = '3643576549'
    endTag = 'name=中科大西区东门'
    waypointTags = ('landmark=1958-WEST', 'amenity=coffee')
    
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)
    raise NotImplementedError("Override me")
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

"""
把你试图借助启发式函数（heuristic）解决的现有搜索问题（problem）转化为一个新的搜索问题（newSearchProblem），
使得对 newSearchProblem 运行一致代价搜索算法（UCS）等同于对 problem 结合 heuristic 运行 A * 算法。
将一个问题的模型及其额外约束转化为同一类问题的新实例的过程称为归约。
这是一种强大的工具，能让我们用熟悉的方式来描述 “新” 的问题模型。
"""

def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def __init__(self):
            # BEGIN_YOUR_CODE (our solution is 3 line of code, but don't worry if you deviate from this)
            #raise NotImplementedError("Override me")
            self.startLocation = problem.startLocation
            self.endTag = problem.endTag
            self.cityMap = problem.cityMap
            # END_YOUR_CODE

        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return problem.startState()
            raise NotImplementedError("Override me")
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            return problem.isEnd(state)
            raise NotImplementedError("Override me")
            # END_YOUR_CODE

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)   
            successors = problem.successorsAndCosts(state)
            Astarsuccessors = []
            for action, nextstate, cost in successors:
                Astarsuccessors.append((action, nextstate, cost + heuristic.evaluate(nextstate) - heuristic.evaluate(state)))   
            #修改代价：cost + heuristic(nextstates) - heuristic(state)
            return Astarsuccessors      
            raise NotImplementedError("Override me")
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
        self.endLoc = [
            location for location in self.cityMap.geoLocations 
            if endTag in self.cityMap.tags[location]
        ]
        #raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        return min(
            computeDistance(self.cityMap.geoLocations[state.location], self.cityMap.geoLocations[endLoc])
            for endLoc in self.endLoc
        )
        raise NotImplementedError("Override me")
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
        self.shortestPaths = {}

        targetLocations = sorted(
            [location for location, tags in cityMap.tags.items() if endTag in tags]
        )
        
        for target in targetLocations:
            pathProblem = ShortestPathProblem(target, "NoGoal", cityMap)
            pathFinder = UniformCostSearch(verbose=0)
            pathFinder.solve(pathProblem)
            for state, cost in pathFinder.pastCosts.items():
                loc = state.location
                if loc not in self.shortestPaths or self.shortestPaths[loc] > cost:
                    self.shortestPaths[loc] = cost
        #raise NotImplementedError("Override me")
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.shortestPaths[state.location]
        raise NotImplementedError("Override me")
        # END_YOUR_CODE


########################################################################################
# Problem 3f: Plan a Route through Hefei with or without a Heuristic

def getHefeiShortestPathProblem(cityMap: CityMap) -> ShortestPathProblem:
    """
    Create a search problem using the map of Hefei
    """
    startLocation=locationFromTag(makeTag("landmark", "USTC"), cityMap)
    endTag=makeTag("landmark", "Chaohu")
    # BEGIN_YOUR_CODE (our solution is 1 lines of code, but don't worry if you deviate from this)
    return ShortestPathProblem(startLocation, endTag, cityMap)
    raise NotImplementedError("Override me")
    # END_YOUR_CODE

def getHefeiShortestPathProblem_withHeuristic(cityMap: CityMap) -> ShortestPathProblem:
    """
    Create a search problem with Heuristic using the map of Hefei
    """
    startLocation=locationFromTag(makeTag("landmark", "USTC"), cityMap)
    endTag=makeTag("landmark", "Chaohu")
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    heuristic = NoWaypointsHeuristic(endTag, cityMap)
    return aStarReduction(ShortestPathProblem(startLocation, endTag, cityMap), heuristic)
    raise NotImplementedError("Override me")
    # END_YOUR_CODE
    
    
