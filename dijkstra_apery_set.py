import heapq
import numpy as np
from typing import List, Dict
from gcd_for_numpy_array import gcd_for_numpy_array


def dijkstra_apery_set(coefficients: List[int]) -> Dict[int, int]:
    """
        O(N) = O(n*modulus*log(modulus))
    """

    if gcd_for_numpy_array(np.array(coefficients)) != 1:
        raise Exception("Problem is impossible.") 
    
    # tiny optimization
    coefficients = sorted(coefficients, reverse=True)
    modulus = coefficients.pop()

    distances = [float('inf')] * modulus
    
    # distance to residue 0 is 0
    distances[0] = 0

    # create Apery set
    apery = {i:i for i in range(modulus)}

    # heap with (distance, residue) terms
    heap = [(0, 0)]

    visited = [False] * modulus

    while heap:
        distance, residue = heapq.heappop(heap)

        if visited[residue]:
            continue

        visited[residue] = True

        # update weights
        for coefficient in coefficients:
            current_residue = (residue + coefficient) % modulus
            new_distance = distance + coefficient

            if new_distance < distances[current_residue]:
                distances[current_residue] = new_distance
                apery[current_residue] = new_distance
                heapq.heappush(heap, (new_distance, current_residue))

    return apery


if __name__ == "__main__":
    # coefficients = [11453, 11159, 11161, 11152, 12231, 11632, 14255, 13288, 17289, 16378, 12314, 13122]
    # coefficients = [100003, 110007, 110319, 104021, 100533, 109043, 108049, 107657]
    # coefficients = [11100003, 11110007, 11110319, 11104021, 11100533, 11109043, 11108049, 11107657]
    # coefficients = [11111112222, 1111212222, 11111452222, 11116662223]  # killed
    # coefficients = [11453, 11159, 11161, 11152, 12231, 13552, 13522, 11632, 14539, 18995, 14255, 13288, 17289, 16378, 12314, 13122]
    coefficients = [7, 9, 15]

    apery_set = dijkstra_apery_set(coefficients)
    print(f"{apery_set = }")
    print(f"{len(apery_set) = }")

