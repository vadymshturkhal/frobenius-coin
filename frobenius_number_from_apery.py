from dijkstra_apery_set import dijkstra_apery_set


def frobenius_number_from_apery(coefficients, apery_set):
    return max(apery_set.values()) - min(coefficients)


if __name__ == "__main__":
    coefficients = [7, 14, 20]
    coefficients = [7, 9, 15]
    
    apery_set = dijkstra_apery_set(coefficients)
    frobenius_number = frobenius_number_from_apery(coefficients, apery_set)
    print(f"Frobenius number of {coefficients} is {frobenius_number}")
