import numpy as np


def gcd_for_numpy_array(arr: np.ndarray) -> int:
    """
    Calculates the greatest common divisor (GCD) of all elements in a NumPy array.

    The GCD is the largest positive integer that divides each of the numbers
    in the array without a remainder.

    Args:
        arr: A NumPy array of integers. It can contain positive, negative,
             or zero values. The array must not be empty.
             If the array is multi-dimensional, it will be flattened first.

    Returns:
        An integer representing the greatest common divisor of the elements.
        - If the array (after flattening) contains only zeros, the GCD is 0.
        - If the array contains zeros and non-zero numbers, the GCD is that
          of the non-zero numbers.
        - The result is always non-negative.

    Raises:
        ValueError: If the input array `arr` is empty (i.e., arr.size == 0).
        TypeError: If the array does not contain integer types that np.gcd can handle
                   (this error is usually propagated from the underlying np.gcd ufunc).
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
        
    if arr.size == 0:
        raise ValueError("Input array cannot be empty for GCD calculation.")

    # Ensure the array is of an integer type. np.gcd requires integer inputs.
    # If you expect arrays that might be floats but represent integers (e.g., [2.0, 4.0]),
    # you might need to add explicit conversion and checks here, e.g.,
    # if not np.issubdtype(arr.dtype, np.integer):
    #     if np.all(arr == arr.astype(np.int64)): # Check if all are whole numbers
    #         arr = arr.astype(np.int64)
    #     else:
    #         raise TypeError("NumPy array elements must be integers or represent whole numbers for GCD calculation.")
    # However, np.gcd itself will raise a TypeError for float inputs, which is often sufficient.

    # Flatten the array to ensure GCD is calculated over all elements
    # regardless of the original array's shape.
    flat_arr = arr.flatten()

    # np.gcd.reduce applies the gcd ufunc cumulatively.
    # The `initial=0` argument is crucial:
    # 1. It provides a starting value for the accumulation.
    # 2. np.gcd(0, x) returns abs(x) (for integer x). This ensures that:
    #    - For a single element array [n] (after flattening), it effectively computes gcd(0, n) = |n|.
    #    - It correctly handles initial zeros in the flattened array.
    # np.gcd ufunc inherently returns non-negative results (or 0 if both inputs are 0).
    result_np = np.gcd.reduce(flat_arr, initial=0)
    
    # Convert NumPy's scalar integer type (like np.int32 or np.int64) to a standard Python int.
    return int(result_np)


if __name__ == "__main__":
    v = np.array([
        [33, 45, 78]
    ])
    greatest_common_divisor = gcd_for_numpy_array(v)
    print(greatest_common_divisor)
