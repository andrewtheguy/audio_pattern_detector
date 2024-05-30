from utils import is_unique_and_sorted


def test_unique_and_sorted():
    array1 = [1, 2, 3, 4, 5]  # Unique and sorted
    array2 = [1, 2, 2, 4, 5]  # Not unique
    array3 = [1, 4, 2, 3, 5]  # Unique but not sorted
    array4 = [1, 2, 4, 2, 5]  # Not unique

    assert is_unique_and_sorted(array1) == True  # Output: True
    assert is_unique_and_sorted(array2) == False  # Output: False
    assert is_unique_and_sorted(array3) == False  # Output: False
    assert is_unique_and_sorted(array4) == False  # Output: False