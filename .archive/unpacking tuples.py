import timeit

def run_performance_test():
    """
    Runs a performance test for different extended unpacking scenarios in Python.
    """
    # Scenario 1: a is 1 and b is [2, 3]
    setup_code_1 = "data = [1, 2, 3]"
    stmt_1 = "a, *b = data"
    time_1 = timeit.timeit(stmt_1, setup_code_1, number=1_000_000)
    print(f"  Execution time: {time_1:.6f} seconds (1,000,000 iterations)")

    # Scenario 2: a is 1, b is [2, 3], and c is 4
    setup_code_2 = "data = [1, 2, 3, 4]"
    stmt_2 = "a, *b, c = data"
    time_2 = timeit.timeit(stmt_2, setup_code_2, number=1_000_000)
    print(f"  Execution time: {time_2:.6f} seconds (1,000,000 iterations)")

    # Scenario 3: b is [1, 2] and c is 3
    setup_code_3 = "data = [1, 2, 3]"
    stmt_3 = "*b, c = data"
    time_3 = timeit.timeit(stmt_3, setup_code_3, number=1_000_000)
    print(f"  Execution time: {time_3:.6f} seconds (1,000,000 iterations)")

    # Scenario 4: a is 1 and b is []
    setup_code_4 = "data = [1]"
    stmt_4 = "a, *b = data"
    time_4 = timeit.timeit(stmt_4, setup_code_4, number=1_000_000)
    print(f"  Execution time: {time_4:.6f} seconds (1,000,000 iterations)")


if __name__ == "__main__":
    run_performance_test()