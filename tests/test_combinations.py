import sweep_utils


def test_build_combinations_example():
    outputs = sweep_utils.build_combinations([[1, 2, 3], [4, 5]])
    assert outputs[0] == [1, 2, 3, 1, 2, 3]
    assert outputs[1] == [4, 5, 4, 5, 4, 5]


def test_build_combinations_single():
    outputs = sweep_utils.build_combinations([[1, 2]])
    assert outputs[0] == [1, 2]
