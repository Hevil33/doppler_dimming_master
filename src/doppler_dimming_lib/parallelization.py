"""
Utilities for 2D and 3D array parallelization
"""

import datetime
import multiprocessing as mp
import sys
import time
from logging import error, info

import numpy as np


def batch_2d_array(input_array, tot_nprocs):
    return np.array_split(input_array, tot_nprocs)


def merge_2d_batched_array(input_array):
    return np.concatenate(input_array)


def batch_3d_array(input_array, tot_nprocs):
    first_slice = batch_2d_array(input_array[0, :, :], tot_nprocs)
    batched_3d_array = [
        np.empty(shape=(input_array.shape[0], *first_slice[i].shape))
        for i in range(tot_nprocs)
    ]

    for slice_i in range(0, input_array.shape[0]):
        current_slice = batch_2d_array(input_array[slice_i, :, :], tot_nprocs)

        for proc_i in range(tot_nprocs):
            batched_3d_array[proc_i][slice_i, :, :] = current_slice[proc_i]

    return batched_3d_array


def batch_arrays(input_list, tot_nprocs):
    batched_list = []

    for array in input_list:
        if len(array.shape) == 2:
            batched_list.append(batch_2d_array(array, tot_nprocs))
        elif len(array.shape) == 3:
            batched_list.append(batch_3d_array(array, tot_nprocs))
        else:
            error("Cannot batch arrays that are not two- or three-dimensional.")
            sys.exit()

    args = [
        [batched_array[i] for batched_array in batched_list] for i in range(tot_nprocs)
    ]

    return args


def _test_func_to_parallelize(input_array1, input_array2, input_array3, input_3d_array):
    first_axis_integrated = np.zeros_like(input_array1)
    first_axis_integrated = (
        3 * input_array1 - input_array2 - input_array3
    )  # this part equale input_array1

    for i in range(input_3d_array.shape[0]):
        first_axis_integrated += input_3d_array[i]  # add the sum over first axis

    return first_axis_integrated


def main():
    sys.path.append("./")
    # import integrals  # import for debugging only

    # debug
    tot_nprocs = 5
    sample_size = 4  # make nxn matrix for testing

    sample_array = (
        np.reshape(range(sample_size * sample_size), (sample_size, sample_size)) * 1.0
    )  # convert to double

    batched_2d_array = batch_2d_array(sample_array, tot_nprocs)
    merged_2d_array = merge_2d_batched_array(batched_2d_array)

    assert np.array_equal(sample_array, merged_2d_array)

    sample_3d_array = np.repeat(sample_array[np.newaxis, :, :], sample_size, axis=0)

    # make it symmetric in the first axis
    for i in range(sample_3d_array.shape[0]):
        sample_3d_array[i] *= i - (sample_3d_array.shape[0] - 1) / 2

    assert np.array_equal(
        np.mean(sample_3d_array, axis=0), np.zeros_like(sample_3d_array[0])
    )

    sample_3d_array = (
        np.reshape(
            range(sample_size * sample_size * sample_size),
            (sample_size, sample_size, sample_size),
        )
        * 1.0
    )  # unique identifiers, stronger test

    batched_3d_array = batch_3d_array(sample_3d_array, tot_nprocs)

    _test_func_to_parallelize(
        batched_2d_array[0],
        batched_2d_array[0],
        batched_2d_array[0],
        batched_3d_array[0],
    )

    expected_args = [
        [
            batched_2d_array[i],
            batched_2d_array[i],
            batched_2d_array[i],
            batched_3d_array[i],
        ]
        for i in range(tot_nprocs)
    ]
    args = batch_arrays([sample_array] * 3 + [sample_3d_array], tot_nprocs)

    # check batch_arguments equal to expected arguments
    assert len(args) == len(expected_args)
    for proc_i in range(tot_nprocs):
        for arg, expected_arg in zip(args[proc_i], expected_args[proc_i]):
            assert np.array_equal(arg, expected_arg)

    print(f"Starting parallel calculation, {tot_nprocs=}")
    start = time.perf_counter()
    with mp.Pool(processes=tot_nprocs) as p:
        result = p.starmap(
            _test_func_to_parallelize,
            args,
        )
    merged_result = merge_2d_batched_array(result)

    expected_result = _test_func_to_parallelize(
        sample_array,
        sample_array,
        sample_array,
        sample_3d_array,
    )

    print("-" * 30)
    print("arrays:")
    print("2d input array:")
    print(sample_array)
    print("after batch and merge again:")
    print(merged_2d_array)
    print("result of parallel function:")
    print(merged_result)
    print("expected result from non-parallel application:")
    print(expected_result)
    print("-" * 30)
    assert np.array_equal(merged_result, expected_result)

    end = time.perf_counter()
    # info(f"Parallel computation ended in: {end - start} s")
    print(f"Parallel computation ended in: {datetime.timedelta(seconds=end-start)} ")

    print("Test ended successfully")


if __name__ == "__main__":
    main()
