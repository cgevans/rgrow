import numpy as np
from rgrow.sdc.anneal import Anneal


def test_gen_arrays():
    anneal = Anneal(
        initial_hold=10,
        final_hold=20,
        initial_tmp=80,
        delta_time=100,
        final_tmp=100,
    )

    times, temps = anneal.gen_arrays()
    assert len(times) == len(temps)


def test_times_values():
    anneal = Anneal(
        initial_hold=2,
        delta_time=8,
        final_hold=2,
        initial_tmp=100,
        final_tmp=60,
        temperature_adjustment=0,
    )

    times, temps = anneal.gen_arrays()

    expected_times = np.array([2, 4, 6, 8, 10, 12])
    expected_temps = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 60.0])

    assert np.array_equal(times, expected_times)
    assert np.allclose(temps, expected_temps)
