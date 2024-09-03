import numpy as np

MIN = 60
HOUR = MIN * 60


class Anneal:
    """
    An anneal protocol.

    Attributes:
        initial_hold (float): How long to hold the system for before changing
            temperature (in seconds)
        final_hold (float): How long to hold the system for once the
            temperature is finished changing (in seconds)
        delta_time (float): The duration of time during which the temperature
            will be changing (in seconds)

        initial_temperature (float): Temperature of the system before anneal
            starts (in degrees C)
        final_temperature (float): Target temperature, it will be reached at
            the end of the anneal (in degrees C)

        scaffold_count (int): Number of scaffolds to simulate, the higher,
            the more statistically significant, but the longer the anneal will
            take to finish running
        timestep (float): Simulated time cannot be continuous. How big do you
            want each time jump to be ? The smaller, the more accurete the
            system will be, but it will take longer.
    """

    def __init__(
        self,
        initial_hold: float,
        initial_tmp: float,
        delta_time: float,
        final_tmp: float,
        final_hold: float,
        scaffold_count: int = 100,
        timestep: float = 2.0,
    ):
        self.initial_hold = initial_hold
        self.initial_tmp = initial_tmp + 8
        self.delta_time = delta_time
        self.final_tmp = final_tmp + 8
        self.final_hold = final_hold
        self.scaffold_count = scaffold_count

        # How many seconds to spend in each step
        #
        # By default, this is two. This means that if we have a 10 second anneal,
        # The times array will look like:
        # 2, 4, 6, 8, 10
        self.timestep = timestep

    @staticmethod
    def standard_long_anneal(from_tmp=80, final_tmp=20, scaffold_count=100):
        """
        Standard anneal with:
            Rest for 10 minutes, then change temperatures linearly
            for 3 hours, then rest for 45 minutes.

            The system will go from the initial temperature (default 80+8),
            to the final temperature (default 20+8) over the 3 hours.
        """
        # error_delta = 8
        return Anneal(10 * MIN, from_tmp, 3 * HOUR, final_tmp, 45 * MIN, scaffold_count)

    def gen_arrays(self):
        """
        Generate the time and the temperature arrays

        Returns:
            An array of times,
            an array of temperatures
        """
        steps_per_sec = 1 / self.timestep
        number_of_steps = int(self.delta_time * steps_per_sec)

        delta_temperatures = np.linspace(
            self.initial_tmp, self.final_tmp, int(number_of_steps + 1)
        )
        initial_temp = np.repeat(
            self.initial_tmp, int(self.initial_hold * steps_per_sec) - 1
        )
        ending_temp = np.repeat(self.final_tmp, int(self.final_hold * steps_per_sec))
        temperatures = np.concatenate([initial_temp, delta_temperatures, ending_temp])

        total_time = self.initial_hold + self.final_hold + self.delta_time
        times = np.arange(self.timestep, total_time + self.timestep, self.timestep)

        return times, temperatures


class AnnealOutputs:
    """
    The output generated when a system runs an anneal

    Attributes:
        system (SDC): The sdc system that was executed
        canvas_arr list[list[list[int]]:
            This is an array of snapshots, each snapshot contains information
            about the state of the scaffolds at each point in time.
            Take a snapshot with 4 compute domains, total length of 5. It would
            look something like this:
            [
                 -  -  A  B  C  D  E  -  -
                [0, 0, 1, 1, 2, 3, 6, 0, 0],
                [0, 0, 1, 2, 8, 3, 6, 0, 0],
                ...
                [0, 0, 1, 3, 1, 3, 6, 0, 0],
                [0, 0, 1, 1, 3, 3, 6, 0, 0],
            ]
            Each one of the inner-arrays represents one scaffold. The first two
            elements will always be 0, as well as the last two elements (for
            performance reasons), the third element would be the id of whatever
            strand is attached to that scaffolds A domain. The fourth is the id
            of whatever is attached to the B domain, ... If nothing is attached
            then the number will be 0.
        anneal (Anneal): The anneal that was executed (stored in the output
            since some of its data is relevant when measuring error / graphing)
    """

    def __init__(self, system, anneal: Anneal, canvas_arr):
        # The anneal that was executed
        self.system = system
        # Snapshots of the canvas
        self.canvas_arr = canvas_arr
        # The anneal that was executed
        self.anneal = anneal