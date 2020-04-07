from synthplayer.oscillators import *
from synthplayer import params as synth_params

from typing import Generator, List, Sequence, Optional, Tuple, Iterator



class LinearCurve(Oscillator):
    """
    Takes an ordered list of <time,value> tuples to define the curve to sample
    """
    def __init__(self, points:List[Tuple[float,float]], samplerate: int = 0) -> None:
        super().__init__(samplerate)
        self._points = points
        self._increment = 1.0/self.samplerate
        self._t = 0
        self._prev_point = None
        self._next_point = None
        self._point_index = 0

    def append(self,time:float,level:float):
        tup = (time,level)
        self._points.append(tup)

    def update_points(self):
        # At the start, we don't have any points, take the first one
        if self._next_point is None and self._prev_point is None:
            if len(self._points):
                self._next_point = self._points[0]
            else:
                print("Warning: curve with no points being sampled")
        # If we don't have a next point, then we have run off the end
        if self._next_point is None:
            pass
        # If we are moving towards a point and have reached it
        # load the next one

        elif self._t >= self._next_point[0]:
            self._prev_point = self._next_point
            self._point_index += 1
            self._next_point = None
            if self._point_index < len(self._points):
                self._next_point = self._points[self._point_index]
        else:
            pass

    # Sample and increment
    def next_sample(self)->float:
        self.update_points()
        val = self.sample()
        self._t += self._increment
        return val

    # Sample at the current value
    def sample(self)->float:
        # If we have both points, interpolate between
        if self._next_point and self._prev_point:
            t_range = self._next_point[0] - self._prev_point[0]
            if t_range == 0:
                return self._next_point[1]
            t_ind = self._t - self._prev_point[0]
            t_prop = t_ind / t_range
            v_diff = self._next_point[1] - self._prev_point[1]
            return self._prev_point[1] + v_diff * t_prop
        elif self._next_point:
            return self._next_point[1]
        elif self._prev_point:
            return self._prev_point[1]
        else:
            return 0.0




    def blocks(self) -> Generator[List[float], None, None]:
        while True:
            block = []  # type: List[float]
            #print(f"Time: {self._t}, next: {self._next_point}, prev: {self._prev_point}")
            for _ in range(synth_params.norm_osc_blocksize):
                block.append(self.next_sample())
            yield block

if __name__ == "__main__":
    a = LinearCurve([(5.0, -1.),(10., 1.),(20.,-1)],samplerate=10)
    for _ in range(300):
        print(f"{a._t} : {a.next_sample()}")
