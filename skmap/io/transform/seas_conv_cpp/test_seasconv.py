import ctypes
import numpy
import glob
import numpy as np

# Setup the interface with the C++ object
libfile = glob.glob('build/*/seasconv*.so')[0]
seasconv = ctypes.CDLL(libfile)
seasconv.run.restype = ctypes.c_int
seasconv.run.argtypes = [ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_float,
    ctypes.c_float,
    numpy.ctypeslib.ndpointer(dtype=numpy.double),
    numpy.ctypeslib.ndpointer(dtype=numpy.double),
    numpy.ctypeslib.ndpointer(dtype=numpy.double)]

N_years = 3
N_ipy = 4
N_pix = 10
att_seas = 60
att_env = 20
ts_in = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
rec_ts_out = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
qa_out = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
ts_in[0] = np.nan
ts_in[3] = np.nan

# Run seasconv
res = seasconv.run(N_years, N_ipy, N_pix, att_seas, att_env, ts_in, rec_ts_out, qa_out)

print(f'Return value {res}')