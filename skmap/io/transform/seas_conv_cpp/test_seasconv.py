import ctypes
import numpy
import glob
import numpy as np
from skmap.parallel import job
import numpy as np
import gc, psutil, os, sys
sys.path.append(os.getcwd())
import seasconv_ref

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
att_seas = 50
att_env = 20
rec_ts_out = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
qa_out = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
ts_in = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
ts_in[0] = np.nan
ts_in[3] = np.nan

# Run seasconv
res = seasconv.run(N_years, N_ipy, N_pix, att_seas, att_env, ts_in, rec_ts_out, qa_out)

print(f'Single thread')
print(rec_ts_out)

############################
##### Validation
############################


print(f'Reference')

N_col = 1
N_row = 1
n_jobs = 8
ts_in = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
ts_in[0] = np.nan
ts_in[3] = np.nan
raster_data_array = np.broadcast_to(ts_in, shape=(N_col,N_row,N_ipy*N_years))

sc = seasconv_ref.SeasConv(ts=raster_data_array, n_ipy=N_ipy, att_env=att_env, att_seas=att_seas, keep_orig_values=False)
pix_ts_rec_s, qa_s = sc.run()
pix_ts_rec_s = pix_ts_rec_s[0,0,:]

print(pix_ts_rec_s)

############################
##### Parallel
############################

N_col = 1
N_row = 1
n_jobs = 8
ts_in = numpy.arange(0, N_years * N_ipy, 1, numpy.double)
ts_in[0] = np.nan
ts_in[3] = np.nan
raster_data_array = np.broadcast_to(ts_in, shape=(N_col,N_row,N_ipy*N_years))
axis = 0


def run(axis, arr):
    N_pix = arr.shape[0]*arr.shape[1]
    if N_pix > 0:
        rec_ts_out = np.zeros((N_pix, arr.shape[2]), numpy.double)
        qa_out = np.zeros((N_pix, arr.shape[2]), numpy.double)
        res = seasconv.run(N_years, N_ipy, N_pix, att_seas, att_env, np.reshape(arr, (N_pix, arr.shape[2])), rec_ts_out, qa_out)
        print(f'Parallel')
        print(rec_ts_out)
        diff = rec_ts_out[0,:]-pix_ts_rec_s
        print(diff)
        rel_err = np.linalg.norm(diff)/np.linalg.norm(pix_ts_rec_s)
        print(f'Relative error {rel_err:.8f}')
    return arr

chunks = [(axis, sub_arr)
        for sub_arr in np.array_split(raster_data_array, n_jobs)]

result = []
for r in job(run, chunks, joblib_args={'backend':'multiprocessing'}):
    # print(r.shape)
    result.append(r)

result = np.concatenate(result)
print(result.shape)
