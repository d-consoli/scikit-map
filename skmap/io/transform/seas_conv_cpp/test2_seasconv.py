import ctypes, glob, gc, psutil, os, sys, time
import numpy as np
from skmap.parallel import job
sys.path.append(os.getcwd())
import seasconv_ref

############################
##### Setup
############################

# Setup the interface with the C++ object
libfile = glob.glob('build/*/seasconv*.so')[0]
seasconv = ctypes.CDLL(libfile)
seasconv.run.restype = ctypes.c_int
seasconv.run.argtypes = [ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_float,
    ctypes.c_float,
    np.ctypeslib.ndpointer(dtype=np.double),
    np.ctypeslib.ndpointer(dtype=np.double),
    np.ctypeslib.ndpointer(dtype=np.double)]

N_jobs = 4
att_seas = 50
att_env = 20
N_years = 2
N_ipy = 2
N_row = 3
N_col = 1
n_img = N_years * N_ipy
N_pix = N_row * N_col

############################
##### Validation
############################


print(f'##########################')
print(f'########### Reference')
print(f'##########################')

ts_in = np.arange(0, N_years * N_ipy, 1, np.double)
ts_in[0] = np.nan
ts_in[3] = np.nan
raster_data_array = np.broadcast_to(ts_in, shape=(1,1,N_ipy*N_years))

sc = seasconv_ref.SeasConv(ts=raster_data_array, n_ipy=N_ipy, att_env=att_env, att_seas=att_seas, keep_orig_values=False)
pix_ts_rec_s, qa_s = sc.run()
pix_ts_rec_s = pix_ts_rec_s[0,0,:]

print(pix_ts_rec_s)



############################
##### Non-parallel
############################

print(f'##########################')
print(f'########### Non parallel')
print(f'##########################')

rec_ts_out = np.arange(0, N_years * N_ipy, 1, np.double)
qa_out = np.arange(0, N_years * N_ipy, 1, np.double)
ts_in = np.arange(0, N_years * N_ipy, 1, np.double)
ts_in[0] = np.nan
ts_in[3] = np.nan
ts_in = np.repeat(ts_in[np.newaxis,:], N_pix, axis=0)
qa_out = np.repeat(qa_out[np.newaxis,:], N_pix, axis=0)
rec_ts_out = np.repeat(rec_ts_out[np.newaxis,:], N_pix, axis=0)

# Run seasconv
start_time = time.time()
res = seasconv.run(N_years, N_ipy, N_pix, att_seas, att_env, ts_in, rec_ts_out, qa_out)
time_SeasConv = time.time() - start_time
print(f"------ Total time non parallel {time_SeasConv:.2f} s")
rec_ts_out = np.reshape(rec_ts_out, (N_row, N_col, N_ipy*N_years))
print(rec_ts_out)

############################
##### Parallel
############################


print(f'##########################')
print(f'########### Parallel')
print(f'##########################')

ts_in = np.arange(0, N_years * N_ipy, 1, np.double)
ts_in[0] = np.nan
ts_in[3] = np.nan
raster_data_array = np.broadcast_to(ts_in, shape=(N_row, N_col, N_ipy*N_years))
axis = 0

def run(axis, arr):
    tmp_N_pix = arr.shape[0]*arr.shape[1]
    tmp_rec_ts_out = np.zeros((tmp_N_pix, arr.shape[2]), np.double)
    tmp_qa_out = np.zeros((tmp_N_pix, arr.shape[2]), np.double)
    if tmp_N_pix > 0:
        res = seasconv.run(N_years, N_ipy, tmp_N_pix, att_seas, att_env, np.reshape(arr, (tmp_N_pix, arr.shape[2])), tmp_rec_ts_out, tmp_qa_out)
        diff = pix_ts_rec_s-tmp_rec_ts_out
        rel_err = np.linalg.norm(diff)/np.linalg.norm(tmp_rec_ts_out)
        print(f'Relative error {rel_err:.8f}')
        print(tmp_rec_ts_out)
    return np.reshape(tmp_rec_ts_out,(arr.shape[0],arr.shape[1],arr.shape[2]))

start_time = time.time()
chunks = [(axis, sub_arr)
        for sub_arr in np.array_split(raster_data_array, N_jobs)]
result = []
for r in job(run, chunks, joblib_args={'backend':'multiprocessing'}):
    result.append(r)
time_SeasConv = time.time() - start_time
print(f"------ Total time parallel {time_SeasConv:.2f} s")

result = np.concatenate(result)



############################
##### Comparison
############################

rel_err = np.linalg.norm(result-rec_ts_out) / np.linalg.norm(rec_ts_out)
print(f"Relative error parrallel vs. non parallel {rel_err:.8f}")