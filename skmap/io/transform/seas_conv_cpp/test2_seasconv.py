import ctypes, glob, gc, psutil, os, sys, time
import numpy as np
sys.path.append(os.getcwd()+"/../../../..")
sys.path.append(os.getcwd())
from skmap.parallel import job
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

N_jobs = 3
att_seas = 50
att_env = 20
N_years = 3
N_ipy = 23
N_row = 1
N_col = 2
N_img = N_years * N_ipy
N_pix = N_row * N_col
N_nan = np.floor(N_img*N_pix*0.3).astype(int)

############################
##### Validation
############################


# print(f'##########################')
# print(f'########### Reference')
# print(f'##########################')

# ts_in = np.arange(0, N_years * N_ipy, 1, np.double)
# ts_in[0] = np.nan
# ts_in[3] = np.nan
# raster_data_array = np.broadcast_to(ts_in, shape=(1,1,N_ipy*N_years))

# sc = seasconv_ref.SeasConv(ts=raster_data_array, n_ipy=N_ipy, att_env=att_env, att_seas=att_seas, keep_orig_values=False)
# pix_ts_rec_s, qa_s = sc.run()
# pix_ts_rec_s = pix_ts_rec_s[0,0,:]

# print(pix_ts_rec_s)



############################
##### Non-parallel
############################

print(f'----------------------------------')
print(f'            Non parallel')
print(f'----------------------------------')


ts_in = np.arange(0, N_img, 1, np.double)
ts_in[0] = np.nan
ts_in[3] = np.nan
ts_in = np.repeat(ts_in[np.newaxis,:], N_pix, axis=0)
# ts_in += np.random.rand(ts_in.shape[0], ts_in.shape[1])
# ts_in.ravel()[np.random.choice(ts_in.size, N_nan, replace=False)] = np.nan
qa_out = np.zeros(ts_in.shape)
rec_ts_out = np.zeros(ts_in.shape)
print(ts_in.shape)
# Run seasconv
start_time = time.time()
res = seasconv.run(N_years, N_ipy, N_pix, att_seas, att_env, ts_in, rec_ts_out, qa_out)
time_SeasConv = time.time() - start_time
print(f"------ Total time non parallel {time_SeasConv:.2f} s")
rec_ts_out = np.reshape(rec_ts_out, (N_row, N_col, N_img))

############################
##### Parallel
############################


print(f' ')
print(f'----------------------------------')
print(f'            Parallel')
print(f'----------------------------------')

ts_in = np.arange(0, N_img, 1, np.double)
ts_in[0] = np.nan
ts_in[3] = np.nan
ts_in_par = np.broadcast_to(ts_in, shape=(N_row, N_col, N_img))
rec_ts_out_par = np.zeros((N_row, N_col, N_img))
qa_out_par = np.zeros((N_row, N_col, N_img))

def run(ts_in_sub, rec_ts_out_sub, qa_out_sub, N_years, N_ipy, att_seas, att_env):
    tmp_N_pix = ts_in_sub.shape[0]*ts_in_sub.shape[1]
    tmp_N_imp = ts_in_sub.shape[2]
    if tmp_N_pix > 0:
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
        res = seasconv.run(N_years, N_ipy, tmp_N_pix, att_seas, att_env, \
                           np.reshape(ts_in_sub, (tmp_N_pix, tmp_N_imp)), \
                           np.reshape(rec_ts_out_sub, (tmp_N_pix, tmp_N_imp)), \
                           np.reshape(qa_out_sub, (tmp_N_pix, tmp_N_imp)))
        #print('#', tmp_rec_ts_out.shape)
        # diff = pix_ts_rec_s-tmp_rec_ts_out
        # rel_err = np.linalg.norm(diff)/np.linalg.norm(tmp_rec_ts_out)
        # print(f'Relative error {rel_err:.8f}')
        # print(tmp_rec_ts_out)
    return 0

start_time = time.time()
chunks = [(ts_in_sub, rec_ts_out_sub, qa_out_sub, N_years, N_ipy, att_seas, att_env)
        for ts_in_sub, rec_ts_out_sub, qa_out_sub in zip(np.array_split(ts_in_par, N_jobs), np.array_split(rec_ts_out_par, N_jobs), np.array_split(qa_out_par, N_jobs))]

result = []

print("Starting the computation")
# for r in job(run, chunks, joblib_args={"backend":"threading"}):
for r in job(run, chunks, joblib_args={"backend":"threading", "mmap_mode":'w+'}, n_jobs=N_jobs):
    print(f"Done {r}")
time_SeasConv = time.time() - start_time
print(f"------ Total time parallel {time_SeasConv:.2f} s")


# ############################
# ##### Comparison
# ############################


print(rec_ts_out_par)
print(rec_ts_out)

print(f' ')
rel_err = np.linalg.norm(rec_ts_out_par-rec_ts_out) / np.linalg.norm(rec_ts_out)
print(f"Relative error parrallel vs. non parallel {rel_err:.8f}")