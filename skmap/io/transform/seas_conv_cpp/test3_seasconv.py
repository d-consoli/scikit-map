import ctypes, glob, gc, psutil, os, sys, time
import numpy as np
sys.path.append(os.getcwd()+"/../../../..")
sys.path.append(os.getcwd())
from skmap.parallel import job
import seasconv_ref
import threading
import mkl
                
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

N_jobs = 40
att_seas = 50
att_env = 20
N_years = 22
N_ipy = 12
N_row = 4000
N_col = 3000
N_img = N_years * N_ipy
N_pix = N_row * N_col
N_nan = np.floor(N_img*N_pix*0.3).astype(int)

mkl.set_num_threads(N_jobs)
cmp_flag = False

print(f"--- Creating random data")
ts_in = np.arange(0, N_img, 1, np.double)
ts_in = np.repeat(ts_in[np.newaxis,:], N_pix, axis=0)
ts_in += np.random.rand(ts_in.shape[0], ts_in.shape[1])
ts_in.ravel()[np.random.choice(ts_in.size, N_nan, replace=False)] = np.nan
qa_out = np.zeros(ts_in.shape)
rec_ts_out = np.zeros(ts_in.shape)
print(ts_in.shape)

############################
##### Non-parallel
############################

if cmp_flag:
    print(f" ")
    print(f'----------------------------------')
    print(f'            Non parallel')
    print(f'----------------------------------')

    ts_in_np = ts_in.copy()
    rec_ts_out_np = rec_ts_out.copy()
    qa_out_np = qa_out.copy()

    # Run seasconv
    print(f"--- Starting the computation")
    start_time = time.time()
    res = seasconv.run(N_years, N_ipy, N_pix, att_seas, att_env, ts_in_np, rec_ts_out_np, qa_out_np)
    time_SeasConv = time.time() - start_time
    print(f"------ Total time non parallel {time_SeasConv:.2f} s")
    # rec_ts_out_np = np.reshape(rec_ts_out_np, (N_row, N_col, N_img))

############################
##### Parallel
############################


print(f' ')
print(f'----------------------------------')
print(f'            Parallel')
print(f'----------------------------------')

ts_in_p = ts_in
rec_ts_out_p = rec_ts_out
qa_out_p = qa_out

def compute_slice(in_array_slice, out_array1_slice, out_array2_slice, N_years, N_ipy, att_seas, att_env):
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
    N_pix_tmp = in_array_slice.shape[0]
    res = seasconv.run(N_years, N_ipy, N_pix_tmp, att_seas, att_env, in_array_slice, out_array1_slice, out_array2_slice)
    pass



def parallel_compute(num_threads, in_array, out_array1, out_array2, N_years, N_ipy, att_seas, att_env):
    num_rows = in_array.shape[0]
    rows_per_thread = num_rows // num_threads

    # Create a list to hold thread objects
    threads = []

    # Split the data and create threads
    for i in range(num_threads):
        start_row = i * rows_per_thread
        end_row = start_row + rows_per_thread if i < num_threads - 1 else num_rows
        in_slice = in_array[start_row:end_row]
        out1_slice = out_array1[start_row:end_row]
        out2_slice = out_array2[start_row:end_row]

        thread = threading.Thread(target=compute_slice, args=(in_slice, out1_slice, out2_slice, N_years, N_ipy, att_seas, att_env))
        threads.append(thread)

    # Start the threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

print(f"--- Starting the computation")
start_time = time.time()
parallel_compute(N_jobs, ts_in_p, rec_ts_out_p, qa_out_p, N_years, N_ipy, att_seas, att_env)
time_SeasConv = time.time() - start_time
print(f"------ Total time parallel {time_SeasConv:.2f} s")
# rec_ts_out_p = np.reshape(rec_ts_out_p, (N_row, N_col, N_img))


# ############################
# ##### Comparison
# ############################


if cmp_flag:
    print(f' ')
    rel_err = np.linalg.norm(rec_ts_out_np-rec_ts_out_p) / np.linalg.norm(rec_ts_out_np)
    print(f"Relative error parrallel vs. non parallel {rel_err:.8f}")