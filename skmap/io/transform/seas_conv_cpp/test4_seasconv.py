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

N_jobs = 40
att_seas = 50
att_env = 20
N_years = 22
N_ipy = 12
N_row = 40
N_col = 40
N_img = N_years * N_ipy
N_ext = N_img * 2
N_pix = N_row * N_col
N_nan = np.floor(N_img*N_pix*0.3).astype(int)

mkl.set_num_threads(N_jobs)
cmp_flag = True
old_flag = True
new_flag = True

print(f"--- Creating random data")
ts = np.arange(0, N_img, 1, np.double)
ts = np.repeat(ts[np.newaxis,:], N_pix, axis=0)
ts += np.random.rand(ts.shape[0], ts.shape[1])
ts.ravel()[np.random.choice(ts.size, N_nan, replace=False)] = np.nan
qa = np.zeros(ts.shape)
rec_ts_out = np.zeros(ts.shape)
print(ts.shape)

############################
##### Parallel old
############################

if old_flag:

    print(f' ')
    print(f'----------------------------------')
    print(f'            Parallel old')
    print(f'----------------------------------')
    
    ts_po = ts.copy()
    rec_ts_out_po = rec_ts_out.copy()
    qa_po = qa.copy()

    def compute_slice_old(in_array_slice, out_array1_slice, out_array2_slice, N_years, N_ipy, att_seas, att_env):
        libfile_old = glob.glob('build/*/old_sc*.so')[0]
        old_sc = ctypes.CDLL(libfile_old)
        old_sc.run.restype = ctypes.c_int
        old_sc.run.argtypes = [ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_float,
        ctypes.c_float,
        np.ctypeslib.ndpointer(dtype=np.double),
        np.ctypeslib.ndpointer(dtype=np.double),
        np.ctypeslib.ndpointer(dtype=np.double)]
        N_pix_tmp = in_array_slice.shape[0]
        res = old_sc.run(N_years, N_ipy, N_pix_tmp, att_seas, att_env, in_array_slice, out_array1_slice, out_array2_slice)
        pass



    def parallel_compute_old(num_threads, in_array, out_array1, out_array2, N_years, N_ipy, att_seas, att_env):
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

            thread = threading.Thread(target=compute_slice_old, args=(in_slice, out1_slice, out2_slice, N_years, N_ipy, att_seas, att_env))
            threads.append(thread)

        # Start the threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    print(f"--- Starting the computation")
    start_time = time.time()
    parallel_compute_old(N_jobs, ts_po, rec_ts_out_po, qa_po, N_years, N_ipy, att_seas, att_env)
    time_SeasConv = time.time() - start_time
    print(f"------ Total time parallel old {time_SeasConv:.2f} s")
    # rec_ts_out_p = np.reshape(rec_ts_out_p, (N_row, N_col, N_img))


############################
##### Parallel
############################

if new_flag:

    print(f' ')
    print(f'----------------------------------')
    print(f'            Parallel ')
    print(f'----------------------------------')

    ts_p = np.zeros((N_pix,N_ext))
    ts_p[:,0:N_img] = ts.copy()
    qa_p = np.zeros((N_pix,N_ext))

    def compute_slice(ts_array_slice, qa_array_slice, N_years, N_ipy, att_seas, att_env):
        libfile = glob.glob('build/*/seasconv*.so')[0]
        seasconv = ctypes.CDLL(libfile)
        seasconv.runDouble.restype = ctypes.c_int
        seasconv.runDouble.argtypes = [ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_float,
        ctypes.c_float,
        np.ctypeslib.ndpointer(dtype=np.double),
        np.ctypeslib.ndpointer(dtype=np.double)]
        N_pix_tmp = ts_array_slice.shape[0]
        res = seasconv.runDouble(N_years, N_ipy, N_pix_tmp, att_seas, att_env, ts_array_slice, qa_array_slice)
        pass



    def parallel_compute(num_threads, ts_array, qa_array, N_years, N_ipy, att_seas, att_env):
        num_rows = ts_array.shape[0]
        rows_per_thread = num_rows // num_threads

        # Create a list to hold thread objects
        threads = []

        # Split the data and create threads
        for i in range(num_threads):
            start_row = i * rows_per_thread
            end_row = start_row + rows_per_thread if i < num_threads - 1 else num_rows
            ts_slice = ts_array[start_row:end_row]
            qa_slice = qa_array[start_row:end_row]

            thread = threading.Thread(target=compute_slice, args=(ts_slice, qa_slice, N_years, N_ipy, att_seas, att_env))
            threads.append(thread)

        # Start the threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    print(f"--- Starting the computation")
    start_time = time.time()
    parallel_compute(N_jobs, ts_p, qa_p, N_years, N_ipy, att_seas, att_env)
    time_SeasConv = time.time() - start_time
    print(f"------ Total time parallel {time_SeasConv:.2f} s")
    # rec_ts_out_p = np.reshape(rec_ts_out_p, (N_row, N_col, N_img))
    ts_p = ts_p[:,0:N_img]
    qa_p = qa_p[:,0:N_img]

# ############################
# ##### Comparison
# ############################


if cmp_flag:
    print(f' ')
    rel_err = np.linalg.norm(rec_ts_out_po-ts_p) / np.linalg.norm(rec_ts_out_po)
    print(f"Relative error parrallel vs. non parallel {rel_err:.8f}")