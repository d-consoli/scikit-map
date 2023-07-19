'''
Time series recostruction method based on seasonally waighted and renormalized convolution of the available images
'''
try:    
    import numpy as np
    import gc, os, warnings
    import pyfftw

    class SeasConv():    
        """
        :param ts: time seires of a pixel
        :param n_ipy: number of images per year
        :param att_seas: dB of attenuation for images of opposite seasonality
        :param att_env: dB of attenuation for temporarly far images
        :param n_cpu: number of CPU to be used in parallel
        :keep_orig_values: use true to keep the original value of the pixel when it is available; use False in case you want to obtain a smoother time series modifying also available pixels 
        """
        
        def __init__(s,
                     ts:np.array,
                     n_ipy:int,
                     att_seas:float = 40,
                     att_env:float = 30,
                     n_cpu:int = os.cpu_count(),
                     keep_orig_values:bool = True,
                    ):
            try:
                import mkl
                mkl.set_num_threads(n_cpu)
            except:
                pass            
            np.seterr(divide='ignore', invalid='ignore')
            s.keep_orig_values = keep_orig_values
            s.orig_shape = ts.shape
            s.n_cpu = n_cpu
            s.ts = np.reshape(ts,(ts.shape[0]*ts.shape[1],ts.shape[2])).T.copy()
            del ts
            gc.collect()
            s.valid_samp = ~np.isnan(s.ts)
            s.ts[~s.valid_samp] = 0.0
            s.n_imag = s.ts.shape[0]
            if n_ipy*2 > s.n_imag:
                warnings.warn("Not enough images available")
            assert (att_env/10 + att_seas/10) < np.finfo(np.double).precision, "Reduce the total attenuations to avoid numerical issues"
            s._compute_conv_mat_row(att_env, att_seas, n_ipy)
            
        def _compute_conv_mat_row(s, att_env, att_seas, n_ipy):            
            # Compute a triangular basis function with yaerly periodicity
            s.conv_mat_row = np.zeros((s.n_imag))
            base_func = np.zeros((n_ipy,))
            period_y = n_ipy/2.0
            slope_y = att_seas/10/period_y
            for i in np.arange(n_ipy):
                if i <= period_y:
                    base_func[i] = -slope_y*i
                else:
                    base_func[i] = slope_y*(i-period_y)-att_seas/10            
            # Compute the envelop to attenuate temporarly far images
            env_func = np.zeros((s.n_imag,))
            delta_e = s.n_imag
            slope_e = att_env/10/delta_e
            for i in np.arange(delta_e):
                env_func[i] = -slope_e*i
            s.conv_mat_row = 10.0**(np.resize(base_func,s.n_imag) + env_func)
            
            
        def get_conv_vec(s):
            conv_vec = np.zeros(2*s.n_imag-1)
            conv_vec[s.n_imag-1:] = s.conv_mat_row.copy()
            conv_vec[0:s.n_imag] = s.conv_mat_row.copy()[::-1]
            return conv_vec        
        
        def _fftw_toeplitz_matmul(self, data, valid_mask, conv_vec):
            plan = 'FFTW_EXHAUSTIVE'
            N_samp = conv_vec.shape[0]
            N_ext = N_samp*2
            N_fft = (np.floor(N_ext/2)+1).astype(int)
            N_imag = data.shape[1]
            in_ts_forward = pyfftw.empty_aligned((N_ext,N_imag), dtype='float32')
            out_ts_forward = pyfftw.empty_aligned((N_fft,N_imag), dtype='complex64')
            in_conv_forward = pyfftw.empty_aligned(N_ext, dtype='float32')
            out_conv_forward = pyfftw.empty_aligned(N_fft, dtype='complex64')
            out_conv_backward = pyfftw.empty_aligned(N_ext, dtype='float32')
            in_ts_backward = pyfftw.empty_aligned((N_fft,N_imag), dtype='complex64')
            out_ts_backward = pyfftw.empty_aligned((N_ext,N_imag), dtype='float32')
            plan_conv_forward = pyfftw.FFTW(in_conv_forward, out_conv_forward, axes=(0,), flags=(plan,), direction='FFTW_FORWARD',threads=self.n_cpu)
            plan_conv_backward = pyfftw.FFTW(out_conv_forward, out_conv_backward, axes=(0,), flags=(plan,), direction='FFTW_BACKWARD',threads=self.n_cpu)
            plan_ts_forward = pyfftw.FFTW(in_ts_forward, out_ts_forward, axes=(0,), flags=(plan,), direction='FFTW_FORWARD',threads=self.n_cpu)
            plan_ts_backward = pyfftw.FFTW(in_ts_backward, out_ts_backward, axes=(0,), flags=(plan,), direction='FFTW_BACKWARD',threads=self.n_cpu)
            in_conv_forward = np.zeros(N_ext)
            in_conv_forward[0:N_samp] = conv_vec
            in_conv_forward[N_samp:] = np.roll(conv_vec[::-1],1)
            plan_conv_forward(in_conv_forward)
            conv_fft = out_conv_forward.copy()
            in_conv_forward = np.zeros(N_ext)
            in_conv_forward[0:N_samp] = 1
            plan_conv_forward(in_conv_forward)  
            out_conv_forward = conv_fft * out_conv_forward
            plan_conv_backward(out_conv_forward) 
            in_ts_forward = np.concatenate((data,np.zeros((N_samp,N_imag))))
            plan_ts_forward(in_ts_forward)
            in_ts_backward = conv_fft.reshape(-1,1) * out_ts_forward
            plan_ts_backward(in_ts_backward)
            conv = out_ts_backward[0:N_samp,:].copy()
            in_ts_forward = np.concatenate((valid_mask,np.zeros((N_samp,N_imag))))
            plan_ts_forward(in_ts_forward)
            in_ts_backward = conv_fft.reshape(-1,1) * out_ts_forward
            plan_ts_backward(in_ts_backward)
            filled_qa = out_ts_backward[0:N_samp,:]
            filled = conv/filled_qa
            filled_qa /= out_conv_backward.reshape(-1,1)[0:N_samp] # Renormalization of the quality assesmtent vector
            return filled, filled_qa
        
        def run(s):
            # Convolution and normalization
            ts_rec, qa = s._fftw_toeplitz_matmul(s.ts, s.valid_samp.astype(float), s.conv_mat_row)
            if s.keep_orig_values:
                ts_rec[s.valid_samp] = s.ts[s.valid_samp]
                qa[s.valid_samp] = 1.0
            # Return the reconstructed time series and the quality assesment layer
            return np.reshape(ts_rec.T,s.orig_shape), np.reshape(qa.T,s.orig_shape) 
                        
except ImportError as e:
    from .misc import _warn_deps
    _warn_deps(e, 'seasconv')