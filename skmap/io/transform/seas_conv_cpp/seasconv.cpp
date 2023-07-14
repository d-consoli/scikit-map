#include <iostream>
#include <fftw3.h>
#include <chrono>
#include <cmath>
#include <Eigen/Dense>

using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::high_resolution_clock;

void compute_conv_vec(Eigen::VectorXd& conv,
    const unsigned int N_years,
    const unsigned int N_ipy,
    const float att_seas,
    const float att_env) {

    const unsigned int N_samp = N_years * N_ipy;
    Eigen::VectorXd base_func = Eigen::VectorXd::Zero(N_ipy);
    Eigen::VectorXd env_func = Eigen::VectorXd::Zero(N_samp);
    const float period_y = (float) N_ipy / 2.;
    const float slope_y = att_seas/10./period_y;

    for (unsigned int i = 0; i < N_ipy; ++i) {
        if (i <= period_y) 
            base_func(i) = -slope_y*(float)i;
        else
            base_func(i) = slope_y*((float)i-period_y) - att_seas/10.;
    }

    const float delta_e = N_ipy;
    const float slope_e = att_env/10./delta_e;
    for (unsigned int i = 0; i < N_samp; ++i) {
        env_func(i) = -slope_e*(float)i;
    }   

    for (unsigned int i = 0; i < N_samp; ++i) {
        conv(i) = std::pow(10., base_func[i%N_ipy] + env_func[i]);
    }
}


extern "C"
int run(const unsigned int N_years,
    const unsigned int N_ipy,
    const unsigned int N_pix,
    const float att_seas,
    const float att_env,
    const double* ts_in,
    double* rec_ts_out,
    double* qa_out) {


    std::cout << "Num pixels " << N_pix << '\n';

    const unsigned int N_samp = N_years * N_ipy;
    const unsigned int N_ext = N_samp * 2;
    const unsigned int N_fft = N_samp + 1;
    const std::complex<double> im(0.0, 1.0);   

    Eigen::VectorXd conv(N_samp);
    Eigen::VectorXd ts(N_samp);
    Eigen::Vector<bool, Eigen::Dynamic> mask(N_samp);

    compute_conv_vec(conv, N_years, N_ipy, att_seas, att_env);
    for (unsigned int i = 0; i < N_samp; ++i) {
        ts[i] = ts_in[i];
    }
    mask = ts.array().isNaN();
    ts = mask.select(0.0, ts);
    mask = !mask.array(); // Switch from a NaN mask to a validity mask for the next steps

    // Print the results
    std::cout << "--------- Input conv:\n";
    for (unsigned int i = 0; i < N_samp; ++i) {
        std::cout << conv[i] << "\n";
    }

    Eigen::VectorXd conv_ext = Eigen::VectorXd::Zero(N_ext);
    Eigen::VectorXd ts_ext = Eigen::VectorXd::Zero(N_ext);
    Eigen::VectorXd mask_ext = Eigen::VectorXd::Zero(N_ext);
    conv_ext.segment(0,N_samp) = conv;
    conv_ext.segment(N_samp+1,N_samp-1) = conv.reverse().segment(0,N_samp-1);
    ts_ext.segment(0,N_samp) = ts;
    mask_ext.segment(0,N_samp) = mask.cast<double>();
    Eigen::VectorXd conv_ts_ext(N_ext);
    Eigen::VectorXd conv_mask_ext(N_ext);
    Eigen::VectorXd rec_ts_ext(N_ext);
    Eigen::VectorXd rec_ts(N_samp);
    Eigen::VectorXd qa(N_samp);
    Eigen::VectorXcd conv_ext_fft(N_fft);
    Eigen::VectorXcd ts_ext_fft(N_fft);
    Eigen::VectorXcd mask_ext_fft(N_fft);
    Eigen::VectorXcd conv_ts_ext_fft(N_fft);
    Eigen::VectorXcd conv_mask_ext_fft(N_fft);

    std::cout << "Start\n";
    auto t0 = high_resolution_clock::now();

    // Create input and output arrays
    double* in_ts_forward = (double*)fftw_malloc(sizeof(double) * N_ext);
    fftw_complex* out_ts_forward = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N_fft);
    fftw_complex* in_ts_backward = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N_fft);
    double* out_ts_backward = (double*)fftw_malloc(sizeof(double) * N_ext);

    // Create plans for forward and backward DFT (Discrete Fourier Transform)
    fftw_plan plan_ts_forward = fftw_plan_dft_r2c_1d(N_ext, in_ts_forward, out_ts_forward, FFTW_MEASURE);
    fftw_plan plan_ts_backward = fftw_plan_dft_c2r_1d(N_ext, in_ts_backward, out_ts_backward, FFTW_MEASURE);
    

    // @FIXME create a function that perform the FFT/IFFT with one input vector and one output vector,
    // check that the filling of the vectors is perfromed in SIMD
    // call that function with inline directives

    // Compute FFT of the convolution vector
    for (unsigned int i = 0; i < N_ext; ++i) {
        in_ts_forward[i] = conv_ext[i];
    }
    fftw_execute(plan_ts_forward);
    for (unsigned int i = 0; i < N_fft; ++i) {
        conv_ext_fft[i] = out_ts_forward[i][0] + out_ts_forward[i][1]*im;
    }

    // Compute FFT of the time-series
    for (unsigned int i = 0; i < N_ext; ++i) {
        in_ts_forward[i] = ts_ext[i];
    }
    fftw_execute(plan_ts_forward);
    for (unsigned int i = 0; i < N_fft; ++i) {
        ts_ext_fft[i] = out_ts_forward[i][0] + out_ts_forward[i][1]*im;
    }

    // Compute FFT of the validity mask
    for (unsigned int i = 0; i < N_ext; ++i) {
        in_ts_forward[i] = mask_ext[i];
    }
    fftw_execute(plan_ts_forward);
    for (unsigned int i = 0; i < N_fft; ++i) {
        mask_ext_fft[i] = out_ts_forward[i][0] + out_ts_forward[i][1]*im;
    }

    // Convolve the vectors
    conv_ts_ext_fft = conv_ext_fft.array() * ts_ext_fft.array();
    conv_mask_ext_fft = conv_ext_fft.array() * mask_ext_fft.array();

    // Compute IFFT of the time-series
    for (unsigned int i = 0; i < N_fft; ++i) {
        in_ts_backward[i][0] = conv_ts_ext_fft[i].real()/N_ext;
        in_ts_backward[i][1] = conv_ts_ext_fft[i].imag()/N_ext;
    }
    fftw_execute(plan_ts_backward);
    for (unsigned int i = 0; i < N_ext; ++i) {
        conv_ts_ext[i] = out_ts_backward[i];
    }

    // Compute IFFT of the validity mask
    for (unsigned int i = 0; i < N_fft; ++i) {
        in_ts_backward[i][0] = conv_mask_ext_fft[i].real()/N_ext;
        in_ts_backward[i][1] = conv_mask_ext_fft[i].imag()/N_ext;
    }
    fftw_execute(plan_ts_backward);
    for (unsigned int i = 0; i < N_ext; ++i) {
        conv_mask_ext[i] = out_ts_backward[i];
    }

    // Renormalize the result
    rec_ts = conv_ts_ext.segment(0,N_samp).array() / conv_mask_ext.segment(0,N_samp).array();
    qa = conv_mask_ext.segment(0,N_samp) / conv.sum();

    // #############################################################33

    auto t3 = high_resolution_clock::now();
    duration<double> ms2_double = (t3 - t0);
    std::cout << ms2_double.count() << "s Doneee \n";

    // Print the results
    std::cout << "--------- Output rec_ts:\n";
    for (unsigned int i = 0; i < N_samp; ++i) {
        std::cout << rec_ts[i] << "\n";
        rec_ts_out[i] = rec_ts[i];
        qa_out[i] = qa[i];
    }


    // Clean up
    fftw_destroy_plan(plan_ts_forward);
    fftw_free(in_ts_forward);
    fftw_free(out_ts_forward);
    fftw_destroy_plan(plan_ts_backward);
    fftw_free(in_ts_backward);
    fftw_free(out_ts_backward);

    return 0;
}
