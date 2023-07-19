#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <Eigen/Dense>

using NumpyMatReal = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using NumpyMatComplex = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

const std::complex<double> im(0.0, 1.0);

class WrapFFT {
private:
    unsigned int m_N_fft;
    unsigned int m_N_ext;
    unsigned int m_N_pix;
    double* m_in_forward;
    fftw_complex* m_out_forward;
    fftw_complex* m_in_backward;
    double* m_out_backward;
    fftw_plan m_plan_forward;
    fftw_plan m_plan_backward;
    double* m_ts_ext_data;
    double* m_mask_ext_data;    
    std::complex<double>* m_ts_ext_fft_data;
    std::complex<double>* m_mask_ext_fft_data;
    fftw_plan m_fftPlan_fw_ts;
    fftw_plan m_fftPlan_fw_mask;
    fftw_plan m_fftPlan_bw_conv_ts;
    fftw_plan m_fftPlan_bw_conv_mask;

    void computeMultipleFFT(fftw_plan fftPlan_fw, double* data, std::complex<double>* fft_data) {
        // Compute the forward transforms 
        for (unsigned int i = 0; i < m_N_pix; ++i) {
            fftw_execute_dft_r2c(fftPlan_fw, data + i * m_N_ext, reinterpret_cast<fftw_complex*>(fft_data) + i * m_N_fft);
        }
    }

    void computeMultipleIFFT(fftw_plan fftPlan_bw, std::complex<double>* fft_data, double* data) {
        // Compute the backward transforms 
        for (unsigned int i = 0; i < m_N_pix; ++i) {
            fftw_execute_dft_c2r(fftPlan_bw, reinterpret_cast<fftw_complex*>(fft_data) + i * m_N_fft, data + i * m_N_ext);
        }
    }

public:
    // Constructor
    WrapFFT(unsigned int N_fft,
            unsigned int N_ext,
            unsigned int N_pix,
            double* ts_ext_data,
            double* mask_ext_data, 
            std::complex<double>* ts_ext_fft_data, 
            std::complex<double>* mask_ext_fft_data) {

        m_N_fft = N_fft;
        m_N_ext = N_ext;
        m_N_pix = N_pix;
        auto fftw_flags = FFTW_EXHAUSTIVE;

        // Create input and output arrays
        m_ts_ext_data = ts_ext_data;
        m_mask_ext_data = mask_ext_data;    
        m_ts_ext_fft_data = reinterpret_cast<std::complex<double>*>(ts_ext_fft_data);
        m_mask_ext_fft_data = reinterpret_cast<std::complex<double>*>(mask_ext_fft_data);        
        m_in_forward = (double*)fftw_malloc(sizeof(double) * m_N_ext);
        m_out_forward = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * m_N_fft);
        m_in_backward = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * m_N_fft);
        m_out_backward = (double*)fftw_malloc(sizeof(double) * m_N_ext);

        // Create plans for forward and backward DFT (Discrete Fourier Transform)
        m_fftPlan_fw_ts = fftw_plan_dft_r2c_1d(m_N_ext, m_ts_ext_data, reinterpret_cast<fftw_complex*>(m_ts_ext_fft_data), fftw_flags);
        m_fftPlan_fw_mask = fftw_plan_dft_r2c_1d(m_N_ext, m_mask_ext_data, reinterpret_cast<fftw_complex*>(m_mask_ext_fft_data), fftw_flags);
        m_fftPlan_bw_conv_ts = fftw_plan_dft_c2r_1d(m_N_ext, reinterpret_cast<fftw_complex*>(m_ts_ext_fft_data), m_ts_ext_data, fftw_flags);
        m_fftPlan_bw_conv_mask = fftw_plan_dft_c2r_1d(m_N_ext, reinterpret_cast<fftw_complex*>(m_mask_ext_fft_data), m_mask_ext_data, fftw_flags);
        m_plan_forward = fftw_plan_dft_r2c_1d(N_ext, m_in_forward, m_out_forward, fftw_flags);
        m_plan_backward = fftw_plan_dft_c2r_1d(N_ext, m_in_backward, m_out_backward, fftw_flags);
    }

    void computeFFT(const Eigen::VectorXd& vec_in, Eigen::Ref<Eigen::VectorXcd> vec_out) {
        // Compute the forward transform
        for (unsigned int i = 0; i < m_N_ext; ++i) {
            m_in_forward[i] = vec_in(i);
        }
        fftw_execute(m_plan_forward);
        for (unsigned int i = 0; i < m_N_fft; ++i) {
            vec_out(i) = m_out_forward[i][0] + m_out_forward[i][1]*im;
        }
    }

    void computeIFFT(const Eigen::VectorXcd& vec_in, Eigen::Ref<Eigen::VectorXd> vec_out) {
        // Compute the backward transform
        for (unsigned int i = 0; i < m_N_fft; ++i) {
            m_in_backward[i][0] = vec_in[i].real()/m_N_ext;
            m_in_backward[i][1] = vec_in[i].imag()/m_N_ext;
        }
        fftw_execute(m_plan_backward);
        for (unsigned int i = 0; i < m_N_ext; ++i) {
            vec_out(i) = m_out_backward[i];
        }
    }

    void computeTimeserisFFT() {
        // Compute the forward transforms for the time series
        computeMultipleFFT(m_fftPlan_fw_ts, m_ts_ext_data, m_ts_ext_fft_data);
    }

    void computeMaskFFT() {
        // Compute the forward transforms for the time series
        computeMultipleFFT(m_fftPlan_fw_mask, m_mask_ext_data, m_mask_ext_fft_data);
    }

    void computeTimeserisIFFT() {
        // Compute the forward transforms for the time series
        computeMultipleIFFT(m_fftPlan_bw_conv_ts, m_ts_ext_fft_data, m_ts_ext_data);
    }

    void computeMaskIFFT() {
        // Compute the forward transforms for the time series
        computeMultipleIFFT(m_fftPlan_bw_conv_mask, m_mask_ext_fft_data, m_mask_ext_data);
    }

    void clean() {
         // Clean up
        fftw_destroy_plan(m_plan_forward);
        fftw_free(m_in_forward);
        fftw_free(m_out_forward);
        fftw_destroy_plan(m_plan_backward);
        fftw_free(m_in_backward);
        fftw_free(m_out_backward);
        fftw_destroy_plan(m_fftPlan_bw_conv_ts);
        fftw_destroy_plan(m_fftPlan_bw_conv_mask);
        fftw_destroy_plan(m_fftPlan_fw_ts);
        fftw_destroy_plan(m_fftPlan_fw_mask);
    }
};


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

    const float delta_e = N_samp;
    const float slope_e = att_env/10./delta_e;
    for (unsigned int i = 0; i < N_samp; ++i) {
        env_func(i) = -slope_e*(float)i;
    }   

    for (unsigned int i = 0; i < N_samp; ++i) {
        conv(i) = std::pow(10., base_func[i%N_ipy] + env_func[i]);
    }
}

// @FIXME move to fftwf and matricies float (maybe templatizing it)
extern "C"
int run(const unsigned int N_years,
    const unsigned int N_ipy,
    const unsigned int N_pix,
    const float att_seas,
    const float att_env,
    double* ts_in,
    double* rec_ts_out,
    double* qa_out) {

    const unsigned int N_samp = N_years * N_ipy;
    const unsigned int N_ext = N_samp * 2;
    const unsigned int N_fft = N_samp + 1;

    // Link the input/output data to Eigen matrices
    Eigen::Map<NumpyMatReal> ts(ts_in, N_pix, N_samp);
    Eigen::Map<NumpyMatReal> rec_ts(rec_ts_out, N_pix, N_samp);
    Eigen::Map<NumpyMatReal> qa(qa_out, N_pix, N_samp);

    // Create needed variables
    NumpyMatComplex ts_ext_fft(N_pix, N_fft);
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mask = ts.array().isNaN(); // Validity mask
    NumpyMatReal mask_ext = NumpyMatReal::Zero(N_pix, N_ext);
    Eigen::VectorXd conv(N_samp);
    Eigen::VectorXd conv_ext = Eigen::VectorXd::Ones(N_ext);
    Eigen::VectorXcd conv_ext_fft(N_fft);
    Eigen::VectorXd norm_qa = Eigen::VectorXd::Ones(N_ext);
    Eigen::VectorXcd norm_qa_fft(N_fft);
    NumpyMatReal ts_ext = NumpyMatReal::Zero(N_pix, N_ext);
    NumpyMatComplex mask_ext_fft(N_pix, N_fft);

    // Plan the FFT computation
    WrapFFT wrapFFT = WrapFFT(N_fft, N_ext, N_pix, ts_ext.data(), mask_ext.data(), ts_ext_fft.data(), mask_ext_fft.data());

    // Preparing the inputs for the comnputation
    ts = mask.select(0.0, ts);
    ts_ext.block(0, 0, N_pix, N_samp) = ts;
    mask = !mask.array(); // Switch from a NaN mask to a validity mask for the next steps
    mask_ext.block(0, 0, N_pix, N_samp) = mask.cast<double>();
    compute_conv_vec(conv, N_years, N_ipy, att_seas, att_env);
    conv_ext.segment(0,N_samp) = conv;
    conv_ext.segment(N_samp+1,N_samp-1) = conv.reverse().segment(0,N_samp-1);
    norm_qa.segment(N_samp,N_samp) = Eigen::VectorXd::Zero(N_samp);

    // Compute forward transformations
    wrapFFT.computeTimeserisFFT();
    wrapFFT.computeMaskFFT();
    wrapFFT.computeFFT(conv_ext, conv_ext_fft);
    wrapFFT.computeFFT(norm_qa, norm_qa_fft);

    // Convolve the vectors
    ts_ext_fft.array().rowwise() *= conv_ext_fft.array().transpose();
    mask_ext_fft.array().rowwise() *= conv_ext_fft.array().transpose();
    norm_qa_fft.array() *= conv_ext_fft.array();

    // Compute forward transformations    
    wrapFFT.computeTimeserisIFFT();
    wrapFFT.computeMaskIFFT();
    wrapFFT.computeIFFT(norm_qa_fft, norm_qa);
    wrapFFT.clean();

    // Renormalize the result
    rec_ts = ts_ext.block(0, 0, N_pix, N_samp).array() / mask_ext.block(0, 0, N_pix, N_samp).array();
    qa = mask_ext.block(0, 0, N_pix, N_samp).array().rowwise() / norm_qa.segment(0,N_samp).array().transpose() / N_ext;

    return 0;
}
