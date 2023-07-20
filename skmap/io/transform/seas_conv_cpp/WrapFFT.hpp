#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <Eigen/Dense>


template <class T>
class WrapFFT {
private:
    
    unsigned int m_N_fft;
    unsigned int m_N_ext;
    unsigned int m_N_pix;
    T* m_in_forward;
    fftw_complex* m_out_forward;
    fftw_complex* m_in_backward;
    T* m_out_backward;
    fftw_plan m_plan_forward;
    fftw_plan m_plan_backward;
    T* m_ts_ext_data;
    T* m_mask_ext_data;    
    std::complex<T>* m_ts_ext_fft_data;
    std::complex<T>* m_mask_ext_fft_data;
    fftw_plan m_fftPlan_fw_ts;
    fftw_plan m_fftPlan_fw_mask;
    fftw_plan m_fftPlan_bw_conv_ts;
    fftw_plan m_fftPlan_bw_conv_mask;

    void computeMultipleFFT(fftw_plan fftPlan_fw, T* data, std::complex<T>* fft_data) {
        // Compute the forward transforms 
        for (unsigned int i = 0; i < m_N_pix; ++i) {
            fftw_execute_dft_r2c(fftPlan_fw, data + i * m_N_ext, reinterpret_cast<fftw_complex*>(fft_data) + i * m_N_fft);
        }
    }

    void computeMultipleIFFT(fftw_plan fftPlan_bw, std::complex<T>* fft_data, T* data) {
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
            T* ts_ext_data,
            T* mask_ext_data, 
            std::complex<T>* ts_ext_fft_data, 
            std::complex<T>* mask_ext_fft_data) {

        m_N_fft = N_fft;
        m_N_ext = N_ext;
        m_N_pix = N_pix;
        auto fftw_flags = FFTW_EXHAUSTIVE;
        
        // Save the first row of the inputs since the planning could modify them
        T tmp_ts_ext_data[m_N_ext];
        T tmp_mask_ext_data[m_N_ext];
        for(unsigned int i = 0; i < m_N_ext; ++i) {
            tmp_ts_ext_data[i] = ts_ext_data[i];
            tmp_mask_ext_data[i] = mask_ext_data[i];
        }

        // Create input and output arrays
        m_ts_ext_data = ts_ext_data;
        m_mask_ext_data = mask_ext_data;    
        m_ts_ext_fft_data = reinterpret_cast<std::complex<T>*>(ts_ext_fft_data);
        m_mask_ext_fft_data = reinterpret_cast<std::complex<T>*>(mask_ext_fft_data);        
        m_in_forward = (T*)fftw_malloc(sizeof(T) * m_N_ext);
        m_out_forward = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * m_N_fft);
        m_in_backward = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * m_N_fft);
        m_out_backward = (T*)fftw_malloc(sizeof(T) * m_N_ext);

        // Create plans for forward and backward DFT (Discrete Fourier Transform)
        m_fftPlan_fw_ts = fftw_plan_dft_r2c_1d(m_N_ext, m_ts_ext_data, reinterpret_cast<fftw_complex*>(m_ts_ext_fft_data), fftw_flags);
        m_fftPlan_fw_mask = fftw_plan_dft_r2c_1d(m_N_ext, m_mask_ext_data, reinterpret_cast<fftw_complex*>(m_mask_ext_fft_data), fftw_flags);
        m_fftPlan_bw_conv_ts = fftw_plan_dft_c2r_1d(m_N_ext, reinterpret_cast<fftw_complex*>(m_ts_ext_fft_data), m_ts_ext_data, fftw_flags);
        m_fftPlan_bw_conv_mask = fftw_plan_dft_c2r_1d(m_N_ext, reinterpret_cast<fftw_complex*>(m_mask_ext_fft_data), m_mask_ext_data, fftw_flags);
        m_plan_forward = fftw_plan_dft_r2c_1d(N_ext, m_in_forward, m_out_forward, fftw_flags);
        m_plan_backward = fftw_plan_dft_c2r_1d(N_ext, m_in_backward, m_out_backward, fftw_flags);
        
        for(unsigned int i = 0; i < m_N_ext; ++i) {
            ts_ext_data[i] = tmp_ts_ext_data[i];
            mask_ext_data[i] = tmp_mask_ext_data[i];
        }
    }

    void computeFFT(const Eigen::VectorXd& vec_in, Eigen::Ref<Eigen::VectorXcd> vec_out) {
        // Compute the forward transform
        for (unsigned int i = 0; i < m_N_ext; ++i) {
            m_in_forward[i] = vec_in(i);
        }
        fftw_execute(m_plan_forward);
        const std::complex<T> im(0.0, 1.0);
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
