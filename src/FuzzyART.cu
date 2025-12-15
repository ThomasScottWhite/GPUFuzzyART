#include "../include/FuzzyART.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <algorithm>

__global__ void global_average_pooling_kernel(float *d_in, float *d_out, int Num_Channels, int H, int W)
{
    int channel = blockIdx.x;
    if (channel >= Num_Channels)
        return;

    int thread_id = threadIdx.x;
    int stride = blockDim.x;
    int num_pixels = H * W;

    float sum = 0.0f;
    for (int i = thread_id; i < num_pixels; i += stride)
    {
        sum += d_in[channel * num_pixels + i];
    }

    float val = sum / (float)num_pixels;
    atomicAdd(&d_out[channel], val);
}

void launch_gap_kernel(float *d_in, float *d_out, int Num_Channels, int H, int W)
{
    global_average_pooling_kernel<<<Num_Channels, 256>>>(d_in, d_out, Num_Channels, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA GAP Kernel Error: %s\n", cudaGetErrorString(err));
    }
}

__global__ void complement_coding_kernel(float *d_in, float *d_out, int original_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < original_dim)
    {
        float val = d_in[idx];
        if (val < 0.0f)
            val = 0.0f;
        if (val > 1.0f)
            val = 1.0f;

        d_out[idx] = val;
        d_out[idx + original_dim] = 1.0f - val;
    }
}

__global__ void calculate_categorical_choice(float *d_input, float *d_weights, float *device_cat_activations, int dim, int num_cats, float alpha, float vigilance, float norm_input)
{
    // Each thread computes the choice function for one category
    // And outputs -1.0f if vigilance test fails
    int cat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cat_idx >= num_cats)
        return;

    float norm_w = 0.0f;
    float norm_intersection = 0.0f;

    int offset = cat_idx * dim;

    for (int i = 0; i < dim; ++i)
    {
        float w = d_weights[offset + i];
        float in = d_input[i];

        norm_w += w;
        norm_intersection += fminf(in, w);
    }

    float match_score = norm_intersection / norm_input;

    if (match_score >= vigilance)
    {
        device_cat_activations[cat_idx] = norm_intersection / (alpha + norm_w);
    }
    else
    {
        device_cat_activations[cat_idx] = -1.0f;
    }
}

__global__ void update_weights_kernel(float *d_input, float *d_weights, int dim, int winner_idx, float beta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim)
    {
        int w_idx = winner_idx * dim + idx;
        float w_old = d_weights[w_idx];
        float in = d_input[idx];

        // w_new = beta * (I /\ w_old) + (1-beta) * w_old
        float intersection = fminf(in, w_old);
        d_weights[w_idx] = beta * intersection + (1.0f - beta) * w_old;
    }
}

__global__ void init_new_category_kernel(float *d_input, float *d_weights, int dim, int new_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim)
    {
        d_weights[new_idx * dim + idx] = d_input[idx];
    }
}

FuzzyART::FuzzyART(int input_dim, int max_categories, float vigilance, float choice_alpha, float learning_rate)
    : input_dim_(input_dim), max_categories_(max_categories),
      vigilance_(vigilance), choice_alpha_(choice_alpha), learning_rate_(learning_rate)
{
    art_dim_ = input_dim * 2;
    init_device_memory();
}

FuzzyART::~FuzzyART()
{
    if (d_weights_)
        cudaFree(d_weights_);
    if (device_cat_activations)
        cudaFree(device_cat_activations);
    if (d_input_compl_)
        cudaFree(d_input_compl_);
}

void FuzzyART::init_device_memory()
{
    cudaMalloc(&d_weights_, max_categories_ * art_dim_ * sizeof(float));
    cudaMalloc(&device_cat_activations, max_categories_ * sizeof(float));
    cudaMalloc(&d_input_compl_, art_dim_ * sizeof(float));

    cudaMemset(d_weights_, 0, max_categories_ * art_dim_ * sizeof(float));
}

int FuzzyART::run(float *d_input)
{
    // Complement Codes the input
    int threads = 256;
    int blocks = (input_dim_ + threads - 1) / threads;
    complement_coding_kernel<<<blocks, threads>>>(d_input, d_input_compl_, input_dim_);

    if (num_initialized_categories_ > 0)
    {
        // Calculate choice function for each category
        float norm_input = (float)input_dim_;
        int cat_threads = 256;
        int cat_blocks = (num_initialized_categories_ + cat_threads - 1) / cat_threads;

        calculate_categorical_choice<<<cat_blocks, cat_threads>>>(
            d_input_compl_, d_weights_, device_cat_activations, art_dim_, num_initialized_categories_,
            choice_alpha_, vigilance_, norm_input);

        // Copy category activations back to host to find max
        std::vector<float> cat_activations(num_initialized_categories_);
        cudaMemcpy(cat_activations.data(), device_cat_activations, num_initialized_categories_ * sizeof(float), cudaMemcpyDeviceToHost);

        auto max_iter = std::max_element(cat_activations.begin(), cat_activations.end());
        float max_val = *max_iter;
        int winner_idx = std::distance(cat_activations.begin(), max_iter);

        // If a valid winner, update its weights
        if (max_val > -0.5f)
        {
            int w_threads = 256;
            int w_blocks = (art_dim_ + w_threads - 1) / w_threads;
            update_weights_kernel<<<w_blocks, w_threads>>>(d_input_compl_, d_weights_, art_dim_, winner_idx, learning_rate_);
            return winner_idx;
        }
    }

    // If no winner found, initialize a new category if possible
    if (num_initialized_categories_ < max_categories_)
    {
        int new_idx = num_initialized_categories_;

        int w_threads = 256;
        int w_blocks = (art_dim_ + w_threads - 1) / w_threads;
        init_new_category_kernel<<<w_blocks, w_threads>>>(d_input_compl_, d_weights_, art_dim_, new_idx);

        num_initialized_categories_++;
        return new_idx;
    }

    return -1;
}