#include "../include/CPUFuzzyART.h"
#include <numeric>
#include <iostream>
#include <limits>

CPUFuzzyART::CPUFuzzyART(int input_dim, int max_categories, float vigilance, float choice_alpha, float learning_rate)
    : input_dim_(input_dim), max_categories_(max_categories),
      vigilance_(vigilance), choice_alpha_(choice_alpha), learning_rate_(learning_rate)
{

    art_dim_ = input_dim * 2;
    // Reserve memory to prevent reallocations, but size is 0 initially
    weights_.reserve(max_categories);
}

int CPUFuzzyART::run(const std::vector<float> &input)
{
    // Complement Code Input
    if ((int)input.size() != input_dim_)
    {
        std::cerr << "[Error] CPUFuzzyART input dimension mismatch." << std::endl;
        return -1;
    }

    std::vector<float> I(art_dim_);
    for (int i = 0; i < input_dim_; ++i)
    {
        float val = input[i];
        if (val < 0.0f)
            val = 0.0f;
        if (val > 1.0f)
            val = 1.0f;

        I[i] = val;
        I[i + input_dim_] = 1.0f - val;
    }

    // Choice Function Calculation
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(num_initialized_categories_);

    for (int j = 0; j < num_initialized_categories_; ++j)
    {
        float norm_intersection = 0.0f;
        float norm_w = 0.0f;

        for (int k = 0; k < art_dim_; ++k)
        {
            float w = weights_[j][k];
            norm_intersection += std::min(I[k], w);
            norm_w += w;
        }

        float T = norm_intersection / (choice_alpha_ + norm_w);
        candidates.push_back({T, j});
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto &a, const auto &b)
              { return a.first > b.first; });

    // Resonance Check
    float norm_I = (float)input_dim_;

    for (const auto &cand : candidates)
    {
        int idx = cand.second;

        float norm_intersection = 0.0f;
        for (int k = 0; k < art_dim_; ++k)
        {
            norm_intersection += std::min(I[k], weights_[idx][k]);
        }

        float match_score = norm_intersection / norm_I;

        // Vigilance Check
        if (match_score >= vigilance_)
        {
            // Update Weights
            for (int k = 0; k < art_dim_; ++k)
            {
                float w_old = weights_[idx][k];
                float intersection = std::min(I[k], w_old);
                weights_[idx][k] = learning_rate_ * intersection + (1.0f - learning_rate_) * w_old;
            }
            return idx;
        }
    }

    // Create New Category if no match found
    if (num_initialized_categories_ < max_categories_)
    {
        weights_.push_back(I);
        int new_idx = num_initialized_categories_;
        num_initialized_categories_++;
        return new_idx;
    }

    return -1;
}