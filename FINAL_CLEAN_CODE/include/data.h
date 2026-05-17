// dataset and token batching declarations.
#pragma once

#include "common.h"

struct CharTokenizer {
    std::unordered_map<char, int> char_to_id;
    std::vector<char> id_to_char;
    int vocab_size = 0;

    void build(const std::string& text);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
};

struct Dataset {
    std::vector<int> tokens;
    int seq_len    = 0;
    int batch_size = 0;
    int cursor     = 0;

    void load(const std::string& path, const CharTokenizer& tok,
              int seq_len, int batch_size, bool verbose = true);

    bool next_batch(std::vector<int>& input, std::vector<int>& target);

    void reset() { cursor = 0; }

    int num_batches() const {
        int tokens_per_batch = batch_size * (seq_len + 1);
        return ((int)tokens.size() - 1) / tokens_per_batch;
    }
};

std::string read_file(const std::string& path);
