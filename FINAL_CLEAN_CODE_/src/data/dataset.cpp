// Dataset loading, vocabulary construction, and batch preparation implementation.
#include "data.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>

std::string read_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "Error: cannot open %s\n", path.c_str());
        exit(1);
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

void CharTokenizer::build(const std::string& text) {
    std::set<char> chars(text.begin(), text.end());
    id_to_char.assign(chars.begin(), chars.end());
    std::sort(id_to_char.begin(), id_to_char.end());
    vocab_size = (int)id_to_char.size();
    for (int i = 0; i < vocab_size; ++i)
        char_to_id[id_to_char[i]] = i;
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    ids.reserve(text.size());
    for (char c : text) {
        auto it = char_to_id.find(c);
        if (it != char_to_id.end())
            ids.push_back(it->second);
    }
    return ids;
}

std::string CharTokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    s.reserve(ids.size());
    for (int id : ids)
        if (id >= 0 && id < vocab_size)
            s.push_back(id_to_char[id]);
    return s;
}

void Dataset::load(const std::string& path, const CharTokenizer& tok,
                   int seq_len_, int batch_size_, bool verbose) {
    std::string text = read_file(path);
    tokens = tok.encode(text);
    seq_len = seq_len_;
    batch_size = batch_size_;
    cursor = 0;
    if (verbose) {
        printf("Dataset: %zu tokens, vocab_size=%d, seq_len=%d, batch_size=%d\n",
               tokens.size(), tok.vocab_size, seq_len, batch_size);
    }
}

bool Dataset::next_batch(std::vector<int>& input, std::vector<int>& target) {
    int chunk = seq_len + 1;
    int total_needed = batch_size * chunk;
    if (cursor + total_needed > (int)tokens.size())
        return false;

    input.resize(batch_size * seq_len);
    target.resize(batch_size * seq_len);

    for (int b = 0; b < batch_size; ++b) {
        int start = cursor + b * chunk;
        for (int t = 0; t < seq_len; ++t) {
            input[b * seq_len + t]  = tokens[start + t];
            target[b * seq_len + t] = tokens[start + t + 1];
        }
    }
    cursor += total_needed;
    return true;
}
