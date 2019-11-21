#include <fstream>
#include <iostream>
#include <vector>

// todo: A reader compatible image and text

class TxtDataReader {
public:
  TxtDataReader(const std::string &filename, int64_t batch_size,
                uint64_t inputs_size = 1, uint64_t start = 0,
                uint64_t end = UINT64_MAX)
      : _batch_size(batch_size), _total_length(0), _inputs_size(inputs_size) {
    _file.open(filename);
    _file.seekg(_file.end);
    _total_length = _file.tellg();
    _file.seekg(_file.beg);
    read_file_to_vec(start, end);
  }

  void reset_current_line() { _current_line = 0; }

  const std::vector<std::string> &get_lines() { return _lines; }

  void read_file_to_vec(const size_t start, const size_t end) {
    std::string line;
    size_t count = 0;
    _lines.clear();
    while (std::getline(_file, line)) {
      if (count >= start && count <= end) {
        _lines.push_back(line);
      }
      count++;
    }
  }

  bool get_next_batch(std::vector<std::vector<float> > *data, char delim = ';',
                      char delim2 = ' ', bool drop=false) {
    data->clear();
    data->resize(_inputs_size);
    int size = 0;
    bool rest = false;
    while (_current_line < _lines.size()) {
      std::vector<std::string> line;
      split(_lines[_current_line], delim, &line);
      for (size_t i = 0; i < line.size(); ++i) {
        std::vector<float> num;
        split_to_numeric(line[i], delim2, &num);
        (*data)[i].insert((*data)[i].end(), num.begin(), num.end());
      }
      ++size;
      ++_current_line;
      if (size >= _batch_size) {
        return true;
      } else {
        rest = size > 0;
      }
    }
    if (drop) {
      return false;
    } else {
      return rest;
    }
  }

private:
  void split(const std::string &str, char sep, std::vector<std::string> *pieces,
             bool ignore_null = true) {
    pieces->clear();
    if (str.empty()) {
      if (!ignore_null) {
        pieces->push_back(str);
      }
      return;
    }
    size_t pos = 0;
    size_t next = str.find(sep, pos);
    while (next != std::string::npos) {
      pieces->push_back(str.substr(pos, next - pos));
      pos = next + 1;
      next = str.find(sep, pos);
    }
    if (!str.substr(pos).empty()) {
      pieces->push_back(str.substr(pos));
    }
  }

  void split_to_numeric(const std::string &str, char sep,
                        std::vector<float> *fs) {
    std::vector<std::string> pieces;
    split(str, sep, &pieces);
    std::transform(pieces.begin(), pieces.end(), std::back_inserter(*fs),
                   [](const std::string &v) { return std::stof(v); });
  }

private:
  std::fstream _file;
  int64_t _batch_size;
  uint64_t _total_length;
  uint64_t _inputs_size;
  uint64_t _current_line;
  std::vector<std::string> _lines;
};


// void ReadBinaryFile(const std::string &filename, std::string *contents) {
//   std::ifstream fin(filename, std::ios::in | std::ios::binary);
//   CHECK(fin.is_open()) << "Cannot open file: " << filename;
//   fin.seekg(0, std::ios::end);
//   auto size = fin.tellg();
//   contents->clear();
//   contents->resize(size);
//   fin.seekg(0, std::ios::beg);
//   fin.read(&(contents->at(0)), contents->size());
//   fin.close();
// }