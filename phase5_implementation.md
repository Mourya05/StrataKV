# StrataKV Phase 5: mapped checkpoints and autoregressive generation

This is a **documentation-only deliverable**: copy the code blocks to their
labeled paths when applying Phase 5. No application sources, local configuration,
or existing build artifacts are changed by creating this document.

## Scope and acceptance criteria

- Open a single `.safetensors` file using Windows file mapping or POSIX `mmap`.
  Parse and validate its little-endian length, bounded JSON header, names,
  FP32/BF16 dtypes, shapes, and relative byte offsets without a JSON library.
- Return `std::unordered_map<std::string, Tensor>` with safe storage lifetimes,
  including after an individual tensor is moved out of the map.
- Use zero-copy mapped FP32 storage **when compatible with the backend's
  alignment and endianness requirements**. Convert BF16 correctly into FP32.
- Load all parameters used by the current one-block architecture, including
  embeddings, final normalization, and the language-model head.
- Prefill the prompt once; repeatedly sample logits, embed the sampled ID, and
  decode one new token using the cached prefix. Stop at EOS or the requested
  limit, and reclaim pages on success and failure.
- Reject invalid input/checkpoints rather than silently generating from dummy
  weights. Preserve CPU-only/CUDA GEMM dispatch without external dependencies.

## Essential compatibility decisions

1. **A Tensor change is required.** The current destructor always calls
   `_aligned_free`, so pointing it into a mapped file would be invalid. The
   replacement below retains move-only semantics and shares a storage owner.
2. **BF16 bytes are not `float` objects.** They cannot be directly exposed as a
   `float*` to the current backend. BF16 tensors are expanded once into aligned
   FP32 allocations. No BF16 arithmetic support is claimed.
3. **Safetensors does not promise 64-byte tensor alignment.** FP32 tensors that
   are not suitably aligned, or a big-endian host, fall back to one aligned
   conversion/copy. `require_mapped_f32=true` rejects this fallback for F32.
   Mapping uses private, writable copy-on-write pages, so mutable Tensor APIs
   cannot write through to the checkpoint file. CPU page faults and CUDA
   transfers are still real costs; mapping is not GPU residency.
4. **Exactly one component appends tokens.** Current
   `PagedAttention::forward` already calls `page_table.append_token()` for each
   row it caches. Generator delegates that action through TransformerBlock and
   never appends the same token manually. The final sampled token need not be
   forwarded if generation is stopping and the sequence is being freed.
5. **Real weights must match the actual architecture.** This engine currently
   has one pre-norm block, full multi-head Q/K/V, a two-linear GELU FFN, and no
   positional encoding. It cannot run arbitrary Llama or other multi-layer,
   RoPE/GQA/SwiGLU checkpoints by renaming keys. The exact supported schema is
   specified below. Inputs/outputs are token IDs: a tokenizer, text rendering,
   sharded checkpoint index, and model-config JSON reader are not supplied.

## Required Tensor storage adaptation

Replace these two existing files when applying this implementation. Owned
allocations remain 64-byte aligned; mapped aliases must also be 64-byte aligned.
This version additionally handles safetensors scalar shape `[]` (one element)
and zero-sized dimensions (zero elements), checks overflow, and uses portable
aligned allocation. Existing layers must still receive positive-size matrices;
the generator and sampler enforce that contract.

## `include/stratakv/tensor.h`

```cpp
#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace stratakv {
class Tensor {
public:
    explicit Tensor(const std::vector<std::size_t>& shape);
    ~Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Lifetime and alignment are checked. The owner must keep the entire
    // referenced float range alive; data may be private copy-on-write storage.
    static Tensor from_external(const std::vector<std::size_t>& shape,
                                float* data, std::shared_ptr<void> owner);
    float* data() noexcept { return data_ptr_; }
    const float* data() const noexcept { return data_ptr_; }
    std::size_t num_elements() const noexcept { return total_elements_; }
    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    const std::vector<std::size_t>& strides() const noexcept { return strides_; }
    std::size_t ndim() const noexcept { return shape_.size(); }
    std::size_t get_flat_index(const std::vector<std::size_t>& idx) const;
    float& operator[](std::size_t i) noexcept { return data_ptr_[i]; }
    float operator[](std::size_t i) const noexcept { return data_ptr_[i]; }

private:
    struct AliasTag {};
    Tensor(const std::vector<std::size_t>& shape, float* data,
           std::shared_ptr<void> owner, AliasTag);
    void compute_strides();
    float* data_ptr_ = nullptr;
    std::size_t total_elements_ = 0;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::shared_ptr<void> owner_;
};
} // namespace stratakv
```

## `src/cpu/tensor.cpp`

```cpp
#include "stratakv/tensor.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace stratakv {
namespace {
void release_aligned(void* p) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
} // namespace

void Tensor::compute_strides() {
    const auto limit = static_cast<std::size_t>(
        (std::numeric_limits<std::ptrdiff_t>::max)()) / sizeof(float);
    strides_.resize(shape_.size());
    std::size_t count = 1;
    for (std::size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = count;
        if (shape_[i] != 0 && count > limit / shape_[i])
            throw std::overflow_error("Tensor: size/stride overflow");
        count *= shape_[i];
    }
    total_elements_ = count;
}

Tensor::Tensor(const std::vector<std::size_t>& shape) : shape_(shape) {
    compute_strides();
    const std::size_t bytes = (std::max)(total_elements_, std::size_t{1}) * sizeof(float);
    void* allocation = nullptr;
#ifdef _WIN32
    allocation = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&allocation, 64, bytes) != 0) allocation = nullptr;
#endif
    if (!allocation) throw std::bad_alloc();
    // shared_ptr invokes the deleter even if its control-block allocation fails.
    owner_ = std::shared_ptr<void>(allocation, release_aligned);
    data_ptr_ = static_cast<float*>(allocation);
    std::memset(data_ptr_, 0, bytes);
}

Tensor::Tensor(const std::vector<std::size_t>& shape, float* data,
               std::shared_ptr<void> owner, AliasTag)
    : data_ptr_(data), shape_(shape), owner_(std::move(owner)) {
    compute_strides();
    if (!owner_ || !data || reinterpret_cast<std::uintptr_t>(data) % 64 != 0)
        throw std::invalid_argument("Tensor: invalid owner or unaligned alias");
}

Tensor Tensor::from_external(const std::vector<std::size_t>& shape,
                             float* data, std::shared_ptr<void> owner) {
    return Tensor(shape, data, std::move(owner), AliasTag{});
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_ptr_(std::exchange(other.data_ptr_, nullptr)),
      total_elements_(std::exchange(other.total_elements_, 0)),
      shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      owner_(std::move(other.owner_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        owner_ = std::move(other.owner_);
        data_ptr_ = std::exchange(other.data_ptr_, nullptr);
        total_elements_ = std::exchange(other.total_elements_, 0);
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
    }
    return *this;
}

std::size_t Tensor::get_flat_index(const std::vector<std::size_t>& idx) const {
    if (idx.size() != shape_.size())
        throw std::invalid_argument("Tensor: index rank mismatch");
    std::size_t flat = 0;
    for (std::size_t i = 0; i < idx.size(); ++i) {
#ifndef NDEBUG
        if (idx[i] >= shape_[i])
            throw std::out_of_range("Tensor: index at dimension " + std::to_string(i));
#endif
        flat += idx[i] * strides_[i];
    }
    return flat;
}
} // namespace stratakv
```

<!-- phase5:loader -->

## `include/stratakv/io/safetensors.h`

```cpp
#pragma once
#include "stratakv/tensor.h"
#include <filesystem>
#include <string>
#include <unordered_map>

namespace stratakv::io {
using WeightMap = std::unordered_map<std::string, Tensor>;

// F32 is aliased if little-endian and 64-byte aligned, otherwise copied.
// BF16 is always expanded to FP32. true rejects non-mappable F32 tensors.
// All returned aliases retain the mapping through Tensor's storage owner.
WeightMap load_safetensors(const std::filesystem::path& path,
                          bool require_mapped_f32 = false);
} // namespace stratakv::io
```

## `src/io/safetensors.cpp`

```cpp
#include "stratakv/io/safetensors.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace stratakv::io {
namespace {
[[noreturn]] void fail(const std::string& reason) {
    throw std::runtime_error("safetensors: " + reason);
}

struct MappedFile {
    unsigned char* data = nullptr;
    std::size_t size = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int file = -1;
#endif
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() noexcept {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
        if (data) munmap(data, size);
        if (file >= 0) close(file);
#endif
    }
};

std::shared_ptr<MappedFile> map_file(const std::filesystem::path& path) {
    auto m = std::make_shared<MappedFile>();
    std::uintmax_t length = 0;
#ifdef _WIN32
    m->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (m->file == INVALID_HANDLE_VALUE) fail("CreateFileW failed");
    LARGE_INTEGER bytes;
    if (!GetFileSizeEx(m->file, &bytes) || bytes.QuadPart < 0)
        fail("GetFileSizeEx failed");
    length = static_cast<std::uintmax_t>(bytes.QuadPart);
#else
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    m->file = open(path.c_str(), flags);
    if (m->file < 0) fail("open failed: errno=" + std::to_string(errno));
    struct stat status {};
    if (fstat(m->file, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0)
        fail("fstat failed or file is not regular");
    length = static_cast<std::uintmax_t>(status.st_size);
#endif
    if (length < 8 || length > static_cast<std::uintmax_t>(
            (std::numeric_limits<std::ptrdiff_t>::max)()))
        fail("file too small or too large for this address space");
    m->size = static_cast<std::size_t>(length);
#ifdef _WIN32
    m->mapping = CreateFileMappingW(m->file, nullptr, PAGE_WRITECOPY, 0, 0, nullptr);
    if (!m->mapping) fail("CreateFileMappingW failed");
    m->data = static_cast<unsigned char*>(
        MapViewOfFile(m->mapping, FILE_MAP_COPY, 0, 0, 0));
    if (!m->data) fail("MapViewOfFile failed");
#else
    void* view = mmap(nullptr, m->size, PROT_READ | PROT_WRITE, MAP_PRIVATE, m->file, 0);
    if (view == MAP_FAILED) fail("mmap failed: errno=" + std::to_string(errno));
    m->data = static_cast<unsigned char*>(view);
#endif
    return m;
}

// All integers in safetensors storage are little-endian, independent of host.
std::uint64_t le64(const unsigned char* p) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
        value |= static_cast<std::uint64_t>(p[i]) << (8u * i);
    return value;
}
bool little_endian() noexcept {
    const std::uint16_t one = 1;
    return *reinterpret_cast<const unsigned char*>(&one) == 1;
}

void validate_utf8(std::string_view text) {
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i++]);
        if (lead < 0x80) continue;
        unsigned tail = 0;
        std::uint32_t cp = 0, minimum = 0;
        if (lead >= 0xC2 && lead <= 0xDF) { tail = 1; cp = lead & 0x1Fu; minimum = 0x80; }
        else if (lead >= 0xE0 && lead <= 0xEF) { tail = 2; cp = lead & 0x0Fu; minimum = 0x800; }
        else if (lead >= 0xF0 && lead <= 0xF4) { tail = 3; cp = lead & 0x07u; minimum = 0x10000; }
        else fail("invalid UTF-8");
        if (tail > text.size() - i) fail("truncated UTF-8");
        while (tail--) {
            const auto c = static_cast<unsigned char>(text[i++]);
            if ((c & 0xC0u) != 0x80u) fail("invalid UTF-8 continuation");
            cp = (cp << 6) | (c & 0x3Fu);
        }
        if (cp < minimum || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
            fail("invalid UTF-8 codepoint");
    }
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) out.push_back(static_cast<char>(cp));
    else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 63u)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 63u)));
        out.push_back(static_cast<char>(0x80u | (cp & 63u)));
    } else {
        out.push_back(static_cast<char>(0xF0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 63u)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 63u)));
        out.push_back(static_cast<char>(0x80u | (cp & 63u)));
    }
}

struct Entry {
    std::string name, dtype;
    std::vector<std::size_t> shape;
    std::size_t begin = 0, end = 0, count = 0;
};

// Schema-directed JSON parser: no regex, generic DOM, recursion, or deps.
// Unknown tensor fields/types and duplicate keys are errors, not skipped bytes.
class Parser {
public:
    explicit Parser(std::string_view header) : text_(header) {}
    std::vector<Entry> parse();
private:
    void whitespace() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' ||
               text_[pos_] == '\r' || text_[pos_] == '\t')) ++pos_;
    }
    bool consume(char c) {
        whitespace();
        if (pos_ < text_.size() && text_[pos_] == c) { ++pos_; return true; }
        return false;
    }
    void expect(char c) {
        if (!consume(c)) fail("expected '" + std::string(1, c) +
                               "' at header byte " + std::to_string(pos_));
    }
    unsigned hex4();
    std::string string();
    std::size_t number();
    std::vector<std::size_t> array(std::size_t maximum);
    void metadata();
    Entry tensor(std::string name);
    std::string_view text_;
    std::size_t pos_ = 0;
};

unsigned Parser::hex4() {
    unsigned value = 0;
    for (unsigned i = 0; i < 4; ++i) {
        if (pos_ == text_.size()) fail("truncated unicode escape");
        const char c = text_[pos_++];
        unsigned digit;
        if (c >= '0' && c <= '9') digit = static_cast<unsigned>(c - '0');
        else if (c >= 'a' && c <= 'f') digit = static_cast<unsigned>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') digit = static_cast<unsigned>(c - 'A' + 10);
        else fail("invalid unicode escape");
        value = (value << 4) | digit;
    }
    return value;
}

std::string Parser::string() {
    expect('"');
    std::string result;
    while (pos_ < text_.size()) {
        const auto c = static_cast<unsigned char>(text_[pos_++]);
        if (c == '"') return result;
        if (c < 0x20) fail("unescaped control character");
        if (c != '\\') { result.push_back(static_cast<char>(c)); continue; }
        if (pos_ == text_.size()) fail("truncated string escape");
        switch (text_[pos_++]) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        case 'u': {
            std::uint32_t cp = hex4();
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (text_.size() - pos_ < 2 || text_[pos_] != '\\' || text_[pos_ + 1] != 'u')
                    fail("missing low surrogate");
                pos_ += 2;
                const auto low = hex4();
                if (low < 0xDC00 || low > 0xDFFF) fail("invalid low surrogate");
                cp = 0x10000u + ((cp - 0xD800u) << 10) + (low - 0xDC00u);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) fail("unpaired low surrogate");
            append_utf8(result, cp);
            break;
        }
        default: fail("invalid string escape");
        }
    }
    fail("unterminated string");
}

std::size_t Parser::number() {
    whitespace();
    const auto start = pos_;
    std::size_t value = 0;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        const auto digit = static_cast<std::size_t>(text_[pos_++] - '0');
        if (value > ((std::numeric_limits<std::size_t>::max)() - digit) / 10)
            fail("integer overflow");
        value = value * 10 + digit;
    }
    if (pos_ == start || (text_[start] == '0' && pos_ - start != 1))
        fail("expected nonnegative JSON integer without leading zeros");
    return value;
}

std::vector<std::size_t> Parser::array(std::size_t maximum) {
    expect('[');
    std::vector<std::size_t> result;
    if (consume(']')) return result;
    do {
        if (result.size() == maximum) fail("array/rank limit exceeded");
        result.push_back(number());
        if (consume(']')) return result;
        expect(',');
    } while (true);
}

void Parser::metadata() {
    expect('{');
    std::unordered_set<std::string> keys;
    if (consume('}')) return;
    do {
        if (!keys.insert(string()).second) fail("duplicate metadata key");
        expect(':');
        (void)string(); // Metadata is strictly string-to-string.
        if (consume('}')) return;
        expect(',');
    } while (true);
}

Entry Parser::tensor(std::string name) {
    Entry e;
    e.name = std::move(name);
    expect('{');
    unsigned seen = 0;
    if (consume('}')) fail("missing tensor fields");
    do {
        const auto key = string();
        expect(':');
        unsigned bit = 0;
        if (key == "dtype") bit = 1;
        else if (key == "shape") bit = 2;
        else if (key == "data_offsets") bit = 4;
        else fail("unknown tensor field: " + key);
        if (seen & bit) fail("duplicate tensor field: " + key);
        seen |= bit;
        if (bit == 1) e.dtype = string();
        else if (bit == 2) e.shape = array(64);
        else {
            const auto offsets = array(2);
            if (offsets.size() != 2) fail("data_offsets must contain two integers");
            e.begin = offsets[0];
            e.end = offsets[1];
        }
        if (consume('}')) break;
        expect(',');
    } while (true);
    if (seen != 7) fail("missing tensor fields");
    return e;
}

std::vector<Entry> Parser::parse() {
    validate_utf8(text_);
    // Safetensors requires the header to begin with '{', not whitespace/BOM.
    if (text_.empty() || text_.front() != '{') fail("header must start with '{'");
    expect('{');
    std::vector<Entry> entries;
    std::unordered_set<std::string> names;
    if (!consume('}')) {
        do {
            auto name = string();
            if (!names.insert(name).second) fail("duplicate top-level key");
            expect(':');
            if (name == "__metadata__") metadata();
            else entries.push_back(tensor(std::move(name)));
            if (consume('}')) break;
            expect(',');
        } while (true);
    }
    whitespace();
    if (pos_ != text_.size()) fail("trailing JSON data");
    return entries;
}

void validate_entries(std::vector<Entry>& entries, std::size_t payload_bytes) {
    const auto limit = static_cast<std::size_t>(
        (std::numeric_limits<std::ptrdiff_t>::max)()) / sizeof(float);
    for (auto& e : entries) {
        if (e.dtype != "F32" && e.dtype != "BF16") fail("unsupported dtype: " + e.dtype);
        // Work backwards to validate representable row-major strides too.
        std::size_t count = 1;
        for (std::size_t i = e.shape.size(); i-- > 0;) {
            if (e.shape[i] != 0 && count > limit / e.shape[i])
                fail("tensor shape overflow");
            count *= e.shape[i];
        }
        e.count = count;
        const std::size_t width = e.dtype == "F32" ? 4u : 2u;
        if (e.begin > e.end || e.end > payload_bytes ||
            e.end - e.begin != count * width)
            fail("invalid tensor byte range: " + e.name);
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return a.begin < b.begin || (a.begin == b.begin && a.end < b.end);
    });
    std::size_t cursor = 0;
    for (const auto& e : entries) {
        if (e.begin != cursor) fail("overlapping tensors or hole in payload");
        cursor = e.end;
    }
    if (cursor != payload_bytes) fail("unreferenced/truncated payload");
}
} // namespace

WeightMap load_safetensors(const std::filesystem::path& path,
                           bool require_mapped_f32) {
    static_assert(CHAR_BIT == 8 && sizeof(float) == 4 &&
                  std::numeric_limits<float>::is_iec559,
                  "Safetensors requires IEEE-754 binary32 and 8-bit bytes");
    auto mapped = map_file(path);
    const std::uint64_t header64 = le64(mapped->data);
    constexpr std::uint64_t header_limit = 100u * 1024u * 1024u;
    if (header64 < 2 || header64 > header_limit || header64 > mapped->size - 8)
        fail("invalid or oversized header length");
    const auto header_bytes = static_cast<std::size_t>(header64);
    const std::size_t data_start = 8u + header_bytes;
    const std::string_view header(
        reinterpret_cast<const char*>(mapped->data + 8), header_bytes);
    auto entries = Parser(header).parse();
    validate_entries(entries, mapped->size - data_start);

    WeightMap result;
    result.reserve(entries.size());
    for (const auto& e : entries) {
        unsigned char* source = mapped->data + data_start + e.begin;
        const bool alias = e.dtype == "F32" && e.count != 0 && little_endian() &&
            reinterpret_cast<std::uintptr_t>(source) % 64u == 0;
        if (require_mapped_f32 && e.dtype == "F32" && e.count != 0 && !alias)
            fail("F32 tensor is not directly mappable: " + e.name);
        if (alias) {
            result.emplace(e.name, Tensor::from_external(
                e.shape, reinterpret_cast<float*>(source), mapped));
            continue;
        }
        Tensor converted(e.shape);
        for (std::size_t i = 0; i < e.count; ++i) {
            std::uint32_t bits;
            if (e.dtype == "BF16") {
                bits = (static_cast<std::uint32_t>(source[2 * i]) |
                       (static_cast<std::uint32_t>(source[2 * i + 1]) << 8)) << 16;
            } else {
                bits = static_cast<std::uint32_t>(source[4 * i]) |
                      (static_cast<std::uint32_t>(source[4 * i + 1]) << 8) |
                      (static_cast<std::uint32_t>(source[4 * i + 2]) << 16) |
                      (static_cast<std::uint32_t>(source[4 * i + 3]) << 24);
            }
            std::memcpy(converted.data() + i, &bits, sizeof(bits));
        }
        result.emplace(e.name, std::move(converted));
    }
    return result;
}
} // namespace stratakv::io
```

Header rank is limited to 64 and header size to 100 MiB. Tensor fields may appear
in any order. JSON string escapes, Unicode surrogate pairs, raw UTF-8,
string-to-string `__metadata__`, scalar/empty tensors, and arbitrarily ordered
tensor entries are supported. Duplicate keys, fractional/negative dimensions,
overflow, unsupported dtypes, holes, overlap, unknown fields, and truncated data
throw exceptions. Metadata is validated but is not returned as a weight tensor.
The operating system must allow private copy-on-write mapping; callers must not
modify or truncate the checkpoint externally while it remains mapped (especially
on POSIX). This is not an untrusted-file sandbox or a promise to survive external
file truncation. OS virtual-memory commitments may be large on Windows.

<!-- phase5:generation -->

## `include/stratakv/generation/sampler.h`

```cpp
#pragma once
#include "stratakv/tensor.h"

namespace stratakv::generation {
class Sampler {
public:
    // Greedy logits argmax; equal maxima select the lowest token ID.
    int sample(const Tensor& logits) const;
};
} // namespace stratakv::generation
```

## `src/generation/sampler.cpp`

```cpp
#include "stratakv/generation/sampler.h"
#include <climits>
#include <cmath>
#include <stdexcept>

namespace stratakv::generation {
int Sampler::sample(const Tensor& logits) const {
    const auto& shape = logits.shape();
    if (!(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1)))
        throw std::invalid_argument("Sampler: expected [vocab] or [1,vocab]");
    const std::size_t vocab = shape.back();
    if (vocab == 0 || vocab > static_cast<std::size_t>(INT_MAX) ||
        logits.num_elements() != vocab || !logits.data())
        throw std::invalid_argument("Sampler: empty or invalid vocabulary/storage");
    std::size_t best = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
        if (!std::isfinite(logits[i]))
            throw std::invalid_argument("Sampler: logits must all be finite");
        if (logits[i] > logits[best]) best = i;
    }
    return static_cast<int>(best); // No softmax is needed for argmax.
}
} // namespace stratakv::generation
```

## `include/stratakv/generation/generator.h`

```cpp
#pragma once
#include "stratakv/generation/sampler.h"
#include "stratakv/io/safetensors.h"
#include "stratakv/models/transformer_block.h"
#include <cstddef>
#include <vector>

namespace stratakv::generation {
struct GeneratorConfig {
    std::size_t hidden_dim = 0;
    std::size_t intermediate_dim = 0;
    std::size_t num_heads = 0;
    int total_blocks = 0;
    int block_size = 0;
    std::size_t vocab_size = 0;
    int eos_token_id = -1; // Disabled when -1.
};

class Generator {
public:
    Generator(GeneratorConfig config, io::WeightMap weights);
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens);
    std::size_t free_block_count() const noexcept { return pool_.free_count(); }

private:
    Tensor embed(const int* ids, std::size_t count) const;
    Tensor forward(const Tensor& input);
    // Declaration order ensures validation precedes all internal allocations.
    GeneratorConfig config_;
    models::TransformerBlock block_;
    memory::BlockPool pool_;
    memory::PageTable pages_;
    Sampler sampler_;
    Tensor embeddings_; // [vocab, hidden], owns moved checkpoint tensor.
    layers::RMSNorm final_norm_;
    layers::Linear lm_head_; // GEMM-ready [hidden, vocab].
    bool retry_attention_ = false;
};
} // namespace stratakv::generation
```

## `src/generation/generator.cpp`

```cpp
#include "stratakv/generation/generator.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace stratakv::generation {
namespace {
std::size_t checked_elements(std::initializer_list<std::size_t> dims) {
    const auto byte_limit = (std::min)(
        (std::numeric_limits<std::size_t>::max)() - 63u,
        static_cast<std::size_t>((std::numeric_limits<std::ptrdiff_t>::max)()));
    const auto limit = byte_limit / sizeof(float);
    std::size_t n = 1;
    for (auto d : dims) {
        if (d == 0 || n > limit / d)
            throw std::overflow_error("Generator: invalid/overflowing tensor size");
        n *= d;
    }
    return n;
}

GeneratorConfig validate_config(GeneratorConfig c) {
    if (c.hidden_dim == 0 || c.intermediate_dim == 0 || c.num_heads == 0 ||
        c.vocab_size == 0 || c.vocab_size > static_cast<std::size_t>(INT_MAX) ||
        c.total_blocks <= 0 || c.block_size <= 0 ||
        c.hidden_dim % c.num_heads != 0 || c.eos_token_id < -1 ||
        (c.eos_token_id >= 0 &&
         static_cast<std::size_t>(c.eos_token_id) >= c.vocab_size))
        throw std::invalid_argument("Generator: invalid configuration");
    checked_elements({c.hidden_dim, c.hidden_dim});
    checked_elements({c.hidden_dim, c.intermediate_dim});
    checked_elements({c.vocab_size, c.hidden_dim});
    checked_elements({static_cast<std::size_t>(c.total_blocks),
                      static_cast<std::size_t>(c.block_size), c.hidden_dim});
    return c;
}

Tensor take(io::WeightMap& weights, const char* key,
            std::initializer_list<std::size_t> expected) {
    auto it = weights.find(key); // Never operator[]: Tensor has no default ctor.
    if (it == weights.end())
        throw std::invalid_argument(std::string("Missing weight: ") + key);
    const Tensor& source = it->second;
    if (source.shape() != std::vector<std::size_t>(expected) ||
        source.num_elements() != checked_elements(expected) || !source.data())
        throw std::invalid_argument(std::string("Wrong shape/storage: ") + key);
    for (std::size_t i = 0; i < source.num_elements(); ++i)
        if (!std::isfinite(source[i]))
            throw std::invalid_argument(std::string("Non-finite weight: ") + key);
    Tensor result = std::move(it->second);
    weights.erase(it);
    return result;
}

void load_linear(io::WeightMap& weights, const char* key, layers::Linear& layer) {
    const auto in = layer.input_dim(), out = layer.output_dim();
    Tensor disk = take(weights, key, {out, in});
    Tensor& dest = layer.weight(); // Transpose into existing allocation once.
    for (std::size_t o = 0; o < out; ++o)
        for (std::size_t i = 0; i < in; ++i)
            dest[i * out + o] = disk[o * in + i];
    if (Tensor* bias = layer.bias())
        std::fill_n(bias->data(), bias->num_elements(), 0.0f);
}

// Existing PageTable can pop a pool block before vector/map growth throws.
// Snapshots recover even unregistered blocks that free_sequence cannot see.
class SequenceGuard {
public:
    SequenceGuard(memory::PageTable& pages, memory::BlockPool& pool)
        : pages_(pages), pool_(pool), saved_pages_(pages), saved_pool_(pool) {}
    SequenceGuard(const SequenceGuard&) = delete;
    SequenceGuard& operator=(const SequenceGuard&) = delete;
    ~SequenceGuard() noexcept {
        try {
            if (pages_.has_sequence(0)) pages_.free_sequence(0, pool_);
        } catch (...) { /* Snapshot rollback also handles cleanup failure. */ }
        pages_ = std::move(saved_pages_);
        pool_ = std::move(saved_pool_);
    }
private:
    memory::PageTable& pages_;
    memory::BlockPool& pool_;
    memory::PageTable saved_pages_;
    memory::BlockPool saved_pool_;
};
static_assert(std::is_nothrow_move_assignable_v<memory::PageTable> &&
              std::is_nothrow_move_assignable_v<memory::BlockPool>,
              "Sequence rollback must not throw");
} // namespace

Generator::Generator(GeneratorConfig config, io::WeightMap weights)
    : config_(validate_config(config)),
      block_(config_.hidden_dim, config_.intermediate_dim, config_.num_heads,
             config_.total_blocks, config_.block_size),
      pool_(config_.total_blocks, config_.block_size),
      embeddings_(take(weights, "tok_embeddings.weight",
                       {config_.vocab_size, config_.hidden_dim})),
      final_norm_(take(weights, "norm.weight", {config_.hidden_dim})),
      lm_head_(config_.hidden_dim, config_.vocab_size) {
    const auto h = config_.hidden_dim;
    block_.input_norm() = layers::RMSNorm(take(weights, "block.input_norm.weight", {h}));
    block_.post_attention_norm() = layers::RMSNorm(
        take(weights, "block.post_attention_norm.weight", {h}));
    auto& attn = block_.attention();
    load_linear(weights, "block.attn.q_proj.weight", attn.query_projection());
    load_linear(weights, "block.attn.k_proj.weight", attn.key_projection());
    load_linear(weights, "block.attn.v_proj.weight", attn.value_projection());
    load_linear(weights, "block.attn.o_proj.weight", attn.output_projection());
    load_linear(weights, "block.ffn.up_proj.weight", block_.ffn_up());
    load_linear(weights, "block.ffn.down_proj.weight", block_.ffn_down());
    load_linear(weights, "lm_head.weight", lm_head_);
    if (!weights.empty())
        throw std::invalid_argument("Unexpected weight: " + weights.begin()->first);
}

Tensor Generator::embed(const int* ids, std::size_t count) const {
    const auto h = config_.hidden_dim;
    checked_elements({count, h});
    Tensor input({count, h});
    for (std::size_t row = 0; row < count; ++row)
        std::copy_n(embeddings_.data() + static_cast<std::size_t>(ids[row]) * h,
                    h, input.data() + row * h);
    return input;
}

Tensor Generator::forward(const Tensor& input) {
    if (retry_attention_) {
        // A failed initial cache allocation can leave only one KV tensor alive.
        // Build first (may throw), then transfer learned projections noexcept.
        layers::PagedAttention clean(config_.hidden_dim, config_.num_heads,
                                     config_.total_blocks, config_.block_size);
        auto& old = block_.attention();
        clean.query_projection() = std::move(old.query_projection());
        clean.key_projection() = std::move(old.key_projection());
        clean.value_projection() = std::move(old.value_projection());
        clean.output_projection() = std::move(old.output_projection());
        old = std::move(clean);
    }
    retry_attention_ = true;
    Tensor output = block_.forward(input, 0, pages_, pool_);
    retry_attention_ = false;
    return output;
}

std::vector<int> Generator::generate(const std::vector<int>& prompt,
                                     int max_new_tokens) {
    if (prompt.empty() || max_new_tokens < 0)
        throw std::invalid_argument("Generator: nonempty prompt and nonnegative max required");
    for (int id : prompt)
        if (id < 0 || static_cast<std::size_t>(id) >= config_.vocab_size)
            throw std::invalid_argument("Generator: prompt token outside vocabulary");
    const auto extra = static_cast<std::size_t>(max_new_tokens);
    if (prompt.size() > static_cast<std::size_t>(INT_MAX) - extra)
        throw std::overflow_error("Generator: prompt plus max exceeds INT_MAX");
    if (max_new_tokens == 0 || prompt.back() == config_.eos_token_id) return prompt;
    const auto required = prompt.size() + extra;
    const auto block_size = static_cast<std::size_t>(config_.block_size);
    const auto blocks = required / block_size + (required % block_size != 0);
    if (blocks > static_cast<std::size_t>(config_.total_blocks))
        throw std::length_error("Generator: prompt plus max exceeds KV capacity");
    checked_elements({prompt.size(), config_.hidden_dim});
    checked_elements({prompt.size(), config_.intermediate_dim});
    std::vector<int> result;
    result.reserve(required);
    result.insert(result.end(), prompt.begin(), prompt.end());
    SequenceGuard guard(pages_, pool_); // Armed before sequence allocation.
    pages_.allocate_sequence(0, pool_);
    Tensor last({1, config_.hidden_dim});
    {
        Tensor hidden = forward(embed(prompt.data(), prompt.size()));
        std::copy_n(hidden.data() + (prompt.size() - 1) * config_.hidden_dim,
                    config_.hidden_dim, last.data());
    } // Retain only the last prefill row.
    for (int step = 0; step < max_new_tokens; ++step) {
        Tensor normalized = final_norm_.forward(last);
        Tensor logits = lm_head_.forward(normalized);
        const int next = sampler_.sample(logits);
        result.push_back(next);
        if (next == config_.eos_token_id || step + 1 == max_new_tokens) break;
        // Attention appends/caches this row. Never manually append the token
        // and never reuse the previous hidden state as the next embedding.
        last = forward(embed(&next, 1));
    }
    return result; // Guard restores allocator state, including on exceptions.
}
} // namespace stratakv::generation
```

### Exact checkpoint schema and ownership

Pass `Generator(config, std::move(weights))`, or pass the map returned by
`io::load_safetensors(path)` directly. All required weights must be present;
extra weights, wrong shapes, and non-finite values are rejected.

| Tensor name | On-disk shape |
| --- | --- |
| `tok_embeddings.weight` | `[vocab_size, hidden_dim]` |
| `block.input_norm.weight` | `[hidden_dim]` |
| `block.post_attention_norm.weight` | `[hidden_dim]` |
| `block.attn.q_proj.weight` | `[hidden_dim, hidden_dim]` |
| `block.attn.k_proj.weight` | `[hidden_dim, hidden_dim]` |
| `block.attn.v_proj.weight` | `[hidden_dim, hidden_dim]` |
| `block.attn.o_proj.weight` | `[hidden_dim, hidden_dim]` |
| `block.ffn.up_proj.weight` | `[intermediate_dim, hidden_dim]` |
| `block.ffn.down_proj.weight` | `[hidden_dim, intermediate_dim]` |
| `norm.weight` | `[hidden_dim]` |
| `lm_head.weight` | `[vocab_size, hidden_dim]` |

Projection weights use conventional disk `[out,in]` order. `load_linear`
transposes them once into Phase 4's physical `[in,out]` layout. Embeddings and
gammas move without an additional copy. No bias keys or tied-head aliases are
implicitly substituted. Transposition and BF16 expansion need real allocations;
the entire inference path is not advertised as zero-copy.

`generate` returns **prompt plus generated IDs**, including a sampled EOS. An
empty prompt, invalid ID, negative limit, invalid config, or impossible capacity
request throws. Zero new tokens or a prompt ending in EOS returns the validated
prompt without allocating pages. Capacity checking conservatively budgets
`prompt.size() + max_new_tokens`, even though the final sampled ID need not be
cached. Each call uses sequence 0 privately, reuses layer-local KV storage, and
restores the page/pool state before returning. Snapshot rollback costs
O(total_blocks) auxiliary allocator storage; it also covers existing PageTable
allocation-failure windows that `free_sequence()` alone cannot recover. A block
forward exception rebuilds attention on the next forward while preserving its
learned projections, recovering partial KV-cache allocation. This API is
single-threaded and not thread-safe.

<!-- phase5:integration -->

## `CMakeLists.txt` integration

Append the three new sources to the existing list. The Tensor replacement
keeps its existing source path, so do not list it twice:

```cmake
set(STRATAKV_SOURCES
    src/cpu/tensor.cpp
    src/cpu/gemm_cpu.cpp
    src/memory/block_pool.cpp
    src/memory/page_table.cpp
    src/layers/linear.cpp
    src/layers/rmsnorm.cpp
    src/layers/attention.cpp
    src/models/transformer_block.cpp
    src/io/safetensors.cpp
    src/generation/sampler.cpp
    src/generation/generator.cpp
)
```

Keep existing CUDA dispatch/definitions and the C++17 standard. Windows file
mapping is provided by the platform SDK; POSIX mapping needs no third-party
package. The mapped loader and Tensor allocation are portable; **the existing
CPU GEMM still contains Windows `_aligned_malloc/_aligned_free` calls**. To
build the full engine on POSIX, its local `AlignedScratch` allocator must use
`posix_memalign/free` (as demonstrated in Tensor) instead. This document does
not claim the pre-existing GEMM is already portable. The CPU path also requires
AVX2/FMA-capable hardware; there is no scalar dispatch in the current backend.

### Calling the generator (inside an application function)

```cpp
#include "stratakv/generation/generator.h"
#include <iostream>

// Replace dimensions, IDs, and path with those of a checkpoint trained for
// the exact one-block architecture/schema above. These are NOT Llama defaults.
void run_tokens(const std::filesystem::path& checkpoint) {
    stratakv::generation::GeneratorConfig cfg;
    cfg.hidden_dim = 8;
    cfg.intermediate_dim = 16;
    cfg.num_heads = 2;
    cfg.total_blocks = 32;
    cfg.block_size = 16;
    cfg.vocab_size = 32;
    cfg.eos_token_id = 2;
    stratakv::generation::Generator engine(cfg, stratakv::io::load_safetensors(checkpoint));
    const auto tokens = engine.generate({1, 5, 7}, 16);
    for (int id : tokens) std::cout << id << ' ';
    std::cout << '\n';
}
```

## Reproducible zero-dependency smoke tests

The following optional test file generates its own tiny, explicitly synthetic
checkpoints. It requires no downloaded model, network access, tokenizer, Python
package, or testing framework. It checks loading plus the actual generation
flow, not language quality. Run it with an existing scratch directory as its
sole argument; it creates/overwrites only files prefixed `stratakv_phase5_` there.

## `tests/phase5_smoke.cpp`

```cpp
#include "stratakv/generation/generator.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
int checks = 0;
void check(bool ok) {
    ++checks;
    if (!ok) throw std::runtime_error("check failed: " + std::to_string(checks));
}
template<class F> void rejects(F operation) {
    bool threw = false;
    try { operation(); } catch (const std::exception&) { threw = true; }
    check(threw);
}
std::vector<unsigned char> fp32(const std::vector<float>& values) {
    std::vector<unsigned char> bytes;
    for (float v : values) {
        std::uint32_t bits;
        std::memcpy(&bits, &v, 4);
        for (unsigned i = 0; i < 4; ++i)
            bytes.push_back(static_cast<unsigned char>((bits >> (8 * i)) & 255u));
    }
    return bytes;
}
void write_file(const std::filesystem::path& file, std::string header,
                const std::vector<unsigned char>& payload, bool aligned = true) {
    if (aligned) while ((8 + header.size()) % 64 != 0) header += ' ';
    else if ((8 + header.size()) % 64 == 0) header += ' ';
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    const auto length = static_cast<std::uint64_t>(header.size());
    for (unsigned i = 0; i < 8; ++i) out.put(static_cast<char>((length >> (8 * i)) & 255u));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!payload.empty())
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
}

void make_model(const std::filesystem::path& path, bool bf16) {
    std::string header = "{";
    std::vector<unsigned char> payload;
    auto add = [&](const char* name, std::vector<std::size_t> shape,
                   const std::vector<float>& values) {
        if (header.size() > 1) header += ',';
        header += '"' + std::string(name) + "\":{\"dtype\":\"" +
                  (bf16 ? "BF16" : "F32") + "\",\"shape\":[";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (i) header += ',';
            header += std::to_string(shape[i]);
        }
        const auto begin = payload.size();
        const auto bytes = fp32(values);
        if (bf16) {
            for (std::size_t i = 0; i < values.size(); ++i) {
                payload.push_back(bytes[4 * i + 2]);
                payload.push_back(bytes[4 * i + 3]);
            }
        } else payload.insert(payload.end(), bytes.begin(), bytes.end());
        header += "],\"data_offsets\":[" + std::to_string(begin) + ',' +
                  std::to_string(payload.size()) + "]}";
    };
    add("tok_embeddings.weight", {3, 2}, {1, 0, 0, 1, -1, 0});
    for (auto name : {"block.input_norm.weight", "block.post_attention_norm.weight", "norm.weight"})
        add(name, {2}, {1, 1});
    for (auto name : {"block.attn.q_proj.weight", "block.attn.k_proj.weight",
                      "block.attn.v_proj.weight", "block.attn.o_proj.weight"})
        add(name, {2, 2}, {1, 0, 0, 1});
    add("block.ffn.up_proj.weight", {4, 2}, std::vector<float>(8, 0));
    add("block.ffn.down_proj.weight", {2, 4}, std::vector<float>(8, 0));
    add("lm_head.weight", {3, 2}, {0, 0, 2, 0, 0, 2});
    write_file(path, header + '}', payload);
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 || !std::filesystem::is_directory(argv[1]))
            throw std::runtime_error("supply an existing scratch directory");
        const auto root = std::filesystem::path(argv[1]);
        const auto file = root / "stratakv_phase5_tensor.safetensors";
        const std::string header =
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
        write_file(file, header, fp32({1.5f, -2.0f}));
        {
            auto weights = stratakv::io::load_safetensors(file, true);
            check(weights.size() == 1);
            stratakv::Tensor retained = std::move(weights.at("x"));
            weights.clear();
            check(retained[0] == 1.5f && retained[1] == -2.0f);
            retained[0] = 99; // Private mapping must not modify the disk file.
            check(stratakv::io::load_safetensors(file, true).at("x")[0] == 1.5f);
        }
        write_file(file, header, fp32({1.5f, -2.0f}), false);
        check(stratakv::io::load_safetensors(file).at("x")[1] == -2);
        rejects([&] { (void)stratakv::io::load_safetensors(file, true); });
        write_file(file,
            R"({"x":{"data_offsets":[0,4],"shape":[2],"dtype":"BF16"}})",
            {0x80, 0x3F, 0x20, 0xC0});
        auto bf = stratakv::io::load_safetensors(file);
        check(bf.at("x")[0] == 1 && bf.at("x")[1] == -2.5f);

        write_file(file, R"({"z":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", fp32({7, 9}));
        {
            auto ordered = stratakv::io::load_safetensors(file);
            check(ordered.at("a")[0] == 7 && ordered.at("z")[0] == 9);
        }
        write_file(file, R"({"\uD83D\uDE00":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})", fp32({7}));
        check(stratakv::io::load_safetensors(file).count("\xF0\x9F\x98\x80") == 1);

        write_file(file, R"({"__metadata__":{"note":"ok"},"\u0078":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})", fp32({3}));
        check(stratakv::io::load_safetensors(file).at("x").ndim() == 0);
        write_file(file, R"({"x":{"dtype":"F32","shape":[0,2],"data_offsets":[0,0]}})", {});
        check(stratakv::io::load_safetensors(file).at("x").num_elements() == 0);

        for (const std::string bad : {
            R"({"x":{"dtype":"F16","shape":[2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[02],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[18446744073709551616],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[1,9]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"y":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}} true)",
            R"({"\uD800":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})",
            R"({"__metadata__":{"bad":3},"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})"}) {
            write_file(file, bad, fp32({1, 2}));
            rejects([&] { (void)stratakv::io::load_safetensors(file); });
        }
        { std::ofstream short_file(file, std::ios::binary | std::ios::trunc); short_file << "123"; }
        rejects([&] { (void)stratakv::io::load_safetensors(file); });
        { std::ofstream bad_length(file, std::ios::binary | std::ios::trunc);
          for (unsigned i = 0; i < 8; ++i) bad_length.put(static_cast<char>(255)); }
        rejects([&] { (void)stratakv::io::load_safetensors(file); });
        write_file(file, "{}", fp32({1})); // Payload bytes cannot be unreferenced.
        rejects([&] { (void)stratakv::io::load_safetensors(file); });

        stratakv::generation::Sampler sampler;
        stratakv::Tensor logits({3});
        logits[0] = -2; logits[1] = 4; logits[2] = 4;
        check(sampler.sample(logits) == 1);
        logits[2] = (std::numeric_limits<float>::quiet_NaN)();
        rejects([&] { (void)sampler.sample(logits); });
        stratakv::Tensor empty({0}), batched({2, 3});
        rejects([&] { (void)sampler.sample(empty); });
        rejects([&] { (void)sampler.sample(batched); });

        stratakv::generation::GeneratorConfig cfg{2, 4, 1, 8, 2, 3, 2};
        for (bool use_bf16 : {false, true}) {
            make_model(file, use_bf16);
            stratakv::generation::Generator engine(cfg, stratakv::io::load_safetensors(file));
            for (int repeat = 0; repeat < 2; ++repeat) {
                check(engine.generate({0}, 4) == std::vector<int>({0, 1, 2}));
                check(engine.free_block_count() == 8);
            }
            check(engine.generate({0, 0, 0, 0, 0}, 4) ==
                  std::vector<int>({0, 0, 0, 0, 0, 1, 2}));
            check(engine.free_block_count() == 8);
            check(engine.generate({0}, 1) == std::vector<int>({0, 1}));
            check(engine.generate({0}, 0) == std::vector<int>({0}));
            check(engine.generate({2}, 4) == std::vector<int>({2}));
            rejects([&] { (void)engine.generate({}, 1); });
            rejects([&] { (void)engine.generate({3}, 1); });
            rejects([&] { (void)engine.generate({0}, -1); });
            rejects([&] { (void)engine.generate({0}, 20); });
            check(engine.free_block_count() == 8);
        }
        make_model(file, false);
        auto missing = stratakv::io::load_safetensors(file);
        missing.erase("lm_head.weight");
        rejects([&] { stratakv::generation::Generator bad(cfg, std::move(missing)); });
        std::cout << "Phase 5: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Phase 5 failure: " << e.what() << '\n';
        return 1;
    }
}
```

Add this optional target after defining `stratakv`:

```cmake
include(CTest)
if(BUILD_TESTING)
    add_executable(phase5_smoke tests/phase5_smoke.cpp)
    target_link_libraries(phase5_smoke PRIVATE stratakv)
    add_test(NAME phase5_smoke
             COMMAND phase5_smoke "${CMAKE_CURRENT_BINARY_DIR}")
endif()
```

After applying the code, verify on Windows from the repository root:

```powershell
cmake -S . -B build-phase5-cpu -DSTRATAKV_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build-phase5-cpu --config Release
ctest --test-dir build-phase5-cpu -C Release --output-on-failure
```

Use a separate CUDA-enabled build for CUDA validation, with the architecture
and toolkit appropriate to the existing setup. A successful CPU run is not
evidence that the CUDA path was tested.

<!-- phase5:verification -->

## Verification actually performed for this deliverable

The nine complete C++ file blocks were extracted into an isolated temporary
source tree and compiled against the current Phase 4 headers/sources. The
workspace application files and its existing `build/` outputs were not edited.

| Check | Actual result |
| --- | --- |
| MSVC 19.44, C++17, `/W4`, CPU AVX2, optimization disabled and bounds checks enabled | Build passed; **58 smoke checks passed** |
| MSVC C++17, `/O2 /DNDEBUG`, CPU AVX2 | Build passed; **58 smoke checks passed** |
| Additional temporary generation/allocator fault-injection harness against the extracted code | **1,145 checks passed**, including recovery at 179 warm-cache and 186 cold-cache failure positions |
| Repository CMake configuration with the documented source list and optional CTest target, in the scratch tree | CPU Release configure/build passed; **CTest 1/1 passed** |
| Existing `stratakv_run` rebuilt with the adapted Tensor | CPU GEMM ran; Phase 3 and Phase 4 smoke assertions passed |
| Markdown/file extraction validation | Nine expected file blocks, balanced fences, and no trailing whitespace |

The included 58-check smoke executable covers mapped FP32 lifetime/COW,
misalignment fallback and strict rejection, BF16 conversion, field ordering,
Unicode, scalar/empty tensors, malformed/truncated input, greedy ties, invalid
logits, missing parameters, deterministic EOS, bounded generation, multi-page
prefill, repeated calls, and allocator reclamation. The larger fault-injection
harness was a separate temporary verification aid, not an extra application
dependency or a committed file. MSBuild emitted its temporary-output-directory
warning MSB8029 during scratch builds; the builds and tests still completed.

**Not verified here:** CUDA execution, the POSIX mapping branch on a POSIX host,
big-endian execution, or language quality with a trained checkpoint. No real
checkpoint was supplied. The tests' small synthetic model demonstrates the
loading/generation contract, not general model compatibility or production
performance. Applying this document changes Tensor's storage ABI; rebuild all
consumers together rather than linking it with stale objects.
