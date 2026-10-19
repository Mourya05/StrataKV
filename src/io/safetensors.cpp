#include "stratakv/io/safetensors.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <chrono>
#include <stdexcept>
#include <string_view>
#include <thread>
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
constexpr std::size_t kParallelThreshold = 4u * 1024u * 1024u;
constexpr std::size_t kF16ParallelThreshold = 1024u * 1024u;
// Match the loader's existing parallel_for bound. hardware_concurrency() can
// report hundreds of logical CPUs on Windows hosts; spawning that many workers
// for a memory-bandwidth-bound decode is slower than a bounded worker set.
constexpr std::size_t kMaxDecodeThreads = 8u;

std::uint16_t le16(const unsigned char* p) noexcept;
float half_to_float(std::uint16_t value) noexcept;

template <typename Function>
void parallel_for(std::size_t count, Function function) {
    const unsigned workers = std::thread::hardware_concurrency();
    if (count < kParallelThreshold || workers < 2) {
        function(0, count);
        return;
    }
    const std::size_t thread_count =
        (std::min)(std::size_t{8}, static_cast<std::size_t>(workers));
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    const std::size_t chunk = (count + thread_count - 1) / thread_count;
    for (std::size_t begin = 0; begin < count; begin += chunk) {
        const std::size_t end = (std::min)(count, begin + chunk);
        threads.emplace_back([=, &function] { function(begin, end); });
    }
    for (auto& thread : threads)
        thread.join();
}

void decode_f16_range(const unsigned char* source, float* destination,
                      std::size_t begin, std::size_t end) noexcept {
    for (std::size_t i = begin; i < end; ++i)
        destination[i] = half_to_float(le16(source + 2u * i));
}

void decode_f16(const unsigned char* source, float* destination,
                std::size_t count) {
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    if (count < kF16ParallelThreshold || hardware_threads < 2) {
        decode_f16_range(source, destination, 0, count);
        return;
    }

    // Each worker owns a disjoint element range. The mapped source is read-only
    // and the destination was fully allocated before any worker starts.
    const std::size_t thread_count = (std::min)(
        kMaxDecodeThreads, static_cast<std::size_t>(hardware_threads));
    const std::size_t chunk =
        (count + thread_count - 1u) / thread_count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    try {
        for (std::size_t begin = 0; begin < count; begin += chunk) {
            const std::size_t end = (std::min)(count, begin + chunk);
            workers.emplace_back(decode_f16_range, source, destination,
                                 begin, end);
        }
    } catch (...) {
        for (auto& worker : workers)
            worker.join();
        throw;
    }
    for (auto& worker : workers)
        worker.join();
}

void decode_bf16_range(const unsigned char* source, float* destination,
                       std::size_t begin, std::size_t end) noexcept {
    for (std::size_t i = begin; i < end; ++i) {
        const std::uint16_t value = le16(source + 2u * i);
        const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16u;
        std::memcpy(destination + i, &bits, sizeof(bits));
    }
}

void decode_bf16(const unsigned char* source, float* destination,
                 std::size_t count) {
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    if (count < kF16ParallelThreshold || hardware_threads < 2) {
        decode_bf16_range(source, destination, 0, count);
        return;
    }

    const std::size_t thread_count = (std::min)(
        kMaxDecodeThreads, static_cast<std::size_t>(hardware_threads));
    const std::size_t chunk =
        (count + thread_count - 1u) / thread_count;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    try {
        for (std::size_t begin = 0; begin < count; begin += chunk) {
            const std::size_t end = (std::min)(count, begin + chunk);
            workers.emplace_back(decode_bf16_range, source, destination,
                                 begin, end);
        }
    } catch (...) {
        for (auto& worker : workers)
            worker.join();
        throw;
    }
    for (auto& worker : workers)
        worker.join();
}

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

std::uint16_t le16(const unsigned char* p) noexcept {
    return static_cast<std::uint16_t>(
        static_cast<std::uint16_t>(p[0]) |
        (static_cast<std::uint16_t>(p[1]) << 8u));
}

float half_to_float(std::uint16_t value) noexcept {
    const std::uint32_t sign =
        static_cast<std::uint32_t>(value & 0x8000u) << 16u;
    const std::uint32_t exponent = (value >> 10u) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    std::uint32_t bits = sign;

    if (exponent == 0) {
        if (mantissa != 0) {
            // Half subnormals start at 2^-14 before normalization. The
            // previous -1 origin produced incorrect values for every
            // subnormal payload.
            int unbiased_exponent = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                --unbiased_exponent;
            }
            mantissa &= 0x03ffu;
            bits |= static_cast<std::uint32_t>(
                unbiased_exponent + 127) << 23u;
            bits |= mantissa << 13u;
        }
    } else if (exponent == 0x1fu) {
        bits |= 0x7f800000u | (mantissa << 13u);
    } else {
        bits |= (exponent + 112u) << 23u;
        bits |= mantissa << 13u;
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool little_endian() noexcept {
    const std::uint16_t one = 1;
    return *reinterpret_cast<const unsigned char*>(&one) == 1;
}

bool is_norm_weight(const std::string& name) noexcept {
    return name.find("norm") != std::string::npos ||
        name.find("Norm") != std::string::npos ||
        name.find("gamma") != std::string::npos ||
        name.find("Gamma") != std::string::npos;
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
    for (auto& e : entries) {
        if (e.dtype != "F32" && e.dtype != "F16" && e.dtype != "BF16" &&
            e.dtype != "bfloat16")
            fail("unsupported dtype: " + e.dtype);
        const auto element_size =
            e.dtype == "F32" ? sizeof(float) : sizeof(std::uint16_t);
        const auto limit = static_cast<std::size_t>(
            (std::numeric_limits<std::ptrdiff_t>::max)()) / element_size;
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

static LoadBreakdown g_last_load_breakdown{};

LoadBreakdown last_load_breakdown() noexcept {
    return g_last_load_breakdown;
}

WeightMap load_safetensors(const std::filesystem::path& path,
                           bool require_mapped_f32) {
    static_assert(CHAR_BIT == 8 && sizeof(float) == 4 &&
                  std::numeric_limits<float>::is_iec559,
                  "Safetensors requires IEEE-754 binary32 and 8-bit bytes");
    g_last_load_breakdown = {};
    const auto mmap_start = std::chrono::steady_clock::now();
    auto mapped = map_file(path);
    g_last_load_breakdown.mmap_open_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - mmap_start).count();
    const std::uint64_t header64 = le64(mapped->data);
    constexpr std::uint64_t header_limit = 100u * 1024u * 1024u;
    if (header64 < 2 || header64 > header_limit || header64 > mapped->size - 8)
        fail("invalid or oversized header length");
    const auto header_bytes = static_cast<std::size_t>(header64);
    const std::size_t data_start = 8u + header_bytes;
    const std::string_view header(
        reinterpret_cast<const char*>(mapped->data + 8), header_bytes);
    const auto header_start = std::chrono::steady_clock::now();
    auto entries = Parser(header).parse();
    validate_entries(entries, mapped->size - data_start);
    g_last_load_breakdown.header_parse_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - header_start).count();

    WeightMap result;
    result.reserve(entries.size());
    const auto decode_start = std::chrono::steady_clock::now();
    for (const auto& e : entries) {
        // Safetensors offsets are relative to the payload immediately after
        // the eight-byte header length and JSON header.
        const unsigned char* source =
            mapped->data + data_start + e.begin;
        const bool is_f16 = e.dtype == "F16";
        const bool is_bf16 = e.dtype == "BF16" || e.dtype == "bfloat16";
        const bool keep_fp32 = is_norm_weight(e.name);
        const bool alias = keep_fp32 && e.dtype == "F32" && e.count != 0 &&
            little_endian() &&
            reinterpret_cast<std::uintptr_t>(source) % 64u == 0;
        if (require_mapped_f32 && e.dtype == "F32" && e.count != 0 && !alias)
            fail("F32 tensor is not directly mappable: " + e.name);
        if (alias) {
            result.emplace(e.name, Tensor::from_external(
                e.shape, const_cast<float*>(
                    reinterpret_cast<const float*>(source)), mapped));
            continue;
        }
        const DataType dtype = DataType::FP32;
        Tensor converted(e.shape, dtype);
        if (is_f16) {
            // F16 checkpoints dominate load time. Decode directly into the
            // fully allocated FP32 tensor using disjoint worker ranges.
            decode_f16(source, converted.data(), e.count);
        } else if (is_bf16) {
            // BF16 checkpoints use the same bounded worker architecture. The
            // mapped source and FP32 destination ranges are disjoint per worker.
            decode_bf16(source, converted.data(), e.count);
        } else {
            parallel_for(e.count, [&](std::size_t begin, std::size_t end) {
                for (std::size_t i = begin; i < end; ++i) {
                    float value = 0.0f;
                    std::uint32_t bits =
                        static_cast<std::uint32_t>(source[4 * i]) |
                        (static_cast<std::uint32_t>(source[4 * i + 1]) << 8) |
                        (static_cast<std::uint32_t>(source[4 * i + 2]) << 16) |
                        (static_cast<std::uint32_t>(source[4 * i + 3]) << 24);
                    if (!little_endian())
                        bits = ((bits & 0x000000ffu) << 24u) |
                            ((bits & 0x0000ff00u) << 8u) |
                            ((bits & 0x00ff0000u) >> 8u) |
                            ((bits & 0xff000000u) >> 24u);
                    std::memcpy(&value, &bits, sizeof(value));
                    converted[i] = value;
                }
            });
        }
        result.emplace(e.name, std::move(converted));
    }
    g_last_load_breakdown.tensor_decode_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decode_start).count();
    return result;
}

Tensor load_linear_bf16_transposed(const Tensor& source,
                                   std::size_t out_features,
                                   std::size_t in_features) {
    if (source.ndim() != 2 ||
        source.shape() != std::vector<std::size_t>{
            out_features, in_features} ||
        (source.dtype() != DataType::BF16 &&
         source.dtype() != DataType::FP32)) {
        throw std::invalid_argument(
            "safetensors: expected a BF16 or FP32 [out, in] linear tensor");
    }

    Tensor destination({in_features, out_features}, DataType::BF16);
    auto* dst = static_cast<std::uint16_t*>(destination.raw_data());
    const auto* src_bf16 = source.dtype() == DataType::BF16
        ? static_cast<const std::uint16_t*>(source.raw_data()) : nullptr;
    const auto* src_fp32 = source.dtype() == DataType::FP32
        ? static_cast<const float*>(source.raw_data()) : nullptr;
    constexpr std::size_t tile_size = 32;
    const std::size_t tile_rows =
        (out_features + tile_size - 1) / tile_size;
    const unsigned hardware_threads = std::thread::hardware_concurrency();
    const std::size_t thread_count = hardware_threads > 1
        ? (std::min)(static_cast<std::size_t>(hardware_threads), tile_rows)
        : 1;
    std::atomic<std::size_t> next_tile{0};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t worker = 0; worker < thread_count; ++worker) {
        workers.emplace_back([&] {
            std::uint16_t tile_values[32][32];
            for (;;) {
                const std::size_t out_tile = next_tile.fetch_add(
                    1, std::memory_order_relaxed);
                if (out_tile >= tile_rows)
                    break;
                const std::size_t out_begin = out_tile * tile_size;
                const std::size_t out_end =
                    (std::min)(out_features, out_begin + tile_size);
                for (std::size_t in_begin = 0; in_begin < in_features;
                     in_begin += tile_size) {
                    const std::size_t in_end =
                        (std::min)(in_features, in_begin + tile_size);
                    for (std::size_t out = out_begin; out < out_end; ++out) {
                        for (std::size_t in = in_begin; in < in_end; ++in) {
                            const std::size_t source_index =
                                out * in_features + in;
                            if (src_bf16 != nullptr) {
                                tile_values[out - out_begin][in - in_begin] =
                                    src_bf16[source_index];
                            } else {
                                std::uint32_t bits;
                                std::memcpy(&bits, &src_fp32[source_index],
                                            sizeof(bits));
                                tile_values[out - out_begin][in - in_begin] =
                                    static_cast<std::uint16_t>(bits >> 16u);
                            }
                        }
                    }
                    for (std::size_t out = out_begin; out < out_end; ++out)
                        for (std::size_t in = in_begin; in < in_end; ++in)
                            dst[in * out_features + out] =
                                tile_values[out - out_begin][in - in_begin];
                }
            }
        });
    }
    for (auto& worker : workers)
        worker.join();
    return destination;
}
} // namespace stratakv::io
