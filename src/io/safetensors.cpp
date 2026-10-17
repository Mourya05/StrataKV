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
