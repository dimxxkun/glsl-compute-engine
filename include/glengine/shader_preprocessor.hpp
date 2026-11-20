#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace glengine {

enum class ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessControl,
    TessEvaluation,
    Unknown
};

std::string_view shaderStageName(ShaderStage stage) noexcept;
ShaderStage inferShaderStage(const std::filesystem::path& path);

struct PreprocessOptions {
    std::vector<std::filesystem::path> includeDirs;
    std::vector<std::pair<std::string, std::string>> defines;
    bool emitLineDirectives = true;
    std::size_t maxIncludeDepth = 16;
};

struct PreprocessResult {
    std::string source;
    ShaderStage stage = ShaderStage::Unknown;
    std::vector<std::filesystem::path> includedFiles;
    std::vector<std::string> sourceFileTable;
};

class ShaderPreprocessor {
public:
    PreprocessResult preprocessFile(
        const std::filesystem::path& filePath,
        const PreprocessOptions& options = {}
    ) const;

    PreprocessResult preprocessSource(
        std::string_view source,
        const std::filesystem::path& virtualPath,
        const PreprocessOptions& options = {}
    ) const;
};

} // namespace glengine
