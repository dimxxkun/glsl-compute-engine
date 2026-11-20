#include "glengine/shader_preprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace glengine {

namespace {

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string trimLeft(std::string_view sv) {
    std::size_t idx = 0;
    while (idx < sv.size() && (sv[idx] == ' ' || sv[idx] == '\t')) {
        ++idx;
    }
    return std::string(sv.substr(idx));
}

std::string stripCarriageReturn(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return line;
}

std::string readTextFile(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

bool parseIncludeDirective(std::string_view line, std::string& outIncludePath) {
    const std::string trimmed = trimLeft(line);
    constexpr std::string_view kPrefix = "#include";
    if (!trimmed.starts_with(kPrefix)) {
        return false;
    }

    std::string rest = trimLeft(std::string_view(trimmed).substr(kPrefix.size()));
    if (rest.size() < 2) {
        return false;
    }

    char openDelim = rest.front();
    char closeDelim = '\0';
    if (openDelim == '"') {
        closeDelim = '"';
    } else if (openDelim == '<') {
        closeDelim = '>';
    } else {
        return false;
    }

    const std::size_t endPos = rest.find(closeDelim, 1);
    if (endPos == std::string::npos || endPos <= 1) {
        return false;
    }

    outIncludePath = rest.substr(1, endPos - 1);
    return true;
}

bool isVersionDirective(std::string_view line) {
    const std::string trimmed = trimLeft(line);
    return trimmed.starts_with("#version");
}

std::filesystem::path resolveIncludePath(
    const std::string& includeToken,
    const std::filesystem::path& currentFileDir,
    const std::vector<std::filesystem::path>& includeDirs
) {
    if (!currentFileDir.empty()) {
        const std::filesystem::path localCandidate = currentFileDir / includeToken;
        if (std::filesystem::exists(localCandidate)) {
            return std::filesystem::weakly_canonical(localCandidate);
        }
    }

    for (const auto& dir : includeDirs) {
        const std::filesystem::path candidate = dir / includeToken;
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::weakly_canonical(candidate);
        }
    }

    throw std::runtime_error(
        "Unresolved shader #include \"" + includeToken + "\" from directory \"" +
        currentFileDir.string() + "\""
    );
}

struct PreprocessContext {
    const PreprocessOptions& options;
    std::vector<std::filesystem::path> includedFiles;
    std::vector<std::string> sourceFileTable;
    std::unordered_set<std::string> activeIncludeStack;

    std::size_t registerSourceFile(const std::filesystem::path& path) {
        const std::string normalized = path.generic_string();
        for (std::size_t i = 0; i < sourceFileTable.size(); ++i) {
            if (sourceFileTable[i] == normalized) {
                return i;
            }
        }
        sourceFileTable.push_back(normalized);
        return sourceFileTable.size() - 1;
    }

    void appendDefines(std::ostringstream& out) const {
        for (const auto& [name, value] : options.defines) {
            out << "#define " << name;
            if (!value.empty()) {
                out << ' ' << value;
            }
            out << '\n';
        }
    }

    void processInternal(
        std::string_view source,
        const std::filesystem::path& currentPath,
        std::size_t depth,
        bool injectDefinesAfterVersion,
        std::ostringstream& out
    ) {
        if (depth > options.maxIncludeDepth) {
            throw std::runtime_error(
                "Exceeded maximum shader #include depth (" +
                std::to_string(options.maxIncludeDepth) + ") at: " + currentPath.string()
            );
        }

        const std::size_t fileId = registerSourceFile(currentPath);
        const std::filesystem::path currentDir = currentPath.parent_path();

        std::istringstream iss{std::string(source)};
        std::string rawLine;
        std::size_t lineNumber = 0;
        bool definesInjected = !injectDefinesAfterVersion || options.defines.empty();

        while (std::getline(iss, rawLine)) {
            ++lineNumber;
            const std::string line = stripCarriageReturn(std::move(rawLine));

            if (!definesInjected && isVersionDirective(line)) {
                out << line << '\n';
                appendDefines(out);
                if (options.emitLineDirectives) {
                    out << "#line " << (lineNumber + 1) << ' ' << fileId << '\n';
                }
                definesInjected = true;
                continue;
            }

            std::string includeTarget;
            if (parseIncludeDirective(line, includeTarget)) {
                const std::filesystem::path resolved =
                    resolveIncludePath(includeTarget, currentDir, options.includeDirs);
                const std::string canonicalKey = resolved.generic_string();

                if (activeIncludeStack.contains(canonicalKey)) {
                    throw std::runtime_error("Cyclic shader #include detected: " + canonicalKey);
                }

                if (!definesInjected) {
                    appendDefines(out);
                    definesInjected = true;
                }

                const std::string includedSource = readTextFile(resolved);
                includedFiles.push_back(resolved);
                const std::size_t childFileId = registerSourceFile(resolved);

                if (options.emitLineDirectives) {
                    out << "#line 1 " << childFileId << '\n';
                }

                activeIncludeStack.insert(canonicalKey);
                processInternal(includedSource, resolved, depth + 1, false, out);
                activeIncludeStack.erase(canonicalKey);

                if (options.emitLineDirectives) {
                    out << "#line " << (lineNumber + 1) << ' ' << fileId << '\n';
                }
                continue;
            }

            // If no #version directive was encountered on line 1, inject defines before first code line.
            if (!definesInjected) {
                const std::string trimmed = trimLeft(line);
                if (!trimmed.empty() && !trimmed.starts_with("//")) {
                    appendDefines(out);
                    if (options.emitLineDirectives) {
                        out << "#line " << lineNumber << ' ' << fileId << '\n';
                    }
                    definesInjected = true;
                }
            }

            out << line << '\n';
        }

        if (!definesInjected) {
            appendDefines(out);
        }
    }
};

} // namespace

std::string_view shaderStageName(ShaderStage stage) noexcept {
    switch (stage) {
        case ShaderStage::Vertex:         return "vertex";
        case ShaderStage::Fragment:       return "fragment";
        case ShaderStage::Compute:        return "compute";
        case ShaderStage::Geometry:       return "geometry";
        case ShaderStage::TessControl:    return "tess_control";
        case ShaderStage::TessEvaluation: return "tess_evaluation";
        case ShaderStage::Unknown:        return "unknown";
    }
    return "unknown";
}

ShaderStage inferShaderStage(const std::filesystem::path& path) {
    const std::string ext = toLowerAscii(path.extension().string());
    if (ext == ".vert" || ext == ".vs") {
        return ShaderStage::Vertex;
    }
    if (ext == ".frag" || ext == ".fs") {
        return ShaderStage::Fragment;
    }
    if (ext == ".comp" || ext == ".cs") {
        return ShaderStage::Compute;
    }
    if (ext == ".geom" || ext == ".gs") {
        return ShaderStage::Geometry;
    }
    if (ext == ".tesc") {
        return ShaderStage::TessControl;
    }
    if (ext == ".tese") {
        return ShaderStage::TessEvaluation;
    }
    return ShaderStage::Unknown;
}

PreprocessResult ShaderPreprocessor::preprocessFile(
    const std::filesystem::path& filePath,
    const PreprocessOptions& options
) const {
    const std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(filePath);
    const std::string source = readTextFile(canonicalPath);
    return preprocessSource(source, canonicalPath, options);
}

PreprocessResult ShaderPreprocessor::preprocessSource(
    std::string_view source,
    const std::filesystem::path& virtualPath,
    const PreprocessOptions& options
) const {
    PreprocessContext ctx{options, {}, {}, {}};
    const std::string rootKey = virtualPath.generic_string();
    ctx.activeIncludeStack.insert(rootKey);

    std::ostringstream out;
    ctx.processInternal(source, virtualPath, 0, true, out);

    PreprocessResult result{};
    result.source = out.str();
    result.stage = inferShaderStage(virtualPath);
    result.includedFiles = std::move(ctx.includedFiles);
    result.sourceFileTable = std::move(ctx.sourceFileTable);
    return result;
}

} // namespace glengine
