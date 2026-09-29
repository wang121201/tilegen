#pragma once
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace tilegen_source {
inline const std::string& legacy_root_prefix() {
    static const std::string value =
        "/Users/wgs/Documents/Codex/2026-09-14/"
        "hbserve-memgen-gtsim-alignment/work/";
    return value;
}

inline const std::filesystem::path& configured_root() {
    namespace fs = std::filesystem;
    static const fs::path value = [] {
        const char* raw = std::getenv("TILEGEN_XMU_SOURCE_ROOT");
        if (raw == nullptr || *raw == '\0')
            throw std::runtime_error(
                "TILEGEN_XMU_SOURCE_ROOT is required for pinned source inputs");
        const fs::path requested(raw);
        if (!requested.is_absolute() || fs::is_symlink(fs::symlink_status(requested)) ||
            !fs::is_directory(requested))
            throw std::runtime_error(
                "TILEGEN_XMU_SOURCE_ROOT must be an absolute, existing, non-symlink directory");
        return fs::canonical(requested);
    }();
    return value;
}

inline std::string resolve(const std::string& logical) {
    namespace fs = std::filesystem;
    const auto& root = configured_root();
    fs::path candidate;
    const auto& legacy = legacy_root_prefix();
    if (logical.compare(0, legacy.size(), legacy) == 0) {
        const fs::path relative(logical.substr(legacy.size()));
        if (relative.empty() || relative.is_absolute())
            throw std::runtime_error("invalid legacy pinned source suffix");
        for (const auto& part : relative)
            if (part == "..")
                throw std::runtime_error("pinned source path traversal rejected");
        candidate = root / relative;
    } else {
        candidate = fs::path(logical);
        if (!candidate.is_absolute())
            throw std::runtime_error("pinned source path must be absolute");
    }
    if (fs::is_symlink(fs::symlink_status(candidate)) || !fs::is_regular_file(candidate))
        throw std::runtime_error("pinned source must resolve to an existing non-symlink file");
    const auto physical = fs::canonical(candidate);
    const auto relative = fs::relative(physical, root);
    if (relative.empty() || relative.is_absolute() || *relative.begin() == "..")
        throw std::runtime_error("pinned source resolved outside TILEGEN_XMU_SOURCE_ROOT");
    return physical.string();
}
}
