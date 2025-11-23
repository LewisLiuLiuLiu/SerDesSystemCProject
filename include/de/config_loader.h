#ifndef SERDES_CONFIG_LOADER_H
#define SERDES_CONFIG_LOADER_H

#include "common/parameters.h"
#include <string>

namespace serdes {

class ConfigLoader {
public:
    ConfigLoader() = default;
    ~ConfigLoader() = default;
    
    // Load configuration from file (JSON or YAML)
    static bool loadFromFile(const std::string& filepath, SystemParams& params);
    
    // Load default configuration
    static SystemParams load_default();
    
private:
    static bool loadJSON(const std::string& filepath, SystemParams& params);
    static bool loadYAML(const std::string& filepath, SystemParams& params);
};

} // namespace serdes

#endif // SERDES_CONFIG_LOADER_H
