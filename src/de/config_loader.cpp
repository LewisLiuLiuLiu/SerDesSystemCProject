#include "de/config_loader.h"
#include <fstream>
#include <iostream>

namespace serdes {

bool ConfigLoader::loadFromFile(const std::string& filepath, SystemParams& params) {
    // Detect file format by extension
    if (filepath.find(".json") != std::string::npos) {
        return loadJSON(filepath, params);
    } else if (filepath.find(".yaml") != std::string::npos || filepath.find(".yml") != std::string::npos) {
        return loadYAML(filepath, params);
    }
    
    std::cerr << "Unknown file format: " << filepath << std::endl;
    return false;
}

bool ConfigLoader::loadJSON(const std::string& filepath, SystemParams& params) {
    // TODO: Implement JSON loading using nlohmann/json
    std::cout << "Loading JSON config from: " << filepath << std::endl;
    // For now, use default parameters
    return true;
}

bool ConfigLoader::loadYAML(const std::string& filepath, SystemParams& params) {
    // TODO: Implement YAML loading using yaml-cpp
    std::cout << "Loading YAML config from: " << filepath << std::endl;
    // For now, use default parameters
    return true;
}

} // namespace serdes
