#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <iomanip>
#include "App.h"
#include "nlohmann/json.hpp"
#include "scene.h"


using json = nlohmann::json;

struct PerSocketData {
    bool isViewer;
};

void logResult(const Result& result, const Tool& tool) {
    static std::mutex ioMutex;
    static uint64_t simCount = 0;
    static uint64_t successCount = 0;
    static auto startTime = std::chrono::steady_clock::now();

    // Log results to file
    {
        std::lock_guard<std::mutex> lock(ioMutex);
        std::ofstream out("sim_results.tsv", std::ios::app);
        out << result.eucledianDistance << "\t" << tool.segments.size() << "\n";
    }

    // Log progress every 1000 sims
    simCount++;
    if (result.completion == 1) successCount++;

    if (simCount % 1000 == 0) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        std::cout << "[Server] Sims: " << simCount 
                  << " | Success: " << (successCount / 1000.0) * 100.0 << "%"
                  << " | Rate: " << 1000.0 / elapsed << " s/s" << std::endl;
        successCount = 0;
        startTime = now;
    }
}

int main() {
    auto app = uWS::App();

    app.ws<PerSocketData>("/*", {
        .compression = uWS::SHARED_COMPRESSOR,
        .maxPayloadLength = 16 * 1024 * 1024,
        .idleTimeout = 16,

        .open = [](auto *ws) {
            auto* data = (PerSocketData*)ws->getUserData();
            data->isViewer = false;
            std::cout << "[Server] Connection opened." << std::endl;
        },
        .message = [&app](auto *ws, std::string_view message, uWS::OpCode opCode) {
            if (message == "IDENT_VIEWER") {
                auto* data = (PerSocketData*)ws->getUserData();
                data->isViewer = true;
                ws->subscribe("visuals");
                std::cout << "[Server] Visualizer subscribed to 'visuals'." << std::endl;
                return;
            }

            try {
                // BROADCAST to all viewers in the topic
                app.publish("visuals", message, opCode, false);

                json scene_data = json::parse(message);
                auto scene_objs = scene_data["scene"].get<std::vector<SceneObject>>();
                auto tool_data = scene_data["tool"].get<Tool>();
                
                Scene scene = create_scene(scene_objs, tool_data);
                Result res = simulate_scene(scene);
                
                logResult(res, tool_data);
                ws->send(json(res).dump(), opCode, false);

                cleanup_scene(scene);
            } catch (const std::exception& e) {
                std::cerr << "[Server] Error: " << e.what() << std::endl;
            }
        },
        .close = [](auto *ws, int code, std::string_view message) {
            std::cout << "[Server] Connection closed." << std::endl;
        }
    }).listen(3000, [](auto *token) {
        if (token) std::cout << "[Server] Simulation Engine listening on port 3000" << std::endl;
    }).run();

    return 0;
}
