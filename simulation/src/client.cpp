#define _USE_MATH_DEFINES
#include <cmath>
// #include <math.h>
#include <string>
#include <vector>
// #include <stdio.h>
// #include <unistd.h>
#include "raylib.h"
#include "imgui.h"
#include "rlImGui.h"
#include "scene.h"
// #include "box2d/id.h"
// #include "box2d/math_functions.h"
// #include "box2d/box2d.h"

const std::string JSON_PATH = "last_scene.json";

enum VisualState { EMPTY, ENV_LOADED, TOOL_PLACED, SIMULATING };
const float kScale = 80.0f; // Pixels per meter

void DrawScene(Scene& scene, bool drawTool) {
    // Draw Walls
    for (const auto& w : scene.walls) {
        Vector2 position = { w.x * kScale, 720.0f - w.y * kScale };
        Vector2 size = { w.width * kScale, w.height * kScale };
        Rectangle wall = {  position.x + size.x / 2.0f,
                            position.y - size.y / 2.0f, 
                            size.x, size.y };
        DrawRectanglePro(wall, {wall.width/2.0f, wall.height/2.0f}, -w.rotation, BLACK);
    }

    // Draw Balls
    for (size_t i = 0; i < scene.balls.size(); ++i) {
        b2Vec2 pos = b2Body_GetPosition(scene.ballIds[i]);
        DrawCircleV({pos.x * kScale, 720.0f - pos.y * kScale}, scene.balls[i].radius * kScale, RED);
    }

    // Draw Tool
    if (drawTool && B2_IS_NON_NULL(scene.toolId)) {
        b2Rot rot = b2Body_GetRotation(scene.toolId);
        float angle = b2Rot_GetAngle(rot) * (180.0f / (float)M_PI);
        for (const auto& seg : scene.toolSegments) {
            b2Vec2 p = b2Body_GetWorldPoint(scene.toolId, {seg.x, seg.y});
            Rectangle r = { p.x * kScale, 720.0f - p.y * kScale, seg.length * kScale, 0.1f * kScale };
            DrawRectanglePro(r, {r.width/2.0f, r.height/2.0f}, -angle - seg.rotation, DARKGRAY);
        }
    }
}

int main() {
    InitWindow(1280, 720, "ML Robo - Tool Design Visualizer");
    rlImGuiSetup(true);
    SetTargetFPS(60);

    Scene scene = {};
    VisualState state = EMPTY;
    bool isAutomatic = true;
    float stateTimer = 0.0f;
    char buffer[100000] = "";   // For scene_definition_string (JSON)

    while (!WindowShouldClose()) {
        // --- Update Scene Definition ---
        static double lastFileTime = 0.0;
        double currentFileTime = GetFileModTime(JSON_PATH.c_str());

        if (currentFileTime > lastFileTime) {
            lastFileTime = currentFileTime;
            
            int fileSize = 0;
            unsigned char* fileData = LoadFileData(JSON_PATH.c_str(), &fileSize);
            if (fileData && fileSize < sizeof(buffer)) {
                memset(buffer, 0, sizeof(buffer));
                memcpy(buffer, fileData, fileSize);
                UnloadFileData(fileData);

                try {
                    if (B2_IS_NON_NULL(scene.worldId)) cleanup_scene(scene);
                    json d = json::parse(buffer);
                    auto objs = d["scene"].get<std::vector<SceneObject>>();
                    Tool emptyTool = {0, 0, {}};
                    scene = create_scene(objs, emptyTool); // Initialize scene without tool first
                    state = ENV_LOADED;
                    stateTimer = 1.0f; // 1 second delay before showing tool
                } catch (const std::exception& e) {
                    printf("Error parsing JSON: %s\n", e.what());
                }
            }
        }
        
        // --- Transition Handling ---
        float dt = GetFrameTime();
        bool triggerNext = false;
        if (isAutomatic && state != EMPTY && state != SIMULATING) {
            stateTimer -= dt;
            if (stateTimer <= 0) triggerNext = true;
        }

        // --- Rendering ---
        BeginDrawing();
        ClearBackground(RAYWHITE);
        if (state != EMPTY) DrawScene(scene, state >= TOOL_PLACED);

        rlImGuiBegin();
        ImGui::Begin("Controller");

        // Display current state
        const char* stateNames[] = {"EMPTY", "ENV_LOADED", "TOOL_PLACED", "SIMULATING"};
        ImGui::Text("Current State: %s", stateNames[state]);
        ImGui::ProgressBar((float)state / 3.0f);
        
        // Mode Selection
        ImGui::Separator();
        ImGui::Checkbox("Automatic Transitions", &isAutomatic);

        if (!isAutomatic) {
            ImGui::BeginDisabled(state == EMPTY || state == SIMULATING);
            if (ImGui::Button("NEXT STEP >>", ImVec2(-1, 40))) {
                triggerNext = true;
            }
            ImGui::EndDisabled();
        }

        if (ImGui::Button("Reset View")) {
            if (B2_IS_NON_NULL(scene.worldId)) cleanup_scene(scene);
            state = EMPTY;
        }

        // Display last received JSON
        ImGui::Separator();
        ImGui::Text("Last Received JSON:");
        ImGui::InputTextMultiline("##json", buffer, sizeof(buffer), ImVec2(-1, 250));

        // Help Message
        if (state == EMPTY) {
            ImGui::TextColored({1,0,0,1}, "Waiting for JSON from Python/Server...");
        }
        
        ImGui::End();
        rlImGuiEnd();
        EndDrawing();

        // --- Simulation Step ---
        if (triggerNext) {
            if (state == ENV_LOADED) {
                try {
                    json d = json::parse(buffer);
                    Tool t = d["tool"].get<Tool>();
                    add_tool_to_scene(scene, t);
                    state = TOOL_PLACED;
                    stateTimer = 1.0f;
                } catch (const std::exception& e) {
                    printf("Error parsing tool JSON: %s\n", e.what());
                }
            } else if (state == TOOL_PLACED) {
                state = SIMULATING;
            }
        }

        if (state == SIMULATING) b2World_Step(scene.worldId, dt, scene.subStepCount);
    }

    cleanup_scene(scene);
    rlImGuiShutdown();
    CloseWindow();

    return 0;
}
