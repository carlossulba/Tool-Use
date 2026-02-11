#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

#include "box2d/box2d.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// --- Physical Constants ---
namespace PhysicsConfig {
    const float k_gravity = -9.81f;
    const float k_toolThickness = 0.1f;
    const float k_toolDensity = 5.0f;
    const float k_ballDensity = 1.0f;
    const float k_friction = 0.3f;
    const float k_collisionThreshold = 0.0001f;
}

// --- State information ---
struct SceneObject {
    int type;   // 0: Wall, 1: Ball
    float x, y;
    float width, height; // half-width and half-height
    float rotation;
};

struct ToolSegment {
    float angle;
    float length;
};

struct Tool {
    float x, y;
    std::vector<ToolSegment> segments;
};

struct Wall {
    float x, y;
    float width, height;
    float rotation;
};

struct Ball {
    float radius;
};

struct Segment {
    float x, y;
    float length;
    float rotation;
};

struct Result {
    int completion;         // 1 if balls touched, 0 otherwise
    int steps;              // steps taken
    int maxSteps;           // timeout limit
    float initialDistance;  // distance at start
    float eucledianDistance; // actual distance
    float actualDistance;   // path finding distance
};

// --- JSON conversions ---
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SceneObject, type, x, y, width, height, rotation)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ToolSegment, angle, length)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Tool, x, y, segments)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Result, completion, steps, maxSteps, initialDistance, eucledianDistance, actualDistance)


// --- Scene ---
struct Scene {
    b2WorldId worldId;
    int timeout;
    int timeStep;   // Timesteps per second
    int subStepCount;

    std::vector<b2BodyId> wallIds;
    std::vector<Wall> walls;
    std::vector<b2BodyId> ballIds;
    std::vector<Ball> balls;

    b2BodyId toolId;
    std::vector<Segment> toolSegments;
};

inline float DegToRad(float deg) { return deg * (static_cast<float>(M_PI) / 180.0f); }


Scene create_scene(std::vector<SceneObject>& scene_objects, Tool& tool) {
    /*  Creates a Box2D world from the given scene objects and tool definition. 
        Returns a Scene structure.
    */
   // Initialize Box2D world
    b2WorldDef worldDef = b2DefaultWorldDef();
    worldDef.gravity = {0.0f, PhysicsConfig::k_gravity};
    b2WorldId worldId = b2CreateWorld(&worldDef);

    Scene scene = {worldId, 60, 120, 4 };

    // Create Static Walls and Dynamic Balls
    for (const auto& obj : scene_objects) {
        b2BodyDef bodyDef = b2DefaultBodyDef();
        bodyDef.position = { obj.x, obj.y };

        if (obj.type == 0) { // Wall (Static)
            b2BodyId id = b2CreateBody(worldId, &bodyDef);
            float rad = DegToRad(obj.rotation);
            
            b2Polygon box = b2MakeOffsetBox(
                obj.width / 2.0f, obj.height / 2.0f,
                { obj.width / 2.0f, obj.height / 2.0f },
                b2MakeRot(rad)
            );

            b2ShapeDef shapeDef = b2DefaultShapeDef();
            b2CreatePolygonShape(id, &shapeDef, &box);
            
            scene.wallIds.push_back(id);
            scene.walls.push_back({obj.x, obj.y, obj.width, obj.height, obj.rotation});

        } else if (obj.type == 1) { // Ball (Dynamic)
            bodyDef.type = b2_dynamicBody;
            b2BodyId id = b2CreateBody(worldId, &bodyDef);

            b2Circle circle = {{0.0f, 0.0f}, obj.width}; // width is used as radius
            
            b2ShapeDef shapeDef = b2DefaultShapeDef();
            shapeDef.density = PhysicsConfig::k_ballDensity;
            shapeDef.material.friction = PhysicsConfig::k_friction;

            b2CreateCircleShape(id, &shapeDef, &circle);
            
            scene.ballIds.push_back(id);
            scene.balls.push_back({obj.width});
        }
    }

    // Create Tool
    b2BodyDef toolDef = b2DefaultBodyDef();
    toolDef.type = b2_dynamicBody;
    toolDef.position = { tool.x, tool.y };
    scene.toolId = b2CreateBody(worldId, &toolDef);

    float curX = 0.0f, curY = 0.0f;
    for (const auto& seg : tool.segments) {
        float rad = DegToRad(seg.angle);
        float c = cosf(rad), s = sinf(rad);

        // Calculate center of the segment relative to tool origin
        float centerX = curX + c * seg.length / 2.0f;
        float centerY = curY + s * seg.length / 2.0f;

        b2Polygon segmentBox = b2MakeOffsetBox(
            seg.length / 2.0f, PhysicsConfig::k_toolThickness / 2.0f,
            { centerX, centerY },
            b2MakeRot(rad)
        );

        b2ShapeDef shapeDef = b2DefaultShapeDef();
        shapeDef.density = PhysicsConfig::k_toolDensity;
        shapeDef.material.friction = PhysicsConfig::k_friction;

        b2CreatePolygonShape(scene.toolId, &shapeDef, &segmentBox);
        scene.toolSegments.push_back({centerX, centerY, seg.length, seg.angle});

        curX += c * seg.length;
        curY += s * seg.length;
    }

    return scene;
}

Result simulate_scene(Scene& scene) {
    const int maxSteps = scene.timeout * scene.timeStep;
    const float timeStep = 1.0f / static_cast<float>(scene.timeStep);

    // We only consider one type of task for now: two balls that need to touch each other.
    float combinedRadius = (scene.balls.size() >= 2) ? 
        (scene.balls[0].radius + scene.balls[1].radius) : 0.0f;

    auto getDist = [&]() {
        b2Vec2 p1 = b2Body_GetPosition(scene.ballIds[0]);
        b2Vec2 p2 = b2Body_GetPosition(scene.ballIds[1]);
        return sqrtf(powf(p1.x - p2.x, 2.0f) + powf(p1.y - p2.y, 2.0f)) - combinedRadius;
    };

    float initialDistance = getDist();
    float currentDistance = initialDistance;
    int steps = 0;
    int completion = 0;

    // Simulation Loop
    for (; steps < maxSteps; ++steps) {
        b2World_Step(scene.worldId, timeStep, scene.subStepCount);
        currentDistance = getDist();

        if (currentDistance <= PhysicsConfig::k_collisionThreshold) {
            completion = 1;
            break;
        }
    }

    float eucledianDistance = std::max(0.0f, currentDistance);
    float actualDistance = 0.0f;    // Not implemented yet (A* could be used here)
    return { completion, steps, maxSteps, initialDistance, eucledianDistance, actualDistance };
}

void cleanup_scene(Scene& scene) {
    b2DestroyWorld(scene.worldId);
}

void add_tool_to_scene(Scene& scene, Tool& tool) {
    /*  Adds a tool to a scene. */
    
    // Destroy existing tool
    if (B2_IS_NON_NULL(scene.toolId)) {
        b2DestroyBody(scene.toolId);
        scene.toolSegments.clear();
    }

    b2BodyDef toolDef = b2DefaultBodyDef();
    toolDef.type = b2_dynamicBody;
    toolDef.position = { tool.x, tool.y };
    scene.toolId = b2CreateBody(scene.worldId, &toolDef);

    // Add tool segments one by one
    float curX = 0.0f, curY = 0.0f;
    for (const auto& seg : tool.segments) {
        float rad = seg.angle * (static_cast<float>(M_PI) / 180.0f);
        float c = cosf(rad), s = sinf(rad);
        float centerX = curX + c * seg.length / 2.0f;
        float centerY = curY + s * seg.length / 2.0f;
        
        b2Polygon segmentBox = b2MakeOffsetBox(
            seg.length / 2.0f, 0.1f / 2.0f,     // toolSegmentThickness = 0.1f
            { centerX, centerY }, 
            b2MakeRot(rad)
        );

        b2ShapeDef shapeDef = b2DefaultShapeDef();
        shapeDef.density = 5.0f;
        b2CreatePolygonShape(scene.toolId, &shapeDef, &segmentBox);

        scene.toolSegments.push_back({centerX, centerY, seg.length, seg.angle});
        curX += c * seg.length;
        curY += s * seg.length;
    }
}
