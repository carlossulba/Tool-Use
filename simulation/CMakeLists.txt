cmake_minimum_required(VERSION 4.2.3)
project(SimulationEnvironment VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB SOURCES "src/*.cpp")

add_subdirectory(external)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(SimulationServer
  src/server.cpp
)

add_executable(SimulationClient
  src/client.cpp
)

target_link_libraries(SimulationServer PUBLIC box2d uWebSockets nlohmann_json::nlohmann_json)
target_link_libraries(SimulationClient PUBLIC box2d nlohmann_json::nlohmann_json raylib imgui rlimgui)
