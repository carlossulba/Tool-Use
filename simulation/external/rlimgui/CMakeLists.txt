cmake_minimum_required(VERSION 4.2.3)

include(FetchContent)
FetchContent_Declare(
  rlimgui_content
  GIT_REPOSITORY https://github.com/raylib-extras/rlImGui
  GIT_TAG main
  GIT_SHALLOW ON
  GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(rlimgui_content)

add_library(rlimgui STATIC ${rlimgui_content_SOURCE_DIR}/rlImGui.cpp)
target_link_libraries(rlimgui PRIVATE imgui raylib)
target_include_directories(rlimgui INTERFACE ${rlimgui_content_SOURCE_DIR})
