cmake_minimum_required(VERSION 4.2.3)

include(FetchContent)
FetchContent_Declare(
  imgui_content
  GIT_REPOSITORY https://github.com/ocornut/imgui
  GIT_TAG v1.92.5
  GIT_SHALLOW ON
  GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(imgui_content)

add_library(
  imgui STATIC
  ${imgui_content_SOURCE_DIR}/imgui.cpp
  ${imgui_content_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_content_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_content_SOURCE_DIR}/imgui_tables.cpp
)
target_include_directories(imgui INTERFACE ${imgui_content_SOURCE_DIR})
