cmake_minimum_required(VERSION 4.2.3)

include(FetchContent)
FetchContent_Declare(
  box2d_content
  GIT_REPOSITORY https://github.com/erincatto/box2d
  GIT_TAG v3.1.1
  GIT_SHALLOW ON
  GIT_SUBMODULES ""
)
FetchContent_MakeAvailable(box2d_content)
