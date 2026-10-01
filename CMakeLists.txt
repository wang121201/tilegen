cmake_minimum_required(VERSION 3.16)
project(tilegen_trace_cosim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
add_executable(tilegen_native
  source/work/tilegen-full-r1/canonical-full-runtime-r4/streaming.cpp
  source/work/tilegen-full-r1/core-native-copy-r2/src/gpu.cpp
  source/work/tilegen-full-r1/core-native-copy-r2/src/scheduler.cpp
  source/work/tilegen-full-r1/core-native-copy-r2/src/simulator.cpp
  source/work/tilegen-full-r1/core-native-copy-r2/src/simulator_session.cpp
  source/work/tilegen-full-r1/core-native-copy-r2/src/sm.cpp
  source/work/tilegen-full-r1/core-native-copy-r2/src/subpartition.cpp
  source/work/hbfsim-latest/upstream/src/app/system_config.cpp
  source/work/hbfsim-latest/upstream/src/physical/address_heatmap.cpp
  source/work/hbfsim-latest/upstream/src/physical/base_die_link.cpp
  source/work/hbfsim-latest/upstream/src/physical/external/cxl_ssd.cpp
  source/work/hbfsim-latest/upstream/src/physical/external/external_backing_device.cpp
  source/work/hbfsim-latest/upstream/src/physical/hbm/hbm_device.cpp
  source/work/hbfsim-latest/upstream/src/physical/hbf/hbf_device.cpp
  source/work/hbfsim-latest/upstream/src/physical/resource_calendar.cpp
  source/work/hbfsim-latest/upstream/src/physical/gap_calendar.cpp
  source/work/hbfsim-latest/upstream/src/host/hbf_gc.cpp
  source/work/hbfsim-latest/upstream/src/host/hbf_zones.cpp
  source/work/hbfsim-latest/upstream/src/host/hbf_wear_snapshot.cpp
  source/work/hbfsim-latest/upstream/src/host/hbf_controller.cpp
  source/work/hbfsim-latest/upstream/src/host/hbf_persistent_image.cpp
  source/work/hbfsim-latest/upstream/src/physical/simulation_session.cpp
)
target_include_directories(tilegen_native PRIVATE
  source/work/tilegen-full-r1/core-native-copy-r2/include
  source/work/tilegen-evaluation-r1/driver
  source/work/tilegen-norm-shared-r1/adapter
  source/work/gddr6-support/delivery-stage/sources/third_party/gtsim-prepared/third_party
  source/work/hbfsim-latest/upstream/src
  source/work/gddr6-support/hbf_coupling
)
target_compile_definitions(tilegen_native PRIVATE TILEGEN_DIRTY_SECTOR_MODE=2)
target_compile_options(tilegen_native PRIVATE -Wall -Wextra)
target_link_libraries(tilegen_native PRIVATE ZLIB::ZLIB Threads::Threads)

# The replay executable links the existing HBFSIM sources, but no GPU simulator TUs.
get_target_property(TILEGEN_INCLUDES tilegen_native INCLUDE_DIRECTORIES)
get_target_property(TILEGEN_SOURCES tilegen_native SOURCES)
set(REPLAY_HBFSIM_SOURCES ${TILEGEN_SOURCES})
list(FILTER REPLAY_HBFSIM_SOURCES INCLUDE REGEX "/hbfsim-latest/")
add_executable(tilegen_replay source/trace_replay_main.cpp ${REPLAY_HBFSIM_SOURCES})
target_include_directories(tilegen_replay PRIVATE ${TILEGEN_INCLUDES})
target_compile_definitions(tilegen_replay PRIVATE TILEGEN_DIRTY_SECTOR_MODE=2)
target_compile_options(tilegen_replay PRIVATE -Wall -Wextra)
target_link_libraries(tilegen_replay PRIVATE ZLIB::ZLIB Threads::Threads)

# TileGen-owned post-cache service adapter. HBFSim sources are linked unchanged; no HBFSim edits.
add_executable(programgraph_hbfsim_service source/programgraph_hbfsim_service_main.cpp ${REPLAY_HBFSIM_SOURCES})
target_include_directories(programgraph_hbfsim_service PRIVATE source ${TILEGEN_INCLUDES})
target_compile_definitions(programgraph_hbfsim_service PRIVATE TILEGEN_DIRTY_SECTOR_MODE=2)
target_compile_options(programgraph_hbfsim_service PRIVATE -Wall -Wextra)
target_link_libraries(programgraph_hbfsim_service PRIVATE ZLIB::ZLIB Threads::Threads)

include(CTest)
if(BUILD_TESTING)
  get_target_property(TILEGEN_INCLUDES tilegen_native INCLUDE_DIRECTORIES)
  foreach(TEST_NAME dirty32_test direct_cache_smoke native_trace_test cta_validation_test trace_replay_test stage_replay_test direct_phase_profile_test replay_phase_report_test replay_backend_test)
    add_executable(${TEST_NAME} tests/${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE source ${TILEGEN_INCLUDES})
    target_compile_definitions(${TEST_NAME} PRIVATE TILEGEN_DIRTY_SECTOR_MODE=2)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME}-artifacts")
  endforeach()
  target_sources(replay_backend_test PRIVATE ${REPLAY_HBFSIM_SOURCES})
  target_link_libraries(replay_backend_test PRIVATE Threads::Threads)
  foreach(HBF_TEST native_trace_test trace_replay_test stage_replay_test)
  target_sources(${HBF_TEST} PRIVATE
    source/work/hbfsim-latest/upstream/src/physical/hbm/hbm_device.cpp
    source/work/hbfsim-latest/upstream/src/physical/resource_calendar.cpp
    source/work/hbfsim-latest/upstream/src/physical/gap_calendar.cpp
    source/work/hbfsim-latest/upstream/src/physical/address_heatmap.cpp)
  endforeach()
endif()
