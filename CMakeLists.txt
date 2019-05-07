cmake_minimum_required (VERSION 2.8)
project (CUDASmallRayTracer)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON) 

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "Configuration types" FORCE)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

file(GLOB SRC_FILES
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)


include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/
  ${CMAKE_SOURCE_DIR}/src/
)

#CMake built-in module
find_package(CUDA QUIET REQUIRED)

#######################################################################################

#set (USE_MSVC_RUNTIME_LIBRARY_DLL true)
#IF (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
#    #We statically link to reduce dependancies
#    FOREACH(flag_var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RETAIL)
#        IF(${flag_var} MATCHES "/MD")
#            STRING(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
#        ENDIF(${flag_var} MATCHES "/MD")
#        IF(${flag_var} MATCHES "/MDd")
#            STRING(REGEX REPLACE "/MDd" "/MTd" ${flag_var} "${${flag_var}}")
#        ENDIF(${flag_var} MATCHES "/MDd")
#    ENDFOREACH(flag_var)
#ENDIF (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)

#######################################################################################

foreach(f ${SRC_FILES})
    # Get the path of the file relative to ${DIRECTORY},
    # then alter it (not compulsory)
    file(RELATIVE_PATH SRCGR ${CMAKE_SOURCE_DIR} ${f})
    set(SRCGR "Sources/${SRCGR}")

    # Extract the folder, ie remove the filename part
    string(REGEX REPLACE "(.*)(/[^/]*)$" "\\1" SRCGR ${SRCGR})

    # Source_group expects \\ (double antislash), not / (slash)
    string(REPLACE / \\ SRCGR ${SRCGR})
    source_group("${SRCGR}" FILES ${f})
endforeach()

set(ENABLE_HIDECONSOLE_BUILD TRUE CACHE BOOL "TRUE to hide console for Windows sys")

if(WINDOWS)
  set(GUI_TYPE WIN32)
elseif(MACOSX)
  set(GUI_TYPE MACOSX_BUNDLE)
endif()

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()


#--------------------------------------------------------------------
# preproc
#--------------------------------------------------------------------
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
add_definitions(-D__STDC_FORMAT_MACROS)
add_definitions(-DBX_CONFIG_ENABLE_MSVC_LEVEL4_WARNINGS=1)
add_definitions(-D__STDC_LIMIT_MACROS)
add_definitions(-D__STDC_CONSTANT_MACROS)
add_definitions(-DIMGUI_DISABLE_OBSOLETE_FUNCTIONS)

if(MSVC)
#add_definitions(-DWIN32)
#add_definitions(-D_WIN32)
#add_definitions(-DNOMINMAX)
endif()

add_definitions(-DUSE_DL_PREFIX)
add_definitions(-DGLEW_STATIC)

SET(LINK_OPTIONS " ")

####################################### Add Executable ###########################################
#Use static cuda link
SET(EXE_NAME "smallptCUDA")
cuda_add_executable(${EXE_NAME} "${CMAKE_SOURCE_DIR}/src/smallptCUDA.cu" ${SRC_FILES})
#--------------------------------------------------------------------
# output dirs
#--------------------------------------------------------------------
set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin )
set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin )
set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_SOURCE_DIR}/bin )
set_target_properties(${EXE_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${EXE_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX "RelWithDebInfo")
set_target_properties(${EXE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

#--------------------------------------------------------------------
# Open the console window in visual studio projects
#--------------------------------------------------------------------
if(MSVC)
set_target_properties(${EXE_NAME} PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE")
set_target_properties(${EXE_NAME} PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:CONSOLE")
endif()

#######################################################################################
##For reference
##https://cmake.org/cmake/help/latest/manual/cmake-properties.7.html#target-properties
#######################################################################################
