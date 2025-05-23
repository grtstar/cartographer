# Copyright 2016 The Cartographer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO(hrapp): Take from the cmake master branch and simplified. As soon as we
# require a CMakeVersion that ships with this, remove again.

#.rst:
# FindLua
# -------
#
#
#
# Locate Lua library This module defines
#
# ::
#
#   LUA_FOUND          - if false, do not try to link to Lua
#   LUA_LIBRARIES      - both lua and lualib
#   LUA_INCLUDE_DIR    - where to find lua.h
#   LUA_VERSION_STRING - the version of Lua found
#   LUA_VERSION_MAJOR  - the major version of Lua
#   LUA_VERSION_MINOR  - the minor version of Lua
#   LUA_VERSION_PATCH  - the patch version of Lua
#
#
#
# Note that the expected include convention is
#
# ::
#
#   #include "lua.h"
#
# and not
#
# ::
#
#   #include <lua/lua.h>
#
# This is because, the lua location is not standardized and may exist in
# locations other than lua/

#=============================================================================
# Copyright 2007-2009 Kitware, Inc.
# Copyright 2013 Rolf Eike Beer <eike@sf-mail.de>
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

unset(_lua_include_subdirs)
unset(_lua_library_names)
unset(_lua_append_versions)

# this is a function only to have all the variables inside go away automatically
function(_lua_set_version_vars)
    set(LUA_VERSIONS5 5.3 5.2)

    if (Lua_FIND_VERSION_EXACT)
        if (Lua_FIND_VERSION_COUNT GREATER 1)
            set(_lua_append_versions ${Lua_FIND_VERSION_MAJOR}.${Lua_FIND_VERSION_MINOR})
        endif ()
    elseif (Lua_FIND_VERSION)
        # once there is a different major version supported this should become a loop
        if (NOT Lua_FIND_VERSION_MAJOR GREATER 5)
            if (Lua_FIND_VERSION_COUNT EQUAL 1)
                set(_lua_append_versions ${LUA_VERSIONS5})
            else ()
                foreach (subver IN LISTS LUA_VERSIONS5)
                    if (NOT subver VERSION_LESS ${Lua_FIND_VERSION})
                        list(APPEND _lua_append_versions ${subver})
                    endif ()
                endforeach ()
            endif ()
        endif ()
    else ()
        # once there is a different major version supported this should become a loop
        set(_lua_append_versions ${LUA_VERSIONS5})
    endif ()

    list(APPEND _lua_include_subdirs "include/lua" "include")


    foreach (ver IN LISTS _lua_append_versions)
        string(REGEX MATCH "^([0-9]+)\\.([0-9]+)$" _ver "${ver}")
        list(APPEND _lua_include_subdirs
             include/lua${CMAKE_MATCH_1}${CMAKE_MATCH_2}
             include/lua${CMAKE_MATCH_1}.${CMAKE_MATCH_2}
             include/lua-${CMAKE_MATCH_1}.${CMAKE_MATCH_2}
        )
        list(APPEND _lua_library_names
             lua${CMAKE_MATCH_1}${CMAKE_MATCH_2}
             lua${CMAKE_MATCH_1}.${CMAKE_MATCH_2}
             lua-${CMAKE_MATCH_1}.${CMAKE_MATCH_2}
             lua.${CMAKE_MATCH_1}.${CMAKE_MATCH_2}
        )
    endforeach ()

    set(_lua_include_subdirs "${_lua_include_subdirs}" PARENT_SCOPE)
    set(_lua_library_names "${_lua_library_names}" PARENT_SCOPE)
    set(_lua_append_versions "${_lua_append_versions}" PARENT_SCOPE)
endfunction(_lua_set_version_vars)

function(_lua_check_header_version _hdr_file)
    # At least 5.[012] have different ways to express the version
    # so all of them need to be tested. Lua 5.2 defines LUA_VERSION
    # and LUA_RELEASE as joined by the C preprocessor, so avoid those.
    file(STRINGS "${_hdr_file}" lua_version_strings
         REGEX "^#define[ \t]+LUA_(RELEASE[ \t]+\"Lua [0-9]|VERSION([ \t]+\"Lua [0-9]|_[MR])).*")

    string(REGEX REPLACE ".*;#define[ \t]+LUA_VERSION_MAJOR[ \t]+\"([0-9])\"[ \t]*;.*" "\\1" LUA_VERSION_MAJOR ";${lua_version_strings};")
    if (LUA_VERSION_MAJOR MATCHES "^[0-9]+$")
        string(REGEX REPLACE ".*;#define[ \t]+LUA_VERSION_MINOR[ \t]+\"([0-9])\"[ \t]*;.*" "\\1" LUA_VERSION_MINOR ";${lua_version_strings};")
        string(REGEX REPLACE ".*;#define[ \t]+LUA_VERSION_RELEASE[ \t]+\"([0-9])\"[ \t]*;.*" "\\1" LUA_VERSION_PATCH ";${lua_version_strings};")
        set(LUA_VERSION_STRING "${LUA_VERSION_MAJOR}.${LUA_VERSION_MINOR}.${LUA_VERSION_PATCH}")
    else ()
        string(REGEX REPLACE ".*;#define[ \t]+LUA_RELEASE[ \t]+\"Lua ([0-9.]+)\"[ \t]*;.*" "\\1" LUA_VERSION_STRING ";${lua_version_strings};")
        if (NOT LUA_VERSION_STRING MATCHES "^[0-9.]+$")
            string(REGEX REPLACE ".*;#define[ \t]+LUA_VERSION[ \t]+\"Lua ([0-9.]+)\"[ \t]*;.*" "\\1" LUA_VERSION_STRING ";${lua_version_strings};")
        endif ()
        string(REGEX REPLACE "^([0-9]+)\\.[0-9.]*$" "\\1" LUA_VERSION_MAJOR "${LUA_VERSION_STRING}")
        string(REGEX REPLACE "^[0-9]+\\.([0-9]+)[0-9.]*$" "\\1" LUA_VERSION_MINOR "${LUA_VERSION_STRING}")
        string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]).*" "\\1" LUA_VERSION_PATCH "${LUA_VERSION_STRING}")
    endif ()
    foreach (ver IN LISTS _lua_append_versions)
        if (ver STREQUAL "${LUA_VERSION_MAJOR}.${LUA_VERSION_MINOR}")
            set(LUA_VERSION_MAJOR ${LUA_VERSION_MAJOR} PARENT_SCOPE)
            set(LUA_VERSION_MINOR ${LUA_VERSION_MINOR} PARENT_SCOPE)
            set(LUA_VERSION_PATCH ${LUA_VERSION_PATCH} PARENT_SCOPE)
            set(LUA_VERSION_STRING ${LUA_VERSION_STRING} PARENT_SCOPE)
            return()
        endif ()
    endforeach ()
endfunction(_lua_check_header_version)

_lua_set_version_vars()

if (LUA_INCLUDE_DIR AND EXISTS "${LUA_INCLUDE_DIR}/lua.h")
    message(STATUS "Found Lua include dir: ${LUA_INCLUDE_DIR}")
    _lua_check_header_version("${LUA_INCLUDE_DIR}/lua.h")
endif ()

if (NOT LUA_VERSION_STRING)
    foreach (subdir IN LISTS _lua_include_subdirs)
        unset(LUA_INCLUDE_PREFIX CACHE)
        find_path(LUA_INCLUDE_PREFIX ${subdir}/lua.h
          HINTS
            ENV LUA_DIR
          PATHS
          ~/Library/Frameworks
          /Library/Frameworks
          /sw # Fink
          /opt/local # DarwinPorts
          /opt/csw # Blastwave
          /opt
        )
        message(STATUS "SYSROOT: $ENV{SYSROOT}/${subdir}/lua.h")
        message(STATUS "Found LUA_INCLUDE_PREFIX: ${LUA_INCLUDE_PREFIX}")
        # Check if $ENV{SYSROOT}/${subdir}/lua.h exists and set LUA_INCLUDE_PREFIX accordingly
        if(EXISTS "$ENV{SYSROOT}/${subdir}/lua.h")
            set(LUA_INCLUDE_PREFIX "$ENV{SYSROOT}")
            message(STATUS "Found lua.h in SYSROOT: $ENV{SYSROOT}/${subdir}/lua.h")
        endif()
        if (LUA_INCLUDE_PREFIX)
            _lua_check_header_version("${LUA_INCLUDE_PREFIX}/${subdir}/lua.h")
            if (LUA_VERSION_STRING)
                set(LUA_INCLUDE_DIR "${LUA_INCLUDE_PREFIX}/${subdir}")
                break()
            endif ()
        endif ()
    endforeach ()
endif ()
unset(_lua_include_subdirs)
unset(_lua_append_versions)

find_library(LUA_LIBRARY
  NAMES ${_lua_library_names} lua
  HINTS
    ENV LUA_DIR
  PATH_SUFFIXES lib
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /sw
  /opt/local
  /opt/csw
  /opt
)
unset(_lua_library_names)

if (LUA_LIBRARY)
    # include the math library for Unix
    if (UNIX AND NOT APPLE AND NOT BEOS)
        find_library(LUA_MATH_LIBRARY m)
        set(LUA_LIBRARIES "${LUA_LIBRARY};${LUA_MATH_LIBRARY}")
    # For Windows and Mac, don't need to explicitly include the math library
    else ()
        set(LUA_LIBRARIES "${LUA_LIBRARY}")
    endif ()
endif ()

# handle the QUIETLY and REQUIRED arguments and set LUA_FOUND to TRUE if
# all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Lua
                                  REQUIRED_VARS LUA_LIBRARIES LUA_INCLUDE_DIR
                                  VERSION_VAR LUA_VERSION_STRING)
mark_as_advanced(LUA_INCLUDE_DIR LUA_LIBRARY LUA_MATH_LIBRARY)

if (NOT LUA_FOUND)
  MESSAGE(FATAL_ERROR "Did not find Lua >= 5.2.")
endif ()


