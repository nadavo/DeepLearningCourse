# Install script for directory: /home/nadavo@st.technion.ac.il/git-repo/eladtools

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/nadavo@st.technion.ac.il/torch/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "0")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/eladtools/scm-1/lua/eladtools" TYPE FILE FILES
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/EarlyStop.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/GlobalDominantPooling.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/NetConversion.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/ODCT.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/Optimizer.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/RecurrentLayer.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/SSU.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/SelectPoint.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/SpatialBottleNeck.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/SpatialConvolutionDCT.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/SpatialNMS.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/init.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/testSwallowBN.lua"
    "/home/nadavo@st.technion.ac.il/git-repo/eladtools/utils.lua"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/nadavo@st.technion.ac.il/git-repo/eladtools/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/nadavo@st.technion.ac.il/git-repo/eladtools/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
