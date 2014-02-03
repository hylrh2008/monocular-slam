# - Try to find eigen lib
# Once done this will define
#
#  EIGEN_FOUND - system has eigen lib
#  EIGEN_INCLUDE_DIR - the eigen include directory

# Copyright (c) 2006, 2007 Montel Laurent, <montel@kde.org>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

if (EIGEN_INCLUDE_DIRS)

  # in cache already
  set(EIGEN_FOUND TRUE)

else (EIGEN_INCLUDE_DIRS)

  find_path(EIGEN_INCLUDE_DIRS NAMES
    Eigen
    PATHS
    /usr/include/eigen3
    )

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Eigen DEFAULT_MSG EIGEN_INCLUDE_DIRS )


  mark_as_advanced(EIGEN_INCLUDE_DIRS)

endif(EIGEN_INCLUDE_DIRS)

