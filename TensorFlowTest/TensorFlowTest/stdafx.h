// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#if defined(_MSC_VER) && !defined(COMPILER_MSVC)
#  define COMPILER_MSVC // Set MSVC visibility of exported symbols in the shared library.
#endif
#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable : 4190)
#endif

#include "targetver.h"
#include <Windows.h>
#include <iostream>
#include <ctime>
// TODO: reference additional headers your program requires here
