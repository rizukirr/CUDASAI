#pragma once

#include "array.h"
#include <cstdio>

// Wait for GPU work, copy values to the host, and print shape and nested
// values. Returns false for invalid arguments or a CUDA synchronization error.
bool print_array(const CdsArray *array, FILE *output = stdout);
