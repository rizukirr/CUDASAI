#pragma once

#include "kernel.h"

template <typename T> void test_array(Cds *root);
template <typename T> void test_cds_to_meta(Cds *root);
template <typename T> void test_view(Cds *root);
template <typename T> void test_copy(Cds *root);
template <typename T> void test_random(Cds *root);
template <typename T> void test_checkpoint(Cds *root);
template <typename T> void test_fill(Cds *root, CdsContext *context);
template <typename T> void test_add(Cds *root, CdsContext *context);
template <typename T> void test_scale(Cds *root, CdsContext *context);
template <typename T> void test_linear(Cds *root, CdsContext *context);
template <typename T> void test_softmax(Cds *root, CdsContext *context);
template <typename T> void test_context(Cds *root);
void test_host();
