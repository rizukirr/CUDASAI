#include "array.h"
#include "helper.h"
#include "kernel.h"
#include <cstdio>
#include <cstdlib>

constexpr size_t in_features = 3, hidden = 4, out_features = 2;

typedef struct {
  CdsArray *w1, *b1;
  CdsArray *gamma, *beta;
  CdsArray *w2, *b2;
} Model;

static bool model_init(const CdsContext *ctx, Cds *params, Model *m) {
  const size_t w1_shape[] = {hidden, in_features}, h_shape[] = {hidden};
  const size_t w2_shape[] = {out_features, hidden}, o_shape[] = {out_features};
  const float one = 1.0f;

  m->w1 = cds_array_rand(params, w1_shape, 2, CDS_TYPE_FLOAT32);
  m->b1 = cds_array_rand(params, h_shape, 1, CDS_TYPE_FLOAT32);
  m->gamma = cds_array(params, h_shape, 1, CDS_TYPE_FLOAT32, nullptr);
  m->beta = cds_array(params, h_shape, 1, CDS_TYPE_FLOAT32, nullptr);
  m->w2 = cds_array_rand(params, w2_shape, 2, CDS_TYPE_FLOAT32);
  m->b2 = cds_array_rand(params, o_shape, 1, CDS_TYPE_FLOAT32);

  return m->w1 && m->b1 && m->gamma && m->beta && m->w2 && m->b2 &&
         cds_fill(ctx, m->gamma, &one) == CDS_OK;
}

static CdsArray *forward(const CdsContext *ctx, Cds *acts, const Model *m,
                         const CdsArray *x, size_t batch) {
  const size_t h_shape[] = {batch, hidden};
  const size_t y_shape[] = {batch, out_features};

  CdsArray *h = cds_array(acts, h_shape, 2, CDS_TYPE_FLOAT32, nullptr);
  CdsArray *logits = cds_array(acts, y_shape, 2, CDS_TYPE_FLOAT32, nullptr);
  CdsArray *y = cds_array(acts, y_shape, 2, CDS_TYPE_FLOAT32, nullptr);

  if (cds_linear(ctx, x, m->w1, m->b1, CDS_ACTIVATION_GELU, h) != CDS_OK ||
      cds_layernorm(ctx, h, m->gamma, m->beta, 1e-5, h) != CDS_OK ||
      cds_linear(ctx, h, m->w2, m->b2, CDS_ACTIVATION_NONE, logits) != CDS_OK ||
      cds_softmax(ctx, logits, y) != CDS_OK)
    return nullptr;

  return y;
}

int main() {
  CdsContext *ctx = cds_context_new(32 << 20);
  Cds *params = cds_new(1 << 20);
  Cds *acts = cds_new(1 << 20);
  Model model{};
  int status = 1;
  srand(42);

  if (!ctx || !params || !acts || !model_init(ctx, params, &model)) {
    fprintf(stderr, "setup failed: %s\n", cds_last_error_message());
    goto done;
  }

  {
    const size_t batch = 4;
    const size_t x_shape[] = {batch, in_features};
    const float x_data[batch][in_features] = {
        {0.0f, 1.0f, 2.0f},
        {0.3f, 4.0f, 5.0f},
        {0.6f, 7.0f, 8.0f},
        {0.9f, 1.0f, 2.0f},
    };

    CdsArray *x = cds_array(acts, x_shape, 2, CDS_TYPE_FLOAT32, x_data);
    CdsArray *y = x ? forward(ctx, acts, &model, x, batch) : nullptr;
    if (!y) {
      fprintf(stderr, "forward failed: %s\n", cds_last_error_message());
      goto done;
    }

    puts("y = forward(x)");
    if (!print_array(y))
      goto done;
  }
  status = 0;

done:
  cds_free(acts);
  cds_free(params);
  cds_context_free(ctx);
  return status;
}
