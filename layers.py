import numpy as np

# ============================================================
#   Multi‑Channel Conv2D Layer
# ============================================================
class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = None

    def im2col(self, x, f):
        C, H, W = x.shape
        out_h = H - f + 1
        out_w = W - f + 1

        col = np.zeros((C * f * f, out_h * out_w))

        col_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = x[:, i:i+f, j:j+f].reshape(-1)
                col[:, col_idx] = patch
                col_idx += 1

        return col, out_h, out_w

    def forward(self, x):
        self.last_input = x
        C, H, W = x.shape
        f = self.filter_size

        if self.filters is None:
            scale = np.sqrt(2.0 / (C * f * f))
            self.filters = np.random.randn(self.num_filters, C * f * f) * scale

        col, out_h, out_w = self.im2col(x, f)

        out = np.dot(self.filters, col)
        out = out.reshape(self.num_filters, out_h, out_w)

        self.last_col = col
        self.last_output = out   # ⭐ REQUIRED FOR RELU BACKWARD

        return out


    def backward_no_update(self, d_out):
        f = self.filter_size
        C, H, W = self.last_input.shape
        out_h, out_w = d_out.shape[1:]

        d_out_flat = d_out.reshape(self.num_filters, -1)

        if not hasattr(self, "d_filters"):
            self.d_filters = np.zeros_like(self.filters)

        self.d_filters += np.dot(d_out_flat, self.last_col.T)

        d_col = np.dot(self.filters.T, d_out_flat)

        d_input = np.zeros_like(self.last_input)

        col_idx = 0
        for i in range(out_h):
            for j in range(out_w):
                patch = d_col[:, col_idx].reshape(C, f, f)
                d_input[:, i:i+f, j:j+f] += patch
                col_idx += 1

        return d_input

    def apply_update(self, lr):
        self.filters -= lr * self.d_filters
        self.d_filters = np.zeros_like(self.filters)


# ============================================================
#   MaxPool2 (2×2 pooling)
# ============================================================
class MaxPool2:
    def forward(self, x):
        self.last_input = x
        C, H, W = x.shape

        out_h = H // 2
        out_w = W // 2
        output = np.zeros((C, out_h, out_w))

        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    region = x[c, i*2:i*2+2, j*2:j*2+2]
                    output[c, i, j] = np.max(region)

        self.last_output = output
        return output

    def backward(self, d_out):
        C, H, W = self.last_input.shape
        d_input = np.zeros_like(self.last_input)

        for c in range(C):
            for i in range(H // 2):
                for j in range(W // 2):
                    region = self.last_input[c, i*2:i*2+2, j*2:j*2+2]
                    max_val = np.max(region)

                    for m in range(2):
                        for n in range(2):
                            if region[m, n] == max_val:
                                d_input[c, i*2+m, j*2+n] = d_out[c, i, j]

        return d_input


# ============================================================
#   Dense Layer
# ============================================================
class Dense:
    def __init__(self, input_size, output_size):
        scale = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(output_size, input_size) * scale
        self.biases = np.zeros(output_size)

    def forward(self, x):
        self.last_input = x
        return np.dot(self.weights, x) + self.biases

    def backward_no_update(self, d_out):
        self.d_weights = np.outer(d_out, self.last_input)
        self.d_biases = d_out
        return np.dot(self.weights.T, d_out)

    def apply_update(self, lr):
        self.weights -= lr * self.d_weights
        self.biases -= lr * self.d_biases



