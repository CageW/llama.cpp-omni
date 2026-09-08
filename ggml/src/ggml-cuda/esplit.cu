#include "esplit.cuh"
#include "ggml-backend-impl.h"
#include "mmid.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------------------------------------------
// buffer type

struct ggml_backend_cuda_esplit_buffer_type_context {
    int              main_device;
    std::vector<int> devices;   // shard owners in order; shard i = experts [i*per, (i+1)*per)
    std::string      name;
};

struct ggml_backend_cuda_esplit_buffer_context {
    ~ggml_backend_cuda_esplit_buffer_context() {
        for (ggml_tensor_extra_gpu * extra : tensor_extras) {
            if (aliases.count(extra) == 0) {   // aliased shards belong to the model instance that loaded them first
                for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
                    if (extra->data_device[id] != nullptr) {
                        CUDA_CHECK(cudaFree(extra->data_device[id]));
                    }
                }
            }
            delete extra;
        }
    }
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::set<ggml_tensor_extra_gpu *>    aliases;
};

// omni: GGML_CUDA_ESPLIT_SHARE=1 lets several model instances in one process (one per LLM card, each with its own
// dense weights) share one copy of the expert shards: the first instance to create an expert tensor registers its
// device pointers under the tensor name, later instances with the same name/shape/type/devices alias them and skip
// the upload. Memory: N x dense + 1 x experts instead of N x everything.
struct esplit_shared_entry {
    std::vector<int> devices;
    int64_t          ne[4];
    ggml_type        type;
    void *           data_device[GGML_CUDA_MAX_DEVICES];
};
static std::mutex & esplit_registry_mutex() { static std::mutex m; return m; }
static std::map<std::string, esplit_shared_entry> & esplit_registry() { static std::map<std::string, esplit_shared_entry> r; return r; }
static bool esplit_share_enabled() {
    static const bool v = getenv("GGML_CUDA_ESPLIT_SHARE") != nullptr;
    return v;
}

static int64_t esplit_per_shard(const ggml_tensor * t, int n_dev) {
    return (t->ne[2] + n_dev - 1) / n_dev;
}

static void esplit_range(const ggml_tensor * t, int n_dev, int i, int64_t * e0, int64_t * e1) {
    const int64_t per = esplit_per_shard(t, n_dev);
    *e0 = std::min<int64_t>(t->ne[2], (int64_t) i * per);
    *e1 = std::min<int64_t>(t->ne[2], (int64_t) (i + 1) * per);
}

static size_t esplit_shard_nbytes(const ggml_tensor * t, int64_t n_exp) {
    // contiguous 3-D tensor: nb[2] is the size of one expert matrix; pad the tail like the row-split buffer does
    size_t size = (size_t) n_exp * t->nb[2];
    if (t->ne[0] % MATRIX_ROW_PADDING != 0) {
        size += ggml_row_size(t->type, MATRIX_ROW_PADDING - t->ne[0] % MATRIX_ROW_PADDING);
    }
    return size;
}

static const char * ggml_backend_cuda_esplit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ((ggml_backend_cuda_esplit_buffer_type_context *) buft->context)->name.c_str();
}

bool ggml_backend_buft_is_cuda_esplit(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_esplit_buffer_type_get_name;
}

static void ggml_backend_cuda_esplit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete (ggml_backend_cuda_esplit_buffer_context *) buffer->context;
}

static void * ggml_backend_cuda_esplit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return (void *) 0x1000;   // per-device pointers live in the tensor extras
}

static enum ggml_status ggml_backend_cuda_esplit_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_ASSERT(tensor->view_src == nullptr && "views of expert-split tensors are not supported");
    GGML_ASSERT(ggml_is_contiguous(tensor) && tensor->ne[3] == 1 && "expert-split tensors must be contiguous 3-D");
    auto * ctx      = (ggml_backend_cuda_esplit_buffer_context *) buffer->context;
    auto * buft_ctx = (ggml_backend_cuda_esplit_buffer_type_context *) buffer->buft->context;
    const int n_dev = (int) buft_ctx->devices.size();

    ggml_tensor_extra_gpu * extra = new ggml_tensor_extra_gpu{};
    ctx->tensor_extras.push_back(extra);
    if (esplit_share_enabled() && tensor->name[0] != 0) {
        std::lock_guard<std::mutex> lock(esplit_registry_mutex());
        auto it = esplit_registry().find(tensor->name);
        if (it != esplit_registry().end()) {
            const esplit_shared_entry & e = it->second;
            const bool same = e.devices == buft_ctx->devices && e.type == tensor->type &&
                e.ne[0] == tensor->ne[0] && e.ne[1] == tensor->ne[1] && e.ne[2] == tensor->ne[2] && e.ne[3] == tensor->ne[3];
            GGML_ASSERT(same && "expert-split share: tensor with the same name but different shape/type/devices");
            memcpy(extra->data_device, e.data_device, sizeof(extra->data_device));
            ctx->aliases.insert(extra);
            tensor->extra = extra;
            return GGML_STATUS_SUCCESS;
        }
    }
    for (int i = 0; i < n_dev; ++i) {
        int64_t e0, e1;
        esplit_range(tensor, n_dev, i, &e0, &e1);
        if (e1 == e0) {
            continue;
        }
        const int    id   = buft_ctx->devices[i];
        const size_t size = esplit_shard_nbytes(tensor, e1 - e0);
        ggml_cuda_set_device(id);
        char * buf;
        CUDA_CHECK(cudaMalloc((void **) &buf, size));
        CUDA_CHECK(cudaMemset(buf, 0, size));
        extra->data_device[id] = buf;
    }
    if (esplit_share_enabled() && tensor->name[0] != 0) {
        std::lock_guard<std::mutex> lock(esplit_registry_mutex());
        esplit_shared_entry e;
        e.devices = buft_ctx->devices; e.type = tensor->type;
        for (int k = 0; k < 4; ++k) { e.ne[k] = tensor->ne[k]; }
        memcpy(e.data_device, extra->data_device, sizeof(e.data_device));
        esplit_registry()[tensor->name] = e;
    }
    tensor->extra = extra;
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_esplit_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor) && "expert-split tensors must be set in their entirety");
    auto * buft_ctx = (ggml_backend_cuda_esplit_buffer_type_context *) buffer->buft->context;
    auto * extra    = (ggml_tensor_extra_gpu *) tensor->extra;
    if (((ggml_backend_cuda_esplit_buffer_context *) buffer->context)->aliases.count(extra)) {
        return;   // shared shards were uploaded by the first model instance
    }
    const int n_dev = (int) buft_ctx->devices.size();
    for (int i = 0; i < n_dev; ++i) {
        int64_t e0, e1;
        esplit_range(tensor, n_dev, i, &e0, &e1);
        if (e1 == e0) {
            continue;
        }
        const int id = buft_ctx->devices[i];
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaMemcpyAsync(extra->data_device[id], (const char *) data + e0*tensor->nb[2], (size_t) (e1 - e0)*tensor->nb[2],
                                   cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_esplit_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
    auto * buft_ctx = (ggml_backend_cuda_esplit_buffer_type_context *) buffer->buft->context;
    auto * extra    = (ggml_tensor_extra_gpu *) tensor->extra;
    const int n_dev = (int) buft_ctx->devices.size();
    for (int i = 0; i < n_dev; ++i) {
        int64_t e0, e1;
        esplit_range(tensor, n_dev, i, &e0, &e1);
        if (e1 == e0) {
            continue;
        }
        const int id = buft_ctx->devices[i];
        ggml_cuda_set_device(id);
        CUDA_CHECK(cudaMemcpyAsync((char *) data + e0*tensor->nb[2], extra->data_device[id], (size_t) (e1 - e0)*tensor->nb[2],
                                   cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
}

static void ggml_backend_cuda_esplit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer); GGML_UNUSED(value);
}

static const ggml_backend_buffer_i ggml_backend_cuda_esplit_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_esplit_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_esplit_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_esplit_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_esplit_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_esplit_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_esplit_buffer_clear,
    /* .reset           = */ NULL,
};

static ggml_backend_buffer_t ggml_backend_cuda_esplit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // like the row-split buffer: device memory is allocated per tensor in init_tensor
    return ggml_backend_buffer_init(buft, ggml_backend_cuda_esplit_buffer_interface, new ggml_backend_cuda_esplit_buffer_context(), size);
}

static size_t ggml_backend_cuda_esplit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

static size_t ggml_backend_cuda_esplit_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    auto * ctx = (ggml_backend_cuda_esplit_buffer_type_context *) buft->context;
    GGML_ASSERT(ggml_is_contiguous(tensor));
    const int n_dev = (int) ctx->devices.size();
    size_t total = 0;
    for (int i = 0; i < n_dev; ++i) {
        int64_t e0, e1;
        esplit_range(tensor, n_dev, i, &e0, &e1);
        if (e1 > e0) {
            total += esplit_shard_nbytes(tensor, e1 - e0);
        }
    }
    return total;
}

static bool ggml_backend_cuda_esplit_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_esplit_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_esplit_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_esplit_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_esplit_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL,
    /* .get_alloc_size   = */ ggml_backend_cuda_esplit_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_esplit_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_cuda_esplit_buffer_type(int main_device, const int * devices, int n_devices) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::map<std::pair<int, std::vector<int>>, ggml_backend_buffer_type> buft_map;

    std::vector<int> devs(devices, devices + n_devices);
    for (int d : devs) {
        if (d < 0 || d >= ggml_backend_cuda_get_device_count()) {
            return nullptr;
        }
    }
    if (n_devices <= 0 || main_device < 0 || main_device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }
    auto key = std::make_pair(main_device, devs);
    auto it = buft_map.find(key);
    if (it != buft_map.end()) {
        return &it->second;
    }
    std::string name = GGML_CUDA_NAME "_ESPLIT" + std::to_string(main_device) + "[";
    for (int d : devs) { name += std::to_string(d) + ","; }
    name += "]";
    auto * ctx = new ggml_backend_cuda_esplit_buffer_type_context{main_device, devs, name};
    ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_esplit_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), main_device),
        /* .context = */ ctx,
    };
    auto result = buft_map.emplace(key, buft);
    return &result.first->second;
}

// ---------------------------------------------------------------------------------------------------------------
// MUL_MAT_ID over expert-split weights

// peer access between the main device and the owners (kernels dereference remote pointers; cudaMemcpyPeer is not used)
static void esplit_enable_peer(int a, int b) {
    static std::mutex mutex;
    static bool done[GGML_CUDA_MAX_DEVICES][GGML_CUDA_MAX_DEVICES] = {};
    std::lock_guard<std::mutex> lock(mutex);
    if (a == b || done[a][b]) {
        return;
    }
    int can = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can, a, b));
    GGML_ASSERT(can && "expert-split: peer access between the main device and an expert owner is not possible");
    ggml_cuda_set_device(a);
    cudaError_t err = cudaDeviceEnablePeerAccess(b, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        (void) cudaGetLastError();
    } else {
        CUDA_CHECK(err);
    }
    done[a][b] = true;
}

static bool esplit_debug() {
    static const bool v = getenv("GGML_CUDA_ESPLIT_DEBUG") != nullptr;
    return v;
}
static void esplit_dbg_sync(const char * what, int device) {
    if (!esplit_debug()) {
        return;
    }
    ggml_cuda_set_device(device);
    cudaError_t err = cudaDeviceSynchronize();
    fprintf(stderr, "[esplit-dbg] %-28s dev=%d -> %s\n", what, device, cudaGetErrorString(err));
    CUDA_CHECK(err);
}

// Every owner device runs this on its own stream: copy the quantized columns of its experts (and the dst-row index
// table and the expert bounds) from the main device into local buffers. All remote reads go through peer access.
static __global__ void k_esplit_gather(const int4 * __restrict__ y_remote, int4 * __restrict__ y_local,
                                       const int32_t * __restrict__ bounds_remote, int32_t * __restrict__ bounds_local,
                                       const int32_t * __restrict__ ids_dst_remote, int32_t * __restrict__ ids_dst_local,
                                       const int e0, const int e1, const int n_expert, const int nblk_k, const int ncols,
                                       const int chunks_per_block) {
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i <= n_expert; i += stride) {
        bounds_local[i] = bounds_remote[i];
    }
    const int col_lo = bounds_remote[e0];
    const int col_hi = bounds_remote[e1];
    for (int64_t i = tid; i < col_hi - col_lo; i += stride) {
        ids_dst_local[col_lo + i] = ids_dst_remote[col_lo + i];
    }
    const int64_t n_chunks = (int64_t) (col_hi - col_lo) * nblk_k * chunks_per_block;
    for (int64_t i = tid; i < n_chunks; i += stride) {
        const int64_t c    = col_lo + i / ((int64_t) nblk_k * chunks_per_block);
        const int64_t rem  = i % ((int64_t) nblk_k * chunks_per_block);
        const int64_t kb   = rem / chunks_per_block;
        const int64_t part = rem % chunks_per_block;
        const int64_t blk  = kb * ncols + c;   // quantize_mmq layout: block index = k_block * ncols + column
        y_local[blk * chunks_per_block + part] = y_remote[blk * chunks_per_block + part];
    }
}

static cudaEvent_t esplit_event(ggml_backend_cuda_context & ctx, int device) {
    if (ctx.esplit_events[device] == nullptr) {
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.esplit_events[device], cudaEventDisableTiming));
    }
    return ctx.esplit_events[device];
}


// ---------------------------------------------------------------------------------------------------------------
// lock-step merged expert computation: K contexts (one per LLM card) run one forward together; at every expert op
// they meet here, and the last one to arrive runs, per owner device, ONE grouped GEMM over the union of all
// participants' (token, expert) columns and scatters the results back into each participant's dst.

static std::atomic<int> g_esplit_participants{1};
void ggml_backend_cuda_esplit_set_participants(int n) { g_esplit_participants.store(n < 1 ? 1 : n); }

#define ESPLIT_MAX_PARTS 8
struct esplit_part {
    ggml_backend_cuda_context * ctx = nullptr;
    int             main = 0;
    cudaStream_t    s_main = nullptr;
    cudaEvent_t     ready = nullptr;
    const int4 *    y = nullptr;        // quantized activations on the participant's main device, layout [kb][ncols]
    const int32_t * ids_dst = nullptr;
    const int32_t * bounds = nullptr;   // n_expert + 1 (global expert index)
    float *         dst = nullptr;
    int             ncols = 0;          // (token, expert) columns of this participant
    long long       s1 = 0;             // dst column stride in elements
};
struct esplit_parts_dev {   // passed to kernels by value
    const int4 *    y[ESPLIT_MAX_PARTS];
    const int32_t * ids_dst[ESPLIT_MAX_PARTS];
    const int32_t * bounds[ESPLIT_MAX_PARTS];
    float *         dst[ESPLIT_MAX_PARTS];
    int             ncols[ESPLIT_MAX_PARTS];
    long long       s1[ESPLIT_MAX_PARTS];
    int             n;
};
struct esplit_slot {
    std::mutex              m;
    std::condition_variable cv;
    uint64_t                generation = 0;
    int                     arrived = 0;
    std::vector<esplit_part> parts;
    cudaEvent_t             done[GGML_CUDA_MAX_DEVICES] = {};
    int                     n_done = 0;
};
static esplit_slot & esplit_slot_for(const char * name) {
    static std::mutex m;
    static std::map<std::string, std::unique_ptr<esplit_slot>> slots;
    std::lock_guard<std::mutex> lock(m);
    auto & p = slots[name];
    if (!p) { p.reset(new esplit_slot()); }
    return *p;
}
static cudaEvent_t esplit_ready_event(ggml_backend_cuda_context & ctx) {
    if (ctx.esplit_ready == nullptr) {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.esplit_ready, cudaEventDisableTiming));
    }
    return ctx.esplit_ready;
}

// per local expert: column offset of every participant inside the merged expert group, and merged expert bounds
static __global__ void k_esplit_merge_bounds(const esplit_parts_dev parts, const int e0, const int n_local, int32_t * __restrict__ off, int32_t * __restrict__ bounds_m) {
    __shared__ int tot[1024];
    for (int e = threadIdx.x; e < n_local; e += blockDim.x) {
        int acc = 0;
        for (int p = 0; p < parts.n; ++p) {
            off[p*n_local + e] = acc;
            acc += parts.bounds[p][e0 + e + 1] - parts.bounds[p][e0 + e];
        }
        tot[e] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < n_local; ++e) { bounds_m[e] = acc; acc += tot[e]; }
        bounds_m[n_local] = acc;
    }
}
// gather every participant's columns of experts [e0, e0+n_local) into the merged layout (grouped by expert, then by
// participant); records for each merged column its participant and its dst column
static __global__ void k_esplit_merge_gather(const esplit_parts_dev parts, const int e0, const int n_local,
        const int32_t * __restrict__ off, const int32_t * __restrict__ bounds_m, int4 * __restrict__ y_st,
        int32_t * __restrict__ colpart, int32_t * __restrict__ coldst, int32_t * __restrict__ ident,
        const int NB, const int nblk_k, const int cpb) {
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, stride = (int64_t) gridDim.x * blockDim.x;
    for (int p = 0; p < parts.n; ++p) {
        const int32_t * b = parts.bounds[p];
        const int col_lo = b[e0], col_hi = b[e0 + n_local];
        const int64_t total = (int64_t) (col_hi - col_lo) * nblk_k * cpb;
        for (int64_t i = tid; i < total; i += stride) {
            const int64_t j   = col_lo + i / ((int64_t) nblk_k * cpb);
            const int64_t rem = i % ((int64_t) nblk_k * cpb);
            const int kb = (int) (rem / cpb), part = (int) (rem % cpb);
            int lo = 0, hi = n_local - 1;   // expert of column j: largest e with b[e0+e] <= j
            while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (b[e0 + mid] <= j) { lo = mid; } else { hi = mid - 1; } }
            const int e    = lo;
            const int dest = bounds_m[e] + off[p*n_local + e] + (int) (j - b[e0 + e]);
            y_st[((int64_t) kb * NB + dest) * cpb + part] = parts.y[p][((int64_t) kb * parts.ncols[p] + j) * cpb + part];
            if (kb == 0 && part == 0) { colpart[dest] = p; coldst[dest] = parts.ids_dst[p][j]; ident[dest] = dest; }
        }
    }
}
static __global__ void k_esplit_merge_scatter(const esplit_parts_dev parts, const float4 * __restrict__ out_st,
        const int32_t * __restrict__ colpart, const int32_t * __restrict__ coldst, const int32_t * __restrict__ bounds_m,
        const int n_local, const int ne01) {
    const int total_cols = bounds_m[n_local];
    const int n4 = ne01 / 4;
    const int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x, stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t i = tid; i < (int64_t) total_cols * n4; i += stride) {
        const int c = (int) (i / n4), r4 = (int) (i % n4);
        const int p = colpart[c];
        float4 * d = (float4 *) (parts.dst[p] + (int64_t) coldst[c] * parts.s1[p]);
        d[r4] = out_st[(int64_t) c * n4 + r4];
    }
}

static void esplit_merged(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const esplit_part & me, const int K,
                          const ggml_backend_cuda_esplit_buffer_type_context * buft_ctx, const ggml_tensor_extra_gpu * extra,
                          const int64_t ne00, const int64_t ne01, const int64_t s01, const int64_t s02, const int64_t s03,
                          const int64_t ne10_padded, const int nblk_k, const bool use_stream_k) {
    esplit_slot & slot = esplit_slot_for(src0->name);
    std::unique_lock<std::mutex> lk(slot.m);
    const uint64_t gen = slot.generation;
    slot.parts.push_back(me);
    slot.arrived++;
    if (slot.arrived < K) {
        if (!slot.cv.wait_for(lk, std::chrono::seconds(10), [&]{ return slot.generation != gen; })) {
            std::string who;
            for (const auto & p : slot.parts) { who += " dev" + std::to_string(p.main) + "(ncols=" + std::to_string(p.ncols) + ")"; }
            GGML_ABORT("expert-split lock-step: %s: only %d of %d participants arrived within 10 s:%s (generation %llu)",
                       src0->name, slot.arrived, K, who.c_str(), (unsigned long long) slot.generation);
        }
        ggml_cuda_set_device(me.main);
        for (int i = 0; i < slot.n_done; ++i) { CUDA_CHECK(cudaStreamWaitEvent(me.s_main, slot.done[i], 0)); }
        return;
    }
    // last arriver: run the merged computation for every shard on this context's streams
    std::vector<esplit_part> parts = slot.parts;
    slot.parts.clear();
    slot.arrived = 0;
    GGML_ASSERT((int) parts.size() <= ESPLIT_MAX_PARTS);
    esplit_parts_dev pd = {};
    pd.n = (int) parts.size();
    int NB = 0;
    for (int p = 0; p < pd.n; ++p) {
        pd.y[p] = parts[p].y; pd.ids_dst[p] = parts[p].ids_dst; pd.bounds[p] = parts[p].bounds; pd.dst[p] = parts[p].dst;
        pd.ncols[p] = parts[p].ncols; pd.s1[p] = parts[p].s1;
        NB += parts[p].ncols;
    }
    const int n_dev = (int) buft_ctx->devices.size();
    const int cc    = ggml_cuda_info().devices[me.main].cc;
    const int cpb   = (int) (sizeof(block_q8_1_mmq) / sizeof(int4));
    // one pool allocation per device (the VMM pool frees in LIFO order, so sub-buffers are carved out of one block)
    std::deque<ggml_cuda_pool_alloc<char>> dbufs;
    auto align_up = [](size_t v) { return (v + 255) & ~(size_t) 255; };
    slot.n_done = 0;
    for (int i = 0; i < n_dev; ++i) {
        int64_t e0, e1;
        esplit_range(src0, n_dev, i, &e0, &e1);
        if (e1 == e0) {
            continue;
        }
        const int d = buft_ctx->devices[i];
        const int n_local = (int) (e1 - e0);
        ggml_cuda_set_device(d);
        cudaStream_t s = (d == me.main) ? me.s_main : ctx.stream(d, 0);
        for (const auto & p : parts) { CUDA_CHECK(cudaStreamWaitEvent(s, p.ready, 0)); }
        const size_t nbytes_y = (size_t) NB * ne10_padded * sizeof(block_q8_1)/QK8_1 + get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        const size_t o_y = 0, o_out = align_up(o_y + nbytes_y), o_off = align_up(o_out + (size_t) NB * ne01 * sizeof(float)),
                     o_bm = align_up(o_off + (size_t) pd.n * n_local * sizeof(int32_t)), o_cp = align_up(o_bm + (n_local + 1) * sizeof(int32_t)),
                     o_cd = align_up(o_cp + (size_t) NB * sizeof(int32_t)), o_id = align_up(o_cd + (size_t) NB * sizeof(int32_t)),
                     o_ti = align_up(o_id + (size_t) NB * sizeof(int32_t)), total = align_up(o_ti + (n_local + 1) * sizeof(int32_t));
        dbufs.emplace_back(ctx.pool(d), total);
        char *    base     = dbufs.back().get();
        char *    y_st     = base + o_y;
        float *   out_st   = (float *)   (base + o_out);
        int32_t * off      = (int32_t *) (base + o_off);
        int32_t * bounds_m = (int32_t *) (base + o_bm);
        int32_t * colpart  = (int32_t *) (base + o_cp);
        int32_t * coldst   = (int32_t *) (base + o_cd);
        int32_t * ident    = (int32_t *) (base + o_id);
        int32_t * tiles    = (int32_t *) (base + o_ti);
        k_esplit_merge_bounds<<<1, 256, 0, s>>>(pd, (int) e0, n_local, off, bounds_m);
        CUDA_CHECK(cudaGetLastError());
        k_esplit_merge_gather<<<2048, 256, 0, s>>>(pd, (int) e0, n_local, off, bounds_m, (int4 *) y_st, colpart, coldst, ident, NB, nblk_k, cpb);
        CUDA_CHECK(cudaGetLastError());
        const int64_t s12q = ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const mmq_args args = {
            (const char *) extra->data_device[d], src0->type, (const int *) y_st, ident, bounds_m, out_st,
            ne00, ne01, NB, s01, NB, ne01,
            n_local, n_local, s02, s12q, 0,
            1, 1, s03, 0, 0,
            use_stream_k, NB, tiles};
        ggml_cuda_mul_mat_q_args(ctx, args, s);
        CUDA_CHECK(cudaGetLastError());
        k_esplit_merge_scatter<<<2048, 256, 0, s>>>(pd, (const float4 *) out_st, colpart, coldst, bounds_m, n_local, (int) ne01);
        CUDA_CHECK(cudaGetLastError());
        cudaEvent_t ev = esplit_event(ctx, d);
        CUDA_CHECK(cudaEventRecord(ev, s));
        slot.done[slot.n_done++] = ev;
    }
    ggml_cuda_set_device(me.main);
    for (int i = 0; i < slot.n_done; ++i) { CUDA_CHECK(cudaStreamWaitEvent(me.s_main, slot.done[i], 0)); }
    slot.generation++;
    lk.unlock();
    slot.cv.notify_all();
}

void ggml_cuda_mul_mat_id_esplit(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_quantized(src0->type) && "expert-split MUL_MAT_ID: only quantized experts (MMQ path)");
    GGML_TENSOR_BINARY_OP_LOCALS

    auto * buft_ctx = (ggml_backend_cuda_esplit_buffer_type_context *) src0->buffer->buft->context;
    auto * extra    = (ggml_tensor_extra_gpu *) src0->extra;
    const int n_dev = (int) buft_ctx->devices.size();
    const int main  = ctx.device;
    GGML_ASSERT(main == buft_ctx->main_device);
    GGML_ASSERT(ne13 == 1 && ne03 == 1);
    GGML_ASSERT(nb12 % nb11 == 0 && nb2 % nb1 == 0);

    for (int d : buft_ctx->devices) {
        esplit_enable_peer(main, d);
        esplit_enable_peer(d, main);
    }
    const int          cc     = ggml_cuda_info().devices[main].cc;
    cudaStream_t       s_main = ctx.stream();
    const float *      src1_d = (const float *) src1->data;
    float *            dst_d  = (float *) dst->data;
    const size_t       ts_src0 = ggml_type_size(src0->type);
    const size_t       ts_src1 = ggml_type_size(src1->type);
    const size_t       ts_dst  = ggml_type_size(dst->type);

    const int64_t n_expert      = ne02;
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ne12;
    const int64_t ne_get_rows   = n_tokens * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    // 1. sort the (token, expert) columns by expert on the main device
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(main), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(main), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(main), n_expert + 1);
    {
        const int si1  = (int) (ids->nb[1] / ggml_element_size(ids));
        const int sis1 = (int) (nb12 / nb11);
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
                                       (int) n_expert, (int) n_tokens, (int) n_expert_used, (int) ne11, si1, sis1, s_main);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2. gather + quantize the activations into the MMQ layout on the main device
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const size_t  nbytes_q    = ne_get_rows*ne10_padded * sizeof(block_q8_1)/QK8_1 + get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8(ctx.pool(main), nbytes_q);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne_get_rows, 1, 1, s_main);
        CUDA_CHECK(cudaGetLastError());
    }
    esplit_dbg_sync("sort+quantize (main)", main);
    ggml_cuda_set_device(main);
    {
        const int K = g_esplit_participants.load();
        if (K > 1) {
            cudaEvent_t ready = esplit_ready_event(ctx);
            CUDA_CHECK(cudaEventRecord(ready, s_main));
            esplit_part me;
            me.ctx = &ctx; me.main = main; me.s_main = s_main; me.ready = ready;
            me.y = (const int4 *) src1_q8.get(); me.ids_dst = ids_dst.get(); me.bounds = expert_bounds.get(); me.dst = dst_d;
            me.ncols = (int) ne_get_rows; me.s1 = (long long) (dst->nb[1] / ts_dst);
            const int64_t s01_ = src0->nb[1] / ts_src0, s02_ = src0->nb[2] / ts_src0, s03_ = src0->nb[3] / ts_src0;
            const bool use_stream_k_ = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;
            esplit_merged(ctx, src0, me, K, buft_ctx, extra, ne00, ne01, s01_, s02_, s03_, ne10_padded, (int) (ne10_padded / (4*QK8_1)), use_stream_k_);
            return;
        }
    }
    cudaEvent_t ev_main = esplit_event(ctx, main);
    CUDA_CHECK(cudaEventRecord(ev_main, s_main));

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s1  = dst->nb[1] / ts_dst;
    const int64_t s2  = dst->nb[2] / ts_dst;
    const int64_t s3  = dst->nb[3] / ts_dst;
    const int64_t s12q = ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13q = ne12 * s12q;
    const int     nblk_k = (int) (ne10_padded / (4*QK8_1));
    const int     chunks_per_block = (int) (sizeof(block_q8_1_mmq) / sizeof(int4));
    static_assert(sizeof(block_q8_1_mmq) % sizeof(int4) == 0, "block_q8_1_mmq must be a multiple of 16 bytes");
    const bool use_stream_k = GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA;

    // 3. every owner device: gather its columns, grouped GEMM over its experts, results straight into dst on main
    std::deque<ggml_cuda_pool_alloc<char>>    y_local;   // deque: pool allocs are not movable
    std::deque<ggml_cuda_pool_alloc<int32_t>> ids_local, bounds_local, tiles_local;
    for (int i = 0; i < n_dev; ++i) {
        int64_t e0, e1;
        esplit_range(src0, n_dev, i, &e0, &e1);
        if (e1 == e0) {
            continue;
        }
        const int    d     = buft_ctx->devices[i];
        cudaStream_t s     = d == main ? s_main : ctx.stream(d, 0);
        ggml_cuda_set_device(d);
        if (d != main) {
            CUDA_CHECK(cudaStreamWaitEvent(s, ev_main, 0));
        }
        y_local.emplace_back(ctx.pool(d), nbytes_q);
        ids_local.emplace_back(ctx.pool(d), ne_get_rows);
        bounds_local.emplace_back(ctx.pool(d), n_expert + 1);
        tiles_local.emplace_back(ctx.pool(d), e1 - e0 + 1);
        const int32_t * bounds_d = bounds_local.back().get();
        const int32_t * ids_d    = ids_local.back().get();
        const char *    y_d      = y_local.back().get();
        if (d != main) {
            k_esplit_gather<<<1024, 256, 0, s>>>((const int4 *) src1_q8.get(), (int4 *) y_local.back().get(),
                expert_bounds.get(), bounds_local.back().get(), ids_dst.get(), ids_local.back().get(),
                (int) e0, (int) e1, (int) n_expert, nblk_k, (int) ne_get_rows, chunks_per_block);
            CUDA_CHECK(cudaGetLastError());
        } else {
            bounds_d = expert_bounds.get();
            ids_d    = ids_dst.get();
            y_d      = src1_q8.get();
        }
        const mmq_args args = {
            (const char *) extra->data_device[d], src0->type, (const int *) y_d, ids_d, bounds_d + e0, dst_d,
            ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
            e1 - e0, e1 - e0, s02, s12q, s2,
            1, 1, s03, s13q, s3,
            use_stream_k, n_tokens, tiles_local.back().get()};
        esplit_dbg_sync("gather (owner)", d);
        ggml_cuda_mul_mat_q_args(ctx, args, s);
        CUDA_CHECK(cudaGetLastError());
        esplit_dbg_sync("grouped mmq (owner)", d);
        if (d != main) {
            CUDA_CHECK(cudaEventRecord(esplit_event(ctx, d), s));
        }
    }
    // 4. the main stream continues only when every owner has written its part of dst
    ggml_cuda_set_device(main);
    for (int i = 0; i < n_dev; ++i) {
        int64_t e0, e1;
        esplit_range(src0, n_dev, i, &e0, &e1);
        const int d = buft_ctx->devices[i];
        if (e1 == e0 || d == main) {
            continue;
        }
        CUDA_CHECK(cudaStreamWaitEvent(s_main, ctx.esplit_events[d], 0));
    }
    // NOTE: the per-device pool buffers are released to their pools here (host-side); the owner streams are ordered
    // behind this op on the main stream through the events above, and each pool is only reused by later ops on the
    // same device stream, so no additional synchronization is needed.
}
