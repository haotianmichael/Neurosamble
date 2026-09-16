// =====================================================================================
// rabitq_search.cu  --  GPU IVF-RaBitQ (1-bit) all-vs-all search for Neurosamble.
//
// Replaces the FAISS search stage: reads the encode fp16 embeddings, builds a GPU
// IVF-RaBitQ index (cuVS fork: Stardust-SJF/cuvs_rabitq, branch cuvs_ivf_rabitq),
// runs all-vs-all top-k search, and writes two flat binary files consumed by
// map_full.py:  <out_nbr> = int64 [N, topk] neighbor ROW ids (global row order),
//               <out_dist> = float32 [N, topk] inner-product scores.
// map_full.py then does the SAME filter + colinear chaining as before.
//
// Single-GPU by design (cuVS indexes are single-GPU). Vectors stay on host
// (~N*D*4 bytes) and are streamed to the GPU at build time (force_streaming).
//
// NOTE: written against the fork's headers/bench but NOT compiled here. Expect to
// adjust include paths / raft template details for your installed cuvs+raft.
// Build: see src/rabitq_gpu/CMakeLists.txt.
// =====================================================================================
#include <cuvs/neighbors/ivf_rabitq.hpp>

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/device_mdspan.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace rabitq = cuvs::neighbors::ivf_rabitq;

// ---- minimal host-side IEEE half -> float (avoids relying on device __half2float) ----
static inline float half_to_float(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1F;
  uint32_t man  = h & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) { f = sign; }
    else {  // subnormal
      exp = 127 - 15 + 1;
      while ((man & 0x400) == 0) { man <<= 1; exp--; }
      man &= 0x3FF;
      f = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1F) {            // inf/nan
    f = sign | 0x7F800000 | (man << 13);
  } else {
    f = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float out; std::memcpy(&out, &f, 4); return out;
}

static const char* argval(int argc, char** argv, const char* key, const char* def) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], key) == 0) return argv[i + 1];
  }
  return def;
}

int main(int argc, char** argv) {
  const std::string emb   = argval(argc, argv, "--emb",   "");
  const std::string outN  = argval(argc, argv, "--out_nbr",  "neighbors.i64");
  const std::string outD  = argval(argc, argv, "--out_dist", "dists.f32");
  const std::string idxf  = argval(argc, argv, "--index", "");     // serialize/reuse (optional)
  const int64_t N   = std::atoll(argval(argc, argv, "--n", "0"));
  const int64_t D   = std::atoll(argval(argc, argv, "--d", "384"));
  const uint32_t nlist  = (uint32_t)std::atol(argval(argc, argv, "--nlist",  "21185"));
  const uint32_t nprobe = (uint32_t)std::atol(argval(argc, argv, "--nprobe", "384"));
  const int      topk   = std::atoi(argval(argc, argv, "--topk",   "128"));
  const uint32_t bits   = (uint32_t)std::atol(argval(argc, argv, "--bits",   "1")); // 1 = pure 1-bit RaBitQ
  const int64_t  B      = std::atoll(argval(argc, argv, "--batch",  "16384"));      // queries per search
  const uint32_t kiters = (uint32_t)std::atol(argval(argc, argv, "--kmeans_iters", "20"));
  if (emb.empty() || N <= 0) { std::fprintf(stderr, "need --emb <fp16 file> --n <N> [--d 384]\n"); return 2; }

  std::fprintf(stderr, "[rabitq][v3] N=%lld D=%lld nlist=%lld nprobe=%lld topk=%lld bits=%lld batch=%lld\n",
               (long long)N, (long long)D, (long long)nlist, (long long)nprobe,
               (long long)topk, (long long)bits, (long long)B);
  std::fflush(stderr);

  raft::device_resources handle;
  // Route the LARGE workspace to managed (UVM) memory so a dataset bigger than
  // GPU RAM still builds via the NON-streaming path (streaming is broken on sm_70).
  static rmm::mr::managed_memory_resource s_managed;
  raft::resource::set_large_workspace_resource(
      handle, rmm::device_async_resource_ref(s_managed));
  cudaStream_t stream = handle.get_stream();

  // ---- 1) load fp16 embeddings -> host float32 [N, D] (row-major) ----
  auto dataset = raft::make_host_matrix<float, int64_t>(N, D);
  {
    std::ifstream in(emb, std::ios::binary);
    if (!in) { std::fprintf(stderr, "[rabitq] cannot open %s\n", emb.c_str()); return 2; }
    const int64_t total = N * D;
    const int64_t CH = 1 << 22;                 // 4M elems/chunk
    std::vector<uint16_t> buf(CH);
    float* dst = dataset.data_handle();
    int64_t done = 0;
    while (done < total) {
      int64_t m = std::min(CH, total - done);
      in.read(reinterpret_cast<char*>(buf.data()), m * (int64_t)sizeof(uint16_t));
      if (in.gcount() != m * (int64_t)sizeof(uint16_t)) { std::fprintf(stderr, "[rabitq] short read\n"); return 2; }
      for (int64_t i = 0; i < m; ++i) dst[done + i] = half_to_float(buf[i]);
      done += m;
    }
    std::fprintf(stderr, "[rabitq] loaded %lld vectors from %s\n", (long long)N, emb.c_str());
  }
  auto dataset_v = raft::make_host_matrix_view<const float, int64_t>(dataset.data_handle(), N, D);

  // ---- 2) build (or load) the GPU IVF-RaBitQ index ----
  rabitq::index<int64_t> index(handle);
  bool have_index = false;
  if (!idxf.empty()) { std::ifstream t(idxf, std::ios::binary); have_index = t.good(); }
  if (have_index) {
    std::fprintf(stderr, "[rabitq] deserialize %s\n", idxf.c_str());
    auto _td=std::chrono::steady_clock::now(); rabitq::deserialize(handle, idxf, &index); std::fprintf(stderr, "[TIME] deserialize = %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-_td).count());
  } else {
    rabitq::index_params ip;
    ip.n_lists         = nlist;
    ip.bits_per_dim    = bits;                                  // 1 = binary RaBitQ
    ip.kmeans_n_iters  = kiters;
    ip.force_streaming = false;                                  // host dataset -> stream to GPU
    ip.metric          = cuvs::distance::DistanceType::L2Expanded;  // normalized vecs: L2-nn == IP-max
    auto _tb=std::chrono::steady_clock::now();
    std::fprintf(stderr, "[rabitq] building index (streaming)...\n");
    index = rabitq::build(handle, ip, dataset_v);
    std::fprintf(stderr, "[rabitq] built: dim=%lld; serialize->deserialize to reorganize...\n", (long long)index.dim());
    // REQUIRED: search only works correctly on a serialized+deserialized index
    std::string tmpf = idxf.empty() ? std::string("/tmp/_rq_reorg.index") : idxf;
    rabitq::serialize(handle, tmpf, index);
    rabitq::index<int64_t> reidx(handle);
    rabitq::deserialize(handle, tmpf, &reidx);
    index = std::move(reidx);
    std::fprintf(stderr, "[rabitq] reorganized (via %s)\n", tmpf.c_str());
    std::fprintf(stderr, "[TIME] build+reorg = %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-_tb).count());
  }

  // ---- 3) all-vs-all search in batches; stream results to disk ----
  auto _ts=std::chrono::steady_clock::now();
  std::ofstream fN(outN, std::ios::binary), fD(outD, std::ios::binary);
  if (!fN || !fD) { std::fprintf(stderr, "[rabitq] cannot open output files\n"); return 2; }

  rmm::device_uvector<float>   d_q(B * D, stream);
  rmm::device_uvector<int64_t> d_nbr(B * topk, stream);
  rmm::device_uvector<float>   d_dist(B * topk, stream);
  std::vector<int64_t> h_nbr(B * topk);
  std::vector<float>   h_dist(B * topk);

  rabitq::search_params sp;
  sp.n_probes = nprobe;

  for (int64_t off = 0; off < N; off += B) {
    int64_t b = std::min(B, N - off);
    // H2D queries = dataset rows [off, off+b)
    cudaMemcpyAsync(d_q.data(), dataset.data_handle() + off * D,
                    (size_t)b * D * sizeof(float), cudaMemcpyHostToDevice, stream);
    auto qv = raft::make_device_matrix_view<const float, int64_t>(d_q.data(), b, D);
    auto nv = raft::make_device_matrix_view<int64_t, int64_t>(d_nbr.data(), b, topk);
    auto dv = raft::make_device_matrix_view<float, int64_t>(d_dist.data(), b, topk);
    rabitq::search(handle, sp, index, qv, nv, dv);
    cudaMemcpyAsync(h_nbr.data(),  d_nbr.data(),  (size_t)b * topk * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_dist.data(), d_dist.data(), (size_t)b * topk * sizeof(float),   cudaMemcpyDeviceToHost, stream);
    handle.sync_stream();
    fN.write(reinterpret_cast<char*>(h_nbr.data()),  (std::streamsize)b * topk * sizeof(int64_t));
    fD.write(reinterpret_cast<char*>(h_dist.data()), (std::streamsize)b * topk * sizeof(float));
    if (((off / B) % 50) == 0)
      std::fprintf(stderr, "[rabitq] searched %lld / %lld\n", (long long)(off + b), (long long)N);
  }
  fN.close(); fD.close();
  std::fprintf(stderr, "[TIME] search+write = %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-_ts).count());
  std::fprintf(stderr, "[rabitq] DONE -> %s , %s\n", outN.c_str(), outD.c_str());
  return 0;
}
