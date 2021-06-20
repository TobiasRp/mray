// Copyright (c) 2020, Tobias Rapp
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Karlsruhe Institute of Technology nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#ifndef MEMORY_H
#define MEMORY_H

#include "../common.h"
#include <cstddef>
#include <vector>
using std::vector;

// Internal utility macros
#define CUT_INTERNAL_TEMP_S1(x) #x
#define CUT_INTERNAL_TEMP_S2(x) CUT_INTERNAL_TEMP_S1(x)
#define CUT_INTERNAL_LOCATION __FILE__ ":" CUT_INTERNAL_TEMP_S2(__LINE__)

namespace cut
{

extern void *allocDevice(size_t size, const char *name);
extern void freeDevice(void *dptr);

#define CUDA_MALLOC(size) cut::allocDevice(size, CUT_INTERNAL_LOCATION)
#define CUDA_FREE(ptr) cut::freeDevice(ptr)

extern void *allocUnified(size_t size, const char *name);
extern void freeUnified(void *dptr);

#define UNIFIED_MALLOC(size) cut::allocUnified(size, CUT_INTERNAL_LOCATION)
#define UNIFIED_FREE(ptr) cut::freeUnified(ptr)

template <typename T> inline T *allocDeviceTyped(size_t size, const char *name)
{
    void *ptr = allocDevice(size * sizeof(T), name);
    return reinterpret_cast<T *>(ptr);
}

template <typename T> inline T *allocUnifiedTyped(size_t size, const char *name)
{
    void *ptr = allocUnified(size * sizeof(T), name);
    return reinterpret_cast<T *>(ptr);
}

#define CUDA_MALLOC_TYPED(T, size) cut::allocDeviceTyped<T>(size, CUT_INTERNAL_LOCATION)
#define UNIFIED_MALLOC_TYPED(T, size) cut::allocUnifiedTyped<T>(size, CUT_INTERNAL_LOCATION)

template <typename T> inline void safeFreeDevice(T *ptr)
{
    if (ptr != nullptr)
        freeDevice(ptr);
}

template <typename T> inline void safeFreeUnified(T *ptr)
{
    if (ptr != nullptr)
        freeUnified(ptr);
}

#define CUDA_SAFE_FREE(ptr) cut::safeFreeDevice(ptr)
#define SAFE_UNIFIED_FREE cut::safeFreeUnified(ptr)

/**
 * @brief reportUnreleasedDeviceMemory prints a list of currently unrealesed, i.e. allocated, device (and unified)
 * CUDA memory.
 */
extern void reportUnreleasedDeviceMemory();

/**
 * @brief assertAllMemoryReleased Instead of printing unreleased memory allocations, this function will assert if
 * CUDA memory is still allocated.
 *
 * @note This function is useful for unit tests and other testing scenarios, but should not be used during normal use.
 */
extern void assertDeviceMemoryReleased();

/**
 * @brief Prints the total device memory.
 */
extern void reportDeviceMemory();

/**
 * @brief getAllocatedDeviceMemory returns the currently allocated device memory.
 * @return Allocated memory in bytes.
 */
extern size_t getAllocatedDeviceMemory();

/**
 * Similar to unique_ptr, but for CUDA device memory.
 *
 * NOT thread safe!
 */
template <typename T> struct dev_ptr
{
    T *m_ptr;

    /**
     * @brief Creates a new smart pointer without allocating memory.
     */
    inline dev_ptr()
        : m_ptr(nullptr)
    {
    }

    /**
     * @brief Takes the given pointer and assumes ownership, i.e. will free it.
     */
    inline explicit dev_ptr(T *ptr)
        : m_ptr(ptr)
    {
    }

    /**
     * @brief Creates a new smart pointer and allocates the specified amount of memory (in elements T).
     */
    inline explicit dev_ptr(size_t size)
        : m_ptr(nullptr)
    {
        resize(size);
    }

    inline explicit dev_ptr(const vector<T> &vec)
        : m_ptr(nullptr)
    {
        resize(vec.size());
        loadFromHost(vec.data(), vec.size());
    }

    ~dev_ptr() { free(); }

    dev_ptr(const dev_ptr &rhs) = delete;
    dev_ptr &operator=(const dev_ptr &rhs) = delete;

    inline dev_ptr(dev_ptr &&rhs) noexcept
    {
        m_ptr = rhs.m_ptr;
        rhs.m_ptr = nullptr;
    }

    inline dev_ptr &operator=(dev_ptr &&rhs)
    {
        m_ptr = rhs.m_ptr;
        rhs.m_ptr = nullptr;
        return *this;
    }

    inline void loadFromHost(const T *h_ptr, size_t size)
    {
        CHECK_CUDA(cudaMemcpy(m_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    inline void loadFromDevice(const T *d_ptr, size_t size)
    {
        CHECK_CUDA(cudaMemcpy(m_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    inline void copyToHost(T *h_ptr, size_t size) const
    {
        CHECK_CUDA(cudaMemcpy(h_ptr, m_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    inline void resize(size_t size)
    {
        free();
        m_ptr = CUDA_MALLOC_TYPED(T, size);
    }

    /**
     * @brief Transfers ownership pf ptr to this.
     */
    inline void set(T *ptr)
    {
        free();
        m_ptr = ptr;
    }

    /**
     * @brief Returns a raw pointer _without_ transfering ownership
     */
    inline T *get() { return m_ptr; }
    inline const T *get() const { return m_ptr; }

    /**
     * @brief free's the owned memory, if any.
     */
    inline void free()
    {
        cut::safeFreeDevice(m_ptr);
        m_ptr = nullptr;
    }

    /**
     * @brief release Returns the pointer, without freeing it and thereby transfering ownership to the caller.
     */
    inline T *release()
    {
        auto res = m_ptr;
        m_ptr = nullptr;
        return res;
    }
};

/**
 * Similar to unique_ptr, but for unified CUDA memory.
 *
 *  NOT thread safe!
 */
template <typename T> struct unified_ptr
{
    T *m_ptr;

    /**
     * @brief Creates a new smart pointer without allocating memory.
     */
    inline unified_ptr()
        : m_ptr(nullptr)
    {
    }

    /**
     * @brief Takes the given pointer and assumes ownership, i.e. will free it.
     */
    inline explicit unified_ptr(T *ptr)
        : m_ptr(ptr)
    {
    }

    /**
     * @brief Creates a new smart pointer and allocates the specified amount of memory (in elements T).
     */
    inline explicit unified_ptr(size_t size)
        : m_ptr(nullptr)
    {
        resize(size);
    }

    inline explicit unified_ptr(const vector<T> &vec)
        : m_ptr(nullptr)
    {
        resize(vec.size());
        loadFromHost(vec.data(), vec.size());
    }

    ~unified_ptr() { free(); }

    unified_ptr(const unified_ptr &rhs) = delete;
    unified_ptr &operator=(const unified_ptr &rhs) = delete;

    inline unified_ptr(unified_ptr &&rhs) noexcept
    {
        m_ptr = rhs.m_ptr;
        rhs.m_ptr = nullptr;
    }

    inline unified_ptr &operator=(unified_ptr &&rhs)
    {
        m_ptr = rhs.m_ptr;
        rhs.m_ptr = nullptr;
        return *this;
    }

    inline void loadFromHost(const T *h_ptr, size_t size)
    {
        memcpy(m_ptr, h_ptr, size * sizeof(T));
        // CHECK_CUDA(cudaMemcpy(m_ptr, h_ptr, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    inline void loadFromDevice(const T *d_ptr, size_t size)
    {
        CHECK_CUDA(cudaMemcpy(m_ptr, d_ptr, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    inline void copyToHost(T *h_ptr, size_t size) const
    {
        CHECK_CUDA(cudaMemcpy(h_ptr, m_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /**
     * @brief resize Resizes the memory.
     * @param size New size in elements (not bytes!)
     */
    inline void resize(size_t size)
    {
        free();
        m_ptr = UNIFIED_MALLOC_TYPED(T, size);
    }

    /**
     * @brief Transfers ownership pf ptr to this.
     */
    inline void set(T *ptr)
    {
        free();
        m_ptr = ptr;
    }

    /**
     * @brief Returns a raw pointer _without_ transfering ownership
     */
    inline T *get() { return m_ptr; }
    inline const T *get() const { return m_ptr; }

    /**
     * @brief free's the owned memory, if any.
     */
    inline void free()
    {
        cut::safeFreeUnified(m_ptr);
        m_ptr = nullptr;
    }

    /**
     * @brief release Returns the pointer, without freeing it and thereby transfering ownership to the caller.
     */
    inline T *release()
    {
        auto res = m_ptr;
        m_ptr = nullptr;
        return res;
    }
};

/**
 * Similar to unique_ptr, but for pitched CUDA device memory.
 * @note This is currently not used and should be tested...
 *
 *  NOT thread safe!
 */
template <typename T> struct dev_pitched_ptr
{
    cudaPitchedPtr m_pptr;

    inline explicit dev_pitched_ptr(uint32_t xdim, uint32_t ydim, uint32_t zdim = 1) { resize(xdim, ydim, zdim); }

    inline ~dev_pitched_ptr()
    {
        if (m_pptr.ptr != nullptr)
            CHECK_CUDA(cudaFree(m_pptr.ptr));
    }

    dev_pitched_ptr(const dev_pitched_ptr &rhs) = delete;
    dev_pitched_ptr &operator=(const dev_pitched_ptr &rhs) = delete;

    inline void resize(uint32_t x, uint32_t y, uint32_t z = 1)
    {
        cudaExtent extent = make_cudaExtent(x * sizeof(T), y, z);
        CHECK_CUDA(cudaMalloc3D(&m_pptr, extent));
    }

    inline void loadFromHost(T *h_ptr, uint32_t x, uint32_t y, uint32_t z = 1)
    {
        // TODO check if this is actually working...

        cudaMemcpy3DParms params;
        memset(&params, 0, sizeof(params));
        params.srcPtr = make_cudaPitchedPtr(h_ptr, sizeof(T) * x, sizeof(T) * x, y);
        params.dstPtr = m_pptr;
        params.extent.width = sizeof(T) * x;
        params.extent.height = y;
        params.extent.depth = z;
        params.kind = cudaMemcpyDeviceToHost;

        CHECK_CUDA(cudaMemcpy3D(&params));
    }

    inline void copyToHost(T *h_ptr, uint32_t x, uint32_t y)
    {
        CHECK_CUDA(
            cudaMemcpy2D(h_ptr, sizeof(T) * x, m_pptr.ptr, m_pptr.pitch, sizeof(T) * x, y, cudaMemcpyDeviceToHost));
    }

    inline void copyToHost(T *h_ptr, uint32_t x, uint32_t y, uint32_t z)
    {
        // TODO check if this is actually working...

        cudaMemcpy3DParms params;
        memset(&params, 0, sizeof(params));
        params.srcPtr = m_pptr;
        params.dstPtr = make_cudaPitchedPtr(h_ptr, sizeof(T) * x, sizeof(T) * x, y);
        params.extent.width = sizeof(T) * x;
        params.extent.height = y;
        params.extent.depth = z;
        params.kind = cudaMemcpyHostToDevice;

        CHECK_CUDA(cudaMemcpy3D(&params));
    }

    inline size_t getPitch() const { return m_pptr.pitch; }

    inline T *get() { return static_cast<T *>(m_pptr.ptr); }
    inline const T *get() const { return static_cast<const T *>(m_pptr.ptr); }

    inline cudaPitchedPtr getPitchedPtr() const { return m_pptr; }

    /**
     * @brief release Returns the pointer, without freeing it and thereby transfering ownership to the caller.
     */
    inline void *release()
    {
        auto res = m_pptr.ptr;
        m_pptr.ptr = nullptr;
        return res;
    }
};

} // namespace cut

#endif // MEMORY_H
