#include <stdlib.h>
#include <sys/mman.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

template <typename Lambda>
long ms(Lambda &&f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

template <int sz>
class mmap_vec {
   public:
    mmap_vec() = default;

    void init(const std::size_t size) {
        this->size = size;
        data = static_cast<int *>(mmap(
            nullptr, sizeof(int) * size, PROT_READ | PROT_WRITE,
            MAP_HUGETLB | (sz << MAP_HUGE_SHIFT) | MAP_PRIVATE | MAP_ANONYMOUS,
            -1, 0));
    }

    void free() {
        __asm__ __volatile__("" : : "r,m"(data) :);
        munmap(data, sizeof(int) * size);
    }

    int &operator[](std::size_t i) { return data[i]; }

    std::size_t sum() {
        std::size_t ans = 0;
#pragma omp parallel for reduction(+ : ans)
        for (std::size_t i = 0; i < size; ++i) {
            ans += static_cast<std::size_t>(data[i]);
        }
        return ans;
    }

    bool ok() { return data != MAP_FAILED; }

   private:
    std::size_t size = 0;
    int *data = nullptr;
};

class hp_vec {
   public:
    hp_vec() = default;

    void init(const std::size_t size) {
        this->size = size;
        posix_memalign(reinterpret_cast<void **>(&data), (1 << 21) * 1,
                       size * sizeof(int));
        madvise(data, sizeof(int) * size, MADV_HUGEPAGE);
    }

    void free() {
        __asm__ __volatile__("" : : "r,m"(data) :);
        std::free(data);
    }

    int &operator[](std::size_t i) { return data[i]; }

    std::size_t sum() {
        std::size_t ans = 0;
#pragma omp parallel for reduction(+ : ans)
        for (std::size_t i = 0; i < size; ++i) {
            ans += static_cast<std::size_t>(data[i]);
        }
        return ans;
    }

    bool ok() { return data != nullptr; }

   private:
    std::size_t size = 0;
    int *data = nullptr;
};

class vec {
   public:
    vec() = default;

    void init(const std::size_t size) {
        this->size = size;
        this->data = static_cast<int *>(malloc(sizeof(int) * size));
    }

    void free() {
        __asm__ __volatile__("" : : "r,m"(data) :);
        ::free(data);
    }

    int &operator[](std::size_t i) { return data[i]; }

    std::size_t sum() {
        std::size_t ans = 0;
#pragma omp parallel for reduction(+ : ans)
        for (std::size_t i = 0; i < size; ++i) {
            ans += static_cast<std::size_t>(data[i]);
        }
        return ans;
    }

    bool ok() { return data != nullptr; }

   private:
    std::size_t size = 0;
    int *data = nullptr;
};

template <typename T>
void run_bench(const std::string &name) {
    constexpr std::size_t size = 16ul * 1024 * 1024 * 1024;

    T v;

    long t_alloc = ms([&] { v.init(size); });
    if (!v.ok()) {
        std::cout << name << " failed to allocate memory" << std::endl;
        std::cout << std::endl;
        return;
    }

    long t_touch = ms([&] {
#pragma omp parallel for
        for (std::size_t i = 0; i < size; ++i) {
            v[i] = static_cast<int>(i);
        }
    });
    long t_sum =
        ms([&] { std::cout << name << " sum: " << v.sum() << std::endl; });
    long t_free = ms([&] { v.free(); });

    std::cout << name << " time alloc: " << t_alloc << " ms" << std::endl;
    std::cout << name << " time touch: " << t_touch << " ms" << std::endl;
    std::cout << name << " time sum: " << t_sum << " ms" << std::endl;
    std::cout << name << " time free: " << t_free << " ms" << std::endl;
    std::cout << std::endl;
}
int main() {
    long t_thp = ms([&] { run_bench<hp_vec>("thp"); });
    long t_2mb_hp = ms([&] { run_bench<mmap_vec<21>>("2mb_hp"); });
    long t_1gb_hp = ms([&] { run_bench<mmap_vec<30>>("1gb_hp"); });
    long t_malloc = ms([&] { run_bench<vec>("malloc"); });

    std::cout << "time thp: " << t_thp << " ms" << std::endl;
    std::cout << "time 2mb_hp: " << t_2mb_hp << " ms" << std::endl;
    std::cout << "time 1gb_hp: " << t_1gb_hp << " ms" << std::endl;
    std::cout << "time malloc: " << t_malloc << " ms" << std::endl;
    return 0;
}
