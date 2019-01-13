#include <chrono>
using namespace std::chrono;

class Timer {
    public:
        void start() {
            ticks[0] = high_resolution_clock::now();
        }

        long long finish_and_get_us() {
            ticks[1] = high_resolution_clock::now();
            auto time_span = duration_cast<microseconds>(ticks[1] - ticks[0]);
            long long us_span = time_span.count();
            return us_span;
        }

    private:
        high_resolution_clock::time_point ticks[2];
};