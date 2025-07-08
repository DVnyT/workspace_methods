#pragma once

#include <atomic>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>

constexpr threadCount = 1;

class ThreadManager
{
private:
        std::array<std::jthread, threadCount> m_workers;
	std::mutex m_mtx;
	std::queue<std::packaged_task<void()>> m_taskQ;
	std::condition_variable m_cv;
	bool stop = false;

	void workerLoop();
public:
	ThreadManager();
};

class Reaper
{
};
