#include "../include/HostPools.hpp"
#include <future>
#include <mutex>
#include <thread>
#include <utility>

ThreadManager::workerLoop()
{
        std::packaged_task<void()> task;
        while ()
        {
                {
                        std::unique_lock lk(m_mtx);
                        cv.wait(lk, [this] { return stop || !m_taskQ.empty(); });

                        if (stop && !m_taskQ.empty())
                        {
                                std::cerr << "Thread " << std::this_thread::get_id() << "returned successfully!"
                                          << std::endl;
                                return;
                        }
                        task = std::move(m_taskQ.front());
                        m_taskQ.pop();
                }
                task();
        }
}

ThreadManager::ThreadManager()
{
        std::generate(m_workers.begin(), m_workers.end(), &workerLoop);
}

template <typename T> void ThreadManager::enqueue(T&& t)
{
        {
                std::unique_lock lk(m_mtx);
                tasks.emplace(std::forward<T>(t));
        }
        cv.notify_one();
}

~ThreadPool()
{
        {
                std::unique_lock<std::mutex> lock(queue_mutex);
                stop = true;
        }
        condition.notify_all();
}
