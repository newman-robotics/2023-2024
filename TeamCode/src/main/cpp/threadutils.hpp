#ifndef CADCAMP23_THREADUTILS_HPP
#define CADCAMP23_THREADUTILS_HPP

#include <mutex>
#include <atomic>
#include <optional>
#include <functional>

/*
Represents an object that may or may not be present at a given time.
Any thread can access the object, but only one thread can set it. A lock
keeps track of which thread can set the object. Additionally, getting
the object resets its value.
*/
template <typename T> class SuppliedObjectHolder {
    std::atomic<std::optional<T>> object;
    std::mutex mtx;
    std::optional<std::thread *> capture_thread;
    std::atomic<bool> done;

    bool owns(std::unique_lock<std::mutex>& lock) {
        return lock.mutex() == &(this->mtx);
    }

public:
    /*
    Creates a SuppliedObjectHolder with no initial value.
     */
    SuppliedObjectHolder() {
        this->object.store({});
        this->done.store(false);
        this->capture_thread = {};
    }
    /*
    Creates a SuppliedObjectHolder with the given initial value.
     */
    explicit SuppliedObjectHolder(T initialValue) : SuppliedObjectHolder() {
        this->object = initialValue;
    }
    /*
    Destroys the SuppliedObjectHolder.
     */
    ~SuppliedObjectHolder() {
        this->done.store(true);
        if (this->capture_thread) this->capture_thread.value()->join();
    }

    /*
    Locks the held object and returns the lock. The lock is
    required to set the value of the held object.
     */
    std::unique_lock<std::mutex> lock() {
        return std::unique_lock<std::mutex>(this->mtx);
    }

    /*
    Locks the held object using the given lock.
     */
    void lock(std::unique_lock<std::mutex> lock) {
        if (not lock and this->owns(lock)) {
            lock.try_lock();
        }
    }

    /*
    Unlocks the held object. No thread can set the value of
    the held object until this object is locked again.
     */
    void unlock(std::unique_lock<std::mutex> lock) {
        if (lock and this->owns(lock)) {
            lock.unlock();
        }
    }

    /*
    Gets the value of the held object. Does not require the
    thread to own the mutex for this object.
     */
    std::optional<T> get() {
        std::optional<T> out = this->object.load();
        this->object.store({});
        return out;
    }

    /*
    Sets the value of the object if the given lock owns the
    mutex for this object.
     */
    void set(T value, std::unique_lock<std::mutex>& lock) {
        if (lock and this->owns(lock)) {
            this->object.store(value);
        }
    }

    /*
    Waits for the held object to have a value. Should not be
    called from the thread that has the lock to avoid deadlocks.
     */
    void wait() {
        while (not this->object.load()) {}
    }

    /*
    Waits for the held object to have a value, then applies the
    specified function to it. The function should accept the
    held object and return nothing. After the function is done
    being applied, the held object is reset.
     */
    void wait_then(std::function<void(T)> func) {
        this->wait();
        func(this->object.load().value());
        this->object.store({});
    }

    /*
    Sets this object's capture function, which is called in an async
    manner every time a new non-empty value is set to the object
     */
    void async_capture(std::function<void(T)> func, std::function<void(void)> onLoad = []{}, std::function<void(void)> onClose = []{}) {
        this->capture_thread = new std::thread([this, func, onLoad, onClose](){
            onLoad();
            while (not this->done) {
                std::optional<T> val;
                if ((val = this->object.load()).has_value()) {
                    func(val.value());
                    this->object.store({});
                }
            }
            onClose();
        });
    }
};

/*To be documented later.*/
template <typename I, typename O> class ChainedAsyncFunction {
    std::thread * func_thread;
    std::atomic<bool> stop;
    bool is_first;
    std::optional<I> last_input;

public:
    std::optional<O> last_output;

    explicit ChainedAsyncFunction(std::function<O(I)> func) {
        this->is_first = true;
        this->stop.store(false);
        this->func_thread = new std::thread([this, func](){
            while (not stop) {
                if (this->last_input) {
                    this->last_output = func(this->last_input);
                }
            }
        });
    }
    ChainedAsyncFunction(std::function<O(I)> func, ChainedAsyncFunction last_func) {
        this->is_first = false;
        this->stop.store(false);
        this->func_thread = new std::thread([this, func, last_func](){
            while (not this->stop) {
                if (last_func.last_output) {
                    this->last_output = func(last_func.last_output);
                }
            }
        });
    }
    ~ChainedAsyncFunction() {
        this->stop.store(true);
        this->func_thread.join();
    }

    void operator()(I input) {
        this->last_input = input;
    }
};

#endif //CADCAMP23_THREADUTILS_HPP