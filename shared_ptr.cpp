#include <iostream>
#include <ostream>

// Implementing the concept of std::shared_ptr. The idea is to create a counter
// object to keep track of the pointers pointing to the target object and
// make sure to destroy it when all pointers are destroyed

class Counter {
private:
  unsigned int count;
public:
  Counter() : count(0) {};
  Counter(const Counter&) = delete;
  Counter& operator=(const Counter&) = delete;

  ~Counter() {}

  void reset() { count = 0; }

  unsigned int get() { return count; }

  void operator++() { count++; }

  void operator--() { count--; }

  friend std::ostream& operator<<(std::ostream& os, const Counter& counter) {
    os << "Count value: " << counter.count << std::endl;
    return os;
  }
};

template<typename T>
class Shared_ptr {
private:
  Counter* counter;
  T* shared_ptr;
public:
  explicit Shared_ptr(T* ptr) {
    shared_ptr = ptr;
    counter = new Counter();
    if (ptr)
      ++(*counter);
  }

  // copy constructor
  Shared_ptr(Shared_ptr<T>& ptr) {
    shared_ptr = ptr.shared_ptr;
    counter = ptr.counter;
    ++(*counter);
  }

  T* get() { return shared_ptr; }

  // Destructor
  ~Shared_ptr() {
    --(*counter);
    if ((*counter).get() == 0) {
      delete counter;
      delete shared_ptr;
    }
  }

  friend std::ostream& operator<<(std::ostream& os, Shared_ptr<T>& ptr) {
    os << "Address pointer: " << ptr.get() << std::endl;
    std::cout << *(ptr.counter) << std::endl;
    return os;
  }
};

int main() {
  Shared_ptr<int> ptr1(new int(5));
  std::cout << "--- ptr1 ---\n";
  std::cout << ptr1;
  {
    Shared_ptr<int> ptr2 = ptr1;
    std::cout << "--- ptr1, ptr2 ---\n";
    std::cout << ptr1;
    std::cout << ptr2;

    {
      Shared_ptr<int> ptr3(ptr2);
      std::cout << "--- ptr1, ptr2, ptr3 ---\n";
      std::cout << ptr1;
      std::cout << ptr2;
      std::cout << ptr3;
    }

    std::cout << "--- ptr3 out of scope. ptr1, ptr2 ---\n";
    std::cout << ptr1;
    std::cout << ptr2;
  }

  std::cout << "--- ptr2, ptr3 out of scope. ptr1 ---\n";
  std::cout << ptr1;

  return 0;
}
