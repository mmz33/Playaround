#include <iostream>

// playaround with operator overloading in C++
// there is a difference between member and global operator overloading
// depending on the left operator, we decide how to implement it. E.g:
// If we have something like 1 + v then we would need operator + overloading
// with parameter (int, obj). Thus, we can just use the member operator +
// overloading function: operator+(const obj& v). We would need to implement
// that as global operator overloading function

template<typename T>
class vec2 {
public:
  T e[2];

  vec2(T x, T y) : e{x, y} {}

  vec2<T> operator-() const {
    return vec2<T>(-e[0], -e[1]);
  }

  vec2<T>& operator+=(T val) {
    e[0] += val;
    e[1] += val;
    return *this;
  }

  vec2<T>& operator+=(const vec2<T>& other) {
    e[0] += other.e[0];
    e[1] += other.e[1];
    return *this;
  }

  vec2<T> operator+(const vec2<T>& other) const {
    std::cout << "calling member operator+ overload function\n";
    return vec2<T>(e[0] + other.e[0], e[1] + other.e[1]);
  }

  vec2<T> operator+(const T val) const {
    return vec2<T>(e[0] + val, e[1] + val);
  }

  T operator[](T i) const {
    return e[i];
  }

  T& operator[](T i) {
    return e[i];
  }

};

// needed global overloading functions

template<typename T>
std::ostream& operator<<(std::ostream& os, const vec2<T>& v) {
  os << '[' << v.e[0] << ", " << v.e[1] << ']';
  return os;
}

template<typename T>
vec2<T> operator+(const vec2<T>& u, const vec2<T>& v) {
  std::cout << "calling global operator+ overload function\n";
  return vec2<T>(u[0] + v[0], u[1] + v[1]);
}

template<typename T>
vec2<T> operator+(const int val, const vec2<T>& v) {
  std::cout << "calling global operator+ overload function\n";
  return vec2<T>(v.e[0] + val, v.e[1] + val); 
}

int main() {
  vec2<int> v {1, 2};
  std::cout << v << '\n';
  auto neg_v = -v;
  std::cout << neg_v << '\n';
  v += 10;
  std::cout << v << '\n';
  v += neg_v;
  std::cout << v << '\n';
  auto res = v + v;
  std::cout << res << '\n';
  res = 1 + res;
  std::cout << res << '\n';
  res = res + 1;
  std::cout << res << '\n';
  return 0;
}
