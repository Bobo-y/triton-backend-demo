#pragma once
#include <cfloat>
#include <limits>
#include <memory>
#include <vector>
#include <type_traits>

template <typename T>
struct Shape2D {
  Shape2D(T w = 0, T h = 0) : width(w), height(h) {}
  T width = 0;
  T height = 0;
};

typedef Shape2D<int> Shape2DI;


template <class T>
struct Rect {
  T l = 0;  // left
  T t = 0;  // top
  T r = 0;  // right
  T b = 0;  // bottom

  Rect(T _l = 0, T _t = 0, T _r = 0, T _b = 0) : l{_l}, t{_t}, r{_r}, b{_b} {}

  T GetWidth() const { return r - l; }

  T GetHeight() const { return b - t; }
};

using RectI = Rect<int>;

template <typename T>
struct Point3D {
  T x = 0;
  T y = 0;
  T z = 0;
  Point3D() : x(T(0)), y(T(0)), z(T(0)){};
  Point3D(T _x, T _y, T _z) : x(_x), y(_y), z(_z){};
};

using Point3DF = Point3D<float>;