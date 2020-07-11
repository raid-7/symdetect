#include "symdetect.hpp"

namespace symdetect::geom {

geom::Segment::Segment(Point a, Point b) : a(std::move(a)), b(std::move(b)) {}

int geom::Segment::lengthSq() const {
    auto l = a - b;
    return l.dot(l);
}

float Segment::length() const {
    return std::sqrt(static_cast<float>(lengthSq()));
}

double Segment::angleDeg(const Segment& u, const Segment& v) {
    auto l1 = u.a - u.b;
    auto l2 = v.a - v.b;
    double cos = l1.ddot(l2) / std::sqrt(l1.ddot(l1) * l2.ddot(l2));
    return std::acos(cos) * 180. / M_PI;
}

}