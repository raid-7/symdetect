#include "symdetect.hpp"

#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <numeric>


namespace symdetect {

float SquareWithCircles::sideLength() const {
    float len = 0.;
    for (int i = 0; i < square.size(); ++i) {
        int j = (i + 1) % 4;
        len += geom::Segment(square[i], square[j]).length();
    }
    return len / 4;
}

cv::Point2f SquareWithCircles::center() const {
    auto sum = std::accumulate(square.begin(), square.end(), cv::Point());
    return cv::Point2f(sum.x, sum.y) / 4;
}


SymbolDetector::SymbolDetector(double t1, double t2, double poly_acc, double circle_acc, bool grayscale_only)
        : t1(t1), t2(t2), poly_acc(poly_acc), circle_acc(circle_acc), grayscale_only(grayscale_only) {}

std::vector<SquareWithCircles>
SymbolDetector::detect(const Mat& source, const cv::_InputOutputArray& dst, bool debug_mode) const {
    // TODO: try out the approach: https://stackoverflow.com/questions/55169645/square-detection-in-image
    ImageStacker dbg;

    auto sz = source.size();
    int thickness = std::max(1, std::min(sz.width, sz.height) / 500);

    Mat filtered = filter(source, sz);
    dbg << filtered;

    Mat can = canny(filtered);
    dbg << can;

    auto contours = findContours(can);
    cv::drawContours(dbg.with(filtered), contours, -1, {0, 0, 255}, thickness);

    auto squares = removeInnerQuads(findSquares(contours, sz));
    cv::drawContours(dbg.with(filtered), squares, -1, {0, 0, 255}, thickness);

    Mat gray;
    cv::cvtColor(filtered, gray, cv::COLOR_BGR2GRAY);
    auto squares_with_circles = squaresWithCircles(gray, squares);
    drawResult(dbg.with(filtered), squares_with_circles, true, thickness);

    if (!debug_mode) {
        filtered.copyTo(dst);
        drawResult(dst, squares_with_circles, false, thickness);
    } else {
        dbg.get().copyTo(dst);
    }

    sortSquares(squares_with_circles);
    return squares_with_circles;
}

void SymbolDetector::drawResult(const cv::_InputOutputArray& dst, const std::vector<SquareWithCircles>& data,
                                bool paint_circles, int thickness) {
    std::vector<Contour> squares;
    std::transform(data.begin(), data.end(), std::back_inserter(squares), [](const auto& sq) {
        return sq.square;
    });
    cv::drawContours(dst, squares, -1, {0, 255, 0}, thickness);

    if (paint_circles) {
        for (const auto& sq : data) {
            drawCircles(dst, sq.circles, {0, 0}, thickness);
        }
    }
}

Mat SymbolDetector::filter(const Mat& source, const cv::Size& target_size) {
    Mat tmp = source, res;
    cv::GaussianBlur(tmp, res, {7, 7}, 0);
    tmp = res;
    cv::resize(tmp, res, target_size, 0, 0, cv::INTER_CUBIC);
    return res;
}

Mat SymbolDetector::canny(const Mat& source) const {
    Mat can;
    if (grayscale_only) {
        Mat grey;
        cv::cvtColor(source, grey, cv::COLOR_BGR2GRAY);
        cv::Canny(grey, can, t1, t2);
    } else {
        auto grey = cv::Mat(source.size(), CV_8U);
        for (int c = 0; c < 3; c++) {
            int ch[] = {c, 0};
            cv::mixChannels(&source, 1, &grey, 1, ch, 1);
            Mat t_can;
            cv::Canny(grey, t_can, t1, t2);
            if (c == 0) {
                can = t_can;
            } else {
                can = cv::max(can, t_can);
            }
        }
    }
    cv::dilate(can, can, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    return can;
}

bool SymbolDetector::isQuadSquare(const Contour& quad) {
    std::vector<int> sq_lengths;
    for (int i = 0; i < 4; ++i) {
        int j = (i + 1) % 4;
        int k = (i + 2) % 4;
        geom::Segment u = {quad[j], quad[i]};
        geom::Segment v = {quad[j], quad[k]};

        double ang = geom::Segment::angleDeg(u, v);
        if (ang < square_angle_min || ang > square_angle_max) {
            return false;
        }

        sq_lengths.push_back(u.lengthSq());
    }

    std::sort(sq_lengths.begin(), sq_lengths.end());
    double sq_ratio = static_cast<double>(sq_lengths.back()) / sq_lengths.front();
    if (sq_ratio > square_side_ratio_max * square_side_ratio_max) {
        return false;
    }

    return true;
}

Mat SymbolDetector::getContourRegionSlice(const Contour& contour, const Mat& source) {
    auto rect = cv::boundingRect(contour);
    return source(rect);
}

bool SymbolDetector::isQuad(const Contour& contour, const cv::Size& im_size) {
    int im_area = im_size.width * im_size.height;
    int ref_area = std::max(im_area / 128, 256);
    return contour.size() == 4 &&
           cv::isContourConvex(contour) &&
           cv::contourArea(contour, false) > ref_area;
}

std::vector<Contour> SymbolDetector::findContours(const Mat& source) {
    std::vector<Contour> res;
    cv::findContours(source, res, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    return res;
}

bool SymbolDetector::isInside(const Contour& inner, const Contour& outer) {
    for (const auto& point : inner) {
        if (cv::pointPolygonTest(outer, point, false) < 0.)
            return false;
    }
    return true;
}

std::vector<Contour> SymbolDetector::removeInnerQuads(const std::vector<Contour>& quads) {
    std::vector<Point> centers;
    std::transform(quads.begin(), quads.end(), std::back_inserter(centers), [](const auto& q) {
        return std::accumulate(q.begin(), q.end(), Point()) / 4;
    });

    std::vector<Contour> result;
    for (int i = 0; i < quads.size(); ++i) {
        bool largest = true;
        const auto& cur = quads[i];
        for (int j = 0; j < quads.size(); ++j) {
            if (i == j)
                continue;
            const auto& superior_candidate = quads[j];
            if (isInside(cur, superior_candidate)) {
                largest = false;
                break;
            }
        }

        if (largest)
            result.push_back(cur);
    }

    return result;
}

std::vector<Contour> SymbolDetector::findSquares(const std::vector<Contour>& contours, const cv::Size& im_size) const {
    std::vector<Contour> quads;
    for (const Contour& contour : contours) {
        Contour appx;
        cv::approxPolyDP(contour, appx, cv::arcLength(contour, true) * poly_acc, true);

        if (isQuad(appx, im_size) && isQuadSquare(appx)) {
            quads.emplace_back(std::move(appx));
        }
    }

    return quads;
}

void SymbolDetector::drawCircles(const cv::_InputOutputArray& img, const std::vector<geom::Circle>& circles,
                                 const Point& offset, int thickness) {
    for (const auto& c : circles) {
        cv::circle(img, c.iCenter(), c.iRadius(), {0, 255, 255}, thickness);
    }
}

std::vector<geom::Circle> SymbolDetector::findCircles(const Mat& source) const {
    auto sz = source.size();
    int max_radius = std::min(sz.width, sz.height) / 2;
    int min_radius = std::max(max_radius / 50, 7);
    int min_dist = min_radius * 2;

    std::vector<cv::Vec3f> h_circles;
    cv::HoughCircles(source, h_circles, cv::HOUGH_GRADIENT_ALT,
                     1.5, min_dist, 300, circle_acc, min_radius, max_radius);

    std::vector<geom::Circle> circles;
    std::transform(h_circles.begin(), h_circles.end(), std::back_inserter(circles), [](auto v) {
        return geom::Circle(v);
    });
    return circles;
}

std::vector<SquareWithCircles>
SymbolDetector::squaresWithCircles(const Mat& source, const std::vector<Contour>& squares) const {
    std::vector<SquareWithCircles> result;
    for (const auto& square : squares) {
        Mat slice = getContourRegionSlice(square, source);
        auto offset = cv::boundingRect(square).tl();
        auto circles = findCircles(slice);
        for (auto& circle : circles) {
            circle = circle.move(offset);
        }

        if (!circles.empty()) {
            result.push_back({square, std::move(circles)});
        }
    }
    return result;
}

float SymbolDetector::sortingParameter(const SquareWithCircles& sq) {
    float side_len = sq.sideLength();
    auto sq_center = sq.center();
    float min_dist = std::numeric_limits<float>::infinity();
    for (const auto& circle : sq.circles) {
        auto c_center = circle.center();
        auto l = sq_center - c_center;
        float dist = std::sqrt(l.dot(l));
        min_dist = std::min(min_dist, dist);
    }
    return min_dist / side_len;
}

void SymbolDetector::sortSquares(std::vector<SquareWithCircles>& squares) {
    std::sort(squares.begin(), squares.end(), [](const auto& a, const auto& b) {
        return sortingParameter(a) < sortingParameter(b);
    });
}

geom::Circle::Circle(cv::Vec3f c) : x(c[0]), y(c[1]), radius(c[2]) {}

geom::Circle::Circle(float x, float y, float radius) : x(x), y(y), radius(radius) {}

Point geom::Circle::iCenter() const {
    return {static_cast<int>(std::round(x)), static_cast<int>(std::round(y))};
}

int geom::Circle::iRadius() const {
    return static_cast<int>(std::round(radius));
}

cv::Point2f geom::Circle::center() const {
    return {x, y};
}

geom::Circle geom::Circle::move(const cv::Point2f& offset) const {
    return {x + offset.x, y + offset.y, radius};
}
}