// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_BINDINGS_UTILS_HPP
#define OPENCV_CORE_BINDINGS_UTILS_HPP

#include <opencv2/core/async.hpp>
#include <opencv2/core/detail/async_promise.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <stdexcept>

namespace cv { namespace utils {
//! @addtogroup core_utils
//! @{

String dumpInputArray(InputArray argument);

String dumpInputArrayOfArrays(InputArrayOfArrays argument);

String dumpInputOutputArray(InputOutputArray argument);

String dumpInputOutputArrayOfArrays(InputOutputArrayOfArrays argument);

static inline
String dumpBool(bool argument)
{
    return (argument) ? String("Bool: True") : String("Bool: False");
}

static inline
String dumpInt(int argument)
{
    return cv::format("Int: %d", argument);
}

static inline
String dumpInt64(int64 argument)
{
    std::ostringstream oss("Int64: ", std::ios::ate);
    oss << argument;
    return oss.str();
}

static inline
String dumpSizeT(size_t argument)
{
    std::ostringstream oss("size_t: ", std::ios::ate);
    oss << argument;
    return oss.str();
}

static inline
String dumpFloat(float argument)
{
    return cv::format("Float: %.2f", argument);
}

static inline
String dumpDouble(double argument)
{
    return cv::format("Double: %.2f", argument);
}

static inline
String dumpCString(const char* argument)
{
    return cv::format("String: %s", argument);
}

static inline
String dumpString(const String& argument)
{
    return cv::format("String: %s", argument.c_str());
}

static inline
String testOverloadResolution(int value, const Point& point = Point(42, 24))
{
    return format("overload (int=%d, point=(x=%d, y=%d))", value, point.x,
                  point.y);
}

static inline
String testOverloadResolution(const Rect& rect)
{
    return format("overload (rect=(x=%d, y=%d, w=%d, h=%d))", rect.x, rect.y,
                  rect.width, rect.height);
}

static inline
String dumpRect(const Rect& argument)
{
    return format("rect: (x=%d, y=%d, w=%d, h=%d)", argument.x, argument.y,
                  argument.width, argument.height);
}

static inline
String dumpTermCriteria(const TermCriteria& argument)
{
    return format("term_criteria: (type=%d, max_count=%d, epsilon=%lf",
                  argument.type, argument.maxCount, argument.epsilon);
}

static inline
String dumpRotatedRect(const RotatedRect& argument)
{
    return format("rotated_rect: (c_x=%f, c_y=%f, w=%f, h=%f, a=%f)",
                  argument.center.x, argument.center.y, argument.size.width,
                  argument.size.height, argument.angle);
}

static inline
RotatedRect testRotatedRect(float x, float y, float w, float h, float angle)
{
    return RotatedRect(Point2f(x, y), Size2f(w, h), angle);
}

static inline
std::vector<RotatedRect> testRotatedRectVector(float x, float y, float w, float h, float angle)
{
    std::vector<RotatedRect> result;
    for (int i = 0; i < 10; i++)
        result.push_back(RotatedRect(Point2f(x + i, y + 2 * i), Size2f(w, h), angle + 10 * i));
    return result;
}

static inline
String dumpRange(const Range& argument)
{
    if (argument == Range::all())
    {
        return "range: all";
    }
    else
    {
        return format("range: (s=%d, e=%d)", argument.start, argument.end);
    }
}

static inline
int testOverwriteNativeMethod(int argument)
{
    return argument;
}

static inline
String testReservedKeywordConversion(int positional_argument, int lambda = 2, int from = 3)
{
    return format("arg=%d, lambda=%d, from=%d", positional_argument, lambda, from);
}

String dumpVectorOfInt(const std::vector<int>& vec);

String dumpVectorOfDouble(const std::vector<double>& vec);

String dumpVectorOfRect(const std::vector<Rect>& vec);

static inline
void generateVectorOfRect(size_t len, CV_OUT std::vector<Rect>& vec)
{
    vec.resize(len);
    if (len > 0)
    {
        RNG rng(12345);
        Mat tmp(static_cast<int>(len), 1, CV_32SC4);
        rng.fill(tmp, RNG::UNIFORM, 10, 20);
        tmp.copyTo(vec);
    }
}

static inline
void generateVectorOfInt(size_t len, CV_OUT std::vector<int>& vec)
{
    vec.resize(len);
    if (len > 0)
    {
        RNG rng(554433);
        Mat tmp(static_cast<int>(len), 1, CV_32SC1);
        rng.fill(tmp, RNG::UNIFORM, -10, 10);
        tmp.copyTo(vec);
    }
}

static inline
void generateVectorOfMat(size_t len, int rows, int cols, int dtype, CV_OUT std::vector<Mat>& vec)
{
    vec.resize(len);
    if (len > 0)
    {
        RNG rng(65431);
        for (size_t i = 0; i < len; ++i)
        {
            vec[i].create(rows, cols, dtype);
            rng.fill(vec[i], RNG::UNIFORM, 0, 10);
        }
    }
}

static inline
void testRaiseGeneralException()
{
    throw std::runtime_error("exception text");
}

static inline
AsyncArray testAsyncArray(InputArray argument)
{
    AsyncPromise p;
    p.setValue(argument);
    return p.getArrayResult();
}

static inline
AsyncArray testAsyncException()
{
    AsyncPromise p;
    try
    {
        CV_Error(Error::StsOk, "Test: Generated async error");
    }
    catch (const cv::Exception& e)
    {
        p.setException(e);
    }
    return p.getArrayResult();
}

static inline
String dumpVec2i(const cv::Vec2i value = cv::Vec2i(42, 24)) {
    return format("Vec2i(%d, %d)", value[0], value[1]);
}

struct CV_EXPORTS_W_SIMPLE ClassWithKeywordProperties {
    CV_PROP_RW int lambda;
    CV_PROP int except;

    explicit ClassWithKeywordProperties(int lambda_arg = 24, int except_arg = 42)
    {
        lambda = lambda_arg;
        except = except_arg;
    }
};

struct CV_EXPORTS_W_PARAMS FunctionParams
{
    CV_PROP_RW int lambda = -1;
    CV_PROP_RW float sigma = 0.0f;

    FunctionParams& setLambda(int value) CV_NOEXCEPT
    {
        lambda = value;
        return *this;
    }

    FunctionParams& setSigma(float value) CV_NOEXCEPT
    {
        sigma = value;
        return *this;
    }
};

static inline String
copyMatAndDumpNamedArguments(InputArray src, OutputArray dst,
                             const FunctionParams& params = FunctionParams())
{
    src.copyTo(dst);
    return format("lambda=%d, sigma=%.1f", params.lambda,
                  params.sigma);
}

namespace nested {
static inline bool testEchoBooleanFunction(bool flag) {
    return flag;
}

class CV_WRAP_AS(ExportClassName) OriginalClassName
{
public:
    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_PROP_RW int int_value;
        CV_PROP_RW float float_value;

        explicit Params(int int_param = 123, float float_param = 3.5f)
        {
            int_value = int_param;
            float_value = float_param;
        }
    };

    explicit OriginalClassName(const OriginalClassName::Params& params = OriginalClassName::Params())
    {
        params_ = params;
    }

    int getIntParam() const
    {
        return params_.int_value;
    }

    float getFloatParam() const
    {
        return params_.float_value;
    }

    static std::string originalName()
    {
        return "OriginalClassName";
    }

    static Ptr<OriginalClassName>
    create(const OriginalClassName::Params& params = OriginalClassName::Params())
    {
        return makePtr<OriginalClassName>(params);
    }

private:
    OriginalClassName::Params params_;
};

typedef OriginalClassName::Params OriginalClassName_Params;
} // namespace nested

namespace fs {
    cv::String getCacheDirectoryForDownloads();
} // namespace fs

//! @}  // core_utils
}  // namespace cv::utils

//! @cond IGNORED

static inline
int setLogLevel(int level)
{
    // NB: Binding generators doesn't work with enums properly yet, so we define separate overload here
    return cv::utils::logging::setLogLevel((cv::utils::logging::LogLevel)level);
}

static inline
int getLogLevel()
{
    return cv::utils::logging::getLogLevel();
}

//! @endcond IGNORED

} // namespaces cv /  utils

#endif // OPENCV_CORE_BINDINGS_UTILS_HPP
