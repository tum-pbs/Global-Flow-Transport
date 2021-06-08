#pragma once

#ifndef _INCLUDE_RENDER_ERRORS
#define _INCLUDE_RENDER_ERRORS

//#include <exception>
#include <stdexcept>
#include <sstream>
//#include <string>

namespace RenderError{

class RenderInternalError : public std::logic_error{
public:
	RenderInternalError(const std::string& msg): std::logic_error(msg){}
};

class RenderError : public std::runtime_error{
public:
	RenderError(const std::string& msg): std::runtime_error(msg){}
};


class CudaError : public RenderError{
public:
	CudaError(const std::string& msg): RenderError(msg){}
};

//https://stackoverflow.com/questions/12261915/how-to-throw-stdexceptions-with-variable-messages
class Formatter
{
public:
    Formatter() {}
    ~Formatter() {}

    template <typename Type>
    Formatter & operator << (const Type & value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const         { return stream_.str(); }
    operator std::string () const   { return stream_.str(); }

    enum ConvertToString 
    {
        to_str
    };
    std::string operator >> (ConvertToString) { return stream_.str(); }

private:
    std::stringstream stream_;

    Formatter(const Formatter &);
    Formatter & operator = (Formatter &);
};

}

#endif //_INCLUDE_RENDER_ERRORS