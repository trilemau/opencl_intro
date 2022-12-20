#include <string>
#include <iostream>
#include <vector>

#include "cl.hpp"

int main()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::cout << "Platform=" << platforms.size() << "\n";


    if (platforms.size() != 1)
    {
        return EXIT_FAILURE;
    }

    auto platform = platforms.front();
    auto platform_info = platform.getInfo<CL_PLATFORM_NAME>();

    std::cout << "Platform info=" << platform_info << "\n";

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    std::cout << "devices=" << devices.size() << "\n";

    if (devices.size() != 1)
    {
        return EXIT_FAILURE;
    }

    auto device = devices.front();
    auto device_info = device.getInfo<CL_DEVICE_VENDOR>();

    std::cout << "device=" << device_info << "\n";

    std::string hello = R"(
__kernel void hello(__global char* data)
{
    data[0] = 'H';
    data[1] = 'E';
    data[2] = 'L';
    data[3] = 'L';
    data[4] = 'O';
}
)";

    std::string array = R"(
__kernel void array(__global int* data, __global int* output)
{
    output[get_global_id(0)] = data[get_globa_id(0)] * 2;
}
)";

    cl::Program::Sources sources(1, std::make_pair(hello.c_str(), hello.size() + 1));
    cl::Context context(device);
    cl::Program program(context, sources);

    auto error = program.build(devices);

    std::cout << "build=" << error << "\n";

    std::string buffer;
    buffer.resize(5);
    cl::Buffer memBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buffer));
    cl::Kernel kernel(program, "hello", &error);
    std::cout << "kernel=" << error << "\n";

    error = kernel.setArg(0, memBuf);
    std::cout << "kernel_setArg=" << error << "\n";

    cl::CommandQueue queue(context, device);
    error = queue.enqueueTask(kernel);
    std::cout << "enqueueTask=" << error << "\n";

    error = queue.enqueueReadBuffer(memBuf, CL_TRUE, 0, sizeof(buffer), (void*) buffer.c_str());
    std::cout << "enqueueReadBuffer=" << error << "\n";

    std::cout << buffer << "\n";

	return EXIT_SUCCESS;
}
